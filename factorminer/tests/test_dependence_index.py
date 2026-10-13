"""The dependence index must reproduce reference Spearman values bit for bit."""

from __future__ import annotations

import copy
import gc
import pickle

import numpy as np
import pytest

from factorminer.application.validation_pipeline import ValidationPipeline
from factorminer.core.factor_library import FactorLibrary
from factorminer.core.parser import parse
from factorminer.domain.dependence import PearsonDependenceMetric, SpearmanDependenceMetric
from factorminer.evaluation.dependence_index import (
    DependenceIndex,
    IndexedSpearmanMetric,
    _immutable,
    indexed_dependence_metric,
)
from factorminer.evaluation.runtime import compute_tree_signals

REFERENCE = SpearmanDependenceMetric()


def _signals(rng, shape, *, nan_rate, ties, dtype=np.float64):
    values = rng.normal(size=shape)
    if ties:
        values = np.round(values * 2) / 2
    values[rng.random(shape) < nan_rate] = np.nan
    return values.astype(dtype)


CASES = [
    # (assets, periods, nan_rate, ties, dtype)
    (12, 300, 0.0, False, np.float64),
    (12, 300, 0.2, False, np.float64),
    (9, 257, 0.35, True, np.float64),
    (5, 513, 0.4, True, np.float64),
    (30, 64, 0.1, True, np.float32),
    (4, 40, 0.5, False, np.float64),
]


@pytest.mark.parametrize("assets,periods,nan_rate,ties,dtype", CASES)
def test_index_matches_reference_bit_for_bit(assets, periods, nan_rate, ties, dtype):
    rng = np.random.default_rng(assets * 1000 + periods)
    panels = [_signals(rng, (assets, periods), nan_rate=nan_rate, ties=ties, dtype=dtype)
              for _ in range(6)]
    # Shared masks exercise the prepared-rank path; a halted asset and a
    # sparse period exercise single-usable-column chunks.
    panels.append(np.where(np.isnan(panels[0]), np.nan, -panels[0] * 3))
    panels[1][0, :] = np.nan
    panels[2][:, periods // 2] = np.nan
    panels[2][:3, periods // 2] = 1.0
    index = DependenceIndex()
    for a in panels:
        for b in panels:
            expected = REFERENCE.compute(a, b)
            assert index.compute(a, b) == expected
            assert index.compute(b, a) == expected
    stats = index.stats()
    assert stats["pair_hits"] > 0 and stats["prepare_misses"] == len(panels)
    if nan_rate:
        assert stats["reranked_periods"] > 0


def test_index_cache_is_bounded_and_forgets_collected_arrays():
    rng = np.random.default_rng(0)
    panel_bytes = 10 * 50 * 9  # float64 ranks + bool mask
    index = DependenceIndex(max_prepared_bytes=2 * panel_bytes)
    keep = [_signals(rng, (10, 50), nan_rate=0.1, ties=False) for _ in range(4)]
    for a in keep:
        for b in keep:
            assert index.compute(a, b) == REFERENCE.compute(a, b)
    assert index.stats()["prepared_signals"] == 2
    assert index.stats()["prepared_bytes"] <= 2 * panel_bytes

    temporary = _signals(rng, (10, 50), nan_rate=0.1, ties=False)
    index.compute(temporary, keep[0])
    pairs_before = index.stats()["cached_pairs"]
    del temporary
    gc.collect()
    assert index.stats()["cached_pairs"] == pairs_before - 1


def test_prepared_rank_budget_is_strict_even_for_one_large_panel():
    rng = np.random.default_rng(12)
    a = _signals(rng, (20, 100), nan_rate=0.1, ties=False)
    b = _signals(rng, (20, 100), nan_rate=0.1, ties=False)
    index = DependenceIndex(max_prepared_bytes=100)
    assert index.compute(a, b) == REFERENCE.compute(a, b)
    assert index.stats()["prepared_bytes"] <= 100


def test_indexed_signals_remain_writeable_and_mutations_invalidate_cache():
    rng = np.random.default_rng(1)
    a = _signals(rng, (6, 20), nan_rate=0.0, ties=False)
    b = _signals(rng, (6, 20), nan_rate=0.0, ties=False)
    index = DependenceIndex()
    first = index.compute(a, b)
    assert a.flags.writeable and b.flags.writeable
    a[:, 0] = b[:, 0]
    second = index.compute(a, b)
    assert second == REFERENCE.compute(a, b)
    assert second != first


def test_indexed_metric_is_a_spearman_metric_and_survives_copying():
    metric = indexed_dependence_metric("spearman")
    assert isinstance(metric, IndexedSpearmanMetric)
    assert isinstance(metric, SpearmanDependenceMetric)
    assert metric.name == "spearman" and metric.describe() == REFERENCE.describe()
    assert isinstance(indexed_dependence_metric("pearson"), PearsonDependenceMetric)
    for clone in (copy.deepcopy(metric), pickle.loads(pickle.dumps(metric))):
        assert clone.index is not metric.index
        assert clone.index.stats()["cached_pairs"] == 0


def _admission_trace(metric, candidates, returns):
    library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.0, dependence_metric=metric)
    pipeline = ValidationPipeline(
        data_tensor={"$close": np.ones_like(returns)},
        returns=returns,
        library=library,
        ic_threshold=0.0001,
        icir_threshold=0.0001,
        replacement_ic_min=0.001,
        replacement_ic_ratio=1.0,
        fast_screen_assets=returns.shape[0],
    )
    trace = []
    for batch_start in range(0, len(candidates), 5):
        batch = candidates[batch_start : batch_start + 5]
        precomputed = [(signals, None) for _, signals in batch]
        results = [
            pipeline.evaluate_candidate(name, "Neg($close)", precomputed=signal)
            for (name, _), signal in zip(batch, precomputed, strict=True)
        ]
        results = pipeline._deduplicate_batch(results)
        from factorminer.architecture.library_services import FactorAdmissionService

        FactorAdmissionService(library).admit_results(results, iteration=batch_start)
        trace.extend(
            (r.factor_name, r.admitted, r.replaced, r.stage_passed, r.rejection_reason,
             r.max_correlation)
            for r in results
        )
    trace.append(tuple(sorted(f.name for f in library.list_factors())))
    trace.append(repr(library.correlation_matrix))
    return trace


def test_admission_and_replacement_decisions_match_reference():
    rng = np.random.default_rng(7)
    assets, periods = 25, 300
    returns = rng.normal(size=(assets, periods))
    base = [rng.normal(size=(assets, periods)) for _ in range(6)]
    candidates = []
    for index in range(40):
        parent = base[index % len(base)]
        # Even candidates strengthen the target loading (replacements); odd
        # ones stay weak (dependence rejections); unrelated noise varies.
        weight = 0.02 * index if index % 2 == 0 else 0.01
        signals = parent + returns * weight
        signals = signals + rng.normal(scale=0.2 if index % 5 else 3.0, size=signals.shape)
        signals[rng.random(signals.shape) < 0.1] = np.nan
        candidates.append((f"c{index:02d}", signals))

    reference = _admission_trace(SpearmanDependenceMetric(), candidates, returns)
    fresh = [(name, signals.copy()) for name, signals in candidates]
    indexed = _admission_trace(IndexedSpearmanMetric(), fresh, returns)
    assert indexed == reference
    admitted = [row for row in reference[:-2] if row[1]]
    assert any(row[2] is not None for row in admitted), "exercise at least one replacement"
    assert any(not row[1] and "dependence" in row[4] for row in reference[:-2])


def test_computed_signals_are_immutable_and_never_alias_inputs():
    data = {"$close": np.arange(40.0).reshape(4, 10)}
    signals = compute_tree_signals(parse("$close"), data, (4, 10))
    assert not signals.flags.writeable and _immutable(signals)
    assert data["$close"].flags.writeable
    data["$close"][:] = 0.0  # the caller's panel changes; the signals do not
    assert signals[0, 1] == 1.0
    assert _immutable(compute_tree_signals(parse("Neg($close)"), data, (4, 10)))
    view = np.zeros((4, 10)).view()
    view.flags.writeable = False
    assert not _immutable(view)  # writable underneath: re-validated on each use


def test_dependence_index_fingerprints_read_only_arrays_once(monkeypatch):
    rng = np.random.default_rng(3)
    candidate = rng.standard_normal((30, 60))
    library = [rng.standard_normal((30, 60)) for _ in range(4)]
    for signals in library:
        signals.flags.writeable = False
    index = DependenceIndex()
    calls: list[int] = []
    original = DependenceIndex._fingerprint
    monkeypatch.setattr(
        DependenceIndex, "_fingerprint",
        staticmethod(lambda a: calls.append(id(a)) or original(a)),
    )
    for _ in range(3):
        for member in library:
            index.compute(candidate, member)
    # The writeable candidate is re-hashed on every call; library members once.
    assert sum(calls.count(id(member)) for member in library) == len(library)
    assert calls.count(id(candidate)) == 3 * len(library)
    # A writeable array is still re-validated: an in-place change invalidates it.
    before = index.compute(candidate, library[0])
    candidate[:] = library[0]
    assert index.compute(candidate, library[0]) != before
