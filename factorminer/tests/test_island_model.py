"""Tests for the island-model mining mode (`architecture/island_model.py`).

Covers the AlphaEvolve/FunSearch-style population-diversity contract:
  - independently-biased islands produce measurably different family
    compositions before migration
  - periodic migration only ever *adds* factors that clear the destination's
    own admission bar (never force-inserted), and family diversity is
    monotonically non-decreasing across a migration round as a result
  - migrated factors that fail the destination's admission bar are rejected

Entirely offline: `MockProvider` (no LLM/network calls) and small synthetic
numpy fixtures (no real market data), matching the repo's `--mock` convention.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from factorminer.agent.llm_interface import MockProvider
from factorminer.architecture.families import infer_family
from factorminer.architecture.island_model import (
    FamilyDiversitySnapshot,
    IslandBiasDescriptor,
    IslandMiningCoordinator,
    IslandPopulation,
    build_island,
)
from factorminer.architecture.library_services import FactorAdmissionService
from factorminer.core.factor_library import Factor, FactorLibrary

# ---------------------------------------------------------------------------
# Minimal mining config (mirrors test_ralph_loop.py's `_TestConfig` pattern)
# ---------------------------------------------------------------------------


@dataclass
class _TestConfig:
    target_library_size: int = 50
    batch_size: int = 8
    max_iterations: int = 50
    ic_threshold: float = 0.001
    icir_threshold: float = 0.0
    correlation_threshold: float = 0.9
    replacement_ic_min: float = 0.10
    replacement_ic_ratio: float = 1.3
    fast_screen_assets: int = 0  # deterministic: no fast screening subsample
    num_workers: int = 1
    output_dir: str = ""
    redundancy_metric: str = "spearman"
    memory_policy: str = "paper"
    memory_regime_lookback_window: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {"target_library_size": self.target_library_size}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="island_model_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def market_data():
    """Small synthetic (M=20, T=80, F=8) tensor + returns, seeded for determinism."""
    rng = np.random.default_rng(42)
    m, t, f = 20, 80, 8
    data_tensor = rng.normal(0, 1, (m, t, f)).astype(np.float64)
    returns = rng.normal(0, 0.02, (m, t)).astype(np.float64)
    return data_tensor, returns


def _make_island(name: str, output_dir: str, bias: IslandBiasDescriptor, data_tensor, returns) -> IslandPopulation:
    config = _TestConfig(output_dir=output_dir)
    return build_island(
        name,
        config,
        data_tensor,
        returns,
        bias,
        llm_provider=MockProvider(cycle=True),
    )


@pytest.fixture
def two_islands(tmp_dir, market_data):
    data_tensor, returns = market_data
    bias_a = IslandBiasDescriptor(name="momentum-island", preferred_families=("Momentum", "Smoothing"))
    bias_b = IslandBiasDescriptor(name="volatility-island", preferred_families=("Volatility", "Extrema"))
    island_a = _make_island("momentum-island", f"{tmp_dir}/a", bias_a, data_tensor, returns)
    island_b = _make_island("volatility-island", f"{tmp_dir}/b", bias_b, data_tensor, returns)
    return island_a, island_b


# ---------------------------------------------------------------------------
# IslandBiasDescriptor
# ---------------------------------------------------------------------------


class TestIslandBiasDescriptor:
    def test_filter_keeps_only_preferred_family_at_full_strength(self):
        bias = IslandBiasDescriptor(name="mom", preferred_families=("Momentum",))
        candidates = [
            ("a", "Neg(Delta($close, 5))"),  # Momentum
            ("b", "Std($returns, 10)"),  # Volatility
        ]
        filtered = bias.filter_candidates(candidates)
        assert filtered == [("a", "Neg(Delta($close, 5))")]

    def test_filter_falls_back_to_full_batch_when_nothing_matches(self):
        bias = IslandBiasDescriptor(name="mom", preferred_families=("Momentum",))
        candidates = [("b", "Std($returns, 10)")]  # Volatility only
        assert bias.filter_candidates(candidates) == candidates

    def test_no_bias_is_a_no_op(self):
        bias = IslandBiasDescriptor(name="neutral")
        candidates = [("a", "Neg(Delta($close, 5))"), ("b", "Std($returns, 10)")]
        assert bias.filter_candidates(candidates) == candidates

    def test_partial_bias_strength_keeps_some_off_family_candidates(self):
        bias = IslandBiasDescriptor(
            name="mom", preferred_families=("Momentum",), family_bias_strength=0.5
        )
        candidates = [
            ("a", "Neg(Delta($close, 5))"),  # Momentum
            ("b", "Std($returns, 10)"),  # Volatility
            ("c", "Var($returns, 10)"),  # Volatility
        ]
        filtered = bias.filter_candidates(candidates)
        assert ("a", "Neg(Delta($close, 5))") in filtered
        # keep round(2 * 0.5) = 1 of the 2 off-family candidates
        assert len(filtered) == 2

    def test_recommended_direction_hints_mention_families_and_regime(self):
        bias = IslandBiasDescriptor(
            name="mom", preferred_families=("Momentum", "Smoothing"), regime_focus="high_vol"
        )
        hints = bias.recommended_direction_hints()
        assert any("Momentum" in h and "Smoothing" in h for h in hints)
        assert any("high_vol" in h for h in hints)

    def test_rejects_out_of_range_strength(self):
        with pytest.raises(ValueError):
            IslandBiasDescriptor(name="bad", family_bias_strength=1.5)


# ---------------------------------------------------------------------------
# FamilyDiversitySnapshot
# ---------------------------------------------------------------------------


class TestFamilyDiversitySnapshot:
    def test_single_family_has_zero_entropy(self):
        snap = FamilyDiversitySnapshot.from_counts("x", {"Momentum": 5})
        assert snap.distinct_families == 1
        assert snap.shannon_entropy == pytest.approx(0.0)

    def test_balanced_two_families_has_entropy_one(self):
        snap = FamilyDiversitySnapshot.from_counts("x", {"Momentum": 3, "Volatility": 3})
        assert snap.distinct_families == 2
        assert snap.shannon_entropy == pytest.approx(1.0)

    def test_empty_counts(self):
        snap = FamilyDiversitySnapshot.from_counts("x", {})
        assert snap.total_factors == 0
        assert snap.distinct_families == 0
        assert snap.shannon_entropy == 0.0


# ---------------------------------------------------------------------------
# (a) Different biases -> measurably different pre-migration family mixes
# ---------------------------------------------------------------------------


def test_biased_islands_diverge_in_family_composition_before_migration(two_islands):
    island_a, island_b = two_islands

    island_a.run_iterations(2)
    island_b.run_iterations(2)

    assert island_a.library.size > 0
    assert island_b.library.size > 0

    div_a = island_a.family_diversity()
    div_b = island_b.family_diversity()

    # Each island's bias should be reflected: island_a stays inside its
    # preferred families, island_b inside its own, disjoint set.
    assert set(div_a.family_counts).issubset({"Momentum", "Smoothing"})
    assert set(div_b.family_counts).issubset({"Volatility", "Extrema"})
    assert div_a.family_counts != div_b.family_counts
    assert set(div_a.family_counts).isdisjoint(set(div_b.family_counts))


# ---------------------------------------------------------------------------
# (b) Migration: cross-pollination + non-decreasing diversity
# ---------------------------------------------------------------------------


def test_migration_cross_pollinates_and_does_not_reduce_diversity(two_islands):
    island_a, island_b = two_islands
    coordinator = IslandMiningCoordinator(
        [island_a, island_b], migration_interval=2, migration_top_k=3
    )

    coordinator.run(total_iterations=2)

    assert len(coordinator.migration_rounds) == 1
    round_ = coordinator.migration_rounds[0]
    accepted = round_.accepted
    assert accepted, "expected at least one migration to be admitted at the destination"

    report = coordinator.diversity_report()
    for name in ("momentum-island", "volatility-island"):
        before = report[name]["before"]
        after = report[name]["after"]
        assert after["distinct_families"] >= before["distinct_families"]
        assert after["shannon_entropy"] >= before["shannon_entropy"] - 1e-9
        # migration should have actually broadened the mix, not just held steady
        assert after["distinct_families"] > before["distinct_families"]

    # Some migrated formula must now show up in both islands' libraries.
    formulas_a = {f.formula for f in island_a.library.list_factors()}
    formulas_b = {f.formula for f in island_b.library.list_factors()}
    shared = formulas_a & formulas_b
    assert shared, "expected at least one factor formula to appear in both islands after migration"

    # And every accepted migration outcome's formula really did land at its destination.
    for outcome in accepted:
        destination_formulas = (
            formulas_a if outcome.destination == "momentum-island" else formulas_b
        )
        assert outcome.formula in destination_formulas


def test_run_requires_at_least_two_islands(two_islands):
    island_a, _ = two_islands
    with pytest.raises(ValueError):
        IslandMiningCoordinator([island_a])


def test_diversity_report_before_run_raises(two_islands):
    coordinator = IslandMiningCoordinator(list(two_islands), migration_interval=2)
    with pytest.raises(RuntimeError):
        coordinator.diversity_report()


# ---------------------------------------------------------------------------
# (c) Migrated factors that fail the destination admission bar are rejected
# ---------------------------------------------------------------------------


def _make_factor(name: str, formula: str, signals: np.ndarray, ic_mean: float = 0.05) -> Factor:
    return Factor(
        id=0,
        name=name,
        formula=formula,
        category=infer_family(formula),
        ic_mean=ic_mean,
        icir=1.0,
        ic_win_rate=0.6,
        max_correlation=0.0,
        batch_number=1,
        signals=signals,
    )


def test_admit_migrant_rejects_factor_below_ic_threshold():
    library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.05)
    service = FactorAdmissionService(library)
    rng = np.random.default_rng(1)
    signals = rng.normal(size=(6, 40))

    weak_migrant = _make_factor("weak", "Delta($close, 5)", signals, ic_mean=0.001)
    admitted, reason = service.admit_migrant(weak_migrant, iteration=1)

    assert admitted is False
    assert "threshold" in reason.lower()
    assert library.size == 0


def test_admit_migrant_rejects_factor_too_correlated_with_destination():
    library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.01)
    service = FactorAdmissionService(library)
    rng = np.random.default_rng(2)
    signals = rng.normal(size=(6, 40))

    existing = _make_factor("existing", "Neg(Delta($close, 5))", signals, ic_mean=0.05)
    library.admit_factor(existing)

    # Same underlying signals as `existing` -> correlation ~1.0 -> must be rejected
    # even though its own IC clears the threshold.
    duplicate_migrant = _make_factor("dup", "Neg(Delta($close, 5))", signals.copy(), ic_mean=0.06)
    admitted, reason = service.admit_migrant(duplicate_migrant, iteration=1)

    assert admitted is False
    assert "correlation" in reason.lower() or "dependence" in reason.lower()
    assert library.size == 1  # destination untouched


def test_admit_migrant_with_no_cached_signals_is_rejected():
    library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.01)
    service = FactorAdmissionService(library)
    migrant = _make_factor("no_signal", "Delta($close, 5)", signals=None, ic_mean=0.5)
    admitted, reason = service.admit_migrant(migrant, iteration=1)
    assert admitted is False
    assert "signal" in reason.lower()


def test_coordinator_try_migrate_factor_rejects_when_destination_saturated(two_islands):
    island_a, island_b = two_islands
    island_a.run_iterations(2)

    top_factor = sorted(
        island_a.library.list_factors(), key=lambda f: f.ic_paper_mean, reverse=True
    )[0]

    # Raise the destination's IC admission floor above the migrant's own paper
    # IC so the normal admission bar (Eq. 10) rejects it outright -- exactly
    # what a native candidate with the same IC would suffer at this island.
    island_b.library.ic_threshold = top_factor.ic_paper_mean + 0.05

    coordinator = IslandMiningCoordinator([island_a, island_b], migration_interval=2)
    outcome = coordinator.try_migrate_factor(
        top_factor, island_b, source_name=island_a.name, iteration=1
    )

    assert outcome.accepted is False
    assert "threshold" in outcome.reason.lower()
    # The migrant's own formula must not have been force-inserted.
    assert top_factor.formula not in {f.formula for f in island_b.library.list_factors()}


# ---------------------------------------------------------------------------
# Coordinator plumbing / validation
# ---------------------------------------------------------------------------


def test_island_name_lookup(two_islands):
    coordinator = IslandMiningCoordinator(list(two_islands), migration_interval=2)
    assert coordinator.island("momentum-island") is two_islands[0]
    with pytest.raises(KeyError):
        coordinator.island("nonexistent")


def test_duplicate_island_names_rejected(tmp_dir, market_data):
    data_tensor, returns = market_data
    bias = IslandBiasDescriptor(name="dup")
    island_1 = _make_island("dup", f"{tmp_dir}/1", bias, data_tensor, returns)
    island_2 = _make_island("dup", f"{tmp_dir}/2", bias, data_tensor, returns)
    with pytest.raises(ValueError):
        IslandMiningCoordinator([island_1, island_2])
