"""Tests for offline RFT dataset export (architecture/rft_export.py).

Covers DiCo reward discrimination, regime task-bank bucketing, JSONL schema
export from FactorLifecycleStore, EvaluationKernel reward-hook registration,
and the honesty statement on the CLI command. No GPU training is exercised
or claimed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from factorminer.architecture.evaluation_kernel import EvaluationKernel
from factorminer.architecture.geometry import LibraryGeometry
from factorminer.architecture.lifecycle import FactorLifecycleStore
from factorminer.architecture.paper_protocol import PaperProtocol
from factorminer.architecture.rft_export import (
    RFT_DATASET_SCHEMA_VERSION,
    DiCoRewardConfig,
    DiCoRewardResult,
    RFTExportConfig,
    build_regime_task_bank,
    build_regime_task_bucket,
    compute_dico_reward,
    compute_max_dependence,
    export_rft_dataset,
    load_rft_dataset,
)
from factorminer.core.factor_library import FactorLibrary
from factorminer.evaluation.regime import MarketRegime, RegimeClassification
from factorminer.utils.config import load_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _panel(seed: int, m: int = 20, t: int = 40, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((m, t)) * scale


def _make_lifecycle_store() -> FactorLifecycleStore:
    """Synthetic multi-iteration lifecycle with varied IC / correlation."""
    store = FactorLifecycleStore(output_dir=None)

    # Iteration 0: three candidates, one admitted.
    store.record(0, "f0", "CsRank($close)", stage="proposed", status="seen")
    store.record(0, "f0", "CsRank($close)", stage="parsed", status="passed")
    store.record(
        0, "f0", "CsRank($close)", stage="fast_screened", status="passed",
        details={"ic_mean": 0.08},
    )
    store.record(
        0, "f0", "CsRank($close)", stage="admitted", status="passed",
        details={"ic_mean": 0.08, "icir": 1.2, "max_correlation": 0.0},
    )

    store.record(0, "f1", "CsRank($volume)", stage="proposed", status="seen")
    store.record(0, "f1", "CsRank($volume)", stage="parsed", status="passed")
    store.record(
        0, "f1", "CsRank($volume)", stage="fast_screened", status="passed",
        details={"ic_mean": 0.05},
    )
    store.record(
        0, "f1", "CsRank($volume)", stage="correlation_rejected", status="failed",
        details={"reason": "corr=0.85 with f0", "max_correlation": 0.85},
    )

    store.record(0, "f2", "Rank(Delta($close, 5))", stage="proposed", status="seen")
    store.record(0, "f2", "Rank(Delta($close, 5))", stage="parsed", status="passed")
    store.record(
        0, "f2", "Rank(Delta($close, 5))", stage="fast_screened", status="passed",
        details={"ic_mean": 0.06},
    )
    store.record(
        0, "f2", "Rank(Delta($close, 5))", stage="admitted", status="passed",
        details={"ic_mean": 0.06, "icir": 0.9, "max_correlation": 0.25},
    )

    # Iteration 1: complementary high-IC + redundant twin of f0.
    store.record(1, "f3", "CsRank(Corr($close, $volume, 20))", stage="proposed", status="seen")
    store.record(1, "f3", "CsRank(Corr($close, $volume, 20))", stage="parsed", status="passed")
    store.record(
        1, "f3", "CsRank(Corr($close, $volume, 20))", stage="fast_screened", status="passed",
        details={"ic_mean": 0.07},
    )
    store.record(
        1, "f3", "CsRank(Corr($close, $volume, 20))", stage="admitted", status="passed",
        details={"ic_mean": 0.07, "icir": 1.0, "max_correlation": 0.15},
    )

    store.record(1, "f4", "CsRank($close)", stage="proposed", status="seen")  # near-dup of f0
    store.record(1, "f4", "CsRank($close)", stage="parsed", status="passed")
    store.record(
        1, "f4", "CsRank($close)", stage="fast_screened", status="passed",
        details={"ic_mean": 0.07},
    )
    store.record(
        1, "f4", "CsRank($close)", stage="correlation_rejected", status="failed",
        details={"reason": "corr=0.95 with f0", "max_correlation": 0.95},
    )

    store.record(1, "f5", "bad((", stage="proposed", status="seen")
    store.record(
        1, "f5", "bad((", stage="parsed", status="failed",
        details={"reason": "parse error"},
    )

    # Iteration 2: another complementary candidate.
    store.record(2, "f6", "TsRank($open, 10)", stage="proposed", status="seen")
    store.record(2, "f6", "TsRank($open, 10)", stage="parsed", status="passed")
    store.record(
        2, "f6", "TsRank($open, 10)", stage="fast_screened", status="passed",
        details={"ic_mean": 0.04},
    )
    store.record(
        2, "f6", "TsRank($open, 10)", stage="admitted", status="passed",
        details={"ic_mean": 0.04, "icir": 0.7, "max_correlation": 0.30},
    )

    return store


# ---------------------------------------------------------------------------
# DiCo reward
# ---------------------------------------------------------------------------


def test_dico_reward_empty_library_is_pure_predictive():
    result = compute_dico_reward(0.05, max_dependence=0.0)
    assert isinstance(result, DiCoRewardResult)
    assert result.complementarity == pytest.approx(1.0)
    assert result.reward == pytest.approx(0.05)
    assert result.library_size == 0


def test_dico_reward_penalizes_redundancy():
    """Equally predictive candidates: redundant one scores strictly lower."""
    predictive = 0.08
    complementary = compute_dico_reward(predictive, max_dependence=0.10)
    redundant = compute_dico_reward(predictive, max_dependence=0.90)

    assert complementary.reward > redundant.reward
    assert complementary.complementarity > redundant.complementarity
    assert redundant.reward == pytest.approx(0.08 * 0.10, rel=1e-9)


def test_dico_reward_discriminates_via_signals():
    """Signal-level path: near-duplicate of an admitted factor loses to an
    orthogonal candidate with the same IC."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((30, 50))
    noise = rng.standard_normal((30, 50)) * 0.05
    orthogonal = rng.standard_normal((30, 50))

    library = [base]
    predictive = 0.06

    red = compute_dico_reward(
        predictive,
        candidate_signals=base + noise,
        library_signals=library,
    )
    comp = compute_dico_reward(
        predictive,
        candidate_signals=orthogonal,
        library_signals=library,
    )

    assert red.max_dependence > 0.7
    assert comp.max_dependence < 0.4
    assert comp.reward > red.reward


def test_dico_reward_weights_and_clip():
    cfg = DiCoRewardConfig(
        predictive_weight=2.0,
        complementarity_weight=1.0,
        clip_reward=(0.0, 0.01),
    )
    result = compute_dico_reward(0.2, max_dependence=0.0, config=cfg)
    # 0.2**2 = 0.04, clipped to 0.01
    assert result.reward == pytest.approx(0.01)


def test_compute_max_dependence_empty_and_bounds():
    cand = _panel(1)
    assert compute_max_dependence(cand, []) == 0.0
    dep = compute_max_dependence(cand, [cand])
    assert 0.99 <= dep <= 1.0


# ---------------------------------------------------------------------------
# Regime task bank
# ---------------------------------------------------------------------------


def test_regime_task_bucket_unknown_without_classification():
    bucket = build_regime_task_bucket(3, None)
    assert bucket.dominant_regime == "unknown"
    assert bucket.task_id == "iter3:unknown"
    assert bucket.regime_mix == {"unknown": 1.0}


def test_regime_task_bank_from_returns_has_named_regimes():
    rng = np.random.default_rng(42)
    # Construct a return panel with a clear bull stretch then a bear stretch.
    t = 200
    m = 15
    rets = np.zeros((m, t))
    rets[:, :100] = 0.002 + rng.normal(0, 0.005, size=(m, 100))
    rets[:, 100:] = -0.003 + rng.normal(0, 0.01, size=(m, 100))

    bank = build_regime_task_bank([0, 1], returns=rets)
    assert set(bank) == {0, 1}
    for bucket in bank.values():
        assert bucket.dominant_regime in {"bull", "bear", "sideways"}
        assert abs(sum(bucket.regime_mix.values()) - 1.0) < 1e-9
        assert bucket.n_periods == t


def test_regime_task_bank_respects_precomputed():
    # Force a non-default label via a hand-built classification.
    labels = np.array([MarketRegime.BEAR.value] * 10 + [MarketRegime.BULL.value] * 5)
    classification = RegimeClassification(
        labels=labels,
        periods={
            MarketRegime.BEAR: labels == MarketRegime.BEAR.value,
            MarketRegime.BULL: labels == MarketRegime.BULL.value,
            MarketRegime.SIDEWAYS: np.zeros(15, dtype=bool),
        },
        stats={
            MarketRegime.BEAR: {"mean_return": -0.01, "volatility": 0.02, "n_periods": 10},
            MarketRegime.BULL: {"mean_return": 0.01, "volatility": 0.01, "n_periods": 5},
            MarketRegime.SIDEWAYS: {"mean_return": 0.0, "volatility": 0.0, "n_periods": 0},
        },
    )
    pre = {1: build_regime_task_bucket(1, classification)}
    bank = build_regime_task_bank([0, 1], returns=None, precomputed=pre)
    assert bank[0].dominant_regime == "unknown"
    assert bank[1].dominant_regime == "bear"
    assert bank[1].regime_mix["bear"] == pytest.approx(10 / 15)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_rft_dataset_schema_and_reward_variance(tmp_path: Path):
    store = _make_lifecycle_store()
    out = tmp_path / "rft_dataset.jsonl"
    result = export_rft_dataset(store, out)

    assert out.exists()
    assert result.n_records >= 5
    assert result.schema_version == RFT_DATASET_SCHEMA_VERSION
    assert result.reward_std > 0.0, "rewards must have non-zero variance"
    assert result.manifest_path is not None
    assert Path(result.manifest_path).exists()

    records = load_rft_dataset(out)
    assert len(records) == result.n_records

    required_top = {
        "schema_version",
        "record_id",
        "iteration",
        "state",
        "action",
        "reward",
        "reward_breakdown",
        "regime_context",
        "metrics",
    }
    for row in records:
        assert required_top <= set(row)
        assert row["schema_version"] == RFT_DATASET_SCHEMA_VERSION
        assert "library_size" in row["state"]
        assert "admitted_formulas" in row["state"]
        assert "formula" in row["action"]
        assert "factor_id" in row["action"]
        assert "dominant_regime" in row["regime_context"]
        assert "task_id" in row["regime_context"]
        assert "predictive_score" in row["reward_breakdown"]
        assert "complementarity" in row["reward_breakdown"]

    # Parse failures are dropped by default.
    formulas = {r["action"]["formula"] for r in records}
    assert "bad((" not in formulas

    # Running library state grows only on admissions.
    admitted_seen = []
    for row in records:
        assert row["state"]["library_size"] == len(row["state"]["admitted_formulas"])
        if row["action"]["admitted"]:
            admitted_seen.append(row["action"]["formula"])
    assert len(admitted_seen) >= 3

    with open(result.manifest_path) as fp:
        manifest = json.load(fp)
    assert manifest["trains_model"] is False
    assert manifest["gpu_required_for_training"] is True
    assert "does NOT train a model" in manifest["honesty"]


def test_export_from_lifecycle_path_roundtrip(tmp_path: Path):
    store = FactorLifecycleStore(output_dir=tmp_path)
    store.record(0, "a", "CsRank($close)", stage="proposed", status="seen")
    store.record(0, "a", "CsRank($close)", stage="parsed", status="passed")
    store.record(
        0, "a", "CsRank($close)", stage="fast_screened", status="passed",
        details={"ic_mean": 0.09},
    )
    store.record(
        0, "a", "CsRank($close)", stage="admitted", status="passed",
        details={"ic_mean": 0.09, "icir": 1.5, "max_correlation": 0.0},
    )
    store.record(0, "b", "CsRank($volume)", stage="proposed", status="seen")
    store.record(0, "b", "CsRank($volume)", stage="parsed", status="passed")
    store.record(
        0, "b", "CsRank($volume)", stage="fast_screened", status="passed",
        details={"ic_mean": 0.04},
    )
    store.record(
        0, "b", "CsRank($volume)", stage="correlation_rejected", status="failed",
        details={"reason": "high corr", "max_correlation": 0.8},
    )

    out = tmp_path / "from_dir.jsonl"
    result = export_rft_dataset(tmp_path, out)
    assert result.n_records == 2
    rows = load_rft_dataset(out)
    rewards = [r["reward"] for r in rows]
    # Admitted low-dependence factor should out-reward the high-corr reject
    # when predictive scores are in the same ballpark — here 0.09*(1-0) vs 0.04*(1-0.8).
    by_id = {r["action"]["factor_id"]: r for r in rows}
    assert by_id["a"]["reward"] > by_id["b"]["reward"]
    assert np.std(rewards) > 0.0


def test_export_enriches_ic_from_session_log(tmp_path: Path):
    """IC-screen failures lack lifecycle ic_mean; session_log supplies it."""
    store = FactorLifecycleStore(output_dir=tmp_path)
    # Lifecycle only has proposed/parsed (mirrors real IC-screen failure path
    # before fast_screened is emitted).
    store.record(0, "a", "CsRank($close)", stage="proposed", status="seen")
    store.record(0, "a", "CsRank($close)", stage="parsed", status="passed")
    store.record(
        0, "a", "CsRank($close)", stage="memory_distilled", status="recorded",
        details={"admitted": False, "rejection_reason": "Paper IC low"},
    )
    store.record(0, "b", "CsRank($volume)", stage="proposed", status="seen")
    store.record(0, "b", "CsRank($volume)", stage="parsed", status="passed")
    store.record(
        0, "b", "CsRank($volume)", stage="memory_distilled", status="recorded",
        details={"admitted": False},
    )

    session_log = {
        "iterations": [],
        "factors": [
            {
                "expression": "CsRank($close)",
                "ic": 0.03,
                "icir": 0.5,
                "max_correlation": 0.0,
                "admitted": False,
                "rejection_reason": "Paper IC 0.03 < threshold",
            },
            {
                "expression": "CsRank($volume)",
                "ic": 0.01,
                "icir": 0.2,
                "max_correlation": 0.0,
                "admitted": False,
            },
        ],
        "summary": {},
    }
    (tmp_path / "session_log.json").write_text(json.dumps(session_log), encoding="utf-8")

    out = tmp_path / "enriched.jsonl"
    result = export_rft_dataset(tmp_path, out)
    rows = load_rft_dataset(out)
    assert result.n_records == 2
    assert result.reward_std > 0.0
    by_id = {r["action"]["factor_id"]: r for r in rows}
    assert by_id["a"]["reward"] == pytest.approx(0.03)
    assert by_id["b"]["reward"] == pytest.approx(0.01)
    assert by_id["a"]["metrics"]["ic"] == pytest.approx(0.03)



def test_export_redundant_vs_complementary_reward_order(tmp_path: Path):
    """End-to-end: equal IC, different dependence → DiCo orders them correctly."""
    store = FactorLifecycleStore(output_dir=None)
    # Seed library admission.
    store.record(0, "lib", "CsRank($close)", stage="admitted", status="passed",
                 details={"ic_mean": 0.05, "max_correlation": 0.0})
    # Two iter-1 candidates, same IC, different max_correlation.
    store.record(1, "red", "CsRank($close)+eps", stage="admitted", status="passed",
                 details={"ic_mean": 0.06, "max_correlation": 0.92})
    store.record(1, "comp", "CsRank($volume)", stage="admitted", status="passed",
                 details={"ic_mean": 0.06, "max_correlation": 0.12})

    out = tmp_path / "order.jsonl"
    # Disable default proposed-only filter concerns — these are admitted rows.
    result = export_rft_dataset(store, out, config=RFTExportConfig())
    rows = load_rft_dataset(out)
    by_id = {r["action"]["factor_id"]: r for r in rows}
    assert "red" in by_id and "comp" in by_id
    assert by_id["comp"]["reward"] > by_id["red"]["reward"]
    assert by_id["comp"]["reward_breakdown"]["complementarity"] > by_id["red"]["reward_breakdown"]["complementarity"]
    assert result.reward_std > 0.0


def test_export_with_regime_returns(tmp_path: Path):
    store = _make_lifecycle_store()
    rng = np.random.default_rng(1)
    returns = rng.normal(0.001, 0.02, size=(10, 180))
    out = tmp_path / "regimed.jsonl"
    result = export_rft_dataset(store, out, returns=returns)
    rows = load_rft_dataset(out)
    assert all(r["regime_context"]["dominant_regime"] != "unknown" or r["regime_context"]["n_periods"] >= 0
               for r in rows)
    assert result.n_records >= 5


# ---------------------------------------------------------------------------
# EvaluationKernel reward hook
# ---------------------------------------------------------------------------
def test_evaluation_kernel_reward_hook_default_and_custom():
    protocol = PaperProtocol.from_config(load_config())
    library = FactorLibrary()
    kernel = EvaluationKernel(protocol=protocol, geometry=LibraryGeometry(library))


    # Default path (no hook): DiCo reward, no behavior change to quality score.
    default = kernel.compute_reward(0.05, max_dependence=0.2)
    assert default.reward == pytest.approx(0.05 * 0.8)

    # Quality score path still works and does not consult the hook.
    signals = _panel(2)
    returns = _panel(3)
    stats = kernel.compute_target_stats(signals, returns)
    quality = kernel.compute_quality_score(
        signals=signals,
        returns=returns,
        target_stats=stats,
        library_signals=[],
    )
    assert "quality_gate" in quality

    # Custom hook registration.
    calls: list[float] = []

    def custom_hook(predictive_score: float, /, **kwargs):
        calls.append(float(predictive_score))
        return DiCoRewardResult(
            reward=float(predictive_score) * 2.0,
            predictive_score=float(predictive_score),
            complementarity=1.0,
            max_dependence=0.0,
            dependence_metric="custom",
            library_size=0,
        )

    kernel.set_reward_hook(custom_hook)
    custom = kernel.compute_reward(0.1)
    assert custom.reward == pytest.approx(0.2)
    assert calls == [0.1]

    kernel.set_reward_hook(None)
    cleared = kernel.compute_reward(0.05, max_dependence=0.0)
    assert cleared.reward == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_export_rft_dataset_help_contains_honesty():
    from factorminer.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["export-rft-dataset", "--help"])
    assert result.exit_code == 0, result.output
    # Click wraps long help lines; normalize whitespace before matching.
    help_text = " ".join(result.output.split())
    assert "does NOT train a model" in help_text
    assert "GRPO" in help_text or "Verl" in help_text or "vLLM" in help_text
    assert "external GPU" in help_text



def test_cli_export_rft_dataset_writes_jsonl(tmp_path: Path):
    from factorminer.cli import main

    # Persist a lifecycle log the CLI can load.
    store = FactorLifecycleStore(output_dir=tmp_path)
    for event_args in [
        (0, "c0", "CsRank($close)", "proposed", "seen", None),
        (0, "c0", "CsRank($close)", "parsed", "passed", None),
        (0, "c0", "CsRank($close)", "fast_screened", "passed", {"ic_mean": 0.08}),
        (0, "c0", "CsRank($close)", "admitted", "passed",
         {"ic_mean": 0.08, "icir": 1.1, "max_correlation": 0.0}),
        (0, "c1", "CsRank($volume)", "proposed", "seen", None),
        (0, "c1", "CsRank($volume)", "parsed", "passed", None),
        (0, "c1", "CsRank($volume)", "fast_screened", "passed", {"ic_mean": 0.05}),
        (0, "c1", "CsRank($volume)", "correlation_rejected", "failed",
         {"reason": "corr", "max_correlation": 0.8}),
        (1, "c2", "TsRank($open, 5)", "proposed", "seen", None),
        (1, "c2", "TsRank($open, 5)", "parsed", "passed", None),
        (1, "c2", "TsRank($open, 5)", "fast_screened", "passed", {"ic_mean": 0.04}),
        (1, "c2", "TsRank($open, 5)", "admitted", "passed",
         {"ic_mean": 0.04, "icir": 0.8, "max_correlation": 0.2}),
    ]:
        it, name, formula, stage, status, details = event_args
        store.record(it, name, formula, stage=stage, status=status, details=details)

    out_jsonl = tmp_path / "cli_rft.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-o", str(tmp_path / "cli_out"),
            "export-rft-dataset",
            str(tmp_path),
            "--output", str(out_jsonl),
            "--mock",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "does NOT train a model" in result.output
    assert out_jsonl.exists()
    rows = load_rft_dataset(out_jsonl)
    assert len(rows) >= 3
    rewards = [r["reward"] for r in rows]
    assert float(np.std(rewards)) > 0.0
    assert all(r["schema_version"] == RFT_DATASET_SCHEMA_VERSION for r in rows)
