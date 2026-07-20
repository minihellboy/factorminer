"""Tests for RD-Agent(Q)-style factor+model co-optimization.

Covers `factorminer.evaluation.model_zoo` (the downstream model fit) and
`factorminer.architecture.model_stage` (the optional, opt-in loop stage that
surfaces it as a JSON-serializable report).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from factorminer.architecture.model_stage import ModelCoOptimizeConfig, ModelCoOptimizeStage
from factorminer.architecture.stages import IterationPayload
from factorminer.core.factor_library import Factor
from factorminer.evaluation.model_zoo import (
    FactorContributionSummary,
    ModelCoOptimizationReport,
    ModelZooConfig,
    ModelZooEvaluator,
)


def _synthetic_panel_fixture(seed: int = 42, assets: int = 24, periods: int = 160):
    """Build a synthetic (assets, periods) factor panel with known informativeness.

    factor_signal    -- genuinely predictive: returns are a noisy linear
                         function of it.
    factor_weak      -- weakly predictive: same relationship, much more noise.
    factor_noise_a/b -- pure iid noise, unrelated to returns.
    """
    rng = np.random.default_rng(seed)
    factor_signal = rng.standard_normal((assets, periods))
    factor_weak = rng.standard_normal((assets, periods))
    factor_noise_a = rng.standard_normal((assets, periods))
    factor_noise_b = rng.standard_normal((assets, periods))

    noise = rng.standard_normal((assets, periods)) * 0.5
    returns = 0.8 * factor_signal + 0.15 * factor_weak + noise

    factor_signals = {
        1: factor_signal,
        2: factor_weak,
        3: factor_noise_a,
        4: factor_noise_b,
    }
    factor_names = {
        1: "alpha_strong",
        2: "alpha_weak",
        3: "noise_a",
        4: "noise_b",
    }
    return factor_signals, factor_names, returns


# ---------------------------------------------------------------------------
# ModelZooEvaluator: model fit + held-out IC/Sharpe + factor ranking
# ---------------------------------------------------------------------------


def test_evaluate_returns_sane_held_out_metrics_for_ridge():
    factor_signals, factor_names, returns = _synthetic_panel_fixture()
    evaluator = ModelZooEvaluator()

    report = evaluator.evaluate(
        factor_signals,
        factor_names,
        returns,
        config=ModelZooConfig(model_kind="ridge", train_fraction=0.7, permutation_repeats=15, seed=42),
        iteration=7,
    )

    assert isinstance(report, ModelCoOptimizationReport)
    assert report.model_kind == "ridge"
    assert report.n_factors == 4
    assert report.n_train_samples > 0
    assert report.n_test_samples > 0
    assert report.generated_at_iteration == 7
    # A genuinely predictive panel should produce a meaningfully positive
    # held-out IC -- not just noise around zero.
    assert report.held_out_ic > 0.15
    assert -1.0 <= report.held_out_ic <= 1.0
    assert np.isfinite(report.held_out_sharpe)
    assert np.isfinite(report.baseline_equal_weight_ic)
    assert len(report.contributions) == 4


def test_evaluate_ranks_predictive_factors_above_noise():
    factor_signals, factor_names, returns = _synthetic_panel_fixture()
    evaluator = ModelZooEvaluator()

    report = evaluator.evaluate(
        factor_signals,
        factor_names,
        returns,
        config=ModelZooConfig(model_kind="ridge", permutation_repeats=25, seed=42),
    )

    by_id = {c.factor_id: c for c in report.contributions}
    assert set(by_id) == {1, 2, 3, 4}

    strong = by_id[1]
    weak = by_id[2]
    noise_a = by_id[3]
    noise_b = by_id[4]

    # The strongly predictive factor must clearly outrank the two pure-noise
    # factors on held-out permutation importance.
    assert strong.permutation_importance_mean > noise_a.permutation_importance_mean
    assert strong.permutation_importance_mean > noise_b.permutation_importance_mean
    assert strong.rank == 1

    # Noise factors should rank at the bottom of the four.
    ranks = sorted(c.rank for c in report.contributions)
    assert ranks == [1, 2, 3, 4]
    noise_ranks = {noise_a.rank, noise_b.rank}
    assert noise_ranks == {3, 4} or max(noise_ranks) == 4

    # Ridge exposes a signed coefficient; xgboost would not.
    assert strong.coefficient is not None
    assert isinstance(strong.ensemble_marginal_delta_ic, float)

    # Weak-but-real signal should still beat at least one pure-noise factor.
    assert weak.permutation_importance_mean >= min(
        noise_a.permutation_importance_mean, noise_b.permutation_importance_mean
    )


def test_evaluate_xgboost_has_no_linear_coefficient():
    factor_signals, factor_names, returns = _synthetic_panel_fixture(assets=20, periods=120)
    evaluator = ModelZooEvaluator()

    report = evaluator.evaluate(
        factor_signals,
        factor_names,
        returns,
        config=ModelZooConfig(model_kind="xgboost", xgb_n_estimators=30, permutation_repeats=10, seed=1),
    )

    assert report.model_kind == "xgboost"
    assert all(c.coefficient is None for c in report.contributions)
    assert all(np.isfinite(c.permutation_importance_mean) for c in report.contributions)


def test_evaluate_report_is_json_serializable():
    factor_signals, factor_names, returns = _synthetic_panel_fixture(assets=16, periods=100)
    evaluator = ModelZooEvaluator()

    report = evaluator.evaluate(factor_signals, factor_names, returns, config=ModelZooConfig(seed=3))
    payload = report.to_dict()

    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["model_kind"] == "ridge"
    assert len(decoded["contributions"]) == 4
    assert all(isinstance(c, dict) for c in decoded["contributions"])
    assert isinstance(payload["contributions"][0], dict)


def test_model_zoo_config_rejects_unknown_model_kind():
    with pytest.raises(ValueError):
        ModelZooConfig(model_kind="not_a_model")


def test_evaluate_handles_single_factor_and_degenerate_input_gracefully():
    rng = np.random.default_rng(0)
    signals = {1: rng.standard_normal((10, 20))}
    names = {1: "solo"}
    returns = rng.standard_normal((10, 20))

    report = ModelZooEvaluator().evaluate(signals, names, returns)
    assert report.n_factors == 1
    assert len(report.contributions) == 1

    empty_report = ModelZooEvaluator().evaluate({}, {}, returns)
    assert empty_report.n_factors == 0
    assert empty_report.contributions == []


# ---------------------------------------------------------------------------
# ModelCoOptimizeStage: opt-in loop stage wiring
# ---------------------------------------------------------------------------


def _make_library(factor_signals: dict[int, np.ndarray], factor_names: dict[int, str]):
    """Build a minimal object exposing `.factors` like `FactorLibrary` does."""
    factors = {
        fid: Factor(
            id=fid,
            name=factor_names[fid],
            formula=f"$formula_{fid}",
            category="Other",
            ic_mean=0.05,
            icir=1.0,
            ic_win_rate=0.6,
            max_correlation=0.1,
            batch_number=1,
            signals=signal,
        )
        for fid, signal in factor_signals.items()
    }
    return SimpleNamespace(factors=factors)


def test_stage_is_noop_when_disabled_by_default():
    factor_signals, factor_names, returns = _synthetic_panel_fixture(assets=12, periods=60)
    loop = SimpleNamespace(library=_make_library(factor_signals, factor_names), returns=returns, iteration=1)
    payload = IterationPayload(iteration=1, batch_size=8, admitted_results=[1, 2, 3])

    stage = ModelCoOptimizeStage()  # default config: enabled=False
    assert stage.config.enabled is False
    stage.run(loop, payload)

    assert "co_optimization_report" not in payload.stage_metrics
    assert stage.last_report is None


def test_stage_respects_every_n_admissions_gate_then_produces_report():
    factor_signals, factor_names, returns = _synthetic_panel_fixture(assets=20, periods=140)
    loop = SimpleNamespace(library=_make_library(factor_signals, factor_names), returns=returns, iteration=1)

    config = ModelCoOptimizeConfig(
        enabled=True,
        every_n_admissions=3,
        min_factors=2,
        model_zoo_config=ModelZooConfig(permutation_repeats=10, seed=42),
    )
    stage = ModelCoOptimizeStage(config=config)

    # First iteration: only 2 admissions accumulated, below the gate of 3.
    payload1 = IterationPayload(iteration=1, batch_size=8, admitted_results=[1, 2])
    stage.run(loop, payload1)
    assert "co_optimization_report" not in payload1.stage_metrics
    assert stage.last_report is None

    # Second iteration: 1 more admission crosses the every_n_admissions=3 gate.
    payload2 = IterationPayload(iteration=2, batch_size=8, admitted_results=[3])
    stage.run(loop, payload2)

    assert "co_optimization_report" in payload2.stage_metrics
    report_dict = payload2.stage_metrics["co_optimization_report"]
    json.dumps(report_dict)  # must be JSON-serializable
    assert report_dict["n_factors"] == len(factor_signals)
    assert report_dict["generated_at_iteration"] == 2
    assert stage.last_report is not None

    # Gate resets after firing.
    payload3 = IterationPayload(iteration=3, batch_size=8, admitted_results=[])
    stage.run(loop, payload3)
    assert "co_optimization_report" not in payload3.stage_metrics


def test_stage_skips_below_min_factors():
    factor_signals, factor_names, returns = _synthetic_panel_fixture(assets=10, periods=50)
    single_signal = {1: factor_signals[1]}
    single_name = {1: factor_names[1]}
    loop = SimpleNamespace(library=_make_library(single_signal, single_name), returns=returns, iteration=1)

    config = ModelCoOptimizeConfig(enabled=True, every_n_admissions=1, min_factors=2)
    stage = ModelCoOptimizeStage(config=config)
    payload = IterationPayload(iteration=1, batch_size=4, admitted_results=[1])
    stage.run(loop, payload)

    assert "co_optimization_report" not in payload.stage_metrics


def test_from_mining_config_builds_disabled_stage_when_attribute_absent():
    mining_config = SimpleNamespace(target_library_size=50)
    stage = ModelCoOptimizeStage.from_mining_config(mining_config)
    assert stage.config.enabled is False


def test_from_mining_config_parses_raw_dict_flag():
    mining_config = SimpleNamespace(
        model_co_optimize={
            "enabled": True,
            "every_n_admissions": 2,
            "model_kind": "lasso",
            "alpha": 0.5,
        }
    )
    stage = ModelCoOptimizeStage.from_mining_config(mining_config)

    assert stage.config.enabled is True
    assert stage.config.every_n_admissions == 2
    assert stage.config.model_zoo_config.model_kind == "lasso"
    assert stage.config.model_zoo_config.alpha == 0.5


def test_stage_run_standalone_against_mock_iteration_payload_end_to_end():
    """(b) acceptance: run standalone against a small mock IterationPayload/library
    fixture (no live LLM) and produce a JSON-serializable report."""
    factor_signals, factor_names, returns = _synthetic_panel_fixture(assets=18, periods=90)
    loop = SimpleNamespace(library=_make_library(factor_signals, factor_names), returns=returns, iteration=5)

    stage = ModelCoOptimizeStage(
        config=ModelCoOptimizeConfig(enabled=True, every_n_admissions=1, min_factors=2)
    )
    payload = IterationPayload(iteration=5, batch_size=8, admitted_results=[10])
    stage.run(loop, payload)

    report = payload.stage_metrics["co_optimization_report"]
    serialized = json.dumps(report)
    assert isinstance(serialized, str)
    assert report["generated_at_iteration"] == 5
    assert isinstance(report["contributions"], list) and len(report["contributions"]) == 4
    for contribution in report["contributions"]:
        assert set(contribution) == set(FactorContributionSummary.__dataclass_fields__)
