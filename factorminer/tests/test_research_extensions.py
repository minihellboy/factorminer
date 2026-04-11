"""Tests for the research extension services."""

from __future__ import annotations

import numpy as np

from factorminer.architecture import ResearchExtensionService


def test_family_context_summarizes_admitted_and_rejected_formulas():
    service = ResearchExtensionService()
    family_context = service.family_context(
        admitted_formulas=[
            {"name": "momentum_a", "formula": "Delta($close, 1)", "ic_mean": 0.05},
            {"name": "smoothing_a", "formula": "Sma($close, 5)", "ic_mean": 0.03},
        ],
        rejected_formulas=[
            {"name": "momentum_b", "formula": "Delta($close, 2)", "ic_mean": -0.01},
            {"name": "other_a", "formula": "Neg($volume)", "ic_mean": 0.0},
        ],
        library_state={"categories": {"Momentum": 4, "Smoothing": 1}},
        memory_signal={
            "recommended_directions": [
                {"template": "CsRank($vwap)", "name": "VWAP"},
            ]
        },
    )

    assert family_context.admitted_count == 2
    assert family_context.rejected_count == 2
    family_names = {family.name for family in family_context.families}
    assert "Momentum" in family_names
    assert "Smoothing" in family_names
    assert "Other" in family_names
    assert "Momentum" in family_context.saturated_families
    assert "VWAP" in family_context.underexplored_families
    assert "Rejection-heavy families" in family_context.prompt_text

    payload = family_context.to_dict()
    assert payload["families"][0]["name"] in family_names


def test_regime_summary_builds_prompt_text_and_metric_means():
    service = ResearchExtensionService()
    labels = np.array(["bull", "bull", "bear", "bull", "bull", "bear"])
    metrics = {
        "volatility": np.array([0.10, 0.12, 0.30, 0.09, 0.11, 0.28], dtype=np.float64),
        "drawdown": np.array([0.03, 0.04, 0.11, 0.02, 0.05, 0.10], dtype=np.float64),
    }

    summary = service.regime_context(labels, metrics, lookback_window=3)

    assert summary.dominant_regime == "bull"
    assert summary.recent_regime == "bull"
    assert summary.counts["bull"] == 4
    assert summary.metrics["volatility"]["bull"]["count"] == 4.0
    assert summary.metrics["volatility"]["bear"]["count"] == 2.0
    assert "Regime theme cues" in summary.prompt_text


def test_dependence_profile_captures_nonlinear_relationship():
    service = ResearchExtensionService()
    assets = 64
    periods = 5
    base = np.linspace(-1.0, 1.0, assets, dtype=np.float64)[:, None]
    candidate = np.repeat(base, periods, axis=1)
    reference = candidate**2 + 0.02 * np.random.default_rng(42).normal(size=candidate.shape)

    profile = service.dependence_profile(candidate, reference)

    assert profile.sample_count > 0
    assert profile.period_count == periods
    assert profile.distance_correlation > 0.1
    assert profile.mutual_information_proxy >= 0.0
    assert profile.combined_score >= profile.distance_correlation
    assert profile.to_dict()["candidate_name"] == "candidate"


def test_ensemble_utility_is_positive_for_incremental_signal():
    service = ResearchExtensionService()
    assets = 48
    periods = 12
    exposure = np.linspace(-1.0, 1.0, assets, dtype=np.float64)[:, None]
    time_signal = np.sin(np.linspace(0.0, 2.0 * np.pi, periods, dtype=np.float64))[None, :]

    existing = exposure * time_signal
    candidate = (exposure**2) - float(np.mean(exposure**2))
    candidate = np.repeat(candidate, periods, axis=1)
    returns = (
        0.7 * existing
        + 0.9 * candidate
        + 0.03 * np.random.default_rng(7).normal(size=existing.shape)
    )

    summary = service.ensemble_utility(candidate, [existing], returns, train_fraction=0.67)

    assert summary.sample_count_train > 0
    assert summary.sample_count_test > 0
    assert summary.delta_r2 > 0.0
    assert summary.max_existing_distance_correlation >= 0.0
    assert "=== ENSEMBLE MARGINAL UTILITY ===" in summary.prompt_text
    assert "Candidate:" in summary.prompt_text


def test_prompt_context_extensions_merge_sections():
    service = ResearchExtensionService()
    labels = np.array(["bull", "bull", "bear", "bull"])
    candidate = np.repeat(np.linspace(-1.0, 1.0, 32, dtype=np.float64)[:, None], 4, axis=1)
    returns = candidate * 0.4

    payload = service.build_prompt_context_extensions(
        admitted_formulas=[{"formula": "Delta($close, 1)", "ic_mean": 0.05}],
        rejected_formulas=[{"formula": "Neg($volume)", "ic_mean": -0.01}],
        regime_labels=labels,
        regime_metrics={"volatility": np.array([0.1, 0.1, 0.3, 0.15], dtype=np.float64)},
        candidate=candidate,
        existing_signals=[candidate * 0.5],
        returns=returns,
        memory_signal={"recommended_directions": [{"template": "CsRank($vwap)"}]},
    )

    assert "family_context" in payload
    assert "regime_context" in payload
    assert "dependence_context" in payload
    assert "ensemble_utility" in payload
    assert "=== FACTOR FAMILY CONTEXT ===" in payload["prompt_text"]
    assert "=== REGIME CONTEXT ===" in payload["prompt_text"]
