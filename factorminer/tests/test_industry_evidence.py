"""Tests for the auditable industry evidence protocol."""

from __future__ import annotations

import json

import numpy as np
import pytest

from factorminer.evaluation.evidence import (
    INDUSTRY_EVIDENCE_VERSION,
    IndustryEvidenceConfig,
    compute_hac_mean_test,
    evaluate_industry_evidence,
    residualize_against_risk_exposures,
)
from factorminer.evaluation.portfolio import PortfolioBacktester


def test_hac_mean_test_accounts_for_positive_serial_dependence():
    rng = np.random.default_rng(7)
    innovations = rng.normal(0.0, 0.03, 500)
    series = np.empty_like(innovations)
    series[0] = innovations[0]
    for index in range(1, len(series)):
        series[index] = 0.85 * series[index - 1] + innovations[index]

    result = compute_hac_mean_test(series, lags=8)
    iid_standard_error = float(np.std(series, ddof=1) / np.sqrt(len(series)))

    assert result.lags == 8
    assert result.n_observations == len(series)
    assert result.standard_error > iid_standard_error
    assert 0.0 <= result.p_value <= 1.0


def test_risk_residualization_removes_static_and_dynamic_exposures():
    rng = np.random.default_rng(11)
    assets, periods = 80, 20
    size = np.linspace(-1.0, 1.0, assets)
    industry = np.where(np.arange(assets) < assets // 2, 0.0, 1.0)
    static_exposures = np.column_stack([size, industry])
    idiosyncratic = rng.normal(0.0, 0.1, (assets, periods))
    signals = 2.5 * size[:, None] - 1.3 * industry[:, None] + idiosyncratic

    result = residualize_against_risk_exposures(
        signals,
        static_exposures,
        exposure_names=["size", "industry_software"],
    )

    assert result.periods_residualized == periods
    assert result.periods_skipped == 0
    assert result.mean_r2 > 0.95
    for period in range(periods):
        residual = result.residual_signals[:, period]
        assert abs(np.corrcoef(residual, size)[0, 1]) < 1e-10
        assert abs(np.corrcoef(residual, industry)[0, 1]) < 1e-10

    dynamic = np.broadcast_to(static_exposures[:, None, :], (assets, periods, 2)).copy()
    dynamic[:, :, 0] *= np.linspace(0.8, 1.2, periods)
    dynamic_result = residualize_against_risk_exposures(signals, dynamic)
    assert dynamic_result.residual_signals.shape == signals.shape


def test_evidence_report_composes_metrics_costs_risk_and_fdr():
    rng = np.random.default_rng(19)
    assets, periods = 100, 120
    style = rng.normal(size=assets)
    idiosyncratic_signal = rng.normal(size=(assets, periods))
    signals = 0.7 * style[:, None] + idiosyncratic_signal
    returns = 0.002 * idiosyncratic_signal + rng.normal(0.0, 0.01, (assets, periods))
    family = {
        "noise_a": rng.normal(0.0, 0.05, periods),
        "noise_b": rng.normal(0.0, 0.05, periods),
    }
    config = IndustryEvidenceConfig(
        cost_bps=(0.0, 20.0),
        primary_cost_bps=20.0,
        bootstrap_n_samples=100,
        bootstrap_block_size=10,
        seed=3,
    )

    report = evaluate_industry_evidence(
        "candidate",
        signals,
        returns,
        config=config,
        risk_exposures=style[:, None],
        exposure_names=["style"],
        family_ic_series=family,
        n_trials=25,
    )
    payload = report.to_dict()

    assert payload["protocol_version"] == INDUSTRY_EVIDENCE_VERSION
    assert payload["metric_contract"]["legacy_ic_alias"] == "spearman_rank_ic"
    assert payload["raw_signal"]["pearson_ic"]["correlation"] == "pearson"
    assert payload["raw_signal"]["rank_ic"]["correlation"] == "spearman_rank"
    assert payload["risk_residual"] is not None
    assert payload["validation_coverage"]["risk_residualization"]["status"] == "measured"
    assert payload["validation_coverage"]["multiple_testing"]["status"] == "measured"
    assert payload["significance"]["fdr"]["factor_adjusted_p_value"] >= 0.0

    cost_curve = payload["portfolio"]["raw_signal"]["cost_curve"]
    assert cost_curve[1]["mean_net_return"] <= cost_curve[0]["mean_net_return"]
    json.dumps(payload, allow_nan=False)


def test_evidence_report_marks_inputs_it_cannot_infer():
    rng = np.random.default_rng(23)
    signals = rng.normal(size=(30, 40))
    returns = rng.normal(0.0, 0.01, size=(30, 40))
    config = IndustryEvidenceConfig(bootstrap_n_samples=20)

    report = evaluate_industry_evidence("raw_only", signals, returns, config=config)

    assert report.validation_coverage["risk_residualization"]["status"] == "not_supplied"
    assert report.validation_coverage["point_in_time_data"]["status"] == "external_required"
    assert report.validation_coverage["walk_forward_freeze"]["status"] == "external_required"
    assert report.risk_residual is None


def test_evidence_report_does_not_mark_empty_measurements_complete():
    signals = np.full((10, 8), np.nan)
    returns = np.zeros_like(signals)

    report = evaluate_industry_evidence(
        "empty",
        signals,
        returns,
        config=IndustryEvidenceConfig(bootstrap_n_samples=10),
    )

    assert report.validation_coverage["ic_rankic"]["status"] == "partial"
    assert report.validation_coverage["serial_dependence"]["status"] == "partial"
    assert report.validation_coverage["turnover_cost_stress"]["status"] == "partial"


def test_cost_path_ignores_nan_returns_outside_selected_assets():
    assets, periods = 20, 8
    signals = np.tile(np.arange(assets, dtype=np.float64)[:, None], (1, periods))
    returns = signals * 0.001
    returns[assets // 2, :] = np.nan

    report = evaluate_industry_evidence(
        "nan_cross_section",
        signals,
        returns,
        config=IndustryEvidenceConfig(bootstrap_n_samples=10),
    )
    gross = report.portfolio["raw_signal"]["gross_return_series"]

    assert np.all(np.isfinite(gross))
    assert np.all(gross > 0.0)


def test_cost_path_does_not_use_future_return_availability_to_select_assets():
    assets, periods = 20, 8
    signals = np.tile(np.arange(assets, dtype=np.float64)[:, None], (1, periods))
    returns = signals * 0.001
    returns[-1, 3] = np.nan

    report = evaluate_industry_evidence(
        "missing_selected_return",
        signals,
        returns,
        config=IndustryEvidenceConfig(bootstrap_n_samples=10),
    )
    gross = np.asarray(report.portfolio["raw_signal"]["gross_return_series"])

    assert np.isnan(gross[3])
    assert np.all(np.isfinite(np.delete(gross, 3)))


@pytest.mark.parametrize("bad_cost", [float("nan"), float("inf"), -1.0])
def test_evidence_config_rejects_non_finite_or_negative_costs(bad_cost):
    with pytest.raises(ValueError, match="transaction-cost levels"):
        IndustryEvidenceConfig(cost_bps=(bad_cost,))


@pytest.mark.parametrize("bad_periods", [float("nan"), float("inf"), 0.0])
def test_evidence_config_rejects_invalid_annualization(bad_periods):
    with pytest.raises(ValueError, match="periods_per_year"):
        IndustryEvidenceConfig(periods_per_year=bad_periods)


def test_portfolio_backtest_retains_rankic_alias_and_adds_pearson():
    rng = np.random.default_rng(29)
    signals = rng.normal(size=(30, 25))
    returns = 0.01 * signals + rng.normal(0.0, 0.01, size=(30, 25))

    result = PortfolioBacktester().quintile_backtest(signals, returns)

    assert result["ic_definition"] == "spearman_rank"
    np.testing.assert_allclose(result["ic_series"], result["rank_ic_series"])
    assert result["pearson_ic_series"].shape == result["rank_ic_series"].shape
    assert result["pearson_ic_mean"] > 0.0


def test_portfolio_membership_does_not_use_future_return_availability():
    signals = np.arange(10, dtype=np.float64)[None, :]
    returns = np.ones_like(signals)
    returns[0, -1] = np.nan

    result = PortfolioBacktester().quintile_backtest(signals, returns)

    assert np.isnan(result["q5_return"])


def test_evidence_rejects_misaligned_panels():
    with pytest.raises(ValueError, match="identical shapes"):
        evaluate_industry_evidence(
            "bad",
            np.zeros((10, 20)),
            np.zeros((10, 19)),
            config=IndustryEvidenceConfig(bootstrap_n_samples=10),
        )
