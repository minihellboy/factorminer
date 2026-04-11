"""Research-mode target, scoring, and model-suite coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorminer.core.factor_library import Factor
from factorminer.data.tensor_builder import TargetSpec, compute_targets
from factorminer.evaluation.portfolio import PortfolioBacktester
from factorminer.evaluation.research import (
    FactorGeometryDiagnostics,
    build_score_vector,
    passes_research_admission,
    run_research_model_suite,
)
from factorminer.evaluation.runtime import DatasetSplit, EvaluationDataset, evaluate_factors
from factorminer.utils.config import load_config


def test_compute_targets_supports_multiple_horizons():
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=5, freq="D").tolist() * 2,
            "asset_id": ["A"] * 5 + ["B"] * 5,
            "open": [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            "high": [11, 12, 13, 14, 15, 21, 22, 23, 24, 25],
            "low": [9, 10, 11, 12, 13, 19, 20, 21, 22, 23],
            "close": [10.5, 11.5, 12.5, 13.5, 14.5, 20.5, 21.5, 22.5, 23.5, 24.5],
            "volume": [1] * 10,
            "amount": [10] * 10,
        }
    )
    out = compute_targets(
        df,
        [
            TargetSpec("paper", 1, 1, "open_to_close", "simple"),
            TargetSpec("h2_close_to_close", 0, 2, "close_to_close", "simple"),
        ],
    )

    a0 = out[(out["asset_id"] == "A")].sort_values("datetime").reset_index(drop=True)
    assert a0.loc[0, "target"] == pytest.approx(11.5 / 11.0 - 1.0)
    assert a0.loc[0, "target_h2_close_to_close"] == pytest.approx(12.5 / 10.5 - 1.0)


def test_evaluate_factors_records_all_target_stats(small_data):
    timestamps = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(50)]
    )
    returns = small_data["$returns"]
    data_tensor = np.stack(
        [
            small_data["$open"],
            small_data["$high"],
            small_data["$low"],
            small_data["$close"],
            small_data["$volume"],
            small_data["$amt"],
            small_data["$vwap"],
            small_data["$returns"],
        ],
        axis=-1,
    )
    alt_returns = -returns
    splits = {
        "train": DatasetSplit(
            name="train",
            indices=np.arange(25),
            timestamps=timestamps[:25],
            returns=returns[:, :25],
            target_returns={"paper": returns[:, :25], "alt": alt_returns[:, :25]},
            default_target="paper",
        ),
        "test": DatasetSplit(
            name="test",
            indices=np.arange(25, 50),
            timestamps=timestamps[25:],
            returns=returns[:, 25:],
            target_returns={"paper": returns[:, 25:], "alt": alt_returns[:, 25:]},
            default_target="paper",
        ),
        "full": DatasetSplit(
            name="full",
            indices=np.arange(50),
            timestamps=timestamps,
            returns=returns,
            target_returns={"paper": returns, "alt": alt_returns},
            default_target="paper",
        ),
    }
    dataset = EvaluationDataset(
        data_dict=small_data,
        data_tensor=data_tensor,
        returns=returns,
        timestamps=timestamps,
        asset_ids=np.array([f"A{i:02d}" for i in range(returns.shape[0])]),
        splits=splits,
        processed_df=pd.DataFrame(),
        target_panels={"paper": returns, "alt": alt_returns},
        default_target="paper",
    )
    factor = Factor(
        id=1,
        name="close_neg",
        formula="Neg($close)",
        category="test",
        ic_mean=0.0,
        icir=0.0,
        ic_win_rate=0.0,
        max_correlation=0.0,
        batch_number=1,
    )

    artifact = evaluate_factors([factor], dataset, signal_failure_policy="reject")[0]

    assert artifact.succeeded
    assert set(artifact.target_stats["train"]) == {"paper", "alt"}
    assert artifact.target_stats["train"]["paper"]["ic_mean"] == pytest.approx(
        -artifact.target_stats["train"]["alt"]["ic_mean"]
    )


def test_research_score_vector_and_admission():
    cfg = load_config(
        overrides={
            "benchmark": {"mode": "research"},
            "research": {
                "enabled": True,
                "horizon_weights": {"h1": 0.7, "h3": 0.3},
            },
        }
    )
    score = build_score_vector(
        target_stats={
            "h1": {
                "ic_mean": 0.08,
                "ic_abs_mean": 0.08,
                "icir": 1.1,
                "turnover": 0.2,
                "ic_series": np.array([0.07, 0.08, 0.09, 0.08, 0.07]),
            },
            "h3": {
                "ic_mean": 0.05,
                "ic_abs_mean": 0.05,
                "icir": 0.8,
                "turnover": 0.1,
                "ic_series": np.array([0.03, 0.05, 0.06, 0.05, 0.04]),
            },
        },
        target_horizons={"h1": 1, "h3": 3},
        research_cfg=cfg.research,
        geometry=FactorGeometryDiagnostics(
            max_abs_correlation=0.2,
            mean_abs_correlation=0.1,
            projection_loss=0.25,
            marginal_span_gain=0.75,
            effective_rank_gain=0.4,
            residual_ic=0.06,
        ),
    )

    assert score.primary_score > 0.0
    assert score.lower_confidence_bound >= 0.0
    admitted, reason = passes_research_admission(
        score,
        cfg.research,
        correlation_threshold=0.5,
    )
    assert admitted is True
    assert "admission" in reason.lower()


def test_research_model_suite_reports_net_ir():
    cfg = load_config(
        overrides={
            "benchmark": {"mode": "research"},
            "research": {
                "enabled": True,
                "selection": {
                    "models": ["ridge", "lasso"],
                    "rolling_train_window": 20,
                    "rolling_test_window": 10,
                    "rolling_step": 10,
                },
                "regimes": {"enabled": False},
                "execution": {"cost_bps": 0.0},
            },
        }
    )
    rng = np.random.default_rng(42)
    t, n = 60, 8
    base = rng.normal(size=(t, n))
    factor_signals = {
        1: base,
        2: rng.normal(size=(t, n)),
        3: 0.5 * base + 0.1 * rng.normal(size=(t, n)),
    }
    returns = 0.03 * base + 0.01 * rng.normal(size=(t, n))

    reports = run_research_model_suite(factor_signals, returns, cfg.research)

    assert "ridge" in reports
    assert reports["ridge"]["available"] is True
    assert "mean_test_net_ir" in reports["ridge"]
    assert reports["ridge"]["selection_stability"] >= 0.0


def test_portfolio_backtest_exposes_raw_series():
    backtester = PortfolioBacktester()
    signal = np.array(
        [
            [1.0, 0.5, -0.2, -1.0, 0.2],
            [1.1, 0.2, -0.3, -0.8, 0.0],
            [0.9, 0.1, -0.5, -1.1, 0.3],
            [1.2, 0.4, -0.1, -0.9, 0.1],
            [1.0, 0.3, -0.4, -1.2, 0.2],
        ]
    )
    returns = np.array(
        [
            [0.03, 0.01, -0.01, -0.02, 0.00],
            [0.02, 0.00, -0.01, -0.03, 0.01],
            [0.01, 0.02, -0.02, -0.01, 0.00],
            [0.03, 0.01, -0.01, -0.02, 0.00],
            [0.02, 0.00, -0.03, -0.01, 0.01],
        ]
    )

    stats = backtester.quintile_backtest(signal, returns, transaction_cost_bps=4.0)

    assert stats["ls_net_series"].shape[0] == signal.shape[0]
    assert stats["turnover_series"].shape[0] == signal.shape[0]
    assert stats["quintile_period_returns"].shape == (signal.shape[0], 5)
