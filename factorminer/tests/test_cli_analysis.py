"""Focused CLI analysis tests for evaluate, combine, and visualize."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from click.testing import CliRunner

from factorminer.cli import main
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import save_library
from factorminer.evaluation.runtime import DatasetSplit, FactorEvaluationArtifact


@dataclass
class _FakeDataset:
    """Small runtime dataset stub sufficient for analysis CLI commands."""

    asset_ids: np.ndarray
    timestamps: np.ndarray
    splits: dict[str, DatasetSplit]

    def get_split(self, name: str) -> DatasetSplit:
        return self.splits[name]


def _make_stats(
    ic_mean: float,
    ic_abs_mean: float,
    icir: float,
    ic_win_rate: float,
    turnover: float,
) -> dict:
    return {
        "ic_mean": ic_mean,
        "ic_paper_mean": abs(ic_mean),
        "ic_abs_mean": ic_abs_mean,
        "icir": icir,
        "ic_paper_icir": abs(icir),
        "ic_win_rate": ic_win_rate,
        "turnover": turnover,
        "ic_series": np.array([ic_mean, -ic_mean / 2.0, ic_mean / 3.0], dtype=np.float64),
        "Q1": -0.02,
        "Q2": -0.01,
        "Q3": 0.0,
        "Q4": 0.01,
        "Q5": 0.02,
        "long_short": 0.04,
        "monotonicity": 1.0,
    }


def _make_artifact(
    factor_id: int,
    name: str,
    train_abs_ic: float,
    test_abs_ic: float,
) -> FactorEvaluationArtifact:
    train_signal = np.full((2, 3), float(factor_id), dtype=np.float64)
    test_signal = np.full((2, 3), float(factor_id) * 10.0, dtype=np.float64)
    full_signal = np.concatenate([train_signal, test_signal], axis=1)

    return FactorEvaluationArtifact(
        factor_id=factor_id,
        name=name,
        formula="Neg($close)",
        category="test",
        parse_ok=True,
        signals_full=full_signal,
        split_signals={
            "train": train_signal,
            "test": test_signal,
            "full": full_signal,
        },
        split_stats={
            "train": _make_stats(0.05 * factor_id, train_abs_ic, 1.0 + factor_id, 0.6, 0.1),
            "test": _make_stats(-0.04 * factor_id, test_abs_ic, 0.8 + factor_id, 0.4, 0.2),
            "full": _make_stats(0.01 * factor_id, max(train_abs_ic, test_abs_ic), 0.9, 0.5, 0.15),
        },
    )


def _make_dataset() -> _FakeDataset:
    timestamps = np.array(
        [
            np.datetime64("2025-01-01"),
            np.datetime64("2025-01-02"),
            np.datetime64("2025-01-03"),
            np.datetime64("2025-01-04"),
            np.datetime64("2025-01-05"),
            np.datetime64("2025-01-06"),
        ]
    )
    returns = np.zeros((2, 3), dtype=np.float64)
    return _FakeDataset(
        asset_ids=np.array(["A", "B"]),
        timestamps=timestamps,
        splits={
            "train": DatasetSplit(
                name="train",
                indices=np.array([0, 1, 2]),
                timestamps=timestamps[:3],
                returns=returns,
            ),
            "test": DatasetSplit(
                name="test",
                indices=np.array([3, 4, 5]),
                timestamps=timestamps[3:],
                returns=returns,
            ),
            "full": DatasetSplit(
                name="full",
                indices=np.array([0, 1, 2, 3, 4, 5]),
                timestamps=timestamps,
                returns=np.zeros((2, 6), dtype=np.float64),
            ),
        },
    )


def _save_test_library(tmp_path) -> str:
    library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)
    library.admit_factor(
        Factor(
            id=0,
            name="factor_one",
            formula="Neg($close)",
            category="test",
            ic_mean=9.99,
            icir=8.88,
            ic_win_rate=0.99,
            max_correlation=0.0,
            batch_number=1,
        )
    )
    library.admit_factor(
        Factor(
            id=0,
            name="factor_two",
            formula="Neg($open)",
            category="test",
            ic_mean=7.77,
            icir=6.66,
            ic_win_rate=0.95,
            max_correlation=0.0,
            batch_number=1,
        )
    )
    base_path = tmp_path / "factor_library"
    save_library(library, base_path, save_signals=False)
    return str(base_path.with_suffix(".json"))


def test_evaluate_recomputes_and_selects_top_k_by_train_split(tmp_path, monkeypatch):
    """`evaluate --period both` should use recomputed train metrics for top-k."""
    library_path = _save_test_library(tmp_path)
    dataset = _make_dataset()
    artifacts = [
        _make_artifact(1, "factor_one", train_abs_ic=0.20, test_abs_ic=0.90),
        _make_artifact(2, "factor_two", train_abs_ic=0.70, test_abs_ic=0.10),
    ]

    monkeypatch.setattr(
        "factorminer.cli._load_runtime_dataset_for_analysis",
        lambda cfg, data_path, mock: dataset,
    )
    monkeypatch.setattr(
        "factorminer.cli._recompute_analysis_artifacts",
        lambda library, dataset_arg, signal_failure_policy: artifacts,
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--cpu",
            "--output-dir",
            str(tmp_path / "out"),
            "evaluate",
            library_path,
            "--mock",
            "--period",
            "both",
            "--top-k",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Evaluating top 1 factors by train paper IC for train/test comparison" in result.output
    assert "factor_two" in result.output
    assert "factor_one" not in result.output
    assert "0.7000" in result.output
    assert "9.9900" not in result.output
    assert "Decay summary (train -> test)" in result.output


def test_combine_uses_fit_split_for_factor_preselection(tmp_path, monkeypatch):
    """`combine` should pre-select factors by fit split rather than eval split."""
    library_path = _save_test_library(tmp_path)
    dataset = _make_dataset()
    artifacts = [
        _make_artifact(1, "factor_one", train_abs_ic=0.20, test_abs_ic=0.90),
        _make_artifact(2, "factor_two", train_abs_ic=0.70, test_abs_ic=0.10),
    ]
    captured_factor_ids: list[int] = []

    monkeypatch.setattr(
        "factorminer.cli._load_runtime_dataset_for_analysis",
        lambda cfg, data_path, mock: dataset,
    )
    monkeypatch.setattr(
        "factorminer.cli._recompute_analysis_artifacts",
        lambda library, dataset_arg, signal_failure_policy: artifacts,
    )

    def _capture_equal_weight(self, factor_signals):
        captured_factor_ids.extend(sorted(factor_signals.keys()))
        return next(iter(factor_signals.values()))

    monkeypatch.setattr(
        "factorminer.evaluation.combination.FactorCombiner.equal_weight",
        _capture_equal_weight,
    )
    monkeypatch.setattr(
        "factorminer.evaluation.portfolio.PortfolioBacktester.quintile_backtest",
        lambda self, combined_signal, returns, transaction_cost_bps=0: {
            "ic_mean": 0.12,
            "icir": 1.23,
            "ls_return": 0.04,
            "monotonicity": 1.0,
            "avg_turnover": 0.10,
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--cpu",
            "--output-dir",
            str(tmp_path / "out"),
            "combine",
            library_path,
            "--mock",
            "--fit-period",
            "train",
            "--eval-period",
            "test",
            "--method",
            "equal-weight",
            "--top-k",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Pre-selected top 1 factors by train paper IC" in result.output
    assert "Fit split:  train" in result.output
    assert "Eval split: test" in result.output
    assert captured_factor_ids == [2]


def test_visualize_defaults_factor_specific_plots_to_split_top_factor(tmp_path, monkeypatch):
    """`visualize` should default factor-specific plots to the split top factor."""
    library_path = _save_test_library(tmp_path)
    dataset = _make_dataset()
    artifacts = [
        _make_artifact(1, "factor_one", train_abs_ic=0.80, test_abs_ic=0.20),
        _make_artifact(2, "factor_two", train_abs_ic=0.30, test_abs_ic=0.90),
    ]
    ic_paths: list[str] = []
    quintile_paths: list[str] = []

    monkeypatch.setattr(
        "factorminer.cli._load_runtime_dataset_for_analysis",
        lambda cfg, data_path, mock: dataset,
    )
    monkeypatch.setattr(
        "factorminer.cli._recompute_analysis_artifacts",
        lambda library, dataset_arg, signal_failure_policy: artifacts,
    )
    monkeypatch.setattr(
        "factorminer.utils.visualization.plot_ic_timeseries",
        lambda ic_series, dates, rolling_window=21, title="", save_path=None: ic_paths.append(save_path),
    )
    monkeypatch.setattr(
        "factorminer.utils.visualization.plot_quintile_returns",
        lambda quintile_returns, title="", save_path=None: quintile_paths.append(save_path),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--cpu",
            "--output-dir",
            str(tmp_path / "viz"),
            "visualize",
            library_path,
            "--mock",
            "--period",
            "test",
            "--ic-timeseries",
            "--quintile",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Defaulted to factor #2 factor_two for factor-specific plots." in result.output
    assert ic_paths and all("factor_2" in path for path in ic_paths)
    assert quintile_paths and all("factor_2" in path for path in quintile_paths)
    assert not any("factor_1" in path for path in ic_paths + quintile_paths)
