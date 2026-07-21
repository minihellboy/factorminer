"""Focused operator and pipeline speed benchmarks."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from factorminer.benchmark.catalogs import build_random_exploration
from factorminer.benchmark.statistics import OperatorSpeedResult, PipelineSpeedResult


def _time_callable(fn: Callable[[], object], repeats: int = 3) -> float:
    started_at = time.perf_counter()
    for _ in range(max(repeats, 1)):
        fn()
    return (time.perf_counter() - started_at) * 1000.0 / max(repeats, 1)


def _build_mock_data_dict(
    n_assets: int = 100,
    n_periods: int = 500,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Build the historical matrix dictionary through the shared data pipeline."""
    from factorminer.data.mock_data import MockConfig, generate_mock_data
    from factorminer.data.preprocessor import preprocess

    raw = generate_mock_data(
        MockConfig(
            num_assets=n_assets,
            num_periods=n_periods,
            frequency="10min",
            plant_alpha=True,
            alpha_strength=0.04,
            alpha_assets_frac=0.4,
            seed=seed,
        )
    )
    processed = preprocess(raw)
    assets = sorted(processed["asset_id"].unique())
    period_count = int(processed.groupby("asset_id").size().min())
    feature_map = {
        "$open": "open",
        "$high": "high",
        "$low": "low",
        "$close": "close",
        "$volume": "volume",
        "$amt": "amount",
        "$vwap": "vwap",
        "$returns": "returns",
    }
    data: dict[str, np.ndarray] = {}
    for feature, column in feature_map.items():
        if column not in processed.columns:
            continue
        pivot = processed.pivot(index="asset_id", columns="datetime", values=column)
        data[feature] = pivot.loc[assets].iloc[:, :period_count].values.astype(np.float64)
    close = data["$close"]
    forward_returns = np.roll(close, -1, axis=1) / close - 1.0
    forward_returns[:, -1] = np.nan
    data["forward_returns"] = forward_returns
    return data


class SpeedBenchmark:
    """Compatibility adapter over the canonical operator runtime."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def run_operator_benchmark(
        self,
        n_assets: int = 500,
        n_periods: int = 2000,
        n_repeats: int = 5,
    ) -> OperatorSpeedResult:
        from factorminer.operators.registry import execute_operator

        assets = min(n_assets, 50)
        periods = min(n_periods, 100)
        left = np.random.RandomState(self.seed).randn(assets, periods)
        right = np.random.RandomState(self.seed + 1).randn(assets, periods)
        runners = {
            "Add": lambda: execute_operator("Add", left, right, backend="numpy"),
            "Mean": lambda: execute_operator("Mean", left, params={"window": 20}, backend="numpy"),
            "Corr": lambda: execute_operator(
                "Corr", left, right, params={"window": 20}, backend="numpy"
            ),
            "CsRank": lambda: execute_operator("CsRank", left, backend="numpy"),
        }
        timings = {
            name: _time_callable(runner, repeats=n_repeats) for name, runner in runners.items()
        }
        return OperatorSpeedResult(timings, n_assets, n_periods, n_repeats)

    def run_full_pipeline_benchmark(
        self,
        n_candidates: int = 200,
        data: dict[str, np.ndarray] | None = None,
    ) -> PipelineSpeedResult:
        from factorminer.core.parser import try_parse
        from factorminer.evaluation.metrics import compute_ic, compute_ic_mean

        inputs = data or _build_mock_data_dict(n_assets=100, n_periods=200, seed=self.seed)
        returns = inputs.get("forward_returns", inputs["$close"])
        entries = build_random_exploration(seed=self.seed, count=n_candidates)
        start = time.perf_counter()
        succeeded = 0
        for entry in entries[:n_candidates]:
            tree = try_parse(entry.formula)
            if tree is None:
                continue
            try:
                compute_ic_mean(compute_ic(tree.evaluate(inputs), returns))
            except Exception:
                continue
            succeeded += 1
        elapsed = time.perf_counter() - start
        return PipelineSpeedResult(
            elapsed,
            succeeded / max(elapsed, 1e-6),
            n_candidates,
        )

    def generate_speed_table(
        self,
        op_result: OperatorSpeedResult,
        pipeline_result: PipelineSpeedResult,
    ) -> str:
        rows = [
            r"\begin{tabular}{lrr}",
            r"\toprule",
            "Task & Time (ms) & Type \\\\",
        ]
        rows.extend(
            f"{name} & {milliseconds:.2f} & operator \\\\"
            for name, milliseconds in op_result.operator_timings_ms.items()
        )
        rows.append(f"Pipeline & {pipeline_result.total_seconds * 1000:.2f} & pipeline \\\\")
        rows.extend([r"\bottomrule", r"\end{tabular}"])
        return "\n".join(rows)
