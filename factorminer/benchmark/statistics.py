"""Statistical comparisons and report-oriented benchmark result contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MethodResult:
    """Flattened metrics for one method in a comparison report."""

    method: str
    library_ic: float = 0.0
    library_icir: float = 0.0
    avg_abs_rho: float = 0.0
    ew_ic: float = 0.0
    ew_icir: float = 0.0
    icw_ic: float = 0.0
    icw_icir: float = 0.0
    lasso_ic: float = 0.0
    lasso_icir: float = 0.0
    xgb_ic: float = 0.0
    xgb_icir: float = 0.0
    n_factors: int = 0
    admission_rate: float = 0.0
    elapsed_seconds: float = 0.0
    avg_turnover: float = 0.0
    ic_series: np.ndarray | None = field(default=None, repr=False)
    run_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("ic_series", None)
        return payload


@dataclass
class DMTestResult:
    """Diebold-Mariano comparison result."""

    dm_statistic: float
    p_value: float
    is_significant: bool
    direction: str
    n_obs: int


@dataclass
class AblationResult:
    """Flattened Phase-2 ablation report."""

    configs: list[str]
    results: dict[str, MethodResult]
    contributions: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "configs": self.configs,
            "results": {name: result.to_dict() for name, result in self.results.items()},
        }


@dataclass
class OperatorSpeedResult:
    """Compatibility container for operator timing results."""

    operator_timings_ms: dict[str, float]
    n_assets: int
    n_periods: int
    n_repeats: int


@dataclass
class PipelineSpeedResult:
    """Compatibility container for pipeline timing results."""

    total_seconds: float
    candidates_per_second: float
    n_candidates: int


@dataclass
class BenchmarkResult:
    """Report-oriented projection of canonical runtime benchmark artifacts."""

    methods: list[str]
    factor_library_metrics: pd.DataFrame
    combination_metrics: pd.DataFrame
    selection_metrics: pd.DataFrame
    speed_metrics: pd.DataFrame
    statistical_tests: dict[str, Any]
    ablation_result: AblationResult | None = None
    raw_method_results: dict[str, list[MethodResult]] = field(default_factory=dict)
    turnover_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    cost_pressure_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    runtime_artifacts: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _metric(frame: pd.DataFrame, method: str, column: str) -> float:
        if frame.empty or column not in frame.columns:
            return 0.0
        row = frame[frame["method"] == method]
        if row.empty or pd.isna(row.iloc[0][column]):
            return 0.0
        return float(row.iloc[0][column])

    def to_markdown_table(self) -> str:
        """Render the comparison in the historical GitHub table shape."""
        lines = [
            "| Method | IC (%) | ICIR | Avg|rho| | EW IC (%) | EW ICIR | "
            "ICW IC (%) | ICW ICIR | Las IC (%) | XGB IC (%) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method in self.methods:
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        f"{self._metric(self.factor_library_metrics, method, 'ic_pct'):.2f}",
                        f"{self._metric(self.factor_library_metrics, method, 'icir'):.3f}",
                        f"{self._metric(self.factor_library_metrics, method, 'avg_abs_rho'):.3f}",
                        f"{self._metric(self.combination_metrics, method, 'ew_ic_pct'):.2f}",
                        f"{self._metric(self.combination_metrics, method, 'ew_icir'):.3f}",
                        f"{self._metric(self.combination_metrics, method, 'icw_ic_pct'):.2f}",
                        f"{self._metric(self.combination_metrics, method, 'icw_icir'):.3f}",
                        f"{self._metric(self.selection_metrics, method, 'lasso_ic_pct'):.2f}",
                        f"{self._metric(self.selection_metrics, method, 'xgb_ic_pct'):.2f}",
                    ]
                )
                + " |"
            )
        return "\n".join(lines) + "\n"

    def to_latex_table(self) -> str:
        """Render a compact Table-1-style LaTeX table."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{HelixFactor vs FactorMiner Runtime Benchmark}",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"Method & IC(\%) & ICIR & Avg$|\rho|$ & EW IC(\%) & EW ICIR & ICW IC(\%) & ICW ICIR & Sel. IC(\%) \\",
            r"\midrule",
        ]
        for method in self.methods:
            selected = max(
                self._metric(self.selection_metrics, method, "lasso_ic_pct"),
                self._metric(self.selection_metrics, method, "xgb_ic_pct"),
            )
            lines.append(
                " & ".join(
                    [
                        method.replace("_", r"\_"),
                        f"{self._metric(self.factor_library_metrics, method, 'ic_pct'):.2f}",
                        f"{self._metric(self.factor_library_metrics, method, 'icir'):.3f}",
                        f"{self._metric(self.factor_library_metrics, method, 'avg_abs_rho'):.3f}",
                        f"{self._metric(self.combination_metrics, method, 'ew_ic_pct'):.2f}",
                        f"{self._metric(self.combination_metrics, method, 'ew_icir'):.3f}",
                        f"{self._metric(self.combination_metrics, method, 'icw_ic_pct'):.2f}",
                        f"{self._metric(self.combination_metrics, method, 'icw_icir'):.3f}",
                        f"{selected:.2f}",
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    def plot_comparison(self, save_path: str) -> None:
        """Plot the main comparison metrics."""
        import matplotlib.pyplot as plt

        columns = [
            (self.factor_library_metrics, "ic_pct", "IC (%)"),
            (self.factor_library_metrics, "icir", "ICIR"),
            (self.combination_metrics, "ew_ic_pct", "EW IC (%)"),
            (self.combination_metrics, "icw_ic_pct", "ICW IC (%)"),
        ]
        fig, axes = plt.subplots(1, len(columns), figsize=(16, 5))
        for axis, (frame, column, title) in zip(axes, columns, strict=True):
            values = [self._metric(frame, method, column) for method in self.methods]
            axis.bar(range(len(values)), values)
            axis.set_xticks(range(len(values)))
            axis.set_xticklabels([m.replace("_", "\n") for m in self.methods], fontsize=7)
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.3)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def generate_full_report(self, save_path: str) -> None:
        """Write a self-contained HTML report from the canonical result frames."""
        sections = [
            ("Factor Library Metrics", self.factor_library_metrics),
            ("Factor Combination Metrics", self.combination_metrics),
            ("Factor Selection Metrics", self.selection_metrics),
            ("Speed Metrics", self.speed_metrics),
        ]
        if not self.turnover_metrics.empty:
            sections.append(("Turnover", self.turnover_metrics))
        if not self.cost_pressure_metrics.empty:
            sections.append(("Cost Pressure", self.cost_pressure_metrics))
        body = "".join(
            f"<h2>{title}</h2>{frame.to_html(index=False, float_format=lambda x: f'{x:.4f}')}"
            for title, frame in sections
        )
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(
            "<!doctype html><meta charset='utf-8'><title>FactorMiner Benchmark</title>"
            "<style>body{font-family:sans-serif;margin:40px}table{border-collapse:collapse}"
            "th,td{padding:6px 10px;border:1px solid #ddd}</style>"
            f"<h1>FactorMiner Runtime Benchmark</h1>{body}",
            encoding="utf-8",
        )


class StatisticalComparisonTests:
    """Paired statistical comparisons used by Phase-2 reports."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.RandomState(seed)

    @staticmethod
    def _paired_valid_series(
        series_1: np.ndarray,
        series_2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        length = min(len(series_1), len(series_2))
        left = np.asarray(series_1[:length], dtype=np.float64)
        right = np.asarray(series_2[:length], dtype=np.float64)
        valid = np.isfinite(left) & np.isfinite(right)
        return left[valid], right[valid]

    def diebold_mariano_test(
        self,
        ic_series_1: np.ndarray,
        ic_series_2: np.ndarray,
        h: int = 1,
        max_lag: int | None = None,
    ) -> DMTestResult:
        """Compare squared IC loss with a Newey-West variance estimate.

        ``max_lag=None`` selects the Newey-West automatic truncation
        ``max(h - 1, floor(4 * (T / 100) ** (2 / 9)))`` so the Bartlett
        correction stays active for one-step forecasts (``h=1``), where the
        classical ``h - 1`` rule degenerates to the iid variance. Pass
        ``max_lag=0`` to force the uncorrected iid variance.
        """
        from scipy.stats import norm

        if h < 1:
            raise ValueError("h must be at least 1")
        if max_lag is not None and max_lag < 0:
            raise ValueError("max_lag must be non-negative")

        series_1, series_2 = self._paired_valid_series(ic_series_1, ic_series_2)
        if len(series_1) < 5:
            return DMTestResult(0.0, 1.0, False, "no_difference", len(series_1))
        differential = series_1**2 - series_2**2
        if np.allclose(differential, 0.0):
            return DMTestResult(0.0, 1.0, False, "no_difference", len(differential))
        mean = float(np.mean(differential))
        variance = float(np.var(differential, ddof=0))
        n_obs = len(differential)
        if max_lag is None:
            max_lag = max(h - 1, int(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))
        max_lag = min(max_lag, n_obs - 1)
        for lag in range(1, max_lag + 1):
            covariance = float(np.mean((differential[lag:] - mean) * (differential[:-lag] - mean)))
            variance += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
        if variance <= 0.0 or not np.isfinite(variance):
            return DMTestResult(0.0, 1.0, False, "no_difference", len(differential))
        statistic = mean / math.sqrt(variance / len(differential))
        p_value = 2.0 * (1.0 - float(norm.cdf(abs(statistic))))
        direction = "no_difference"
        if abs(statistic) >= 1.96:
            direction = "ralph_better" if mean > 0 else "helix_better"
        return DMTestResult(
            float(statistic),
            float(p_value),
            bool(p_value < 0.05),
            direction,
            len(differential),
        )

    def paired_t_test(self, series_1: np.ndarray, series_2: np.ndarray) -> dict[str, Any]:
        from scipy.stats import ttest_rel

        left, right = self._paired_valid_series(series_1, series_2)
        if len(left) < 5:
            return {"t_stat": 0.0, "p_value": 1.0, "mean_diff": 0.0, "n": len(left)}
        statistic, p_value = ttest_rel(left, right)
        if not np.isfinite(statistic) or not np.isfinite(p_value):
            return {"t_stat": 0.0, "p_value": 1.0, "mean_diff": 0.0, "n": len(left)}
        return {
            "t_stat": float(statistic),
            "p_value": float(p_value),
            "mean_diff": float(np.mean(left - right)),
            "n": len(left),
        }

    def bootstrap_ic_difference_ci(
        self,
        series_1: np.ndarray,
        series_2: np.ndarray,
        n_bootstrap: int = 1000,
        block_size: int = 20,
    ) -> tuple[float, float]:
        left, right = self._paired_valid_series(series_1, series_2)
        if len(left) < 5:
            return 0.0, 0.0
        difference = left - right
        block_size = max(1, min(block_size, len(difference) // 2))
        block_count = math.ceil(len(difference) / block_size)
        means = np.empty(n_bootstrap)
        for index in range(n_bootstrap):
            starts = self._rng.randint(
                0,
                len(difference) - block_size + 1,
                size=block_count,
            )
            sampled = np.concatenate([np.arange(start, start + block_size) for start in starts])[
                : len(difference)
            ]
            means[index] = float(np.mean(difference[sampled]))
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

    def wilcoxon_test(self, series_1: np.ndarray, series_2: np.ndarray) -> dict[str, Any]:
        from scipy.stats import wilcoxon

        left, right = self._paired_valid_series(series_1, series_2)
        if len(left) < 5:
            return {"statistic": 0.0, "p_value": 1.0, "n": len(left)}
        try:
            statistic, p_value = wilcoxon(left, right, alternative="two-sided")
        except ValueError:
            statistic, p_value = 0.0, 1.0
        return {"statistic": float(statistic), "p_value": float(p_value), "n": len(left)}

    def run_all_tests(self, ic_helix: np.ndarray, ic_ralph: np.ndarray) -> dict[str, Any]:
        dm = self.diebold_mariano_test(ic_helix, ic_ralph)
        t_test = self.paired_t_test(ic_helix, ic_ralph)
        lower, upper = self.bootstrap_ic_difference_ci(ic_helix, ic_ralph)
        wilcoxon_result = self.wilcoxon_test(ic_helix, ic_ralph)
        helix, ralph = self._paired_valid_series(ic_helix, ic_ralph)
        mean_difference = float(np.mean(helix - ralph)) if len(helix) else 0.0
        return {
            "diebold_mariano": {
                "dm_stat": dm.dm_statistic,
                "p_value": dm.p_value,
                "significant": dm.is_significant,
                "direction": dm.direction,
                "n_obs": dm.n_obs,
            },
            "paired_t_test": t_test,
            "bootstrap_ci_95": {
                "lower": lower,
                "upper": upper,
                "excludes_zero": lower > 0 or upper < 0,
            },
            "wilcoxon": wilcoxon_result,
            "mean_ic_difference": mean_difference,
            "helix_outperforms": mean_difference > 0,
        }
