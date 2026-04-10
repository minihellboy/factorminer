"""Pluggable dependence metrics for library redundancy and replacement logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def _iter_valid_columns(
    signals_a: np.ndarray,
    signals_b: np.ndarray,
):
    if signals_a.shape != signals_b.shape:
        raise ValueError(
            f"Signal shapes must match: {signals_a.shape} vs {signals_b.shape}"
        )

    _, periods = signals_a.shape
    for period in range(periods):
        col_a = signals_a[:, period]
        col_b = signals_b[:, period]
        valid = ~(np.isnan(col_a) | np.isnan(col_b))
        if int(valid.sum()) < 3:
            continue
        yield col_a[valid], col_b[valid]


def _pearson_abs(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if float(denom) < 1e-12:
        return 0.0
    return float(abs(np.sum(x_centered * y_centered) / denom))


def _distance_correlation_abs(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    if x.shape[0] < 3:
        return 0.0

    dist_x = np.abs(x - x.T)
    dist_y = np.abs(y - y.T)

    ax = dist_x - dist_x.mean(axis=0) - dist_x.mean(axis=1, keepdims=True) + dist_x.mean()
    ay = dist_y - dist_y.mean(axis=0) - dist_y.mean(axis=1, keepdims=True) + dist_y.mean()

    dcov2 = float(np.mean(ax * ay))
    dvar_x = float(np.mean(ax * ax))
    dvar_y = float(np.mean(ay * ay))
    if dvar_x <= 0.0 or dvar_y <= 0.0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvar_x * dvar_y)))


@dataclass(frozen=True)
class DependenceMetric(ABC):
    """Abstract pairwise dependence metric over cross-sectional signals."""

    name: str

    @abstractmethod
    def compute(self, signals_a: np.ndarray, signals_b: np.ndarray) -> float:
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "family": "pairwise_time_averaged_dependence"}


@dataclass(frozen=True)
class SpearmanDependenceMetric(DependenceMetric):
    """Mean absolute Spearman rank correlation across time."""

    name: str = "spearman"

    def compute(self, signals_a: np.ndarray, signals_b: np.ndarray) -> float:
        scores: list[float] = []
        for col_a, col_b in _iter_valid_columns(signals_a, signals_b):
            scores.append(_pearson_abs(rankdata(col_a), rankdata(col_b)))
        if not scores:
            return 0.0
        return float(np.mean(scores))


@dataclass(frozen=True)
class PearsonDependenceMetric(DependenceMetric):
    """Mean absolute Pearson correlation across time."""

    name: str = "pearson"

    def compute(self, signals_a: np.ndarray, signals_b: np.ndarray) -> float:
        scores: list[float] = []
        for col_a, col_b in _iter_valid_columns(signals_a, signals_b):
            scores.append(_pearson_abs(col_a, col_b))
        if not scores:
            return 0.0
        return float(np.mean(scores))


@dataclass(frozen=True)
class DistanceCorrelationMetric(DependenceMetric):
    """Mean distance-correlation dependence across time."""

    name: str = "distance_correlation"

    def compute(self, signals_a: np.ndarray, signals_b: np.ndarray) -> float:
        scores: list[float] = []
        for col_a, col_b in _iter_valid_columns(signals_a, signals_b):
            scores.append(_distance_correlation_abs(col_a, col_b))
        if not scores:
            return 0.0
        return float(np.mean(scores))


def build_dependence_metric(name: str | None) -> DependenceMetric:
    """Build one supported dependence metric from config/runtime names."""

    metric_name = str(name or "spearman").strip().lower()
    if metric_name in {"spearman", "rank_correlation"}:
        return SpearmanDependenceMetric()
    if metric_name in {"pearson", "linear_correlation"}:
        return PearsonDependenceMetric()
    if metric_name in {"distance_correlation", "distance", "dcor"}:
        return DistanceCorrelationMetric()
    raise ValueError(
        "Unsupported redundancy/dependence metric "
        f"'{name}'. Expected one of: spearman, pearson, distance_correlation"
    )
