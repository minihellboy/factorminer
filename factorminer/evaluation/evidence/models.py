# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 FactorMiner Team

"""Configuration and immutable result models for factor evidence reports."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np

INDUSTRY_EVIDENCE_VERSION = "industry_evidence_v1"


@dataclass(frozen=True)
class IndustryEvidenceConfig:
    """Configuration for ``evaluate_industry_evidence``."""

    periods_per_year: float = 252.0
    hac_lags: int | None = None
    hac_confidence: float = 0.95
    top_fraction: float = 0.20
    cost_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    primary_cost_bps: float = 10.0
    bootstrap_n_samples: int = 1000
    bootstrap_block_size: int = 20
    bootstrap_confidence: float = 0.95
    fdr_level: float = 0.05
    seed: int = 42

    def __post_init__(self) -> None:
        if not math.isfinite(self.periods_per_year) or self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be finite and > 0")
        if self.hac_lags is not None and self.hac_lags < 0:
            raise ValueError("hac_lags must be >= 0")
        if not 0.0 < self.hac_confidence < 1.0:
            raise ValueError("hac_confidence must be in (0, 1)")
        if not 0.0 < self.top_fraction <= 0.5:
            raise ValueError("top_fraction must be in (0, 0.5]")
        cost_levels = (*self.cost_bps, self.primary_cost_bps)
        if any(not math.isfinite(cost) or cost < 0 for cost in cost_levels):
            raise ValueError("transaction-cost levels must be finite and >= 0")
        if self.bootstrap_n_samples < 1:
            raise ValueError("bootstrap_n_samples must be >= 1")
        if self.bootstrap_block_size < 1:
            raise ValueError("bootstrap_block_size must be >= 1")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("bootstrap_confidence must be in (0, 1)")
        if not 0.0 < self.fdr_level < 1.0:
            raise ValueError("fdr_level must be in (0, 1)")


@dataclass(frozen=True)
class HACMeanTestResult:
    """Newey-West/Bartlett inference for a time-series mean."""

    mean: float
    standard_error: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float
    lags: int
    n_observations: int
    confidence: float


@dataclass(frozen=True)
class ICMetricSummary:
    """One fully labeled IC or RankIC summary."""

    correlation: str
    mean: float
    absolute_mean: float
    mean_absolute: float
    sample_std: float
    icir: float
    annualized_icir: float
    independent_t_stat: float
    win_rate: float
    n_periods: int
    hac: HACMeanTestResult


@dataclass(frozen=True)
class RiskResidualizationResult:
    """Signals left after cross-sectional style/industry neutralization."""

    residual_signals: np.ndarray
    r2_series: np.ndarray
    mean_r2: float
    periods_residualized: int
    periods_skipped: int
    exposure_names: tuple[str, ...]
    weighted: bool
    add_intercept: bool

    def to_dict(self, *, include_signals: bool = False) -> dict[str, Any]:
        """Serialize diagnostics, optionally including the full residual panel."""
        payload: dict[str, Any] = {
            "r2_series": self.r2_series,
            "mean_r2": self.mean_r2,
            "periods_residualized": self.periods_residualized,
            "periods_skipped": self.periods_skipped,
            "exposure_names": list(self.exposure_names),
            "weighted": self.weighted,
            "add_intercept": self.add_intercept,
        }
        if include_signals:
            payload["residual_signals"] = self.residual_signals
        return cast(dict[str, Any], json_safe(payload))


@dataclass(frozen=True)
class IndustryEvidenceReport:
    """Machine-readable factor evidence and explicit validation coverage."""

    protocol_version: str
    factor_name: str
    metric_contract: dict[str, Any]
    raw_signal: dict[str, Any]
    risk_residual: dict[str, Any] | None
    portfolio: dict[str, Any]
    significance: dict[str, Any]
    capacity: dict[str, Any] | None
    validation_coverage: dict[str, dict[str, str]]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-compatible report payload."""
        return cast(dict[str, Any], json_safe(asdict(self)))


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy and non-finite values for strict JSON."""
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
