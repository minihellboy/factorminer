# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 FactorMiner Team

"""IC summaries and autocorrelation-robust mean inference."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.stats import norm

from factorminer.evaluation.evidence.models import (
    HACMeanTestResult,
    ICMetricSummary,
    IndustryEvidenceConfig,
)
from factorminer.evaluation.metrics import (
    compute_ic_abs_mean,
    compute_ic_mean,
    compute_ic_paper_mean,
    compute_ic_win_rate,
    compute_icir,
    compute_pearson_ic,
    compute_rank_ic,
)


def _default_hac_lags(n_observations: int) -> int:
    if n_observations < 2:
        return 0
    bandwidth = int(math.floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0)))
    return min(bandwidth, n_observations - 1)


def compute_hac_mean_test(
    series: np.ndarray,
    *,
    lags: int | None = None,
    confidence: float = 0.95,
) -> HACMeanTestResult:
    """Test a mean with a Bartlett-kernel Newey-West long-run variance."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    clean = np.asarray(series, dtype=np.float64).reshape(-1)
    clean = clean[np.isfinite(clean)]
    n_observations = int(clean.size)
    resolved_lags = _default_hac_lags(n_observations) if lags is None else int(lags)
    if resolved_lags < 0:
        raise ValueError("lags must be >= 0")
    if n_observations:
        resolved_lags = min(resolved_lags, n_observations - 1)
    else:
        resolved_lags = 0

    if n_observations == 0:
        return HACMeanTestResult(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0, 0, confidence)

    mean = float(np.mean(clean))
    if n_observations < 3:
        return HACMeanTestResult(
            mean, 0.0, 0.0, 1.0, mean, mean, resolved_lags, n_observations, confidence
        )

    centered = clean - mean
    long_run_variance = float(np.dot(centered, centered) / n_observations)
    for lag in range(1, resolved_lags + 1):
        autocovariance = float(np.dot(centered[lag:], centered[:-lag]) / n_observations)
        bartlett_weight = 1.0 - lag / (resolved_lags + 1.0)
        long_run_variance += 2.0 * bartlett_weight * autocovariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / n_observations)
    if standard_error <= 1e-15:
        if abs(mean) <= 1e-15:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = math.copysign(np.finfo(np.float64).max, mean)
            p_value = 0.0
    else:
        t_stat = mean / standard_error
        p_value = 2.0 * (1.0 - float(norm.cdf(abs(t_stat))))
    critical_value = float(norm.ppf(0.5 + confidence / 2.0))
    return HACMeanTestResult(
        mean=mean,
        standard_error=standard_error,
        t_stat=float(t_stat),
        p_value=float(np.clip(p_value, 0.0, 1.0)),
        ci_lower=mean - critical_value * standard_error,
        ci_upper=mean + critical_value * standard_error,
        lags=resolved_lags,
        n_observations=n_observations,
        confidence=confidence,
    )


def _summarize_ic(
    series: np.ndarray,
    *,
    correlation: str,
    config: IndustryEvidenceConfig,
) -> ICMetricSummary:
    clean = np.asarray(series, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    n_periods = int(clean.size)
    sample_std = float(np.std(clean, ddof=1)) if n_periods > 1 else 0.0
    icir = compute_icir(clean)
    return ICMetricSummary(
        correlation=correlation,
        mean=compute_ic_mean(clean),
        absolute_mean=compute_ic_paper_mean(clean),
        mean_absolute=compute_ic_abs_mean(clean),
        sample_std=sample_std,
        icir=icir,
        annualized_icir=icir * math.sqrt(config.periods_per_year),
        independent_t_stat=icir * math.sqrt(n_periods),
        win_rate=compute_ic_win_rate(clean),
        n_periods=n_periods,
        hac=compute_hac_mean_test(
            clean,
            lags=config.hac_lags,
            confidence=config.hac_confidence,
        ),
    )


def compute_ic_bundle(
    signals: np.ndarray,
    returns: np.ndarray,
    config: IndustryEvidenceConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Return labeled Pearson and Spearman summaries plus their raw series."""
    pearson_series = compute_pearson_ic(signals, returns)
    rank_series = compute_rank_ic(signals, returns)
    pearson = _summarize_ic(pearson_series, correlation="pearson", config=config)
    rank = _summarize_ic(rank_series, correlation="spearman_rank", config=config)
    return (
        {
            "pearson_ic": asdict(pearson),
            "rank_ic": asdict(rank),
            "pearson_ic_series": pearson_series,
            "rank_ic_series": rank_series,
        },
        pearson_series,
        rank_series,
    )
