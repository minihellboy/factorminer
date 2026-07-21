# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 FactorMiner Team

"""Auditable Tier-0 evidence for formulaic alpha factors.

The module composes FactorMiner's existing validation primitives without
pretending that a single score proves an alpha. It makes the commonly
ambiguous IC contract explicit, adds autocorrelation-robust inference,
supports risk-model-style cross-sectional neutralization, and reports which
parts of an institutional validation protocol were actually measured.

Inputs use FactorMiner's canonical ``(assets, periods)`` orientation. The
caller remains responsible for point-in-time data, universe construction,
corporate actions, and train/test freeze discipline; those properties cannot
be inferred from two numerical arrays.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
from scipy.stats import norm

from factorminer.evaluation.metrics import (
    compute_ic_abs_mean,
    compute_ic_mean,
    compute_ic_paper_mean,
    compute_ic_win_rate,
    compute_icir,
    compute_pearson_ic,
    compute_rank_ic,
)

INDUSTRY_EVIDENCE_VERSION = "industry_evidence_v1"


@dataclass(frozen=True)
class IndustryEvidenceConfig:
    """Configuration for :func:`evaluate_industry_evidence`."""

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
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be > 0")
        if self.hac_lags is not None and self.hac_lags < 0:
            raise ValueError("hac_lags must be >= 0")
        if not 0.0 < self.hac_confidence < 1.0:
            raise ValueError("hac_confidence must be in (0, 1)")
        if not 0.0 < self.top_fraction <= 0.5:
            raise ValueError("top_fraction must be in (0, 0.5]")
        if any(cost < 0 for cost in self.cost_bps) or self.primary_cost_bps < 0:
            raise ValueError("transaction-cost levels must be >= 0")
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
        return cast(dict[str, Any], _json_safe(payload))


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
        return cast(dict[str, Any], _json_safe(asdict(self)))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _aligned_panels(signals: np.ndarray, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal_panel = np.asarray(signals, dtype=np.float64)
    return_panel = np.asarray(returns, dtype=np.float64)
    if signal_panel.ndim != 2 or return_panel.ndim != 2:
        raise ValueError("signals and returns must be 2-D (assets, periods) panels")
    if signal_panel.shape != return_panel.shape:
        raise ValueError(
            f"signals and returns must have identical shapes; got "
            f"{signal_panel.shape} and {return_panel.shape}"
        )
    return signal_panel, return_panel


def _default_hac_lags(n_observations: int) -> int:
    """Andrews/Newey-West-style automatic bandwidth used by many packages."""
    if n_observations < 2:
        return 0
    return min(int(math.floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0))), n_observations - 1)


def compute_hac_mean_test(
    series: np.ndarray,
    *,
    lags: int | None = None,
    confidence: float = 0.95,
) -> HACMeanTestResult:
    """Test a mean with a Bartlett-kernel Newey-West long-run variance.

    Unlike ``sqrt(n) * mean / std``, this statistic allows the IC series to
    be heteroskedastic and serially correlated. It is still an asymptotic
    normal approximation, so short samples should also be inspected with the
    block-bootstrap result emitted by :func:`evaluate_industry_evidence`.
    """
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
        autocovariance = float(
            np.dot(centered[lag:], centered[:-lag]) / n_observations
        )
        bartlett_weight = 1.0 - lag / (resolved_lags + 1.0)
        long_run_variance += 2.0 * bartlett_weight * autocovariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = math.sqrt(long_run_variance / n_observations)
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


def _ic_bundle(
    signals: np.ndarray,
    returns: np.ndarray,
    config: IndustryEvidenceConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
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


def _resolve_exposure_panel(
    exposures: np.ndarray,
    *,
    asset_count: int,
    period_count: int,
) -> np.ndarray:
    panel = np.asarray(exposures, dtype=np.float64)
    if panel.ndim == 2:
        if panel.shape[0] != asset_count:
            raise ValueError(
                f"static exposures must have shape (assets, exposures); got {panel.shape}"
            )
        panel = np.broadcast_to(panel[:, None, :], (asset_count, period_count, panel.shape[1]))
    elif panel.ndim == 3:
        expected_prefix = (asset_count, period_count)
        if panel.shape[:2] != expected_prefix:
            raise ValueError(
                "dynamic exposures must have shape (assets, periods, exposures); "
                f"got {panel.shape}"
            )
    else:
        raise ValueError(
            "exposures must have shape (assets, exposures) or "
            "(assets, periods, exposures)"
        )
    if panel.shape[2] < 1:
        raise ValueError("at least one risk exposure is required")
    return panel


def residualize_against_risk_exposures(
    signals: np.ndarray,
    exposures: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    exposure_names: Sequence[str] | None = None,
    add_intercept: bool = True,
    min_assets: int | None = None,
) -> RiskResidualizationResult:
    """Neutralize each signal cross-section against supplied risk exposures.

    The implementation performs point-in-time OLS or WLS independently in
    each period and returns the residual signal. Numeric styles and one-hot
    industry/country columns may be supplied together. This is an open
    risk-model-style primitive, not a reproduction of proprietary Barra
    exposure estimation, covariance forecasting, or specific-risk models.
    """
    signal_panel = np.asarray(signals, dtype=np.float64)
    if signal_panel.ndim != 2:
        raise ValueError("signals must have shape (assets, periods)")
    asset_count, period_count = signal_panel.shape
    exposure_panel = _resolve_exposure_panel(
        exposures,
        asset_count=asset_count,
        period_count=period_count,
    )
    exposure_count = exposure_panel.shape[2]
    names = tuple(exposure_names or [f"exposure_{index}" for index in range(exposure_count)])
    if len(names) != exposure_count:
        raise ValueError(
            f"exposure_names has {len(names)} entries; expected {exposure_count}"
        )

    weight_panel: np.ndarray | None = None
    if weights is not None:
        raw_weights = np.asarray(weights, dtype=np.float64)
        if raw_weights.ndim == 1 and raw_weights.shape == (asset_count,):
            weight_panel = np.broadcast_to(raw_weights[:, None], signal_panel.shape)
        elif raw_weights.shape == signal_panel.shape:
            weight_panel = raw_weights
        else:
            raise ValueError(
                "weights must have shape (assets,) or (assets, periods); "
                f"got {raw_weights.shape}"
            )

    required_assets = min_assets or max(5, exposure_count + int(add_intercept) + 1)
    residuals = np.full_like(signal_panel, np.nan, dtype=np.float64)
    r2_series = np.full(period_count, np.nan, dtype=np.float64)
    periods_residualized = 0

    for period in range(period_count):
        y = signal_panel[:, period]
        X = exposure_panel[:, period, :]
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        period_weights: np.ndarray | None = None
        if weight_panel is not None:
            period_weights = weight_panel[:, period]
            valid &= np.isfinite(period_weights) & (period_weights > 0)
        if int(valid.sum()) < required_assets:
            continue

        y_valid = y[valid]
        X_valid = X[valid]
        if add_intercept:
            X_valid = np.column_stack([np.ones(len(y_valid)), X_valid])
        if period_weights is None:
            fit_weights = np.ones(len(y_valid), dtype=np.float64)
        else:
            fit_weights = period_weights[valid]
        root_weights = np.sqrt(fit_weights)
        beta, *_ = np.linalg.lstsq(
            X_valid * root_weights[:, None],
            y_valid * root_weights,
            rcond=None,
        )
        period_residuals = y_valid - X_valid @ beta
        residuals[valid, period] = period_residuals

        weighted_mean = float(np.average(y_valid, weights=fit_weights))
        total = float(np.sum(fit_weights * (y_valid - weighted_mean) ** 2))
        unexplained = float(np.sum(fit_weights * period_residuals**2))
        r2_series[period] = 0.0 if total <= 1e-15 else 1.0 - unexplained / total
        periods_residualized += 1

    finite_r2 = r2_series[np.isfinite(r2_series)]
    return RiskResidualizationResult(
        residual_signals=residuals,
        r2_series=r2_series,
        mean_r2=float(np.mean(finite_r2)) if finite_r2.size else 0.0,
        periods_residualized=periods_residualized,
        periods_skipped=period_count - periods_residualized,
        exposure_names=names,
        weighted=weight_panel is not None,
        add_intercept=add_intercept,
    )


def _long_short_path(
    signals: np.ndarray,
    returns: np.ndarray,
    *,
    top_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gross spread, one-way turnover, and traded gross notional."""
    signal_panel, return_panel = _aligned_panels(signals, returns)
    asset_count, period_count = signal_panel.shape
    gross_returns = np.full(period_count, np.nan, dtype=np.float64)
    turnover = np.full(period_count, np.nan, dtype=np.float64)
    traded_notional = np.zeros(period_count, dtype=np.float64)
    previous_weights: np.ndarray | None = None

    for period in range(period_count):
        signal = signal_panel[:, period]
        forward_return = return_panel[:, period]
        valid = np.flatnonzero(np.isfinite(signal) & np.isfinite(forward_return))
        if valid.size < 5:
            previous_weights = None
            continue
        leg_size = max(int(valid.size * top_fraction), 1)
        if 2 * leg_size > valid.size:
            previous_weights = None
            continue
        ordered = valid[np.argsort(signal[valid], kind="stable")]
        short_assets = ordered[:leg_size]
        long_assets = ordered[-leg_size:]
        current_weights = np.zeros(asset_count, dtype=np.float64)
        current_weights[long_assets] = 1.0 / leg_size
        current_weights[short_assets] = -1.0 / leg_size
        gross_returns[period] = float(
            np.dot(current_weights[valid], forward_return[valid])
        )
        if previous_weights is not None:
            gross_trade = float(np.sum(np.abs(current_weights - previous_weights)))
            traded_notional[period] = gross_trade
            turnover[period] = gross_trade / 2.0
        previous_weights = current_weights
    return gross_returns, turnover, traded_notional


def _annualized_sharpe(series: np.ndarray, periods_per_year: float) -> float:
    clean = np.asarray(series, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return 0.0
    standard_deviation = float(np.std(clean, ddof=1))
    if standard_deviation <= 1e-15:
        return 0.0
    return float(np.mean(clean) / standard_deviation * math.sqrt(periods_per_year))


def _cost_stress(
    signals: np.ndarray,
    returns: np.ndarray,
    config: IndustryEvidenceConfig,
) -> tuple[dict[str, Any], dict[float, np.ndarray]]:
    gross, turnover, traded_notional = _long_short_path(
        signals,
        returns,
        top_fraction=config.top_fraction,
    )
    levels = list(dict.fromkeys((*config.cost_bps, config.primary_cost_bps)))
    points: list[dict[str, float]] = []
    net_paths: dict[float, np.ndarray] = {}
    for cost_bps in levels:
        net = np.where(
            np.isfinite(gross),
            gross - float(cost_bps) / 10_000.0 * traded_notional,
            np.nan,
        )
        net_paths[float(cost_bps)] = net
        finite_gross = gross[np.isfinite(gross)]
        finite_net = net[np.isfinite(net)]
        points.append(
            {
                "cost_bps": float(cost_bps),
                "mean_gross_return": (
                    float(np.mean(finite_gross)) if finite_gross.size else 0.0
                ),
                "mean_net_return": float(np.mean(finite_net)) if finite_net.size else 0.0,
                "annualized_net_return": (
                    float(np.mean(finite_net)) * config.periods_per_year
                    if finite_net.size
                    else 0.0
                ),
                "annualized_net_sharpe": _annualized_sharpe(
                    finite_net, config.periods_per_year
                ),
            }
        )

    valid_turnover = turnover[np.isfinite(turnover)]
    mean_traded_notional = (
        float(np.mean(traded_notional[np.isfinite(turnover)]))
        if valid_turnover.size
        else 0.0
    )
    finite_gross = gross[np.isfinite(gross)]
    mean_gross = float(np.mean(finite_gross)) if finite_gross.size else 0.0
    break_even = (
        abs(mean_gross) / mean_traded_notional * 10_000.0
        if mean_traded_notional > 1e-15
        else float("inf")
    )
    return (
        {
            "portfolio_definition": "equal-weight top-minus-bottom quantile spread",
            "gross_exposure": 2.0,
            "turnover_definition": "0.5 * sum(abs(target_weight_t - target_weight_t-1))",
            "initial_entry_charged": False,
            "top_fraction": config.top_fraction,
            "average_one_way_turnover": (
                float(np.mean(valid_turnover)) if valid_turnover.size else 0.0
            ),
            "absolute_spread_break_even_one_way_cost_bps": break_even,
            "cost_curve": points,
            "gross_return_series": gross,
            "turnover_series": turnover,
        },
        net_paths,
    )


def _coverage(status: str, detail: str) -> dict[str, str]:
    return {"status": status, "detail": detail}


def evaluate_industry_evidence(
    factor_name: str,
    signals: np.ndarray,
    forward_returns: np.ndarray,
    *,
    config: IndustryEvidenceConfig | None = None,
    risk_exposures: np.ndarray | None = None,
    risk_weights: np.ndarray | None = None,
    exposure_names: Sequence[str] | None = None,
    family_ic_series: Mapping[str, np.ndarray] | None = None,
    n_trials: int = 1,
    pbo_performance_matrix: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    capacity_config: Any | None = None,
) -> IndustryEvidenceReport:
    """Build a Tier-0 factor evidence report from aligned signal/return panels.

    Optional inputs activate otherwise-unmeasurable gates:

    - ``risk_exposures``: static ``(M, K)`` or point-in-time ``(M, T, K)``
      style/industry exposures;
    - ``family_ic_series``: IC series for the complete tried family, used for
      Benjamini-Hochberg FDR with HAC p-values;
    - ``pbo_performance_matrix``: ``(trials, paths)`` CPCV performance matrix;
    - ``volume``: dollar-volume panel used by FactorMiner's square-root
      capacity estimator.

    The function does not create a walk-forward split after seeing the data.
    Split/freeze construction belongs upstream and remains explicit in
    ``validation_coverage``.
    """
    cfg = config or IndustryEvidenceConfig()
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    signal_panel, return_panel = _aligned_panels(signals, forward_returns)
    warnings: list[str] = []

    raw_metrics, _, rank_series = _ic_bundle(signal_panel, return_panel, cfg)
    raw_portfolio, raw_net_paths = _cost_stress(signal_panel, return_panel, cfg)

    from factorminer.evaluation.significance import (
        BootstrapICTester,
        DeflatedSharpeCalculator,
        FDRController,
        SignificanceConfig,
    )

    significance_config = SignificanceConfig(
        bootstrap_n_samples=cfg.bootstrap_n_samples,
        bootstrap_block_size=cfg.bootstrap_block_size,
        bootstrap_confidence=cfg.bootstrap_confidence,
        fdr_level=cfg.fdr_level,
        seed=cfg.seed,
    )
    bootstrap_tester = BootstrapICTester(significance_config)
    bootstrap_result = bootstrap_tester.compute_ci(factor_name, rank_series)
    primary_net = raw_net_paths[float(cfg.primary_cost_bps)]
    dsr_result = DeflatedSharpeCalculator(significance_config).compute(
        factor_name,
        primary_net,
        n_trials=n_trials,
        annualization_factor=cfg.periods_per_year,
    )
    significance: dict[str, Any] = {
        "rank_ic_block_bootstrap": asdict(bootstrap_result),
        "deflated_sharpe": {
            **asdict(dsr_result),
            "cost_bps": cfg.primary_cost_bps,
            "implementation_note": (
                "FactorMiner's existing DSR uses an expected-maximum null approximation "
                "parameterized by n_trials; retain the complete trial return panel for "
                "independent replication."
            ),
        },
        "fdr": None,
        "pbo": None,
    }

    if family_ic_series is not None:
        family = {
            str(name): np.asarray(values, dtype=np.float64)
            for name, values in family_ic_series.items()
        }
        family[factor_name] = rank_series
        p_values = {
            name: compute_hac_mean_test(
                values,
                lags=cfg.hac_lags,
                confidence=cfg.hac_confidence,
            ).p_value
            for name, values in family.items()
        }
        fdr = FDRController(significance_config).apply_fdr(p_values)
        significance["fdr"] = {
            **asdict(fdr),
            "p_value_method": "two-sided Newey-West mean test",
            "factor_adjusted_p_value": fdr.adjusted_p_values[factor_name],
            "factor_significant": fdr.significant[factor_name],
        }
        warnings.append(
            "Benjamini-Hochberg is most defensible under independent or positively dependent "
            "tests; strongly dependent factor families may need BY or resampling-based control."
        )

    if pbo_performance_matrix is not None:
        from factorminer.evaluation.cross_validation import (
            ProbabilityOfBacktestOverfitting,
        )

        pbo_result = ProbabilityOfBacktestOverfitting(seed=cfg.seed).compute(
            pbo_performance_matrix
        )
        significance["pbo"] = asdict(pbo_result)

    risk_payload: dict[str, Any] | None = None
    risk_portfolio: dict[str, Any] | None = None
    if risk_exposures is not None:
        risk_result = residualize_against_risk_exposures(
            signal_panel,
            risk_exposures,
            weights=risk_weights,
            exposure_names=exposure_names,
        )
        residual_metrics, _, _ = _ic_bundle(
            risk_result.residual_signals,
            return_panel,
            cfg,
        )
        risk_portfolio, _ = _cost_stress(
            risk_result.residual_signals,
            return_panel,
            cfg,
        )
        risk_payload = {
            "method": "point-in-time cross-sectional weighted least squares residual",
            "diagnostics": risk_result.to_dict(include_signals=False),
            "metrics": residual_metrics,
        }

    capacity_payload: dict[str, Any] | None = None
    if volume is not None:
        volume_panel = np.asarray(volume, dtype=np.float64)
        if volume_panel.shape != signal_panel.shape:
            raise ValueError(
                f"volume must have shape {signal_panel.shape}; got {volume_panel.shape}"
            )
        from factorminer.evaluation.capacity import CapacityConfig, CapacityEstimator

        resolved_capacity_config = capacity_config or CapacityConfig()
        estimator = CapacityEstimator(return_panel, volume_panel, resolved_capacity_config)
        capacity_estimate = estimator.estimate(factor_name, signal_panel)
        net_cost = estimator.net_cost_evaluation(factor_name, signal_panel)
        capacity_payload = {
            "model": "square-root market-impact scenario",
            "estimate": asdict(capacity_estimate),
            "base_capital_net_evaluation": asdict(net_cost),
        }
        warnings.append(
            "Capacity is a square-root impact scenario using long-leg liquidity as the shared "
            "two-leg proxy, not an order-book replay or live fill study."
        )

    validation_coverage = {
        "ic_rankic": _coverage(
            "measured",
            "Pearson IC and Spearman RankIC are separate, with exact series and definitions.",
        ),
        "serial_dependence": _coverage(
            "measured",
            "Newey-West inference and a circular block-bootstrap CI are reported.",
        ),
        "turnover_cost_stress": _coverage(
            "measured",
            "Exact long/short target-weight turnover and configurable linear cost stress.",
        ),
        "capacity": _coverage(
            "measured" if volume is not None else "not_supplied",
            (
                "Square-root impact capacity evaluated from the supplied dollar-volume panel."
                if volume is not None
                else "Supply a point-in-time dollar-volume panel to activate capacity stress."
            ),
        ),
        "multiple_testing": _coverage(
            "measured" if family_ic_series is not None else "partial",
            (
                "Family-wide HAC p-values receive Benjamini-Hochberg adjustment; DSR also "
                "uses the declared trial count."
                if family_ic_series is not None
                else "DSR uses n_trials, but family-wide FDR requires every tried factor's IC series."
            ),
        ),
        "selection_overfit": _coverage(
            "measured" if pbo_performance_matrix is not None else "not_supplied",
            (
                "PBO evaluated from the supplied CPCV path matrix."
                if pbo_performance_matrix is not None
                else "Supply the complete trial-by-CPCV-path matrix to estimate PBO."
            ),
        ),
        "risk_residualization": _coverage(
            "measured" if risk_exposures is not None else "not_supplied",
            (
                "Signal was neutralized against supplied point-in-time exposures."
                if risk_exposures is not None
                else "Supply style and one-hot industry/country exposures; no exposure was inferred."
            ),
        ),
        "walk_forward_freeze": _coverage(
            "external_required",
            "This report scores supplied panels; construct train/validation/test freezes upstream.",
        ),
        "point_in_time_data": _coverage(
            "external_required",
            "Dataset timestamps, survivorship, corporate actions, and vendor revisions need provenance.",
        ),
        "independent_execution_replay": _coverage(
            "external_required",
            "Replay frozen weights in an event-driven engine or broker simulator before deployment.",
        ),
    }
    if risk_exposures is None:
        warnings.append("Raw predictive power may be an unmeasured style or industry exposure.")
    if family_ic_series is None:
        warnings.append("A single-factor p-value does not control the factor zoo's false discoveries.")

    return IndustryEvidenceReport(
        protocol_version=INDUSTRY_EVIDENCE_VERSION,
        factor_name=factor_name,
        metric_contract={
            "panel_orientation": "assets_by_periods",
            "forward_return_alignment": "caller_supplied",
            "legacy_ic_alias": "spearman_rank_ic",
            "icir": "mean(IC_t) / sample_std(IC_t), unannualized",
            "annualized_icir": "icir * sqrt(periods_per_year)",
            "independent_t_stat": "icir * sqrt(n_periods)",
            "hac_t_stat": "mean(IC_t) / Newey-West standard_error(mean)",
            "periods_per_year": cfg.periods_per_year,
        },
        raw_signal=raw_metrics,
        risk_residual=risk_payload,
        portfolio={
            "raw_signal": raw_portfolio,
            "risk_residual_signal": risk_portfolio,
        },
        significance=significance,
        capacity=capacity_payload,
        validation_coverage=validation_coverage,
        warnings=tuple(warnings),
    )
