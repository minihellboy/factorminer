# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 FactorMiner Team

"""Cross-sectional risk-exposure residualization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from factorminer.evaluation.evidence.models import RiskResidualizationResult


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
                f"dynamic exposures must have shape (assets, periods, exposures); got {panel.shape}"
            )
    else:
        raise ValueError(
            "exposures must have shape (assets, exposures) or (assets, periods, exposures)"
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
    each period. Numeric styles and one-hot industry/country columns can be
    supplied together. It does not estimate exposures, covariance, or
    specific risk.
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
        raise ValueError(f"exposure_names has {len(names)} entries; expected {exposure_count}")

    weight_panel: np.ndarray | None = None
    if weights is not None:
        raw_weights = np.asarray(weights, dtype=np.float64)
        if raw_weights.ndim == 1 and raw_weights.shape == (asset_count,):
            weight_panel = np.broadcast_to(raw_weights[:, None], signal_panel.shape)
        elif raw_weights.shape == signal_panel.shape:
            weight_panel = raw_weights
        else:
            raise ValueError(
                f"weights must have shape (assets,) or (assets, periods); got {raw_weights.shape}"
            )

    default_minimum = max(5, exposure_count + int(add_intercept) + 1)
    required_assets = default_minimum if min_assets is None else min_assets
    if required_assets < exposure_count + int(add_intercept) + 1:
        raise ValueError("min_assets must exceed the fitted parameter count")

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
        fit_weights = (
            np.ones(len(y_valid), dtype=np.float64)
            if period_weights is None
            else period_weights[valid]
        )
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
