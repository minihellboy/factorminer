# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 FactorMiner Team

"""Long/short portfolio construction and deterministic cost stress."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from factorminer.evaluation.evidence.models import IndustryEvidenceConfig


def aligned_panels(signals: np.ndarray, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return aligned floating-point asset-by-period panels."""
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


def _long_short_path(
    signals: np.ndarray,
    returns: np.ndarray,
    *,
    top_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return gross spread, one-way turnover, and traded gross notional."""
    signal_panel, return_panel = aligned_panels(signals, returns)
    asset_count, period_count = signal_panel.shape
    gross_returns = np.full(period_count, np.nan, dtype=np.float64)
    turnover = np.full(period_count, np.nan, dtype=np.float64)
    traded_notional = np.zeros(period_count, dtype=np.float64)
    previous_weights: np.ndarray | None = None

    for period in range(period_count):
        signal = signal_panel[:, period]
        forward_return = return_panel[:, period]
        eligible = np.flatnonzero(np.isfinite(signal))
        if eligible.size < 5:
            previous_weights = None
            continue
        leg_size = max(int(eligible.size * top_fraction), 1)
        if 2 * leg_size > eligible.size:
            previous_weights = None
            continue
        ordered = eligible[np.argsort(signal[eligible], kind="stable")]
        short_assets = ordered[:leg_size]
        long_assets = ordered[-leg_size:]
        current_weights = np.zeros(asset_count, dtype=np.float64)
        current_weights[long_assets] = 1.0 / leg_size
        current_weights[short_assets] = -1.0 / leg_size
        selected_assets = np.concatenate([short_assets, long_assets])
        if np.all(np.isfinite(forward_return[selected_assets])):
            gross_returns[period] = float(
                np.dot(current_weights[selected_assets], forward_return[selected_assets])
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


def compute_cost_stress(
    signals: np.ndarray,
    returns: np.ndarray,
    config: IndustryEvidenceConfig,
) -> tuple[dict[str, Any], dict[float, np.ndarray]]:
    """Evaluate a target-weight long/short path across linear cost levels."""
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
                "mean_gross_return": (float(np.mean(finite_gross)) if finite_gross.size else 0.0),
                "mean_net_return": float(np.mean(finite_net)) if finite_net.size else 0.0,
                "annualized_net_return": (
                    float(np.mean(finite_net)) * config.periods_per_year if finite_net.size else 0.0
                ),
                "annualized_net_sharpe": _annualized_sharpe(finite_net, config.periods_per_year),
            }
        )

    valid_turnover = turnover[np.isfinite(turnover)]
    mean_traded_notional = (
        float(np.mean(traded_notional[np.isfinite(turnover)])) if valid_turnover.size else 0.0
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
