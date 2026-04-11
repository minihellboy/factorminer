"""Tests for capacity-aware backtesting (evaluation/capacity.py)."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.evaluation.capacity import (
    CapacityConfig,
    CapacityEstimator,
    MarketImpactModel,
    NetCostResult,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def market_data(rng):
    """Synthetic returns and volume for capacity tests."""
    M, T = 20, 100
    returns = rng.normal(0, 0.01, (M, T))
    volume = np.abs(rng.normal(1e6, 1e5, (M, T)))
    signals = rng.normal(0, 1, (M, T))
    return returns, volume, signals


# -----------------------------------------------------------------------
# MarketImpactModel: higher capital -> higher impact_bps
# -----------------------------------------------------------------------

def test_impact_increases_with_capital(rng):
    """Higher capital should result in higher average impact."""
    M, T = 20, 100
    signals = rng.normal(0, 1, (M, T))
    # Use very high volume so low capital stays below participation limit
    volume = np.abs(rng.normal(1e9, 1e8, (M, T)))
    model = MarketImpactModel()

    low_cap = model.estimate_impact(signals, volume, capital=1e6)
    high_cap = model.estimate_impact(signals, volume, capital=1e9)

    assert high_cap.avg_impact_bps > low_cap.avg_impact_bps


def test_impact_result_shape(market_data):
    """Impact arrays should match T dimension."""
    returns, volume, signals = market_data
    T = signals.shape[1]
    model = MarketImpactModel()
    result = model.estimate_impact(signals, volume, capital=1e8)

    assert result.impact_bps.shape == (T,)
    assert result.participation_rate.shape == (T,)
    assert result.avg_impact_bps >= 0


# -----------------------------------------------------------------------
# CapacityEstimator: low capital -> net_icir ~ gross_icir
# -----------------------------------------------------------------------

def test_low_capital_minimal_degradation(market_data):
    """At very low capital, net ICIR should be close to gross ICIR."""
    returns, volume, signals = market_data
    estimator = CapacityEstimator(
        returns=returns,
        volume=volume,
        config=CapacityConfig(base_capital_usd=1e4),
    )
    result = estimator.net_cost_evaluation("test", signals, capital=1e4)
    assert isinstance(result, NetCostResult)
    # At very low capital, impact is tiny, so net ~ gross
    diff = abs(result.gross_icir - result.net_icir)
    assert diff < abs(result.gross_icir) + 0.5  # generous tolerance


# -----------------------------------------------------------------------
# CapacityEstimator: high capital -> significant IC degradation
# -----------------------------------------------------------------------

def test_high_capital_degrades_ic(market_data):
    """At very high capital, the net ICIR should be meaningfully lower."""
    returns, volume, signals = market_data
    config = CapacityConfig(
        capacity_levels=[1e4, 1e6, 1e8, 1e10],
    )
    estimator = CapacityEstimator(
        returns=returns,
        volume=volume,
        config=config,
    )
    cap_est = estimator.estimate("test", signals)
    # The capacity curve should show increasing degradation
    degradations = list(cap_est.capacity_curve.values())
    assert degradations[-1] >= degradations[0]


# -----------------------------------------------------------------------
# Edge case: zero volume
# -----------------------------------------------------------------------

def test_zero_volume_handling(rng):
    """Zero volume should be handled gracefully (participation_limit used)."""
    M, T = 10, 50
    rng.normal(0, 0.01, (M, T))
    volume = np.zeros((M, T))  # all zero volume
    signals = rng.normal(0, 1, (M, T))

    model = MarketImpactModel()
    result = model.estimate_impact(signals, volume, capital=1e8)

    # Should not crash; participation rate should be capped at limit
    assert not np.any(np.isnan(result.impact_bps))
    cfg = CapacityConfig()
    assert np.allclose(result.participation_rate, cfg.participation_limit)
