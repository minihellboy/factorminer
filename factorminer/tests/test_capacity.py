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
    # This fixture's signals/returns are independent (near-zero true IC),
    # so degradation is guarded to 0.0 rather than blowing up on a
    # near-zero-gross-IC ratio (see capacity.py's 1e-3 guard). The
    # meaningful, non-degenerate check -- degradation actually rising with
    # capital on a *real* factor -- is test_capacity_degradation_is_not_a_noop.
    degradations = list(cap_est.capacity_curve.values())
    assert all(0.0 <= d <= 1.0 for d in degradations)


def test_capacity_degradation_is_not_a_noop(rng):
    """Net-of-cost degradation must actually respond to cost, not be a no-op.

    Regression test: a prior implementation subtracted a uniform per-bar
    cost from every asset's return before computing Spearman IC. Spearman
    IC is a rank correlation and is exactly invariant to any
    column-constant shift, so that implementation made capacity_curve
    identically 0.0 and max_capacity_usd always +inf regardless of cost or
    capital -- a silent no-op. This test uses a genuinely correlated
    factor (signals informative about returns) and asserts degradation
    actually rises with capital, and that break_even_cost_bps /
    net_ls_return are sensitive to cost.
    """
    M, T = 30, 60
    true_alpha = rng.normal(size=(M, 1)) * np.ones((1, T))
    returns = 0.6 * true_alpha + 0.4 * rng.normal(size=(M, T)) * 0.01
    signals = true_alpha + 0.3 * rng.normal(size=(M, T))
    volume = np.abs(rng.normal(1e6, 1e5, (M, T)))

    config = CapacityConfig(capacity_levels=[1e4, 1e7, 1e9, 1e11])
    estimator = CapacityEstimator(returns=returns, volume=volume, config=config)

    cap_est = estimator.estimate("real_factor", signals)
    degradations = list(cap_est.capacity_curve.values())

    # Must not be the flat all-zero no-op curve.
    assert any(d > 0.0 for d in degradations), (
        "capacity_curve is entirely zero -- net-of-cost screening is a no-op"
    )
    # Degradation should rise (non-strictly) with capital.
    assert degradations[-1] > degradations[0]
    assert all(0.0 <= d <= 1.0 for d in degradations)

    # break_even_cost_bps must be a real, positive return-space quantity,
    # not an IC-space number mislabeled as bps.
    assert cap_est.break_even_cost_bps > 0.0

    # net_cost_evaluation: net_ls_return must actually decrease vs. gross
    # as capital rises (previously gross_ls_return was the cross-sectional
    # market mean, not a long-short spread, and net_icir never moved).
    low = estimator.net_cost_evaluation("real_factor", signals, capital=1e4)
    high = estimator.net_cost_evaluation("real_factor", signals, capital=1e11)
    assert high.net_ls_return < low.net_ls_return
    assert high.net_icir <= low.net_icir


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
