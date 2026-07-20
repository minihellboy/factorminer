"""Tests for risk-based portfolio construction (evaluation/risk_portfolio.py)."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.evaluation.risk_portfolio import (
    CVaRPortfolioOptimizer,
    HRPOptimizer,
    RiskParityOptimizer,
    RiskPortfolioConfig,
    build_result,
    construct_portfolio,
    effective_n,
    historical_cvar,
    naive_inverse_variance_weights,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_correlated_cluster(
    n_assets: int, n_periods: int, vol: float, rho: float, rng: np.random.Generator
) -> np.ndarray:
    """Simulate a block of `n_assets` returns sharing pairwise correlation `rho`."""
    common = rng.standard_normal(n_periods)
    idio = rng.standard_normal((n_periods, n_assets))
    raw = np.sqrt(rho) * common[:, None] + np.sqrt(1.0 - rho) * idio
    return raw * vol


@pytest.fixture
def two_cluster_returns() -> np.ndarray:
    """Two strongly-correlated clusters engineered so the textbook HRP claim holds.

    Cluster A: 4 higher-vol assets, correlated at rho=0.7 ("strongly" but with
    real diversification benefit within the cluster).
    Cluster B: 4 lower-vol assets, correlated at rho=0.95 (near-duplicate
    bets -- naive inverse-variance weighting over-trusts them because it is
    blind to the redundancy).

    Naive inverse-variance weighting piles weight onto cluster B's near
    identical, low-variance assets. HRP's top-level split instead compares
    each cluster's *diversified* (IVP-within-cluster) variance, which
    discounts cluster B's redundancy less than cluster A's genuine
    diversification -- so HRP hands relatively more weight to cluster A,
    producing a less concentrated (higher effective-N) portfolio than naive
    inverse-variance weighting on this fixture.
    """
    rng = np.random.default_rng(42)
    n_periods = 1500
    cluster_a = _make_correlated_cluster(4, n_periods, vol=0.02, rho=0.7, rng=rng)
    cluster_b = _make_correlated_cluster(4, n_periods, vol=0.01, rho=0.95, rng=rng)
    return np.concatenate([cluster_a, cluster_b], axis=1)


@pytest.fixture
def risk_parity_returns() -> np.ndarray:
    """Five assets with heterogeneous vol/correlation for risk-parity tests."""
    rng = np.random.default_rng(7)
    n_periods = 1000
    n_assets = 5
    vols = np.array([0.005, 0.01, 0.02, 0.03, 0.05])
    common = rng.standard_normal(n_periods)
    idio = rng.standard_normal((n_periods, n_assets))
    raw = 0.4 * common[:, None] + np.sqrt(1.0 - 0.4**2) * idio
    return raw * vols


@pytest.fixture
def tail_risk_returns() -> np.ndarray:
    """Four assets: two stable, two with occasional large left-tail shocks."""
    rng = np.random.default_rng(11)
    n_periods = 1200
    n_assets = 4
    base = rng.standard_normal((n_periods, n_assets)) * np.array([0.01, 0.01, 0.015, 0.015])
    # Inject fat left-tail shocks into assets 2 and 3 (0-indexed).
    shock_periods = rng.choice(n_periods, size=40, replace=False)
    base[shock_periods[:20], 2] -= rng.uniform(0.08, 0.20, size=20)
    base[shock_periods[20:], 3] -= rng.uniform(0.08, 0.20, size=20)
    return base


# ---------------------------------------------------------------------------
# HRPOptimizer
# ---------------------------------------------------------------------------


def test_hrp_weights_are_valid_simplex(two_cluster_returns):
    """HRP weights must be non-negative and sum to 1."""
    weights = HRPOptimizer().optimize(two_cluster_returns)

    assert weights.shape == (8,)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0, atol=1e-8)


def test_hrp_is_less_concentrated_than_naive_ivp(two_cluster_returns):
    """On the two-cluster fixture, HRP should have higher effective-N than IVP.

    This is the textbook HRP claim: naive inverse-variance weighting is
    blind to correlation, so it over-allocates to the cluster of highly
    redundant (near-duplicate) low-variance assets. HRP's hierarchical
    split discounts that redundancy, spreading risk budget more evenly.
    """
    hrp_weights = HRPOptimizer().optimize(two_cluster_returns)
    ivp_weights = naive_inverse_variance_weights(two_cluster_returns)

    assert np.isclose(ivp_weights.sum(), 1.0, atol=1e-8)

    hrp_eff_n = effective_n(hrp_weights)
    ivp_eff_n = effective_n(ivp_weights)

    assert hrp_eff_n > ivp_eff_n + 0.2  # robust margin, not a coin-flip


def test_hrp_single_asset_returns_full_weight():
    """A single-asset panel must trivially get 100% weight."""
    returns = np.random.default_rng(0).standard_normal((50, 1)) * 0.01
    weights = HRPOptimizer().optimize(returns)
    assert weights.shape == (1,)
    assert weights[0] == pytest.approx(1.0)


def test_hrp_rejects_bad_shapes():
    with pytest.raises(ValueError):
        HRPOptimizer().optimize(np.zeros(10))


# ---------------------------------------------------------------------------
# RiskParityOptimizer
# ---------------------------------------------------------------------------


def test_risk_parity_weights_are_valid_simplex(risk_parity_returns):
    weights = RiskParityOptimizer().optimize(risk_parity_returns)
    assert weights.shape == (5,)
    assert np.all(weights >= 0.0)
    assert np.isclose(weights.sum(), 1.0, atol=1e-8)


def test_risk_parity_equalizes_risk_contributions(risk_parity_returns):
    """Each asset's fractional risk contribution should be near 1/N."""
    optimizer = RiskParityOptimizer()
    weights = optimizer.optimize(risk_parity_returns)
    rc = optimizer.risk_contributions(risk_parity_returns, weights)

    n = risk_parity_returns.shape[1]
    target = 1.0 / n
    tolerance = 0.01  # documented tolerance: within 1 percentage point of equal share

    assert np.isclose(rc.sum(), 1.0, atol=1e-6)
    assert np.max(np.abs(rc - target)) < tolerance


def test_risk_parity_beats_equal_weight_dispersion(risk_parity_returns):
    """Risk parity should equalize risk contributions far better than equal weight."""
    optimizer = RiskParityOptimizer()
    n = risk_parity_returns.shape[1]

    rp_weights = optimizer.optimize(risk_parity_returns)
    rp_rc = optimizer.risk_contributions(risk_parity_returns, rp_weights)

    equal_weights = np.full(n, 1.0 / n)
    equal_rc = optimizer.risk_contributions(risk_parity_returns, equal_weights)

    assert np.std(rp_rc) < np.std(equal_rc)


# ---------------------------------------------------------------------------
# CVaRPortfolioOptimizer
# ---------------------------------------------------------------------------


def test_cvar_weights_are_feasible(tail_risk_returns):
    weights = CVaRPortfolioOptimizer().optimize(tail_risk_returns, alpha=0.95)
    assert weights.shape == (4,)
    assert np.all(weights >= -1e-9)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)


def test_cvar_optimal_beats_equal_weight_cvar(tail_risk_returns):
    """CVaR-optimal weights should realize lower CVaR than equal weight on the same data."""
    weights = CVaRPortfolioOptimizer().optimize(tail_risk_returns, alpha=0.95)

    n = tail_risk_returns.shape[1]
    equal_weights = np.full(n, 1.0 / n)

    cvar_optimal = historical_cvar(tail_risk_returns @ weights, alpha=0.95)
    cvar_equal = historical_cvar(tail_risk_returns @ equal_weights, alpha=0.95)

    assert cvar_optimal < cvar_equal
    # The optimizer should avoid the two tail-shock assets in favor of the stable ones.
    assert weights[2] + weights[3] < weights[0] + weights[1]


def test_cvar_respects_target_return_constraint(tail_risk_returns):
    """A binding target-return constraint should be satisfied by the LP solution."""
    mean_returns = tail_risk_returns.mean(axis=0)
    # Pick an achievable target between the min and max single-asset mean return.
    target = float(np.median(mean_returns))

    weights = CVaRPortfolioOptimizer().optimize(
        tail_risk_returns, alpha=0.95, target_return=target
    )
    realized_mean = float(mean_returns @ weights)

    assert np.isclose(weights.sum(), 1.0, atol=1e-6)
    assert realized_mean >= target - 1e-6


def test_cvar_rejects_bad_alpha(tail_risk_returns):
    with pytest.raises(ValueError):
        CVaRPortfolioOptimizer().optimize(tail_risk_returns, alpha=1.5)


# ---------------------------------------------------------------------------
# Shared helpers / convenience entry point
# ---------------------------------------------------------------------------


def test_effective_n_bounds():
    assert effective_n(np.array([1.0])) == pytest.approx(1.0)
    assert effective_n(np.full(4, 0.25)) == pytest.approx(4.0)
    assert effective_n(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_historical_cvar_matches_worst_case_mean():
    returns = np.array([0.05, -0.10, 0.02, -0.20, 0.01, -0.01, 0.03, -0.02, 0.04, -0.30])
    # alpha=0.9 over 10 obs -> worst 1 observation (the biggest loss).
    cvar = historical_cvar(returns, alpha=0.9)
    assert cvar == pytest.approx(0.30)


def test_build_result_reports_diagnostics(two_cluster_returns):
    weights = HRPOptimizer().optimize(two_cluster_returns)
    result = build_result("hrp-test", weights, two_cluster_returns)

    assert result.method == "hrp-test"
    assert result.effective_n == pytest.approx(effective_n(weights))
    assert result.realized_cvar >= 0.0
    payload = result.to_dict()
    assert payload["weights"] == pytest.approx(weights.tolist())
    assert len(payload["asset_ids"]) == weights.shape[0]


@pytest.mark.parametrize("method", ["hrp", "risk_parity", "cvar"])
def test_construct_portfolio_entry_point(two_cluster_returns, method):
    result = construct_portfolio(
        two_cluster_returns, method=method, asset_ids=list(range(100, 108))
    )
    assert result.weights.shape == (8,)
    assert np.isclose(result.weights.sum(), 1.0, atol=1e-6)
    assert result.asset_ids == list(range(100, 108))


def test_construct_portfolio_rejects_unknown_method(two_cluster_returns):
    with pytest.raises(ValueError):
        construct_portfolio(two_cluster_returns, method="not-a-method")


def test_risk_portfolio_config_defaults_are_long_only():
    config = RiskPortfolioConfig()
    assert config.min_weight == 0.0
    assert 0.0 < config.cvar_alpha < 1.0
