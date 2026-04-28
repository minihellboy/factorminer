"""Tests for the evaluation metrics pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.evaluation.metrics import (
    compute_factor_stats,
    compute_ic,
    compute_ic_abs_mean,
    compute_ic_mean,
    compute_ic_paper_icir,
    compute_ic_paper_mean,
    compute_ic_win_rate,
    compute_icir,
    compute_pairwise_correlation,
    compute_quintile_returns,
    compute_turnover,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def perfect_signal(rng):
    """Signal perfectly correlated with returns -> IC should be ~1.0."""
    M, T = 50, 60
    returns = rng.normal(0, 0.01, (M, T))
    signals = returns.copy()  # Perfect correlation
    return signals, returns


@pytest.fixture
def random_signal(rng):
    """Random signal independent of returns -> IC should be ~0."""
    M, T = 50, 80
    returns = rng.normal(0, 0.01, (M, T))
    signals = rng.normal(0, 1.0, (M, T))  # Independent
    return signals, returns


@pytest.fixture
def known_quintile_signal(rng):
    """Signal where high-signal assets have high returns."""
    M, T = 100, 50
    signals = np.tile(np.arange(M, dtype=np.float64).reshape(M, 1), (1, T))
    # Returns correlated with signal rank
    returns = signals * 0.001 + rng.normal(0, 0.001, (M, T))
    return signals, returns


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

class TestIC:
    """Test Information Coefficient computation."""

    def test_perfect_signal_ic_near_one(self, perfect_signal):
        signals, returns = perfect_signal
        ic_series = compute_ic(signals, returns)
        valid = ic_series[~np.isnan(ic_series)]
        assert len(valid) > 0
        # Perfect correlation should give IC close to 1.0
        mean_ic = np.mean(valid)
        assert mean_ic > 0.9, f"Expected IC > 0.9, got {mean_ic}"

    def test_random_signal_ic_near_zero(self, random_signal):
        signals, returns = random_signal
        ic_series = compute_ic(signals, returns)
        valid = ic_series[~np.isnan(ic_series)]
        assert len(valid) > 0
        # Random signal should give IC near 0
        mean_ic = np.mean(np.abs(valid))
        assert mean_ic < 0.2, f"Expected |IC| < 0.2, got {mean_ic}"

    def test_ic_shape(self, perfect_signal):
        signals, returns = perfect_signal
        ic_series = compute_ic(signals, returns)
        assert ic_series.shape == (signals.shape[1],)

    def test_ic_with_nans(self, rng):
        M, T = 30, 20
        signals = rng.normal(0, 1, (M, T))
        returns = rng.normal(0, 0.01, (M, T))
        # Inject NaNs
        signals[0, :] = np.nan
        signals[:, 0] = np.nan
        ic_series = compute_ic(signals, returns)
        assert ic_series.shape == (T,)

    def test_ic_too_few_assets_returns_nan(self):
        # Only 3 assets (below threshold of 5)
        signals = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        returns = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
        ic_series = compute_ic(signals, returns)
        assert np.all(np.isnan(ic_series))


# ---------------------------------------------------------------------------
# ICIR computation
# ---------------------------------------------------------------------------

class TestICIR:
    """Test ICIR = mean(IC) / std(IC)."""

    def test_icir_positive_for_good_signal(self, rng):
        # Use a signal that is correlated but not perfectly, so IC has variance
        M, T = 50, 80
        returns = rng.normal(0, 0.01, (M, T))
        signals = returns + rng.normal(0, 0.005, (M, T))  # Noisy correlation
        ic_series = compute_ic(signals, returns)
        icir = compute_icir(ic_series)
        assert icir > 0, f"Expected positive ICIR, got {icir}"

    def test_icir_near_zero_for_random(self, random_signal):
        signals, returns = random_signal
        ic_series = compute_ic(signals, returns)
        icir = compute_icir(ic_series)
        # Random signal: ICIR should be small in magnitude
        assert abs(icir) < 2.0, f"Expected small ICIR, got {icir}"

    def test_icir_with_few_valid_points(self):
        ic_series = np.array([np.nan, np.nan, 0.05])
        icir = compute_icir(ic_series)
        # Only 1 valid point -> returns 0.0
        assert icir == 0.0

    def test_icir_constant_ic_returns_zero(self):
        ic_series = np.array([0.05, 0.05, 0.05, 0.05])
        icir = compute_icir(ic_series)
        # std = 0 -> returns 0.0
        assert icir == 0.0


# ---------------------------------------------------------------------------
# IC-derived statistics
# ---------------------------------------------------------------------------

class TestICStats:
    """Test IC mean and win rate."""

    def test_ic_mean_signed_and_legacy_abs(self):
        ic_series = np.array([0.1, -0.05, 0.08, -0.03, np.nan])
        result = compute_ic_mean(ic_series)
        expected = np.mean([0.1, -0.05, 0.08, -0.03])
        np.testing.assert_almost_equal(result, expected)
        np.testing.assert_almost_equal(
            compute_ic_abs_mean(ic_series),
            np.mean(np.abs([0.1, -0.05, 0.08, -0.03])),
        )
        np.testing.assert_almost_equal(
            compute_ic_paper_mean(ic_series),
            abs(expected),
        )

    def test_alternating_ic_distinguishes_paper_and_abs_mean(self):
        ic_series = np.array([0.1, -0.1])
        np.testing.assert_almost_equal(compute_ic_abs_mean(ic_series), 0.1)
        np.testing.assert_almost_equal(compute_ic_paper_mean(ic_series), 0.0)
        np.testing.assert_almost_equal(compute_ic_mean(ic_series), 0.0)

    def test_consistently_negative_ic_has_positive_paper_metrics(self):
        ic_series = np.array([-0.1, -0.08, -0.12, -0.09])
        assert compute_ic_mean(ic_series) < 0.0
        assert compute_ic_paper_mean(ic_series) > 0.0
        assert compute_ic_paper_icir(ic_series) > 0.0

    def test_ic_win_rate(self):
        ic_series = np.array([0.1, -0.05, 0.08, -0.03, 0.02, np.nan])
        result = compute_ic_win_rate(ic_series)
        # 3 positive out of 5 valid
        np.testing.assert_almost_equal(result, 0.6)

    def test_ic_mean_all_nan(self):
        ic_series = np.array([np.nan, np.nan, np.nan])
        assert compute_ic_mean(ic_series) == 0.0

    def test_ic_win_rate_all_nan(self):
        ic_series = np.array([np.nan, np.nan])
        assert compute_ic_win_rate(ic_series) == 0.0


# ---------------------------------------------------------------------------
# Pairwise correlation
# ---------------------------------------------------------------------------

class TestPairwiseCorrelation:
    """Test pairwise cross-sectional correlation."""

    def test_identical_signals_correlation_one(self, rng):
        M, T = 30, 40
        signals = rng.normal(0, 1, (M, T))
        corr = compute_pairwise_correlation(signals, signals)
        assert corr > 0.95, f"Expected corr > 0.95 for identical, got {corr}"

    def test_independent_signals_low_correlation(self, rng):
        M, T = 50, 60
        a = rng.normal(0, 1, (M, T))
        b = rng.normal(0, 1, (M, T))
        corr = compute_pairwise_correlation(a, b)
        assert abs(corr) < 0.3, f"Expected low corr, got {corr}"

    def test_negatively_correlated(self, rng):
        M, T = 30, 40
        a = rng.normal(0, 1, (M, T))
        b = -a  # Perfectly negatively correlated
        corr = compute_pairwise_correlation(a, b)
        assert corr < -0.95, f"Expected corr < -0.95, got {corr}"

    def test_correlation_with_nans(self, rng):
        M, T = 30, 20
        a = rng.normal(0, 1, (M, T))
        b = rng.normal(0, 1, (M, T))
        a[:5, :] = np.nan
        corr = compute_pairwise_correlation(a, b)
        # Should still produce a valid number
        assert np.isfinite(corr)


# ---------------------------------------------------------------------------
# Quintile returns
# ---------------------------------------------------------------------------

class TestQuintileReturns:
    """Test quintile return computation."""

    def test_quintile_keys(self, known_quintile_signal):
        signals, returns = known_quintile_signal
        result = compute_quintile_returns(signals, returns)
        assert "Q1" in result
        assert "Q5" in result
        assert "long_short" in result
        assert "monotonicity" in result

    def test_quintile_monotonic_for_known_signal(self, known_quintile_signal):
        signals, returns = known_quintile_signal
        result = compute_quintile_returns(signals, returns)
        # With positively correlated signal, Q5 > Q1
        assert result["long_short"] > 0, (
            f"Expected positive long_short, got {result['long_short']}"
        )
        # Monotonicity should be positive
        assert result["monotonicity"] > 0.5, (
            f"Expected high monotonicity, got {result['monotonicity']}"
        )

    def test_quintile_returns_shape(self, rng):
        M, T = 20, 30
        signals = rng.normal(0, 1, (M, T))
        returns = rng.normal(0, 0.01, (M, T))
        result = compute_quintile_returns(signals, returns, n_quantiles=5)
        # Should have Q1..Q5 plus long_short and monotonicity
        assert len(result) == 7


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------

class TestTurnover:
    """Test portfolio turnover computation."""

    def test_constant_signal_zero_turnover(self):
        M, T = 20, 10
        signals = np.tile(np.arange(M, dtype=np.float64).reshape(M, 1), (1, T))
        turnover = compute_turnover(signals, top_fraction=0.2)
        assert turnover == 0.0

    def test_random_signal_positive_turnover(self, rng):
        M, T = 30, 50
        signals = rng.normal(0, 1, (M, T))
        turnover = compute_turnover(signals, top_fraction=0.2)
        assert 0 <= turnover <= 1.0


# ---------------------------------------------------------------------------
# Comprehensive factor stats
# ---------------------------------------------------------------------------

class TestFactorStats:
    """Test the compute_factor_stats wrapper."""

    def test_factor_stats_keys(self, rng):
        M, T = 30, 40
        signals = rng.normal(0, 1, (M, T))
        returns = rng.normal(0, 0.01, (M, T))
        stats = compute_factor_stats(signals, returns)
        assert stats["metric_version"] == "paper_ic_v2"
        assert "ic_mean" in stats
        assert "ic_paper_mean" in stats
        assert "ic_abs_mean" in stats
        assert "icir" in stats
        assert "ic_paper_icir" in stats
        assert "ic_win_rate" in stats
        assert "Q1" in stats
        assert "long_short" in stats
        assert "turnover" in stats
        assert "ic_series" in stats

    def test_factor_stats_ic_series_shape(self, rng):
        M, T = 20, 30
        signals = rng.normal(0, 1, (M, T))
        returns = rng.normal(0, 0.01, (M, T))
        stats = compute_factor_stats(signals, returns)
        assert stats["ic_series"].shape == (T,)
