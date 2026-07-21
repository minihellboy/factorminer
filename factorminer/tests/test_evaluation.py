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
    compute_ic_vectorized,
    compute_ic_win_rate,
    compute_icir,
    compute_pairwise_correlation,
    compute_pearson_ic,
    compute_quintile_returns,
    compute_rank_ic,
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

    def test_pearson_ic_and_rank_ic_are_explicitly_distinct(self):
        signals = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 100.0])[:, None], (1, 4))
        returns = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0])[:, None], (1, 4))

        pearson = compute_pearson_ic(signals, returns)
        rank = compute_rank_ic(signals, returns)

        np.testing.assert_allclose(rank, 1.0)
        assert np.all(pearson < 0.8)
        np.testing.assert_allclose(compute_ic(signals, returns), rank)

    def test_vectorized_rank_ic_matches_non_finite_contract(self):
        rng = np.random.default_rng(41)
        signals = rng.normal(size=(12, 5))
        returns = rng.normal(size=(12, 5))
        signals[0, 0] = np.inf
        returns[1, 1] = -np.inf

        np.testing.assert_allclose(
            compute_ic_vectorized(signals, returns),
            compute_rank_ic(signals, returns),
            equal_nan=True,
        )


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

    def test_ic_matches_scipy_spearman_per_period(self, rng):
        """Shipped compute_ic must match scipy.stats.spearmanr column-wise."""
        from scipy.stats import spearmanr

        M, T = 40, 25
        signals = rng.integers(-3, 4, size=(M, T)).astype(np.float64)
        returns = rng.normal(0, 0.01, (M, T))
        signals[rng.random(signals.shape) < 0.1] = np.nan
        returns[rng.random(returns.shape) < 0.1] = np.nan
        signals[0, 0] = np.inf
        returns[1, 1] = -np.inf

        got = compute_ic(signals, returns)
        for t in range(T):
            s = signals[:, t]
            r = returns[:, t]
            valid = np.isfinite(s) & np.isfinite(r)
            if valid.sum() < 5:
                assert np.isnan(got[t])
                continue
            exp, _ = spearmanr(s[valid], r[valid])
            if not np.isfinite(exp):
                assert got[t] == 0.0 or np.isnan(got[t])
            else:
                np.testing.assert_allclose(got[t], exp, rtol=1e-10, atol=1e-10)


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
        assert result["long_short"] > 0, f"Expected positive long_short, got {result['long_short']}"
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

    def test_quintiles_preserve_signal_universe_and_invalid_bucket_policy(self):
        signals = np.tile(np.arange(10, dtype=np.float64)[:, None], (1, 2))
        returns = np.tile(np.arange(10, dtype=np.float64)[:, None], (1, 2))
        returns[0, 0] = np.nan

        result = compute_quintile_returns(signals, returns)

        # Q1 at period 0 is skipped as a whole; period 1 still contributes.
        assert result["Q1"] == pytest.approx(0.5)
        assert result["Q5"] == pytest.approx(8.5)


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

    def test_turnover_skips_sparse_columns(self):
        """Columns with too few finite values reset the consecutive window."""
        M, T = 10, 4
        signals = np.arange(M, dtype=np.float64).reshape(M, 1) * np.ones((1, T))
        # Period 1 almost all NaN → break consecutive chain; constant ranks elsewhere.
        signals[:, 1] = np.nan
        signals[0, 1] = 1.0
        turnover = compute_turnover(signals, top_fraction=0.2)
        assert turnover == 0.0

    def test_turnover_matches_reference_across_chunk_boundaries(self, rng):
        signals = rng.integers(-3, 4, size=(40, 130)).astype(np.float64)
        signals[rng.random(signals.shape) < 0.1] = np.nan
        signals[:, 64] = np.nan
        top_fraction = 0.2
        k = int(signals.shape[0] * top_fraction)
        expected: list[float] = []
        previous: set[int] | None = None
        for period in range(signals.shape[1]):
            column = signals[:, period]
            valid = np.isfinite(column)
            if int(valid.sum()) < k:
                previous = None
                continue
            selected = set(
                np.argpartition(np.where(valid, column, -np.inf), -k)[-k:]
            )
            if previous is not None:
                expected.append(1.0 - len(selected & previous) / k)
            previous = selected

        assert compute_turnover(signals, top_fraction) == pytest.approx(
            np.mean(expected), abs=1e-15
        )


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
    def test_factor_stats_labels_legacy_rankic_and_explicit_pearson(self, rng):
        signals = rng.normal(size=(20, 12))
        returns = rng.normal(size=(20, 12))

        stats = compute_factor_stats(signals, returns)

        assert stats["ic_definition"] == "spearman_rank"
        np.testing.assert_allclose(stats["ic_series"], stats["rank_ic_series"])
        assert stats["pearson_ic_series"].shape == stats["rank_ic_series"].shape


# ---------------------------------------------------------------------------
# Batch Spearman / rank columns (evaluation.correlation)
# ---------------------------------------------------------------------------


class TestBatchSpearman:
    """Drive shipped correlation helpers on the real entry points."""

    def test_rank_columns_preserves_nan_and_average_ties(self):
        from factorminer.evaluation.correlation import _rank_columns

        x = np.array(
            [
                [1.0, np.nan, 3.0],
                [1.0, 2.0, 1.0],
                [3.0, 4.0, 2.0],
                [np.nan, 1.0, np.nan],
            ],
            dtype=np.float64,
        )
        ranked = _rank_columns(x)
        # col0: values 1,1,3,nan → average ranks 1.5,1.5,3
        assert np.isnan(ranked[3, 0])
        np.testing.assert_allclose(sorted(ranked[~np.isnan(ranked[:, 0]), 0]), [1.5, 1.5, 3.0])
        # col1 has only 3 finite (>=2)
        assert not np.isnan(ranked[1, 1])

    def test_batch_spearman_identical_is_one(self, rng):
        from factorminer.evaluation.correlation import batch_spearman_correlation

        M, T = 40, 30
        cand = rng.normal(0, 1, (M, T))
        lib = cand[None, :, :].copy()
        corr = batch_spearman_correlation(cand, lib)
        assert corr.shape == (1,)
        assert corr[0] > 0.99

    def test_batch_spearman_matches_period_reference(self, rng):
        from scipy.stats import rankdata

        from factorminer.evaluation.correlation import batch_spearman_correlation

        candidate = rng.integers(-2, 3, size=(20, 9)).astype(np.float64)
        library = rng.integers(-2, 3, size=(3, 20, 9)).astype(np.float64)
        candidate[0:16, 0] = np.nan
        candidate[:, 1] = 1.0
        library[0, 2:5, 2] = np.nan

        expected = []
        for factor in library:
            period_correlations = []
            for period in range(candidate.shape[1]):
                left = candidate[:, period]
                right = factor[:, period]
                left_rank = np.full(left.shape, np.nan)
                right_rank = np.full(right.shape, np.nan)
                left_observed = ~np.isnan(left)
                right_observed = ~np.isnan(right)
                if int(left_observed.sum()) >= 2:
                    left_rank[left_observed] = rankdata(left[left_observed])
                if int(right_observed.sum()) >= 2:
                    right_rank[right_observed] = rankdata(right[right_observed])
                valid = ~(np.isnan(left_rank) | np.isnan(right_rank))
                if int(valid.sum()) < 5:
                    continue
                left_valid = left_rank[valid]
                right_valid = right_rank[valid]
                left_centered = left_valid - left_valid.mean()
                right_centered = right_valid - right_valid.mean()
                denominator = np.sqrt(
                    np.sum(left_centered**2) * np.sum(right_centered**2)
                )
                correlation = 0.0
                if denominator > 1e-12:
                    correlation = float(
                        np.sum(left_centered * right_centered) / denominator
                    )
                period_correlations.append(correlation)
            expected.append(np.mean(period_correlations) if period_correlations else 0.0)

        np.testing.assert_allclose(
            batch_spearman_correlation(candidate, library), expected, atol=1e-15
        )

    def test_batch_spearman_pairwise_symmetric(self, rng):
        from factorminer.evaluation.correlation import batch_spearman_pairwise

        sigs = [rng.normal(0, 1, (25, 20)) for _ in range(3)]
        mat = batch_spearman_pairwise(sigs)
        np.testing.assert_allclose(mat, mat.T)
        np.testing.assert_allclose(np.diag(mat), 1.0)
