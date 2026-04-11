"""Tests for factor combination and selection strategies."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.evaluation.combination import FactorCombiner
from factorminer.evaluation.selection import FactorSelector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def combiner():
    return FactorCombiner()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_signals(rng):
    """Three factor signals of shape (T=50, N=20)."""
    T, N = 50, 20
    return {
        1: rng.normal(0, 1, (T, N)),
        2: rng.normal(0, 1, (T, N)),
        3: rng.normal(0, 1, (T, N)),
    }


@pytest.fixture
def identical_signals(rng):
    """Two identical factor signals."""
    T, N = 30, 10
    sig = rng.normal(0, 1, (T, N))
    return {1: sig.copy(), 2: sig.copy()}


# ---------------------------------------------------------------------------
# Equal weight
# ---------------------------------------------------------------------------

class TestEqualWeight:
    """Test equal-weight combination."""

    def test_output_shape(self, combiner, simple_signals):
        result = combiner.equal_weight(simple_signals)
        T, N = next(iter(simple_signals.values())).shape
        assert result.shape == (T, N)

    def test_single_factor_is_zscore(self, combiner, rng):
        T, N = 30, 10
        sig = rng.normal(5, 2, (T, N))
        signals = {1: sig}
        result = combiner.equal_weight(signals)
        # Should be z-scored: mean ~0 per row
        row_means = np.nanmean(result, axis=1)
        np.testing.assert_array_almost_equal(row_means, np.zeros(T), decimal=10)

    def test_two_identical_factors(self, combiner, identical_signals):
        result = combiner.equal_weight(identical_signals)
        # Average of two identical z-scored signals = same z-scored signal
        single = combiner.equal_weight({1: identical_signals[1]})
        np.testing.assert_array_almost_equal(result, single, decimal=10)

    def test_empty_raises(self, combiner):
        with pytest.raises(ValueError, match="not be empty"):
            combiner.equal_weight({})

    def test_result_is_average(self, combiner, rng):
        """EW of multiple factors should be the average of their z-scores."""
        T, N = 20, 10
        s1 = np.ones((T, N))  # Constant -> z-score = 0
        s2 = np.tile(np.arange(N, dtype=np.float64), (T, 1))  # Variable
        signals = {1: s1, 2: s2}
        result = combiner.equal_weight(signals)
        # s1 z-score = 0 everywhere (constant), s2 z-score is not 0
        # Average should be s2_zscore / 2
        s2_zscore = combiner._cross_sectional_standardize(s2)
        # s1_zscore is 0 (constant cross-section, std=0 -> std=1 fallback)
        s1_zscore = combiner._cross_sectional_standardize(s1)
        expected = np.nanmean(np.stack([s1_zscore, s2_zscore]), axis=0)
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# IC-weighted
# ---------------------------------------------------------------------------

class TestICWeighted:
    """Test IC-weighted combination."""

    def test_output_shape(self, combiner, simple_signals):
        ic_values = {1: 0.05, 2: 0.08, 3: 0.03}
        result = combiner.ic_weighted(simple_signals, ic_values)
        T, N = next(iter(simple_signals.values())).shape
        assert result.shape == (T, N)

    def test_higher_ic_gets_more_weight(self, combiner, rng):
        T, N = 30, 15
        s1 = rng.normal(0, 1, (T, N))
        s2 = rng.normal(0, 1, (T, N))
        signals = {1: s1, 2: s2}

        # Give all weight to factor 1
        ic_values_1 = {1: 1.0, 2: 0.0001}
        result_1 = combiner.ic_weighted(signals, ic_values_1)

        # Give all weight to factor 2
        ic_values_2 = {1: 0.0001, 2: 1.0}
        result_2 = combiner.ic_weighted(signals, ic_values_2)

        # Results should be different (weighted differently)
        assert not np.allclose(result_1, result_2)

    def test_fallback_to_ew_with_nonpositive_ic(self, combiner, simple_signals):
        ic_values = {1: -0.01, 2: 0.0, 3: -0.05}
        result = combiner.ic_weighted(simple_signals, ic_values)
        # Should fall back to equal weight
        ew_result = combiner.equal_weight(simple_signals)
        np.testing.assert_array_almost_equal(result, ew_result)

    def test_empty_raises(self, combiner):
        with pytest.raises(ValueError, match="not be empty"):
            combiner.ic_weighted({}, {})


# ---------------------------------------------------------------------------
# Orthogonal
# ---------------------------------------------------------------------------

class TestOrthogonal:
    """Test orthogonal (Gram-Schmidt) combination."""

    def test_output_shape(self, combiner, simple_signals):
        result = combiner.orthogonal(simple_signals)
        T, N = next(iter(simple_signals.values())).shape
        assert result.shape == (T, N)

    def test_single_factor(self, combiner, rng):
        T, N = 20, 10
        sig = rng.normal(0, 1, (T, N))
        result = combiner.orthogonal({1: sig})
        # Single factor orthogonalized = z-scored version
        zscore = combiner._cross_sectional_standardize(sig)
        np.testing.assert_array_almost_equal(result, zscore)

    def test_orthogonal_different_from_ew(self, combiner, simple_signals):
        ew = combiner.equal_weight(simple_signals)
        ortho = combiner.orthogonal(simple_signals)
        # They should generally differ (unless signals are already orthogonal)
        # Check that the operation at least runs without error
        assert ortho.shape == ew.shape

    def test_empty_raises(self, combiner):
        with pytest.raises(ValueError, match="not be empty"):
            combiner.orthogonal({})


# ---------------------------------------------------------------------------
# Cross-sectional standardization helper
# ---------------------------------------------------------------------------

class TestCrossSectionalStandardize:
    """Test the internal _cross_sectional_standardize method."""

    def test_zero_mean_per_row(self, combiner, rng):
        T, N = 20, 15
        signals = rng.normal(5.0, 2.0, (T, N))
        result = combiner._cross_sectional_standardize(signals)
        row_means = np.nanmean(result, axis=1)
        np.testing.assert_array_almost_equal(row_means, np.zeros(T), decimal=10)

    def test_unit_std_per_row(self, combiner, rng):
        T, N = 20, 30
        signals = rng.normal(10.0, 3.0, (T, N))
        result = combiner._cross_sectional_standardize(signals)
        row_stds = np.nanstd(result, axis=1)
        np.testing.assert_array_almost_equal(row_stds, np.ones(T), decimal=5)

    def test_constant_row_handled(self, combiner):
        signals = np.ones((5, 10))
        result = combiner._cross_sectional_standardize(signals)
        # Constant row: std=0, should be 0 after standardization
        np.testing.assert_array_almost_equal(result, np.zeros((5, 10)))

    def test_nan_handling(self, combiner, rng):
        T, N = 10, 10
        signals = rng.normal(0, 1, (T, N))
        signals[0, 0] = np.nan
        result = combiner._cross_sectional_standardize(signals)
        assert np.isnan(result[0, 0])


# ---------------------------------------------------------------------------
# Gram-Schmidt helper
# ---------------------------------------------------------------------------

class TestGramSchmidt:
    """Test the Gram-Schmidt orthogonalization helper."""

    def test_orthogonal_output(self, rng):
        T, N = 20, 10
        factors = [rng.normal(0, 1, (T, N)) for _ in range(3)]
        ortho = FactorCombiner._gram_schmidt(factors)
        assert len(ortho) == 3

        # Check approximate orthogonality of flattened vectors
        for i in range(len(ortho)):
            for j in range(i + 1, len(ortho)):
                vi = np.where(np.isnan(ortho[i]), 0, ortho[i]).ravel()
                vj = np.where(np.isnan(ortho[j]), 0, ortho[j]).ravel()
                denom = np.sqrt(np.dot(vi, vi) * np.dot(vj, vj))
                if denom > 1e-10:
                    cos_sim = abs(np.dot(vi, vj) / denom)
                    assert cos_sim < 0.01, f"Factors {i} and {j} not orthogonal: cos={cos_sim}"

    def test_single_factor(self, rng):
        T, N = 10, 5
        f = [rng.normal(0, 1, (T, N))]
        ortho = FactorCombiner._gram_schmidt(f)
        assert len(ortho) == 1
        np.testing.assert_array_almost_equal(ortho[0], f[0])

    def test_nan_preserved(self, rng):
        T, N = 10, 5
        f1 = rng.normal(0, 1, (T, N))
        f2 = rng.normal(0, 1, (T, N))
        f1[0, 0] = np.nan
        ortho = FactorCombiner._gram_schmidt([f1, f2])
        assert np.isnan(ortho[0][0, 0])


# ===========================================================================
# Factor Selection Tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixtures for selection tests
# ---------------------------------------------------------------------------

@pytest.fixture
def selector():
    return FactorSelector()


@pytest.fixture
def synthetic_factors(rng):
    """Synthetic factor signals for selection tests.

    Creates 5 factors of shape (T=80, N=30) where factor 0 is predictive
    (correlated with returns) and the rest are noise.
    """
    T, N = 80, 30
    returns = rng.normal(0, 0.02, (T, N))

    signals = {}
    # Factor 0: predictive (signal ~ returns + noise)
    signals[0] = returns + rng.normal(0, 0.01, (T, N))
    # Factors 1-4: pure noise
    for i in range(1, 5):
        signals[i] = rng.normal(0, 1, (T, N))

    return signals, returns


@pytest.fixture
def uniform_factors(rng):
    """5 factors that are all weakly predictive."""
    T, N = 60, 25
    returns = rng.normal(0, 0.02, (T, N))

    signals = {}
    for i in range(5):
        signals[i] = returns * (0.5 + 0.1 * i) + rng.normal(0, 0.05, (T, N))

    return signals, returns


# ---------------------------------------------------------------------------
# _prepare_panel helper tests
# ---------------------------------------------------------------------------

class TestPreparePanel:
    """Test the _prepare_panel static helper."""

    def test_empty_returns_empty(self, selector):
        ids, X, y = selector._prepare_panel({}, np.empty((10, 5)))
        assert ids == []
        assert X.shape == (0, 0)
        assert y.shape == (0,)

    def test_output_shapes(self, selector, rng):
        T, N = 20, 10
        signals = {
            1: rng.normal(0, 1, (T, N)),
            2: rng.normal(0, 1, (T, N)),
        }
        returns = rng.normal(0, 1, (T, N))

        ids, X, y = selector._prepare_panel(signals, returns)
        assert ids == [1, 2]
        # X should be (n_valid_samples, 2), y should be (n_valid_samples,)
        assert X.shape[1] == 2
        assert X.shape[0] == y.shape[0]
        assert X.shape[0] <= T * N

    def test_nan_rows_dropped(self, selector, rng):
        T, N = 10, 5
        signals = {1: np.ones((T, N))}
        returns = np.ones((T, N))
        # Inject NaN into one position
        signals[1][0, 0] = np.nan

        ids, X, y = selector._prepare_panel(signals, returns)
        assert X.shape[0] == T * N - 1  # One row dropped

    def test_ids_sorted(self, selector, rng):
        T, N = 5, 3
        signals = {
            3: rng.normal(0, 1, (T, N)),
            1: rng.normal(0, 1, (T, N)),
            2: rng.normal(0, 1, (T, N)),
        }
        returns = rng.normal(0, 1, (T, N))
        ids, _, _ = selector._prepare_panel(signals, returns)
        assert ids == [1, 2, 3]


# ---------------------------------------------------------------------------
# _composite_icir helper tests
# ---------------------------------------------------------------------------

class TestCompositeICIR:
    """Test the _composite_icir static helper."""

    def test_empty_returns_zero(self, selector, rng):
        T, N = 20, 10
        signals = {1: rng.normal(0, 1, (T, N))}
        returns = rng.normal(0, 1, (T, N))
        assert selector._composite_icir(signals, [], returns) == 0.0

    def test_single_factor(self, selector, rng):
        T, N = 50, 20
        returns = rng.normal(0, 0.02, (T, N))
        signals = {1: returns + rng.normal(0, 0.01, (T, N))}
        icir = selector._composite_icir(signals, [1], returns)
        assert isinstance(icir, float)
        # Predictive signal should have positive ICIR
        assert icir > 0

    def test_noise_factor_low_icir(self, selector, rng):
        T, N = 50, 20
        returns = rng.normal(0, 0.02, (T, N))
        signals = {1: rng.normal(0, 1, (T, N))}
        icir = selector._composite_icir(signals, [1], returns)
        # Pure noise should have ICIR near zero (much lower than predictive)
        assert abs(icir) < 2.0  # Loose bound, noise can have some correlation


# ---------------------------------------------------------------------------
# Lasso selection tests
# ---------------------------------------------------------------------------

class TestLassoSelection:
    """Test L1-regularized Lasso factor selection."""

    def test_empty_signals_returns_empty(self, selector):
        result = selector.lasso_selection({}, np.empty((10, 5)))
        assert result == []

    def test_returns_list_of_tuples(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.lasso_selection(signals, returns, alpha=0.001)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            fid, coef = item
            assert isinstance(fid, int)
            assert isinstance(coef, float)

    def test_sorted_by_abs_coefficient(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.lasso_selection(signals, returns, alpha=0.001)
        if len(result) >= 2:
            abs_coefs = [abs(c) for _, c in result]
            assert abs_coefs == sorted(abs_coefs, reverse=True)

    def test_selects_predictive_factor(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.lasso_selection(signals, returns, alpha=0.0001)
        if result:
            selected_ids = [fid for fid, _ in result]
            # Factor 0 is correlated with returns; it should be selected
            assert 0 in selected_ids

    def test_sparsity_with_high_alpha(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result_low = selector.lasso_selection(signals, returns, alpha=0.0001)
        result_high = selector.lasso_selection(signals, returns, alpha=1.0)
        # Higher alpha should select fewer (or equal) factors
        assert len(result_high) <= len(result_low)

    def test_auto_alpha_via_cv(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        # alpha=None triggers LassoCV
        result = selector.lasso_selection(signals, returns, alpha=None)
        assert isinstance(result, list)

    def test_nonzero_coefficients_only(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.lasso_selection(signals, returns, alpha=0.001)
        for _, coef in result:
            assert abs(coef) > 1e-10


# ---------------------------------------------------------------------------
# Forward stepwise selection tests
# ---------------------------------------------------------------------------

class TestForwardStepwise:
    """Test greedy forward stepwise factor selection."""

    def test_empty_signals_returns_empty(self, selector):
        result = selector.forward_stepwise({}, np.empty((10, 5)))
        assert result == []

    def test_returns_list_of_tuples(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.forward_stepwise(signals, returns, max_factors=3)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_respects_max_factors(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.forward_stepwise(signals, returns, max_factors=2)
        assert len(result) <= 2

    def test_selection_order(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.forward_stepwise(signals, returns, max_factors=5)
        # Each entry should have positive delta (ICIR improvement)
        for _, delta in result:
            assert delta > 0

    def test_no_duplicate_selections(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.forward_stepwise(signals, returns, max_factors=5)
        selected_ids = [fid for fid, _ in result]
        assert len(selected_ids) == len(set(selected_ids))

    def test_predictive_factor_selected_first(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.forward_stepwise(signals, returns, max_factors=5)
        if result:
            # Factor 0 is highly predictive; should likely be selected first
            first_id = result[0][0]
            assert first_id == 0

    def test_single_factor_available(self, selector, rng):
        T, N = 40, 15
        returns = rng.normal(0, 0.02, (T, N))
        signals = {42: returns + rng.normal(0, 0.01, (T, N))}
        result = selector.forward_stepwise(signals, returns, max_factors=5)
        assert len(result) <= 1
        if result:
            assert result[0][0] == 42


# ---------------------------------------------------------------------------
# XGBoost selection tests
# ---------------------------------------------------------------------------

class TestXGBoostSelection:
    """Test XGBoost importance-based factor selection."""

    def test_empty_signals_returns_empty(self, selector):
        result = selector.xgboost_selection({}, np.empty((10, 5)))
        assert result == []

    def test_returns_all_factors(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.xgboost_selection(signals, returns)
        # XGBoost returns importance for all factors
        assert len(result) == len(signals)

    def test_returns_list_of_tuples(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.xgboost_selection(signals, returns)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            fid, importance = item
            assert isinstance(fid, int)
            assert isinstance(importance, float)

    def test_sorted_by_importance(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.xgboost_selection(signals, returns)
        importances = [imp for _, imp in result]
        assert importances == sorted(importances, reverse=True)

    def test_importances_nonnegative(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.xgboost_selection(signals, returns)
        for _, importance in result:
            assert importance >= 0.0

    def test_predictive_factor_has_high_importance(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.xgboost_selection(signals, returns)
        if result:
            # Factor 0 is predictive; it should have the highest importance
            top_id = result[0][0]
            assert top_id == 0

    def test_importances_sum_to_one(self, selector, synthetic_factors):
        signals, returns = synthetic_factors
        result = selector.xgboost_selection(signals, returns)
        total = sum(imp for _, imp in result)
        # Gain importances from XGBoost should sum to ~1
        assert total == pytest.approx(1.0, abs=0.05)
