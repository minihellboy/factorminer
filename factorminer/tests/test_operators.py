"""Tests for all operator categories via the registry."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.operators.registry import (
    execute_operator,
    get_operator,
    implemented_operators,
    list_operators,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _arr(*rows):
    """Build a (M, T) float64 array from nested lists."""
    return np.array(rows, dtype=np.float64)


@pytest.fixture
def x_simple():
    """Simple 2x10 input for operator tests."""
    return _arr(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    )


@pytest.fixture
def y_simple():
    return _arr(
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------

class TestArithmeticOps:
    """Test element-wise arithmetic operators."""

    def test_add(self, x_simple, y_simple):
        result = execute_operator("Add", x_simple, y_simple)
        expected = x_simple + y_simple
        np.testing.assert_array_almost_equal(result, expected)

    def test_sub(self, x_simple, y_simple):
        result = execute_operator("Sub", x_simple, y_simple)
        expected = x_simple - y_simple
        np.testing.assert_array_almost_equal(result, expected)

    def test_mul(self, x_simple, y_simple):
        result = execute_operator("Mul", x_simple, y_simple)
        expected = x_simple * y_simple
        np.testing.assert_array_almost_equal(result, expected)

    def test_neg_negates(self, x_simple):
        result = execute_operator("Neg", x_simple)
        np.testing.assert_array_almost_equal(result, -x_simple)

    def test_neg_double_neg(self, x_simple):
        result = execute_operator("Neg", execute_operator("Neg", x_simple))
        np.testing.assert_array_almost_equal(result, x_simple)

    def test_abs(self):
        x = _arr([-1, -2, 3, 0], [5, -6, 0, -8])
        result = execute_operator("Abs", x)
        np.testing.assert_array_almost_equal(result, np.abs(x))

    def test_sign(self):
        x = _arr([-3, 0, 5], [7, -2, 0])
        result = execute_operator("Sign", x)
        np.testing.assert_array_almost_equal(result, np.sign(x))

    def test_div_by_zero_returns_nan(self):
        x = _arr([1, 2, 3], [4, 5, 6])
        y = _arr([0, 0, 0], [1, 0, 2])
        result = execute_operator("Div", x, y)
        # Where y is 0, result should be NaN
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        assert np.isnan(result[1, 1])
        # Where y is non-zero, should be correct
        np.testing.assert_almost_equal(result[1, 0], 4.0)
        np.testing.assert_almost_equal(result[1, 2], 3.0)

    def test_log_handles_negative(self):
        x = _arr([-1, 0, 1, np.e - 1], [2, -3, 0.5, 10])
        result = execute_operator("Log", x)
        # Log is defined as log(1+|x|)*sign(x)
        expected = np.log1p(np.abs(x)) * np.sign(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_sqrt_handles_negative(self):
        x = _arr([-4, 0, 9, 16], [1, -1, 4, 25])
        result = execute_operator("Sqrt", x)
        expected = np.sqrt(np.abs(x)) * np.sign(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_square(self, x_simple):
        result = execute_operator("Square", x_simple)
        np.testing.assert_array_almost_equal(result, x_simple ** 2)

    def test_inv_zero_returns_nan(self):
        x = _arr([0, 1, 2], [3, 0, 5])
        result = execute_operator("Inv", x)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 1])
        np.testing.assert_almost_equal(result[0, 1], 1.0)

    def test_max_elementwise(self, x_simple, y_simple):
        result = execute_operator("Max", x_simple, y_simple)
        np.testing.assert_array_almost_equal(result, np.fmax(x_simple, y_simple))

    def test_min_elementwise(self, x_simple, y_simple):
        result = execute_operator("Min", x_simple, y_simple)
        np.testing.assert_array_almost_equal(result, np.fmin(x_simple, y_simple))

    def test_clip(self):
        x = _arr([-5, -1, 0, 2, 5], [10, -10, 3, -3, 0])
        result = execute_operator("Clip", x, params={"lower": -3.0, "upper": 3.0})
        np.testing.assert_array_almost_equal(result, np.clip(x, -3.0, 3.0))


# ---------------------------------------------------------------------------
# Statistical operators (rolling window)
# ---------------------------------------------------------------------------

class TestStatisticalOps:
    """Test rolling-window statistical operators."""

    def test_mean_window3(self):
        x = _arr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = execute_operator("Mean", x, params={"window": 3})
        assert result.shape == (1, 10)
        # First 2 values should be NaN
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        # Mean of [1,2,3] = 2.0
        np.testing.assert_almost_equal(result[0, 2], 2.0)
        # Mean of [2,3,4] = 3.0
        np.testing.assert_almost_equal(result[0, 3], 3.0)
        # Mean of [8,9,10] = 9.0
        np.testing.assert_almost_equal(result[0, 9], 9.0)

    def test_std_window3(self):
        x = _arr([1, 2, 3, 4, 5])
        result = execute_operator("Std", x, params={"window": 3})
        assert result.shape == (1, 5)
        # First 2 values NaN
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        # std of [1,2,3] with ddof=1 = 1.0
        np.testing.assert_almost_equal(result[0, 2], 1.0)

    def test_sum_window3(self):
        x = _arr([1, 2, 3, 4, 5])
        result = execute_operator("Sum", x, params={"window": 3})
        np.testing.assert_almost_equal(result[0, 2], 6.0)  # 1+2+3
        np.testing.assert_almost_equal(result[0, 4], 12.0)  # 3+4+5

    def test_ts_max_window3(self):
        x = _arr([3, 1, 4, 1, 5, 9, 2, 6])
        result = execute_operator("TsMax", x, params={"window": 3})
        np.testing.assert_almost_equal(result[0, 2], 4.0)  # max(3,1,4)
        np.testing.assert_almost_equal(result[0, 5], 9.0)  # max(1,5,9)

    def test_ts_min_window3(self):
        x = _arr([3, 1, 4, 1, 5, 9, 2, 6])
        result = execute_operator("TsMin", x, params={"window": 3})
        np.testing.assert_almost_equal(result[0, 2], 1.0)  # min(3,1,4)
        np.testing.assert_almost_equal(result[0, 5], 1.0)  # min(1,5,9)

    def test_ts_rank_basic(self):
        # Ascending series: latest should have high rank
        x = _arr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = execute_operator("TsRank", x, params={"window": 5})
        # At index 4, window=[1,2,3,4,5], latest=5 is the largest
        # count_less = 4 values less than 5, count_valid = 5
        # rank = 4 / (5-1) = 1.0
        np.testing.assert_almost_equal(result[0, 4], 1.0)

    def test_median_window3(self):
        x = _arr([1, 5, 3, 4, 2])
        result = execute_operator("Median", x, params={"window": 3})
        np.testing.assert_almost_equal(result[0, 2], 3.0)  # median(1,5,3)


# ---------------------------------------------------------------------------
# Time-series operators
# ---------------------------------------------------------------------------

class TestTimeseriesOps:
    """Test time-series operators like Delta, Delay, Return."""

    def test_delta_period1_is_diff(self):
        x = _arr([1, 3, 6, 10, 15])
        result = execute_operator("Delta", x, params={"window": 1})
        assert np.isnan(result[0, 0])
        np.testing.assert_almost_equal(result[0, 1], 2.0)  # 3-1
        np.testing.assert_almost_equal(result[0, 2], 3.0)  # 6-3
        np.testing.assert_almost_equal(result[0, 3], 4.0)  # 10-6
        np.testing.assert_almost_equal(result[0, 4], 5.0)  # 15-10

    def test_delay_lags_by_period(self):
        x = _arr([10, 20, 30, 40, 50])
        result = execute_operator("Delay", x, params={"window": 2})
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 1])
        np.testing.assert_almost_equal(result[0, 2], 10.0)
        np.testing.assert_almost_equal(result[0, 3], 20.0)
        np.testing.assert_almost_equal(result[0, 4], 30.0)

    def test_return_period1(self):
        x = _arr([100, 110, 99, 105])
        result = execute_operator("Return", x, params={"window": 1})
        assert np.isnan(result[0, 0])
        np.testing.assert_almost_equal(result[0, 1], 0.10)  # 110/100 - 1
        np.testing.assert_almost_equal(result[0, 2], -0.1, decimal=2)  # 99/110 - 1

    def test_cumsum(self):
        x = _arr([1, 2, 3, 4, 5])
        result = execute_operator("CumSum", x)
        np.testing.assert_array_almost_equal(result[0], [1, 3, 6, 10, 15])

    def test_cummax(self):
        x = _arr([3, 1, 4, 1, 5, 9, 2])
        result = execute_operator("CumMax", x)
        np.testing.assert_array_almost_equal(result[0], [3, 3, 4, 4, 5, 9, 9])

    def test_cummin(self):
        x = _arr([5, 3, 4, 1, 2, 6, 0])
        result = execute_operator("CumMin", x)
        np.testing.assert_array_almost_equal(result[0], [5, 3, 3, 1, 1, 1, 0])


# ---------------------------------------------------------------------------
# Cross-sectional operators
# ---------------------------------------------------------------------------

class TestCrossSectionalOps:
    """Test cross-sectional operators."""

    def test_csrank_produces_percentiles(self):
        # 5 assets, 1 time step; values 1..5
        x = _arr([1], [2], [3], [4], [5])
        result = execute_operator("CsRank", x)
        assert result.shape == (5, 1)
        # Ranks should be [0, 0.25, 0.5, 0.75, 1.0]
        expected = _arr([0], [0.25], [0.5], [0.75], [1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_csrank_nan_handling(self):
        x = _arr([np.nan], [2], [3], [np.nan], [5])
        result = execute_operator("CsRank", x)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[3, 0])
        # Valid ranks for [2, 3, 5] = [0, 0.5, 1.0]
        valid = result[~np.isnan(result)]
        assert len(valid) == 3

    def test_cszscore_zero_mean(self):
        x = _arr([1], [2], [3], [4], [5])
        result = execute_operator("CsZScore", x)
        # Mean of z-scores should be ~0
        np.testing.assert_almost_equal(np.nanmean(result[:, 0]), 0.0, decimal=10)

    def test_csdemean(self):
        x = _arr([10], [20], [30])
        result = execute_operator("CsDemean", x)
        expected = _arr([-10], [0], [10])
        np.testing.assert_array_almost_equal(result, expected)

    def test_csscale_unit_l1(self):
        x = _arr([1], [2], [3])
        result = execute_operator("CsScale", x)
        l1_norm = np.nansum(np.abs(result[:, 0]))
        np.testing.assert_almost_equal(l1_norm, 1.0)


# ---------------------------------------------------------------------------
# Smoothing operators
# ---------------------------------------------------------------------------

class TestSmoothingOps:
    """Test smoothing / moving average operators."""

    def test_sma_equals_mean(self):
        x = _arr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        sma = execute_operator("SMA", x, params={"window": 3})
        mean = execute_operator("Mean", x, params={"window": 3})
        # SMA should equal Mean for non-NaN data
        valid = ~(np.isnan(sma) | np.isnan(mean))
        np.testing.assert_array_almost_equal(sma[valid], mean[valid])

    def test_ema_convergence(self):
        # Constant series: EMA should converge to that constant
        x = _arr([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        result = execute_operator("EMA", x, params={"window": 3})
        # Should all be 5 (for constant input, EMA = constant)
        np.testing.assert_array_almost_equal(result[0], np.full(10, 5.0))

    def test_ema_output_shape(self, x_simple):
        result = execute_operator("EMA", x_simple, params={"window": 5})
        assert result.shape == x_simple.shape


# ---------------------------------------------------------------------------
# Regression operators
# ---------------------------------------------------------------------------

class TestRegressionOps:
    """Test rolling regression operators."""

    def test_slope_of_linear_data(self):
        # Perfectly linear: y = 2*t for each asset
        t_vals = np.arange(20, dtype=np.float64)
        x = np.stack([2 * t_vals, 3 * t_vals], axis=0)  # (2, 20)
        result = execute_operator("TsLinRegSlope", x, params={"window": 5})
        # After window-1 NaNs, slope should be ~2.0 for first asset
        valid_idx = ~np.isnan(result[0])
        if valid_idx.any():
            np.testing.assert_almost_equal(result[0, valid_idx][-1], 2.0, decimal=3)
            np.testing.assert_almost_equal(result[1, valid_idx][-1], 3.0, decimal=3)

    def test_resid_of_linear_is_near_zero(self):
        # Perfectly linear: residuals should be ~0
        t_vals = np.arange(20, dtype=np.float64)
        x = np.stack([2 * t_vals + 1, t_vals + 5], axis=0)
        result = execute_operator("TsLinRegResid", x, params={"window": 5})
        valid = ~np.isnan(result)
        if valid.any():
            np.testing.assert_almost_equal(np.abs(result[valid]).max(), 0.0, decimal=3)


# ---------------------------------------------------------------------------
# Logical operators
# ---------------------------------------------------------------------------

class TestLogicalOps:
    """Test conditional and comparison operators."""

    def test_ifelse_branching(self):
        cond = _arr([1, -1, 1, -1, 0])
        x = _arr([10, 20, 30, 40, 50])
        y = _arr([100, 200, 300, 400, 500])
        result = execute_operator("IfElse", cond, x, y)
        # cond > 0 -> x, else y
        np.testing.assert_almost_equal(result[0, 0], 10)
        np.testing.assert_almost_equal(result[0, 1], 200)
        np.testing.assert_almost_equal(result[0, 2], 30)
        np.testing.assert_almost_equal(result[0, 3], 400)
        np.testing.assert_almost_equal(result[0, 4], 500)  # 0 is not > 0

    def test_greater(self):
        x = _arr([1, 5, 3], [4, 2, 6])
        y = _arr([2, 3, 3], [4, 5, 1])
        result = execute_operator("Greater", x, y)
        expected = _arr([0, 1, 0], [0, 0, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_less(self):
        x = _arr([1, 5, 3])
        y = _arr([2, 3, 3])
        result = execute_operator("Less", x, y)
        expected = _arr([1, 0, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_and(self):
        x = _arr([1, 1, -1, -1])
        y = _arr([1, -1, 1, -1])
        result = execute_operator("And", x, y)
        expected = _arr([1, 0, 0, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_or(self):
        x = _arr([1, 1, -1, -1])
        y = _arr([1, -1, 1, -1])
        result = execute_operator("Or", x, y)
        expected = _arr([1, 1, 1, 0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_not(self):
        x = _arr([1, -1, 0, 5])
        result = execute_operator("Not", x)
        expected = _arr([0, 1, 1, 0])
        np.testing.assert_array_almost_equal(result, expected)


# ---------------------------------------------------------------------------
# NaN propagation
# ---------------------------------------------------------------------------

class TestNaNPropagation:
    """Test NaN handling across operators."""

    def test_add_nan_propagation(self):
        x = _arr([1, np.nan, 3])
        y = _arr([4, 5, np.nan])
        result = execute_operator("Add", x, y)
        assert np.isnan(result[0, 1])
        assert np.isnan(result[0, 2])
        np.testing.assert_almost_equal(result[0, 0], 5.0)

    def test_neg_nan_propagation(self):
        x = _arr([1, np.nan, 3])
        result = execute_operator("Neg", x)
        assert np.isnan(result[0, 1])

    def test_greater_nan_propagation(self):
        x = _arr([1, np.nan, 3])
        y = _arr([0, 1, np.nan])
        result = execute_operator("Greater", x, y)
        assert np.isnan(result[0, 1])
        assert np.isnan(result[0, 2])


# ---------------------------------------------------------------------------
# GPU (torch) vs CPU equivalence
# ---------------------------------------------------------------------------

class TestGPUCPUEquivalence:
    """Test that torch and numpy implementations produce similar results."""

    @pytest.fixture
    def torch_available(self):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            pytest.skip("PyTorch not available")
        return True

    @pytest.mark.parametrize("op_name", ["Add", "Sub", "Mul", "Neg", "Abs", "Sign"])
    def test_arithmetic_equivalence(self, torch_available, x_simple, y_simple, op_name):
        import torch as th

        spec = get_operator(op_name)
        if spec.arity == 1:
            np_result = execute_operator(op_name, x_simple, backend="numpy")
            torch_result = execute_operator(
                op_name, th.tensor(x_simple), backend="torch"
            )
        else:
            np_result = execute_operator(op_name, x_simple, y_simple, backend="numpy")
            torch_result = execute_operator(
                op_name, th.tensor(x_simple), th.tensor(y_simple), backend="torch"
            )
        np.testing.assert_array_almost_equal(
            np_result, torch_result.numpy(), decimal=5
        )

    @pytest.mark.parametrize("op_name", ["Mean", "Std", "TsMax", "TsMin"])
    def test_statistical_equivalence(self, torch_available, x_simple, op_name):
        import torch as th

        np_result = execute_operator(op_name, x_simple, params={"window": 3}, backend="numpy")
        torch_result = execute_operator(
            op_name, th.tensor(x_simple, dtype=th.float64), params={"window": 3}, backend="torch"
        )
        valid = ~(np.isnan(np_result) | np.isnan(torch_result.numpy()))
        if valid.any():
            np.testing.assert_array_almost_equal(
                np_result[valid], torch_result.numpy()[valid], decimal=4
            )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    """Test operator registry functions."""

    def test_list_operators_flat(self):
        ops = list_operators(grouped=False)
        assert isinstance(ops, list)
        assert "Add" in ops
        assert "Neg" in ops
        assert "CsRank" in ops

    def test_list_operators_grouped(self):
        groups = list_operators(grouped=True)
        assert isinstance(groups, dict)
        assert "ARITHMETIC" in groups
        assert "STATISTICAL" in groups

    def test_implemented_operators(self):
        impl = implemented_operators()
        assert len(impl) > 0
        assert "Add" in impl

    def test_get_operator_unknown_raises(self):
        with pytest.raises(KeyError):
            get_operator("FooBarBaz")

    def test_execute_unknown_raises(self):
        with pytest.raises(KeyError):
            execute_operator("UnknownOp", np.ones((2, 3)))
