"""Operators must not fabricate values from missing (NaN) bars.

Real panels carry gaps: instruments listed after the panel starts, trading
suspensions, delistings. A gap is missing information, never a zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.core.expression_plan import OPERATOR_SEMANTICS_VERSION
from factorminer.core.parser import parse

NAN = np.nan


def _eval(formula: str, **panels: np.ndarray) -> np.ndarray:
    data = {f"${name}": np.atleast_2d(np.asarray(values, dtype=np.float64))
            for name, values in panels.items()}
    with np.errstate(all="ignore"):
        return parse(formula).evaluate(data)[0]


def test_semantics_version_marks_the_missing_data_rules():
    assert OPERATOR_SEMANTICS_VERSION == "numpy-expression-v2"


def test_division_by_a_missing_value_is_missing_not_zero():
    out = _eval("Div($close, $open)", close=[1.0, 2.0, 3.0], open=[2.0, NAN, 0.0])
    np.testing.assert_array_equal(out, [0.5, NAN, 0.0])  # 0 only for a vanishing denominator


def test_return_after_a_gap_is_missing():
    out = _eval("Return($close, 1)", close=[10.0, NAN, 11.0, 12.1])
    np.testing.assert_allclose(out, [NAN, NAN, NAN, 0.1], equal_nan=True)


def test_log_return_never_reads_a_price_level_as_a_return():
    out = _eval("LogReturn($close, 1)", close=[10.0, NAN, 11.0])
    assert np.isnan(out[2])


def test_ema_skips_gaps_and_starts_at_the_first_observation():
    close = np.array([NAN, NAN, 1.0, 2.0, NAN, 4.0])
    out = _eval("EMA($close, 3)", close=close)
    alpha = 0.5
    s3 = alpha * 2.0 + (1 - alpha) * 1.0
    s5 = alpha * 4.0 + (1 - alpha) * s3
    np.testing.assert_allclose(out, [NAN, NAN, 1.0, s3, NAN, s5], equal_nan=True)


def test_ema_of_a_gapless_series_is_the_classic_recurrence():
    close = np.arange(1.0, 9.0)
    out = _eval("EMA($close, 4)", close=close)
    expected = [close[0]]
    for value in close[1:]:
        expected.append(0.4 * value + 0.6 * expected[-1])
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("formula", ["Sum($close, 3)", "Prod($close, 3)"])
def test_empty_windows_are_missing(formula):
    out = _eval(formula, close=[1.0, NAN, NAN, NAN, 2.0])
    assert np.isnan(out[3]) and np.isfinite(out[4])


def test_ts_rank_of_a_missing_bar_is_missing():
    out = _eval("TsRank($close, 3)", close=[1.0, 2.0, NAN, 3.0])
    assert np.isnan(out[2]) and out[3] == pytest.approx(2 / 3)  # ranks over the window length


@pytest.mark.parametrize("formula", ["Corr($close, $open, 3)", "Beta($close, $open, 3)"])
def test_correlations_need_observations_but_constant_windows_are_zero(formula):
    close = [1.0, 2.0, 3.0, NAN, NAN, NAN, 5.0, 5.0, 5.0]
    open_ = [2.0, 1.0, 4.0, NAN, NAN, NAN, 5.0, 5.0, 5.0]
    out = _eval(formula, close=close, open=open_)
    assert np.isnan(out[5])
    assert out[8] == 0.0


@pytest.mark.parametrize("name", ["CumSum", "CumProd", "CumMax", "CumMin"])
def test_cumulative_operators_are_missing_on_missing_bars(name):
    out = _eval(f"{name}($close)", close=[NAN, 2.0, NAN, 3.0])
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert np.isfinite(out[1]) and np.isfinite(out[3])


def test_hma_does_not_zero_fill_its_warm_up():
    out = _eval("HMA($close, 4)", close=np.arange(1.0, 11.0))
    finite = out[np.isfinite(out)]
    assert np.isnan(out[:4]).all() and finite.min() > 1.0
