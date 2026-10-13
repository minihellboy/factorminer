"""Vectorized rolling quantile matches NumPy's per-window nanquantile exactly."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from factorminer.core.expression_tree import rolling_nanquantile
from factorminer.operators.statistical import quantile_np


def _reference(x: np.ndarray, window: int, q: float) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for t in range(window - 1, x.shape[1]):
            out[:, t] = np.nanquantile(x[:, t - window + 1:t + 1], q, axis=1)
    return out


@pytest.mark.parametrize("seed", range(40))
def test_rolling_nanquantile_is_bit_identical_to_numpy(seed):
    rng = np.random.default_rng(seed)
    rows, periods, window = int(rng.integers(1, 8)), int(rng.integers(1, 40)), int(rng.integers(1, 15))
    q = float(rng.choice([0.0, 0.5, 1.0, rng.random()]))
    x = rng.normal(size=(rows, periods))
    if seed % 3 == 0:
        x = np.round(x, 1)  # ties
    x[rng.random(x.shape) < rng.random()] = np.nan
    expected = _reference(x, window, q)
    np.testing.assert_array_equal(rolling_nanquantile(x, window, q), expected)
    np.testing.assert_array_equal(quantile_np(x, window, q), expected)


@pytest.mark.parametrize("budget", [7, 64, 400])  # forces time chunks, row blocks, or both
def test_rolling_nanquantile_blocks_large_panels(monkeypatch, budget):
    monkeypatch.setattr("factorminer.core.expression_tree._QUANTILE_BLOCK_ELEMENTS", budget)
    x = np.random.default_rng(0).normal(size=(9, 30))
    x[x > 1.2] = np.nan
    np.testing.assert_array_equal(rolling_nanquantile(x, 5, 0.8), _reference(x, 5, 0.8))
