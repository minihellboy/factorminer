"""Compiled CPU backend using Bottleneck with NumPy fallbacks."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from factorminer.operators import smoothing, statistical, timeseries

try:
    import bottleneck as bn

    BOTTLENECK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional acceleration
    bn = None  # type: ignore[assignment]
    BOTTLENECK_AVAILABLE = False


def backend_available() -> bool:
    """Return whether the compiled Bottleneck backend is importable."""
    return BOTTLENECK_AVAILABLE


def _full_window(result: np.ndarray, window: int) -> np.ndarray:
    out = np.asarray(result, dtype=np.float64).copy()
    if window > 1:
        out[:, : window - 1] = np.nan
    return out


def mean_c(x: np.ndarray, window: int = 10) -> np.ndarray:
    if not BOTTLENECK_AVAILABLE:
        return statistical.mean_np(x, window)
    return _full_window(bn.move_mean(x, window=window, min_count=window, axis=1), int(window))


def std_c(x: np.ndarray, window: int = 10) -> np.ndarray:
    if not BOTTLENECK_AVAILABLE:
        return statistical.std_np(x, window)
    return _full_window(
        bn.move_std(x, window=window, min_count=window, axis=1, ddof=1),
        int(window),
    )


def sum_c(x: np.ndarray, window: int = 10) -> np.ndarray:
    if not BOTTLENECK_AVAILABLE:
        return statistical.sum_np(x, window)
    return _full_window(bn.move_sum(x, window=window, min_count=window, axis=1), int(window))


def ts_max_c(x: np.ndarray, window: int = 10) -> np.ndarray:
    if not BOTTLENECK_AVAILABLE:
        return statistical.ts_max_np(x, window)
    return _full_window(bn.move_max(x, window=window, min_count=window, axis=1), int(window))


def ts_min_c(x: np.ndarray, window: int = 10) -> np.ndarray:
    if not BOTTLENECK_AVAILABLE:
        return statistical.ts_min_np(x, window)
    return _full_window(bn.move_min(x, window=window, min_count=window, axis=1), int(window))


def ts_argmax_c(x: np.ndarray, window: int = 10) -> np.ndarray:
    if not BOTTLENECK_AVAILABLE or not hasattr(bn, "move_argmax"):
        return statistical.ts_argmax_np(x, window)
    return _full_window(
        np.asarray(bn.move_argmax(x, window=window, min_count=window, axis=1), dtype=np.float64),
        int(window),
    )


def ts_argmin_c(x: np.ndarray, window: int = 10) -> np.ndarray:
    if not BOTTLENECK_AVAILABLE or not hasattr(bn, "move_argmin"):
        return statistical.ts_argmin_np(x, window)
    return _full_window(
        np.asarray(bn.move_argmin(x, window=window, min_count=window, axis=1), dtype=np.float64),
        int(window),
    )


def sma_c(x: np.ndarray, window: int = 10) -> np.ndarray:
    if not BOTTLENECK_AVAILABLE:
        return smoothing.sma_np(x, window)
    return mean_c(x, window)


def delta_c(x: np.ndarray, window: int = 1) -> np.ndarray:
    return timeseries.delta_np(x, window)


def corr_c(x: np.ndarray, y: np.ndarray, window: int = 10) -> np.ndarray:
    return timeseries.corr_np(x, y, window)


def cov_c(x: np.ndarray, y: np.ndarray, window: int = 10) -> np.ndarray:
    return timeseries.cov_np(x, y, window)


def beta_c(x: np.ndarray, y: np.ndarray, window: int = 10) -> np.ndarray:
    return timeseries.beta_np(x, y, window)


def resid_c(x: np.ndarray, y: np.ndarray, window: int = 10) -> np.ndarray:
    return timeseries.resid_np(x, y, window)


def wma_c(x: np.ndarray, window: int = 10) -> np.ndarray:
    return timeseries.wma_np(x, window)


def ts_rank_c(x: np.ndarray, window: int = 10) -> np.ndarray:
    return statistical.ts_rank_np(x, window)


C_BACKEND_IMPLS: dict[str, Callable] = {
    "Beta": beta_c,
    "Corr": corr_c,
    "Cov": cov_c,
    "Delta": delta_c,
    "Mean": mean_c,
    "Resid": resid_c,
    "SMA": sma_c,
    "Std": std_c,
    "Sum": sum_c,
    "TsArgMax": ts_argmax_c,
    "TsArgMin": ts_argmin_c,
    "TsMax": ts_max_c,
    "TsMin": ts_min_c,
    "TsRank": ts_rank_c,
    "WMA": wma_c,
}
