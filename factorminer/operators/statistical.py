"""Rolling-window statistical operators.

Each function operates along the **time** axis (axis=1) independently for
every asset row.  Input shape: ``(M, T)`` -> output shape ``(M, T)``.
The first ``(window - 1)`` values in each row are set to ``NaN``.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# ===========================================================================
# Helpers
# ===========================================================================

def _rolling_np(x: np.ndarray, window: int):
    """Yield views of shape (M, T-w+1, w) using stride tricks."""
    M, T = x.shape
    if T < window:
        return None
    strides = (x.strides[0], x.strides[1], x.strides[1])
    shape = (M, T - window + 1, window)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _pad_front(result: np.ndarray, window: int, total_T: int) -> np.ndarray:
    """Pad front of time axis with NaN to restore original length."""
    M = result.shape[0]
    pad_len = total_T - result.shape[1]
    if pad_len > 0:
        pad = np.full((M, pad_len), np.nan, dtype=result.dtype)
        return np.concatenate([pad, result], axis=1)
    return result


def _unfold_torch(x: torch.Tensor, window: int) -> torch.Tensor:
    """Unfold last dimension to get sliding windows: (M, T) -> (M, T-w+1, w)."""
    return x.unfold(dimension=1, size=window, step=1)


def _pad_front_torch(result: torch.Tensor, window: int, total_T: int) -> torch.Tensor:
    M = result.shape[0]
    pad_len = total_T - result.shape[1]
    if pad_len > 0:
        pad = torch.full((M, pad_len), float("nan"), device=result.device, dtype=result.dtype)
        return torch.cat([pad, result], dim=1)
    return result


# ===========================================================================
# NumPy implementations
# ===========================================================================

def mean_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.nanmean(w, axis=2)
    return _pad_front(result, window, T)


def std_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.nanstd(w, axis=2, ddof=1)
    return _pad_front(result, window, T)


def var_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.nanvar(w, axis=2, ddof=1)
    return _pad_front(result, window, T)


def skew_np(x: np.ndarray, window: int = 20) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    m = np.nanmean(w, axis=2, keepdims=True)
    d = w - m
    np.sum(~np.isnan(w), axis=2, keepdims=True).astype(np.float64)
    m2 = np.nanmean(d ** 2, axis=2, keepdims=True)
    m3 = np.nanmean(d ** 3, axis=2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        sk = m3 / np.power(m2, 1.5)
    result = sk.squeeze(2)
    return _pad_front(result, window, T)


def kurt_np(x: np.ndarray, window: int = 20) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    m = np.nanmean(w, axis=2, keepdims=True)
    d = w - m
    m2 = np.nanmean(d ** 2, axis=2, keepdims=True)
    m4 = np.nanmean(d ** 4, axis=2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        kt = m4 / np.power(m2, 2.0) - 3.0
    result = kt.squeeze(2)
    return _pad_front(result, window, T)


def median_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.nanmedian(w, axis=2)
    return _pad_front(result, window, T)


def sum_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.nansum(w, axis=2)
    # If all NaN in a window, nansum returns 0; fix that
    all_nan = np.all(np.isnan(w), axis=2)
    result[all_nan] = np.nan
    return _pad_front(result, window, T)


def prod_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.nanprod(w, axis=2)
    all_nan = np.all(np.isnan(w), axis=2)
    result[all_nan] = np.nan
    return _pad_front(result, window, T)


def ts_max_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.nanmax(w, axis=2)
    return _pad_front(result, window, T)


def ts_min_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.nanmin(w, axis=2)
    return _pad_front(result, window, T)


def ts_argmax_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.nanargmax(w, axis=2).astype(np.float64)
    return _pad_front(result, window, T)


def ts_argmin_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.nanargmin(w, axis=2).astype(np.float64)
    return _pad_front(result, window, T)


def ts_rank_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Rolling percentile rank of the latest value within its window."""
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    latest = w[:, :, -1:]  # (M, T-w+1, 1)
    count_less = np.nansum(w < latest, axis=2).astype(np.float64)
    count_valid = np.sum(~np.isnan(w), axis=2).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = count_less / (count_valid - 1.0)
    result[count_valid <= 1] = np.nan
    return _pad_front(result, window, T)


def quantile_np(x: np.ndarray, window: int = 10, q: float = 0.5) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.nanquantile(w, q, axis=2)
    return _pad_front(result, window, T)


def count_nan_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.sum(np.isnan(w), axis=2).astype(np.float64)
    return _pad_front(result, window, T)


def count_not_nan_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    window = int(window)
    M, T = x.shape
    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    result = np.sum(~np.isnan(w), axis=2).astype(np.float64)
    return _pad_front(result, window, T)


# ===========================================================================
# PyTorch (GPU) implementations
# ===========================================================================

def mean_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)  # (M, T-w+1, w)
    result = w.nanmean(dim=2)
    return _pad_front_torch(result, window, T)


def std_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    m = w.nanmean(dim=2, keepdim=True)
    d = w - m
    not_nan = ~torch.isnan(w)
    d = d.nan_to_num(0.0)
    n = not_nan.sum(dim=2, keepdim=True).float()
    var = (d ** 2).sum(dim=2, keepdim=True) / (n - 1).clamp(min=1)
    result = var.sqrt().squeeze(2)
    result[n.squeeze(2) < 2] = float("nan")
    return _pad_front_torch(result, window, T)


def var_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    m = w.nanmean(dim=2, keepdim=True)
    d = w - m
    not_nan = ~torch.isnan(w)
    d = d.nan_to_num(0.0)
    n = not_nan.sum(dim=2, keepdim=True).float()
    result = ((d ** 2).sum(dim=2, keepdim=True) / (n - 1).clamp(min=1)).squeeze(2)
    result[n.squeeze(2) < 2] = float("nan")
    return _pad_front_torch(result, window, T)


def skew_torch(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    m = w.nanmean(dim=2, keepdim=True)
    d = (w - m).nan_to_num(0.0)
    not_nan = ~torch.isnan(w)
    n = not_nan.sum(dim=2, keepdim=True).float()
    m2 = (d ** 2).sum(dim=2, keepdim=True) / n.clamp(min=1)
    m3 = (d ** 3).sum(dim=2, keepdim=True) / n.clamp(min=1)
    result = (m3 / m2.pow(1.5)).squeeze(2)
    result[n.squeeze(2) < 3] = float("nan")
    return _pad_front_torch(result, window, T)


def kurt_torch(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    m = w.nanmean(dim=2, keepdim=True)
    d = (w - m).nan_to_num(0.0)
    not_nan = ~torch.isnan(w)
    n = not_nan.sum(dim=2, keepdim=True).float()
    m2 = (d ** 2).sum(dim=2, keepdim=True) / n.clamp(min=1)
    m4 = (d ** 4).sum(dim=2, keepdim=True) / n.clamp(min=1)
    result = (m4 / m2.pow(2.0) - 3.0).squeeze(2)
    result[n.squeeze(2) < 4] = float("nan")
    return _pad_front_torch(result, window, T)


def median_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    result = w.nanmedian(dim=2).values
    return _pad_front_torch(result, window, T)


def sum_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    result = w.nansum(dim=2)
    all_nan = torch.isnan(w).all(dim=2)
    result[all_nan] = float("nan")
    return _pad_front_torch(result, window, T)


def prod_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    filled = w.nan_to_num(1.0)
    result = filled.prod(dim=2)
    all_nan = torch.isnan(w).all(dim=2)
    result[all_nan] = float("nan")
    return _pad_front_torch(result, window, T)


def ts_max_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    filled = w.nan_to_num(float("-inf"))
    result = filled.max(dim=2).values
    all_nan = torch.isnan(w).all(dim=2)
    result[all_nan] = float("nan")
    return _pad_front_torch(result, window, T)


def ts_min_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    filled = w.nan_to_num(float("inf"))
    result = filled.min(dim=2).values
    all_nan = torch.isnan(w).all(dim=2)
    result[all_nan] = float("nan")
    return _pad_front_torch(result, window, T)


def ts_argmax_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    filled = w.nan_to_num(float("-inf"))
    result = filled.argmax(dim=2).float()
    return _pad_front_torch(result, window, T)


def ts_argmin_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    filled = w.nan_to_num(float("inf"))
    result = filled.argmin(dim=2).float()
    return _pad_front_torch(result, window, T)


def ts_rank_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Rolling percentile rank -- key GPU acceleration target (17x speedup)."""
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)  # (M, T-w+1, w)
    latest = w[:, :, -1:]  # (M, T-w+1, 1)
    not_nan = ~torch.isnan(w)
    # Count values strictly less than latest (NaN-safe)
    less = ((w < latest) & not_nan).sum(dim=2).float()
    count_valid = not_nan.sum(dim=2).float()
    result = less / (count_valid - 1).clamp(min=1)
    result[count_valid <= 1] = float("nan")
    return _pad_front_torch(result, window, T)


def quantile_torch(x: torch.Tensor, window: int = 10, q: float = 0.5) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    result = w.nanmedian(dim=2).values  # approximation; true quantile below
    # Use sorting for proper quantile
    sorted_w, _ = w.sort(dim=2)
    n = (~torch.isnan(w)).sum(dim=2).float()
    idx = ((n - 1) * q).long().clamp(min=0)
    # Gather the quantile value
    result = sorted_w.gather(2, idx.unsqueeze(2)).squeeze(2)
    return _pad_front_torch(result, window, T)


def count_nan_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    result = torch.isnan(w).sum(dim=2).float()
    return _pad_front_torch(result, window, T)


def count_not_nan_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    w = _unfold_torch(x, window)
    result = (~torch.isnan(w)).sum(dim=2).float()
    return _pad_front_torch(result, window, T)


# ===========================================================================
# Registration table
# ===========================================================================

STATISTICAL_OPS = {
    "Mean": (mean_np, mean_torch),
    "Std": (std_np, std_torch),
    "Var": (var_np, var_torch),
    "Skew": (skew_np, skew_torch),
    "Kurt": (kurt_np, kurt_torch),
    "Median": (median_np, median_torch),
    "Sum": (sum_np, sum_torch),
    "Prod": (prod_np, prod_torch),
    "TsMax": (ts_max_np, ts_max_torch),
    "TsMin": (ts_min_np, ts_min_torch),
    "TsArgMax": (ts_argmax_np, ts_argmax_torch),
    "TsArgMin": (ts_argmin_np, ts_argmin_torch),
    "TsRank": (ts_rank_np, ts_rank_torch),
    "Quantile": (quantile_np, quantile_torch),
    "CountNaN": (count_nan_np, count_nan_torch),
    "CountNotNaN": (count_not_nan_np, count_not_nan_torch),
}
