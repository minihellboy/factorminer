"""Time-series operators along the T axis for each asset row.

Input shape: ``(M, T)`` -> output shape ``(M, T)``.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# ===========================================================================
# NumPy implementations
# ===========================================================================

def delta_np(x: np.ndarray, window: int = 1) -> np.ndarray:
    """x[t] - x[t - period]."""
    window = int(window)
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    if window < T:
        out[:, window:] = x[:, window:] - x[:, :-window]
    return out


def delay_np(x: np.ndarray, window: int = 1) -> np.ndarray:
    """x[t - period] (lag operator)."""
    window = int(window)
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    if window < T:
        out[:, window:] = x[:, :-window]
    return out


def return_np(x: np.ndarray, window: int = 1) -> np.ndarray:
    """x[t] / x[t-d] - 1."""
    window = int(window)
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    if window < T:
        prev = x[:, :-window]
        mask = np.abs(prev) > 1e-10
        out_slice = np.full_like(prev, np.nan)
        out_slice[mask] = x[:, window:][mask] / prev[mask] - 1.0
        out[:, window:] = out_slice
    return out


def log_return_np(x: np.ndarray, window: int = 1) -> np.ndarray:
    """log(x[t] / x[t-d])."""
    window = int(window)
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    if window < T:
        prev = x[:, :-window]
        curr = x[:, window:]
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(np.abs(prev) > 1e-10, curr / prev, np.nan)
            out[:, window:] = np.where(ratio > 0, np.log(ratio), np.nan)
    return out


def corr_np(x: np.ndarray, y: np.ndarray, window: int = 10) -> np.ndarray:
    """Rolling Pearson correlation."""
    window = int(window)
    M, T = x.shape
    if T < window:
        return np.full_like(x, np.nan)

    from factorminer.operators.statistical import _pad_front, _rolling_np

    wx = _rolling_np(x, window)
    wy = _rolling_np(y, window)
    if wx is None or wy is None:
        return np.full_like(x, np.nan)

    mx = np.nanmean(wx, axis=2, keepdims=True)
    my = np.nanmean(wy, axis=2, keepdims=True)
    dx = wx - mx
    dy = wy - my
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.nanmean(dx * dy, axis=2)
        sx = np.sqrt(np.nanmean(dx ** 2, axis=2))
        sy = np.sqrt(np.nanmean(dy ** 2, axis=2))
        result = np.where((sx > 1e-10) & (sy > 1e-10), cov / (sx * sy), np.nan)
    return _pad_front(result, window, T)


def cov_np(x: np.ndarray, y: np.ndarray, window: int = 10) -> np.ndarray:
    """Rolling covariance."""
    window = int(window)
    M, T = x.shape
    if T < window:
        return np.full_like(x, np.nan)

    from factorminer.operators.statistical import _pad_front, _rolling_np

    wx = _rolling_np(x, window)
    wy = _rolling_np(y, window)
    if wx is None or wy is None:
        return np.full_like(x, np.nan)

    mx = np.nanmean(wx, axis=2, keepdims=True)
    my = np.nanmean(wy, axis=2, keepdims=True)
    result = np.nanmean((wx - mx) * (wy - my), axis=2)
    return _pad_front(result, window, T)


def beta_np(x: np.ndarray, y: np.ndarray, window: int = 10) -> np.ndarray:
    """Rolling regression beta: slope of x regressed on y."""
    window = int(window)
    M, T = x.shape
    if T < window:
        return np.full_like(x, np.nan)

    from factorminer.operators.statistical import _pad_front, _rolling_np

    wx = _rolling_np(x, window)
    wy = _rolling_np(y, window)
    if wx is None or wy is None:
        return np.full_like(x, np.nan)

    my = np.nanmean(wy, axis=2, keepdims=True)
    mx = np.nanmean(wx, axis=2, keepdims=True)
    dy = wy - my
    dx = wx - mx
    with np.errstate(invalid="ignore", divide="ignore"):
        var_y = np.nanmean(dy ** 2, axis=2)
        cov_xy = np.nanmean(dx * dy, axis=2)
        result = np.where(var_y > 1e-10, cov_xy / var_y, np.nan)
    return _pad_front(result, window, T)


def resid_np(x: np.ndarray, y: np.ndarray, window: int = 10) -> np.ndarray:
    """Rolling regression residual: x - beta * y - alpha, evaluated at last point."""
    window = int(window)
    M, T = x.shape
    if T < window:
        return np.full_like(x, np.nan)

    from factorminer.operators.statistical import _pad_front, _rolling_np

    wx = _rolling_np(x, window)
    wy = _rolling_np(y, window)
    if wx is None or wy is None:
        return np.full_like(x, np.nan)

    mx = np.nanmean(wx, axis=2, keepdims=True)
    my = np.nanmean(wy, axis=2, keepdims=True)
    dx = wx - mx
    dy = wy - my
    with np.errstate(invalid="ignore", divide="ignore"):
        var_y = np.nanmean(dy ** 2, axis=2, keepdims=True)
        cov_xy = np.nanmean(dx * dy, axis=2, keepdims=True)
        b = np.where(var_y > 1e-10, cov_xy / var_y, 0.0)
        a = mx - b * my
    # Residual at last time step in each window
    result = (wx[:, :, -1:] - b * wy[:, :, -1:] - a).squeeze(2)
    return _pad_front(result, window, T)


def wma_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Linearly weighted moving average."""
    window = int(window)
    M, T = x.shape
    from factorminer.operators.statistical import _pad_front, _rolling_np

    w = _rolling_np(x, window)
    if w is None:
        return np.full_like(x, np.nan)
    weights = np.arange(1, window + 1, dtype=np.float64)
    weights = weights / weights.sum()
    result = np.nansum(w * weights[np.newaxis, np.newaxis, :], axis=2)
    return _pad_front(result, window, T)


def decay_np(x: np.ndarray, window: int = 10) -> np.ndarray:
    """Exponentially decaying sum (linearly decaying weighted average)."""
    return wma_np(x, window)


def cumsum_np(x: np.ndarray) -> np.ndarray:
    return np.nancumsum(x, axis=1)


def cumprod_np(x: np.ndarray) -> np.ndarray:
    filled = np.where(np.isnan(x), 1.0, x)
    return np.cumprod(filled, axis=1)


def cummax_np(x: np.ndarray) -> np.ndarray:
    out = np.copy(x)
    for t in range(1, x.shape[1]):
        out[:, t] = np.fmax(out[:, t - 1], x[:, t])
    return out


def cummin_np(x: np.ndarray) -> np.ndarray:
    out = np.copy(x)
    for t in range(1, x.shape[1]):
        out[:, t] = np.fmin(out[:, t - 1], x[:, t])
    return out


# ===========================================================================
# PyTorch implementations
# ===========================================================================

def delta_torch(x: torch.Tensor, window: int = 1) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    out = torch.full_like(x, float("nan"))
    if window < T:
        out[:, window:] = x[:, window:] - x[:, :-window]
    return out


def delay_torch(x: torch.Tensor, window: int = 1) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    out = torch.full_like(x, float("nan"))
    if window < T:
        out[:, window:] = x[:, :-window]
    return out


def return_torch(x: torch.Tensor, window: int = 1) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    out = torch.full_like(x, float("nan"))
    if window < T:
        prev = x[:, :-window]
        mask = prev.abs() > 1e-10
        r = torch.full_like(prev, float("nan"))
        r[mask] = x[:, window:][mask] / prev[mask] - 1.0
        out[:, window:] = r
    return out


def log_return_torch(x: torch.Tensor, window: int = 1) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    out = torch.full_like(x, float("nan"))
    if window < T:
        prev = x[:, :-window]
        curr = x[:, window:]
        mask = prev.abs() > 1e-10
        ratio = torch.full_like(prev, float("nan"))
        ratio[mask] = curr[mask] / prev[mask]
        lr = torch.full_like(prev, float("nan"))
        pos = ratio > 0
        lr[pos] = ratio[pos].log()
        out[:, window:] = lr
    return out


def corr_torch(x: torch.Tensor, y: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    from factorminer.operators.statistical import _pad_front_torch, _unfold_torch

    wx = _unfold_torch(x, window)
    wy = _unfold_torch(y, window)
    mx = wx.nanmean(dim=2, keepdim=True)
    my = wy.nanmean(dim=2, keepdim=True)
    dx = (wx - mx).nan_to_num(0.0)
    dy = (wy - my).nan_to_num(0.0)
    not_nan = ~(torch.isnan(wx) | torch.isnan(wy))
    n = not_nan.sum(dim=2).float()
    cov = (dx * dy * not_nan).sum(dim=2) / n.clamp(min=1)
    sx = ((dx ** 2 * not_nan).sum(dim=2) / n.clamp(min=1)).sqrt()
    sy = ((dy ** 2 * not_nan).sum(dim=2) / n.clamp(min=1)).sqrt()
    result = torch.where((sx > 1e-10) & (sy > 1e-10), cov / (sx * sy),
                         torch.tensor(float("nan"), device=x.device))
    return _pad_front_torch(result, window, T)


def cov_torch(x: torch.Tensor, y: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    from factorminer.operators.statistical import _pad_front_torch, _unfold_torch

    wx = _unfold_torch(x, window)
    wy = _unfold_torch(y, window)
    mx = wx.nanmean(dim=2, keepdim=True)
    my = wy.nanmean(dim=2, keepdim=True)
    dx = (wx - mx).nan_to_num(0.0)
    dy = (wy - my).nan_to_num(0.0)
    not_nan = ~(torch.isnan(wx) | torch.isnan(wy))
    n = not_nan.sum(dim=2).float()
    result = (dx * dy * not_nan).sum(dim=2) / n.clamp(min=1)
    return _pad_front_torch(result, window, T)


def beta_torch(x: torch.Tensor, y: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    from factorminer.operators.statistical import _pad_front_torch, _unfold_torch

    wx = _unfold_torch(x, window)
    wy = _unfold_torch(y, window)
    mx = wx.nanmean(dim=2, keepdim=True)
    my = wy.nanmean(dim=2, keepdim=True)
    dx = (wx - mx).nan_to_num(0.0)
    dy = (wy - my).nan_to_num(0.0)
    not_nan = ~(torch.isnan(wx) | torch.isnan(wy))
    n = not_nan.sum(dim=2).float()
    var_y = (dy ** 2 * not_nan).sum(dim=2) / n.clamp(min=1)
    cov_xy = (dx * dy * not_nan).sum(dim=2) / n.clamp(min=1)
    result = torch.where(var_y > 1e-10, cov_xy / var_y,
                         torch.tensor(float("nan"), device=x.device))
    return _pad_front_torch(result, window, T)


def resid_torch(x: torch.Tensor, y: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    from factorminer.operators.statistical import _pad_front_torch, _unfold_torch

    wx = _unfold_torch(x, window)
    wy = _unfold_torch(y, window)
    mx = wx.nanmean(dim=2, keepdim=True)
    my = wy.nanmean(dim=2, keepdim=True)
    dx = (wx - mx).nan_to_num(0.0)
    dy = (wy - my).nan_to_num(0.0)
    not_nan = ~(torch.isnan(wx) | torch.isnan(wy))
    n = not_nan.sum(dim=2, keepdim=True).float()
    var_y = (dy ** 2 * not_nan).sum(dim=2, keepdim=True) / n.clamp(min=1)
    cov_xy = (dx * dy * not_nan).sum(dim=2, keepdim=True) / n.clamp(min=1)
    b = torch.where(var_y > 1e-10, cov_xy / var_y, torch.zeros_like(var_y))
    a = mx - b * my
    result = (wx[:, :, -1:] - b * wy[:, :, -1:] - a).squeeze(2)
    return _pad_front_torch(result, window, T)


def wma_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    window = int(window)
    M, T = x.shape
    from factorminer.operators.statistical import _pad_front_torch, _unfold_torch

    w = _unfold_torch(x, window)
    weights = torch.arange(1, window + 1, dtype=x.dtype, device=x.device).float()
    weights = weights / weights.sum()
    filled = w.nan_to_num(0.0)
    result = (filled * weights.unsqueeze(0).unsqueeze(0)).sum(dim=2)
    return _pad_front_torch(result, window, T)


def decay_torch(x: torch.Tensor, window: int = 10) -> torch.Tensor:
    return wma_torch(x, window)


def cumsum_torch(x: torch.Tensor) -> torch.Tensor:
    return x.nan_to_num(0.0).cumsum(dim=1)


def cumprod_torch(x: torch.Tensor) -> torch.Tensor:
    return x.nan_to_num(1.0).cumprod(dim=1)


def cummax_torch(x: torch.Tensor) -> torch.Tensor:
    filled = x.nan_to_num(float("-inf"))
    return filled.cummax(dim=1).values


def cummin_torch(x: torch.Tensor) -> torch.Tensor:
    filled = x.nan_to_num(float("inf"))
    return filled.cummin(dim=1).values


# ===========================================================================
# Registration table
# ===========================================================================

TIMESERIES_OPS = {
    "Delta": (delta_np, delta_torch),
    "Delay": (delay_np, delay_torch),
    "Return": (return_np, return_torch),
    "LogReturn": (log_return_np, log_return_torch),
    "Corr": (corr_np, corr_torch),
    "Cov": (cov_np, cov_torch),
    "Beta": (beta_np, beta_torch),
    "Resid": (resid_np, resid_torch),
    "WMA": (wma_np, wma_torch),
    "Decay": (decay_np, decay_torch),
    "CumSum": (cumsum_np, cumsum_torch),
    "CumProd": (cumprod_np, cumprod_torch),
    "CumMax": (cummax_np, cummax_torch),
    "CumMin": (cummin_np, cummin_torch),
}
