"""Rolling linear-regression operators.

Each function regresses x against a simple time index [0, 1, ..., window-1]
within a rolling window along axis=1.  Input/output shape: ``(M, T)``.
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

def _linreg_components_np(x: np.ndarray, window: int):
    """Compute slope, intercept, and fitted values for rolling OLS vs time index."""
    window = int(window)
    M, T = x.shape

    from factorminer.operators.statistical import _pad_front, _rolling_np

    w = _rolling_np(x, window)
    if w is None:
        nan = np.full_like(x, np.nan)
        return nan, nan, nan, nan

    t_idx = np.arange(window, dtype=np.float64)  # (window,)
    t_mean = t_idx.mean()
    t_var = ((t_idx - t_mean) ** 2).sum()

    x_mean = np.nanmean(w, axis=2, keepdims=True)  # (M, T-w+1, 1)
    # covariance of x with t_idx
    cov_xt = np.nansum((w - x_mean) * (t_idx - t_mean), axis=2)  # (M, T-w+1)

    slope = cov_xt / t_var  # (M, T-w+1)
    intercept = x_mean.squeeze(2) - slope * t_mean

    # Fitted value at the last time step in window (t = window - 1)
    fitted = slope * (window - 1) + intercept

    # Residual at last time step
    residual = w[:, :, -1] - fitted

    # R-squared
    ss_res_all = w - (slope[:, :, np.newaxis] * t_idx + intercept[:, :, np.newaxis])
    ss_res = np.nansum(ss_res_all ** 2, axis=2)
    ss_tot = np.nansum((w - x_mean) ** 2, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, np.nan)

    slope = _pad_front(slope, window, T)
    intercept = _pad_front(intercept, window, T)
    fitted = _pad_front(fitted, window, T)
    residual = _pad_front(residual, window, T)
    r2 = _pad_front(r2, window, T)

    return slope, intercept, fitted, residual, r2


def ts_linreg_np(x: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling linear-regression fitted value."""
    _, _, fitted, _, _ = _linreg_components_np(x, window)
    return fitted


def ts_linreg_slope_np(x: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling linear-regression slope."""
    slope, _, _, _, _ = _linreg_components_np(x, window)
    return slope


def ts_linreg_intercept_np(x: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling linear-regression intercept."""
    _, intercept, _, _, _ = _linreg_components_np(x, window)
    return intercept


def ts_linreg_resid_np(x: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling linear-regression residual at the last time step."""
    _, _, _, residual, _ = _linreg_components_np(x, window)
    return residual


def rsquare_np(x: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling linear-regression R-squared against the time index."""
    _, _, _, _, r2 = _linreg_components_np(x, window)
    return r2


# ===========================================================================
# PyTorch implementations
# ===========================================================================

def _linreg_components_torch(x: torch.Tensor, window: int):
    """Vectorized rolling OLS on GPU."""
    window = int(window)
    M, T = x.shape

    from factorminer.operators.statistical import _pad_front_torch, _unfold_torch

    w = _unfold_torch(x, window)  # (M, T-w+1, window)

    t_idx = torch.arange(window, dtype=x.dtype, device=x.device)
    t_mean = t_idx.mean()
    t_var = ((t_idx - t_mean) ** 2).sum()

    x_mean = w.nanmean(dim=2, keepdim=True)
    # Handle NaN: replace with 0 for summation
    w_filled = w.nan_to_num(0.0)
    not_nan = ~torch.isnan(w)
    not_nan.sum(dim=2, keepdim=True).float()

    # Recompute mean with nan handling
    cov_xt = ((w_filled - x_mean.nan_to_num(0.0)) * (t_idx - t_mean) * not_nan).sum(dim=2)

    slope = cov_xt / t_var
    intercept = x_mean.squeeze(2) - slope * t_mean

    fitted = slope * (window - 1) + intercept
    residual = w[:, :, -1] - fitted

    # R-squared
    fitted_all = slope.unsqueeze(2) * t_idx + intercept.unsqueeze(2)
    ss_res = ((w_filled - fitted_all) ** 2 * not_nan).sum(dim=2)
    ss_tot = ((w_filled - x_mean.nan_to_num(0.0)) ** 2 * not_nan).sum(dim=2)
    r2 = torch.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot,
                     torch.tensor(float("nan"), device=x.device))

    slope = _pad_front_torch(slope, window, T)
    intercept = _pad_front_torch(intercept, window, T)
    fitted = _pad_front_torch(fitted, window, T)
    residual = _pad_front_torch(residual, window, T)
    r2 = _pad_front_torch(r2, window, T)

    return slope, intercept, fitted, residual, r2


def ts_linreg_torch(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    _, _, fitted, _, _ = _linreg_components_torch(x, window)
    return fitted


def ts_linreg_slope_torch(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    slope, _, _, _, _ = _linreg_components_torch(x, window)
    return slope


def ts_linreg_intercept_torch(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    _, intercept, _, _, _ = _linreg_components_torch(x, window)
    return intercept


def ts_linreg_resid_torch(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    _, _, _, residual, _ = _linreg_components_torch(x, window)
    return residual


def rsquare_torch(x: torch.Tensor, window: int = 20) -> torch.Tensor:
    _, _, _, _, r2 = _linreg_components_torch(x, window)
    return r2


# ===========================================================================
# Registration table
# ===========================================================================

REGRESSION_OPS = {
    "TsLinReg": (ts_linreg_np, ts_linreg_torch),
    "TsLinRegSlope": (ts_linreg_slope_np, ts_linreg_slope_torch),
    "TsLinRegIntercept": (ts_linreg_intercept_np, ts_linreg_intercept_torch),
    "TsLinRegResid": (ts_linreg_resid_np, ts_linreg_resid_torch),
    "Slope": (ts_linreg_slope_np, ts_linreg_slope_torch),
    "Rsquare": (rsquare_np, rsquare_torch),
    "Resi": (ts_linreg_resid_np, ts_linreg_resid_torch),
}
