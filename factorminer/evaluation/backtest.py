"""Full backtesting utilities for factor evaluation.

Provides time-series splitting, rolling and cumulative IC computation,
factor return attribution, and drawdown analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

# ------------------------------------------------------------------
# Time-series splitting
# ------------------------------------------------------------------

@dataclass
class SplitWindow:
    """Indices for a single train/test split."""
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def train_test_split(
    T: int,
    train_ratio: float = 0.7,
) -> SplitWindow:
    """Simple contiguous train/test split.

    Parameters
    ----------
    T : int
        Total number of time steps.
    train_ratio : float
        Fraction of data used for training (default 70%).

    Returns
    -------
    SplitWindow
    """
    split = int(T * train_ratio)
    return SplitWindow(
        train_start=0,
        train_end=split,
        test_start=split,
        test_end=T,
    )


def rolling_splits(
    T: int,
    train_window: int,
    test_window: int,
    step: int = 1,
) -> list[SplitWindow]:
    """Generate rolling-window train/test splits.

    Parameters
    ----------
    T : int
        Total number of time steps.
    train_window : int
        Size of training window.
    test_window : int
        Size of testing window.
    step : int
        Step size between consecutive windows.

    Returns
    -------
    list of SplitWindow
    """
    splits: list[SplitWindow] = []
    start = 0
    while start + train_window + test_window <= T:
        splits.append(SplitWindow(
            train_start=start,
            train_end=start + train_window,
            test_start=start + train_window,
            test_end=start + train_window + test_window,
        ))
        start += step
    return splits


# ------------------------------------------------------------------
# IC computation
# ------------------------------------------------------------------

def compute_ic_series(
    signal: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """Compute cross-sectional Spearman IC at each time step.

    Parameters
    ----------
    signal : ndarray of shape (T, N)
        Factor signal values.
    returns : ndarray of shape (T, N)
        Forward returns.

    Returns
    -------
    ndarray of shape (T,)
        IC values; NaN where computation is not possible.
    """
    T = signal.shape[0]
    ics = np.full(T, np.nan)
    for t in range(T):
        x = signal[t]
        y = returns[t]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 5:
            continue
        corr, _ = spearmanr(x[valid], y[valid])
        if np.isfinite(corr):
            ics[t] = corr
    return ics


def compute_rolling_ic(
    signal: np.ndarray,
    returns: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Compute rolling-window average IC.

    Parameters
    ----------
    signal : ndarray of shape (T, N)
    returns : ndarray of shape (T, N)
    window : int
        Rolling window size.

    Returns
    -------
    ndarray of shape (T,)
        Rolling mean IC; NaN where window is insufficient.
    """
    ic_series = compute_ic_series(signal, returns)
    T = len(ic_series)
    rolling_ic = np.full(T, np.nan)
    for t in range(window - 1, T):
        window_ics = ic_series[t - window + 1: t + 1]
        finite = window_ics[np.isfinite(window_ics)]
        if len(finite) >= 1:
            rolling_ic[t] = float(np.mean(finite))
    return rolling_ic


def compute_cumulative_ic(
    signal: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """Compute cumulative (expanding-window) mean IC.

    Parameters
    ----------
    signal : ndarray of shape (T, N)
    returns : ndarray of shape (T, N)

    Returns
    -------
    ndarray of shape (T,)
        Expanding-window mean IC.
    """
    ic_series = compute_ic_series(signal, returns)
    T = len(ic_series)
    cumulative = np.full(T, np.nan)
    running_sum = 0.0
    running_count = 0
    for t in range(T):
        if np.isfinite(ic_series[t]):
            running_sum += ic_series[t]
            running_count += 1
        if running_count > 0:
            cumulative[t] = running_sum / running_count
    return cumulative


def compute_ic_stats(ic_series: np.ndarray) -> dict:
    """Compute summary statistics for an IC series.

    Parameters
    ----------
    ic_series : ndarray of shape (T,)

    Returns
    -------
    dict with keys: ic_mean, ic_std, icir, ic_win_rate, ic_max, ic_min.
    """
    finite = ic_series[np.isfinite(ic_series)]
    if len(finite) < 2:
        return {
            "ic_mean": 0.0,
            "ic_std": 0.0,
            "icir": 0.0,
            "ic_win_rate": 0.0,
            "ic_max": 0.0,
            "ic_min": 0.0,
        }
    ic_mean = float(np.mean(finite))
    ic_std = float(np.std(finite, ddof=1))
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": ic_mean / ic_std if ic_std > 1e-12 else 0.0,
        "ic_win_rate": float(np.mean(finite > 0)),
        "ic_max": float(np.max(finite)),
        "ic_min": float(np.min(finite)),
    }


# ------------------------------------------------------------------
# Factor return attribution
# ------------------------------------------------------------------

def factor_return_attribution(
    factor_signals: dict[int, np.ndarray],
    returns: np.ndarray,
) -> dict[int, dict]:
    """Attribute portfolio returns to individual factors.

    For each factor, computes the IC series, ICIR, and the mean return of
    the top-quintile (Q5) minus bottom-quintile (Q1) long-short portfolio.

    Parameters
    ----------
    factor_signals : dict[int, ndarray]
        Mapping from factor ID to (T, N) signal array.
    returns : ndarray of shape (T, N)

    Returns
    -------
    dict mapping factor_id -> attribution dict with keys:
        ic_mean, icir, ic_win_rate, ls_return
    """
    results: dict[int, dict] = {}
    for fid, signal in factor_signals.items():
        ic_series = compute_ic_series(signal, returns)
        stats = compute_ic_stats(ic_series)

        # Compute long-short return
        T, N = signal.shape
        ls_returns = np.full(T, np.nan)
        for t in range(T):
            sig_t = signal[t]
            ret_t = returns[t]
            valid = np.isfinite(sig_t) & np.isfinite(ret_t)
            n_valid = valid.sum()
            if n_valid < 5:
                continue
            valid_sigs = sig_t[valid]
            valid_rets = ret_t[valid]
            k = max(1, n_valid // 5)
            sorted_idx = np.argsort(valid_sigs)
            q1_ret = np.mean(valid_rets[sorted_idx[:k]])
            q5_ret = np.mean(valid_rets[sorted_idx[-k:]])
            ls_returns[t] = q5_ret - q1_ret

        stats["ls_return"] = float(np.nanmean(ls_returns))
        results[fid] = stats
    return results


# ------------------------------------------------------------------
# Drawdown analysis
# ------------------------------------------------------------------

@dataclass
class DrawdownResult:
    """Results of drawdown analysis."""
    max_drawdown: float
    max_drawdown_start: int
    max_drawdown_end: int
    drawdown_series: np.ndarray
    recovery_periods: list[tuple[int, int, int]]  # (start, trough, end)


def compute_drawdown(cumulative_returns: np.ndarray) -> DrawdownResult:
    """Compute drawdown statistics from a cumulative return series.

    Parameters
    ----------
    cumulative_returns : ndarray of shape (T,)
        Cumulative returns (can be from cumsum of period returns).

    Returns
    -------
    DrawdownResult
    """
    cumulative_returns = np.asarray(cumulative_returns, dtype=np.float64)
    T = len(cumulative_returns)

    # Running maximum
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown_series = cumulative_returns - running_max

    # Max drawdown
    max_dd_idx = np.argmin(drawdown_series)
    max_dd = float(drawdown_series[max_dd_idx])
    # Find the peak before the max drawdown
    peak_idx = int(np.argmax(cumulative_returns[:max_dd_idx + 1]))

    # Identify recovery periods (peak -> trough -> recovery)
    recovery_periods: list[tuple[int, int, int]] = []
    i = 0
    while i < T:
        # Find start of drawdown (where dd becomes negative)
        if drawdown_series[i] < -1e-12:
            start = i - 1 if i > 0 else 0
            # Find trough
            j = i
            trough = i
            while j < T and drawdown_series[j] < -1e-12:
                if drawdown_series[j] < drawdown_series[trough]:
                    trough = j
                j += 1
            end = j if j < T else T - 1
            recovery_periods.append((start, trough, end))
            i = j
        else:
            i += 1

    return DrawdownResult(
        max_drawdown=max_dd,
        max_drawdown_start=peak_idx,
        max_drawdown_end=max_dd_idx,
        drawdown_series=drawdown_series,
        recovery_periods=recovery_periods,
    )


def compute_sharpe_ratio(
    returns_series: np.ndarray,
    annualization_factor: float = 252.0,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute annualized Sharpe ratio.

    Parameters
    ----------
    returns_series : ndarray of shape (T,)
        Period returns.
    annualization_factor : float
        Number of periods per year (252 for daily).
    risk_free_rate : float
        Annualized risk-free rate.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    finite = returns_series[np.isfinite(returns_series)]
    if len(finite) < 2:
        return 0.0
    rf_period = risk_free_rate / annualization_factor
    excess = finite - rf_period
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)
    if std_excess < 1e-12:
        return 0.0
    return float(mean_excess / std_excess * np.sqrt(annualization_factor))


def compute_calmar_ratio(
    returns_series: np.ndarray,
    annualization_factor: float = 252.0,
) -> float:
    """Compute Calmar ratio (annualized return / max drawdown).

    Parameters
    ----------
    returns_series : ndarray of shape (T,)
    annualization_factor : float

    Returns
    -------
    float
        Calmar ratio; 0 if max drawdown is zero.
    """
    finite = returns_series[np.isfinite(returns_series)]
    if len(finite) < 2:
        return 0.0
    cumulative = np.cumsum(finite)
    dd = compute_drawdown(cumulative)
    if abs(dd.max_drawdown) < 1e-12:
        return 0.0
    annualized_return = float(np.mean(finite)) * annualization_factor
    return annualized_return / abs(dd.max_drawdown)
