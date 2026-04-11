"""Portfolio construction and quintile backtesting.

Implements quintile-sorted long-short portfolio backtesting with
transaction cost pressure testing, following the FactorMiner paper methodology.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


class PortfolioBacktester:
    """Backtest factor signals using quintile portfolios."""

    # ------------------------------------------------------------------
    # Main backtest
    # ------------------------------------------------------------------

    def quintile_backtest(
        self,
        combined_signal: np.ndarray,
        returns: np.ndarray,
        transaction_cost_bps: float = 0,
    ) -> dict:
        """Run quintile portfolio backtest.

        At each time step t, sort assets into 5 quintiles by signal strength.
        Q5 = highest signal (long), Q1 = lowest signal (short).

        Parameters
        ----------
        combined_signal : ndarray of shape (T, N)
            Composite factor signal.
        returns : ndarray of shape (T, N)
            Forward returns aligned with the signal.
        transaction_cost_bps : float
            One-way transaction cost in basis points (1 bp = 0.01%).

        Returns
        -------
        dict with keys:
            q1_return .. q5_return : float
                Average annualized return per quintile.
            ls_return : float
                Average long-short return (Q5 - Q1).
            ls_cumulative : ndarray
                Cumulative long-short return series.
            ic_mean : float
            icir : float
            ic_win_rate : float
                Fraction of periods with IC > 0.
            monotonicity : float
                1.0 if perfect Q1 < Q2 < ... < Q5 ordering of mean returns.
            avg_turnover : float
                Mean daily turnover of the long quintile.
        """
        combined_signal = np.asarray(combined_signal, dtype=np.float64)
        returns = np.asarray(returns, dtype=np.float64)
        T, N = combined_signal.shape
        cost_frac = transaction_cost_bps / 10000.0

        # Per-period quintile returns
        quintile_returns = np.full((T, 5), np.nan)
        for t in range(T):
            sig_t = combined_signal[t]
            ret_t = returns[t]
            valid = np.isfinite(sig_t) & np.isfinite(ret_t)
            n_valid = valid.sum()
            if n_valid < 5:
                continue
            ranks = _rank_array(sig_t[valid])
            boundaries = np.linspace(0, 1, 6)
            for q in range(5):
                mask = (ranks >= boundaries[q]) & (ranks < boundaries[q + 1])
                if q == 4:
                    mask = (ranks >= boundaries[q]) & (ranks <= boundaries[q + 1])
                if mask.sum() > 0:
                    quintile_returns[t, q] = np.mean(ret_t[valid][mask])

        # Turnover for cost adjustment
        turnover = self.compute_turnover(combined_signal, top_fraction=0.2)
        avg_turnover = float(np.nanmean(turnover))

        # Long-short return (Q5 - Q1) with transaction costs
        ls_raw = quintile_returns[:, 4] - quintile_returns[:, 0]
        ls_cost = 2.0 * cost_frac * turnover  # both legs
        ls_net = np.where(
            np.isfinite(ls_raw),
            ls_raw - ls_cost,
            np.nan,
        )
        ls_cumulative = np.nancumsum(np.where(np.isfinite(ls_net), ls_net, 0.0))

        # IC series (cross-sectional Spearman rank correlation)
        ic_series = np.full(T, np.nan)
        for t in range(T):
            sig_t = combined_signal[t]
            ret_t = returns[t]
            valid = np.isfinite(sig_t) & np.isfinite(ret_t)
            if valid.sum() < 5:
                continue
            corr, _ = spearmanr(sig_t[valid], ret_t[valid])
            if np.isfinite(corr):
                ic_series[t] = corr

        finite_ic = ic_series[np.isfinite(ic_series)]
        if len(finite_ic) > 1:
            ic_mean = float(np.mean(finite_ic))
            ic_std = float(np.std(finite_ic, ddof=1))
            icir = ic_mean / ic_std if ic_std > 1e-12 else 0.0
            ic_win_rate = float(np.mean(finite_ic > 0))
        else:
            ic_mean = 0.0
            icir = 0.0
            ic_win_rate = 0.0

        # Mean quintile returns
        q_means = [float(np.nanmean(quintile_returns[:, q])) for q in range(5)]

        # Monotonicity: fraction of adjacent quintile pairs in correct order
        correct_pairs = sum(
            1 for i in range(4) if q_means[i] < q_means[i + 1]
        )
        monotonicity = correct_pairs / 4.0

        return {
            "q1_return": q_means[0],
            "q2_return": q_means[1],
            "q3_return": q_means[2],
            "q4_return": q_means[3],
            "q5_return": q_means[4],
            "ls_return": float(np.nanmean(ls_net)),
            "ls_gross_return": float(np.nanmean(ls_raw)),
            "ls_cumulative": ls_cumulative,
            "ls_gross_series": ls_raw,
            "ls_net_series": ls_net,
            "quintile_period_returns": quintile_returns,
            "turnover_series": turnover,
            "ic_series": ic_series,
            "ic_mean": ic_mean,
            "icir": icir,
            "ic_win_rate": ic_win_rate,
            "monotonicity": monotonicity,
            "avg_turnover": avg_turnover,
        }

    # ------------------------------------------------------------------
    # Cost pressure testing
    # ------------------------------------------------------------------

    def cost_pressure_test(
        self,
        combined_signal: np.ndarray,
        returns: np.ndarray,
        cost_settings: list[float] | None = None,
    ) -> dict[float, dict]:
        """Run backtest under multiple transaction cost settings (in bps).

        Paper Figure 9: Test at 1, 4, 7, 10, 11 bps.

        Parameters
        ----------
        combined_signal : ndarray of shape (T, N)
        returns : ndarray of shape (T, N)
        cost_settings : list of float or None
            Transaction cost levels in basis points.
            Defaults to [1, 4, 7, 10, 11].

        Returns
        -------
        dict mapping cost_bps -> backtest result dict.
        """
        if cost_settings is None:
            cost_settings = [1.0, 4.0, 7.0, 10.0, 11.0]

        results: dict[float, dict] = {}
        for cost_bps in cost_settings:
            results[cost_bps] = self.quintile_backtest(
                combined_signal, returns, transaction_cost_bps=cost_bps,
            )
        return results

    # ------------------------------------------------------------------
    # Turnover computation
    # ------------------------------------------------------------------

    def compute_turnover(
        self,
        signal: np.ndarray,
        top_fraction: float = 0.2,
    ) -> np.ndarray:
        """Compute daily turnover of the top/bottom quintile portfolios.

        Turnover is defined as the fraction of assets that change between
        consecutive rebalance periods in the top-quintile portfolio.

        Parameters
        ----------
        signal : ndarray of shape (T, N)
        top_fraction : float
            Fraction of assets in each quintile (default 0.2 = top 20%).

        Returns
        -------
        ndarray of shape (T,)
            Per-period turnover ratios.  First period is 0.
        """
        signal = np.asarray(signal, dtype=np.float64)
        T, N = signal.shape
        turnover = np.zeros(T)
        prev_top: np.ndarray | None = None

        for t in range(T):
            sig_t = signal[t]
            valid = np.isfinite(sig_t)
            n_valid = valid.sum()
            if n_valid < 5:
                prev_top = None
                continue

            k = max(1, int(n_valid * top_fraction))
            valid_idx = np.where(valid)[0]
            valid_vals = sig_t[valid_idx]
            # Indices of top-k assets
            top_idx = valid_idx[np.argpartition(valid_vals, -k)[-k:]]
            top_set = np.zeros(N, dtype=bool)
            top_set[top_idx] = True

            if prev_top is not None:
                changed = np.sum(top_set != prev_top)
                turnover[t] = changed / (2.0 * k)  # normalize by portfolio size
            prev_top = top_set

        return turnover


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _rank_array(x: np.ndarray) -> np.ndarray:
    """Compute percentile ranks in [0, 1] for a 1-D array.

    Ties receive the average rank.
    """
    n = len(x)
    if n == 0:
        return x.copy()
    order = x.argsort()
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # Handle ties by averaging
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks / max(n - 1, 1)
