"""Portfolio construction and quintile backtesting.

Implements quintile-sorted long-short portfolio backtesting with
transaction cost pressure testing, following the FactorMiner paper methodology.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorminer.evaluation.metrics import compute_icir, compute_pearson_ic, compute_rank_ic

_PORTFOLIO_BLOCK_SIZE = 128


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

        boundaries = np.linspace(0.0, 1.0, 6)
        quintile_returns = np.full((T, 5), np.nan, dtype=np.float64)
        for start in range(0, T, _PORTFOLIO_BLOCK_SIZE):
            stop = start + _PORTFOLIO_BLOCK_SIZE
            signal_block = combined_signal[start:stop]
            return_block = returns[start:stop]
            eligible = np.isfinite(signal_block)
            eligible_count = eligible.sum(axis=1)
            ranks = pd.DataFrame(np.where(eligible, signal_block, np.nan)).rank(
                axis=1, method="average", na_option="keep"
            )
            percentile = np.divide(
                ranks.to_numpy(dtype=np.float64, copy=False) - 1.0,
                np.maximum(eligible_count - 1, 1)[:, None],
            )
            usable_period = eligible_count >= 5
            for q in range(5):
                lower, upper = boundaries[q], boundaries[q + 1]
                if q == 4:
                    members = (percentile >= lower) & (percentile <= upper)
                else:
                    members = (percentile >= lower) & (percentile < upper)
                has_members = members.any(axis=1)
                finite_returns = np.all(
                    np.where(members, np.isfinite(return_block), True), axis=1
                )
                take = usable_period & has_members & finite_returns
                totals = np.where(members, return_block, 0.0).sum(axis=1)
                counts = members.sum(axis=1)
                means = np.divide(
                    totals,
                    counts,
                    out=np.zeros(signal_block.shape[0], dtype=np.float64),
                    where=counts > 0,
                )
                rows = start + np.flatnonzero(take)
                quintile_returns[rows, q] = means[take]

        # Turnover for cost adjustment
        turnover = self.compute_turnover(combined_signal, top_fraction=0.2)
        avg_turnover = _finite_mean_or_nan(turnover)

        # Long-short return (Q5 - Q1) with transaction costs
        ls_raw = quintile_returns[:, 4] - quintile_returns[:, 0]
        ls_cost = 2.0 * cost_frac * turnover  # both legs
        ls_net = np.where(
            np.isfinite(ls_raw),
            ls_raw - ls_cost,
            np.nan,
        )
        ls_cumulative = np.nancumsum(np.where(np.isfinite(ls_net), ls_net, 0.0))

        # Historical ``ic_*`` fields remain Spearman RankIC. Explicit
        # Pearson/Rank fields prevent ambiguity in new benchmark artifacts.
        rank_ic_series = compute_rank_ic(combined_signal.T, returns.T)
        pearson_ic_series = compute_pearson_ic(combined_signal.T, returns.T)
        ic_series = rank_ic_series

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

        finite_pearson_ic = pearson_ic_series[np.isfinite(pearson_ic_series)]
        pearson_ic_mean = float(np.mean(finite_pearson_ic)) if finite_pearson_ic.size else 0.0
        pearson_icir = compute_icir(pearson_ic_series)

        # Mean quintile returns
        q_means = [_finite_mean_or_nan(quintile_returns[:, q]) for q in range(5)]

        # Monotonicity: fraction of adjacent quintile pairs in correct order
        correct_pairs = sum(1 for i in range(4) if q_means[i] < q_means[i + 1])
        monotonicity = correct_pairs / 4.0

        return {
            "q1_return": q_means[0],
            "q2_return": q_means[1],
            "q3_return": q_means[2],
            "q4_return": q_means[3],
            "q5_return": q_means[4],
            "ls_return": _finite_mean_or_nan(ls_net),
            "ls_gross_return": _finite_mean_or_nan(ls_raw),
            "ls_cumulative": ls_cumulative,
            "ls_gross_series": ls_raw,
            "ls_net_series": ls_net,
            "quintile_period_returns": quintile_returns,
            "turnover_series": turnover,
            "ic_definition": "spearman_rank",
            "ic_series": ic_series,
            "ic_mean": ic_mean,
            "icir": icir,
            "ic_win_rate": ic_win_rate,
            "rank_ic_series": rank_ic_series,
            "rank_ic_mean": ic_mean,
            "rank_icir": icir,
            "pearson_ic_series": pearson_ic_series,
            "pearson_ic_mean": pearson_ic_mean,
            "pearson_icir": pearson_icir,
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
                combined_signal,
                returns,
                transaction_cost_bps=cost_bps,
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


def _finite_mean_or_nan(values: np.ndarray) -> float:
    """Return the finite-value mean, or NaN without an empty-slice warning."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


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
