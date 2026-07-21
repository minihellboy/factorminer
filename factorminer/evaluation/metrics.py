"""Core evaluation metrics for alpha factors.

Provides vectorized, production-quality implementations of Information
Coefficient (IC), ICIR, quintile analysis, turnover, and comprehensive
factor statistics used by the validation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------

_EVALUATION_BLOCK_SIZE = 128


def _validate_panel_pair(signals: np.ndarray, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two aligned floating-point ``(assets, periods)`` panels."""
    signal_panel = np.asarray(signals, dtype=np.float64)
    return_panel = np.asarray(returns, dtype=np.float64)
    if signal_panel.ndim != 2 or return_panel.ndim != 2:
        raise ValueError("signals and returns must both be 2-D (assets, periods) panels")
    if signal_panel.shape != return_panel.shape:
        raise ValueError(
            f"signals and returns must have identical shapes; got "
            f"{signal_panel.shape} and {return_panel.shape}"
        )
    return signal_panel, return_panel


def _column_average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks per column while retaining missing entries."""
    return pd.DataFrame(values).rank(method="average", na_option="keep").to_numpy(
        dtype=np.float64, copy=False
    )


def _compute_cross_sectional_correlation(
    signals: np.ndarray,
    returns: np.ndarray,
    *,
    rank: bool,
    min_assets: int = 5,
) -> np.ndarray:
    """Compute a Pearson correlation, optionally after ranking each cross-section."""
    signal_panel, return_panel = _validate_panel_pair(signals, returns)
    series = np.full(signal_panel.shape[1], np.nan, dtype=np.float64)
    for start in range(0, signal_panel.shape[1], _EVALUATION_BLOCK_SIZE):
        stop = start + _EVALUATION_BLOCK_SIZE
        left_block = signal_panel[:, start:stop]
        right_block = return_panel[:, start:stop]
        valid = np.isfinite(left_block) & np.isfinite(right_block)
        observations = valid.sum(axis=0)

        left = np.where(valid, left_block, np.nan)
        right = np.where(valid, right_block, np.nan)
        if rank:
            left = _column_average_ranks(left)
            right = _column_average_ranks(right)

        left_values = np.where(valid, left, 0.0)
        right_values = np.where(valid, right, 0.0)
        left_mean = np.divide(
            left_values.sum(axis=0),
            observations,
            out=np.zeros_like(observations, dtype=np.float64),
            where=observations > 0,
        )
        right_mean = np.divide(
            right_values.sum(axis=0),
            observations,
            out=np.zeros_like(observations, dtype=np.float64),
            where=observations > 0,
        )
        left_centered = np.where(valid, left - left_mean, 0.0)
        right_centered = np.where(valid, right - right_mean, 0.0)
        numerator = np.sum(left_centered * right_centered, axis=0)
        denominator = np.sqrt(
            np.sum(left_centered**2, axis=0) * np.sum(right_centered**2, axis=0)
        )
        correlation = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-12,
        )
        usable = observations >= min_assets
        series_block = series[start:stop]
        series_block[usable] = correlation[usable]
    return series


def compute_pearson_ic(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Compute cross-sectional Pearson IC for each period.

    This is the conventional *linear* Information Coefficient. It is kept
    separate from :func:`compute_rank_ic` because the two answer different
    questions and are reported separately by Qlib/AlphaBench-style tooling.
    """
    return _compute_cross_sectional_correlation(signals, returns, rank=False)


def compute_rank_ic(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Compute cross-sectional Spearman RankIC for each period."""
    return _compute_cross_sectional_correlation(signals, returns, rank=True)


def compute_ic(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Compute the historical FactorMiner IC field (Spearman RankIC).

    ``compute_ic`` is retained as a compatibility alias. FactorMiner has
    historically used Spearman correlation for fields named ``ic_*``. New
    code should call :func:`compute_rank_ic` or :func:`compute_pearson_ic`
    explicitly and label the result accordingly.

    Parameters
    ----------
    signals : np.ndarray, shape (M, T)
        Factor signals for M assets over T periods.
    returns : np.ndarray, shape (M, T)
        Forward returns for M assets over T periods.

    Returns
    -------
    np.ndarray, shape (T,)
        Spearman rank correlation per period. NaN where fewer than 5
        valid (non-NaN) asset pairs exist.
    """
    return compute_rank_ic(signals, returns)


def compute_ic_vectorized(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Compatibility implementation of FactorMiner's Spearman RankIC.

    Ranks are computed per-column, then Pearson correlation on ranks is
    computed with the same numerical contract as :func:`compute_rank_ic`.

    Parameters
    ----------
    signals : np.ndarray, shape (M, T)
    returns : np.ndarray, shape (M, T)

    Returns
    -------
    np.ndarray, shape (T,)
    """
    return compute_rank_ic(signals, returns)


# ---------------------------------------------------------------------------
# IC-derived statistics
# ---------------------------------------------------------------------------

METRIC_VERSION = "paper_ic_v2"


def compute_icir(ic_series: np.ndarray) -> float:
    """Compute ICIR = mean(IC) / std(IC).

    Parameters
    ----------
    ic_series : np.ndarray
        IC time series (may contain NaN).

    Returns
    -------
    float
        ICIR value.  Returns 0.0 if std is near zero or too few valid points.
    """
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) < 3:
        return 0.0
    std = float(np.std(valid, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(valid)) / std


def compute_ic_mean(ic_series: np.ndarray) -> float:
    """Compute signed mean IC, ``mean(IC_t)``.

    Parameters
    ----------
    ic_series : np.ndarray

    Returns
    -------
    float
    """
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid))


def compute_ic_paper_mean(ic_series: np.ndarray) -> float:
    """Compute the paper IC summary, ``abs(mean(IC_t))``."""
    return abs(compute_ic_mean(ic_series))


def compute_ic_abs_mean(ic_series: np.ndarray) -> float:
    """Compute legacy diagnostic IC, ``mean(abs(IC_t))``."""
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(np.abs(valid)))


def compute_ic_paper_icir(ic_series: np.ndarray) -> float:
    """Compute the paper ICIR summary, ``abs(mean(IC_t)) / std(IC_t)``."""
    return abs(compute_icir(ic_series))


def compute_ic_win_rate(ic_series: np.ndarray) -> float:
    """Fraction of periods with positive IC.

    Parameters
    ----------
    ic_series : np.ndarray

    Returns
    -------
    float
        Win rate in [0, 1].
    """
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid > 0))


# ---------------------------------------------------------------------------
# Cross-factor correlation
# ---------------------------------------------------------------------------


def compute_pairwise_correlation(
    signals_a: np.ndarray,
    signals_b: np.ndarray,
) -> float:
    """Time-averaged cross-sectional Spearman correlation between two factors.

    rho(a, b) = (1/|T|) * sum_t Corr_rank(s_t^a, s_t^b)

    Parameters
    ----------
    signals_a : np.ndarray, shape (M, T)
    signals_b : np.ndarray, shape (M, T)

    Returns
    -------
    float
        Average cross-sectional Spearman correlation.
    """
    correlations = compute_rank_ic(signals_a, signals_b)
    valid = correlations[np.isfinite(correlations)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


# ---------------------------------------------------------------------------
# Quintile analysis
# ---------------------------------------------------------------------------


def compute_quintile_returns(
    signals: np.ndarray,
    returns: np.ndarray,
    n_quantiles: int = 5,
) -> dict:
    """Sort assets into quintiles by factor signal, compute average returns.

    Parameters
    ----------
    signals : np.ndarray, shape (M, T)
    returns : np.ndarray, shape (M, T)
    n_quantiles : int
        Number of quantile buckets (default 5 for quintiles).

    Returns
    -------
    dict
        Keys: Q1..Q{n}, long_short, monotonicity.
        Q1 is lowest signal quintile, Q{n} is highest.
    """
    signals, returns = _validate_panel_pair(signals, returns)
    _, period_count = signals.shape
    n_quantiles = int(n_quantiles)
    result: dict = {}
    return_sums = np.zeros(n_quantiles, dtype=np.float64)
    return_counts = np.zeros(n_quantiles, dtype=np.int64)
    for start in range(0, period_count, _EVALUATION_BLOCK_SIZE):
        stop = start + _EVALUATION_BLOCK_SIZE
        signal_block = signals[:, start:stop]
        return_block = returns[:, start:stop]
        eligible = np.isfinite(signal_block)
        eligible_count = eligible.sum(axis=0)
        ranks = _column_average_ranks(np.where(eligible, signal_block, np.nan))
        quantile_labels = np.ceil(
            np.divide(
                ranks,
                eligible_count,
                out=np.full_like(ranks, np.nan),
                where=eligible_count > 0,
            )
            * n_quantiles
        )
        quantile_labels = np.clip(quantile_labels, 1, n_quantiles)
        usable_period = eligible_count >= n_quantiles

        for q in range(1, n_quantiles + 1):
            members = quantile_labels == q
            has_members = members.any(axis=0)
            finite_returns = np.all(
                np.where(members, np.isfinite(return_block), True), axis=0
            )
            take = usable_period & has_members & finite_returns
            totals = np.where(members, return_block, 0.0).sum(axis=0)
            counts = members.sum(axis=0)
            period_returns = np.divide(
                totals,
                counts,
                out=np.zeros(signal_block.shape[1], dtype=np.float64),
                where=counts > 0,
            )
            return_sums[q - 1] += period_returns[take].sum()
            return_counts[q - 1] += np.count_nonzero(take)

    means: dict[int, float] = {}
    for q in range(1, n_quantiles + 1):
        means[q] = (
            float(return_sums[q - 1] / return_counts[q - 1])
            if return_counts[q - 1]
            else 0.0
        )
        result[f"Q{q}"] = means[q]

    # Long-short: top quintile minus bottom quintile
    result["long_short"] = means[n_quantiles] - means[1]

    # Monotonicity: Spearman corr between quintile index and mean return
    q_indices = np.arange(1, n_quantiles + 1, dtype=np.float64)
    q_returns = np.array([means[q] for q in range(1, n_quantiles + 1)])
    if np.std(q_returns) < 1e-12:
        result["monotonicity"] = 0.0
    else:
        rq = rankdata(q_indices)
        rr = rankdata(q_returns)
        rq_m = rq - rq.mean()
        rr_m = rr - rr.mean()
        denom = np.sqrt((rq_m**2).sum() * (rr_m**2).sum())
        result["monotonicity"] = float((rq_m * rr_m).sum() / denom) if denom > 1e-12 else 0.0

    return result


# ---------------------------------------------------------------------------
# Turnover
# ---------------------------------------------------------------------------


def compute_turnover(signals: np.ndarray, top_fraction: float = 0.2) -> float:
    """Compute average portfolio turnover rate.

    Turnover measures the fraction of top-ranked assets that change
    between consecutive periods.

    Parameters
    ----------
    signals : np.ndarray, shape (M, T)
    top_fraction : float
        Fraction of assets in the "top" bucket (default 0.2 = top quintile).

    Returns
    -------
    float
        Average turnover rate in [0, 1].
    """
    asset_count, period_count = signals.shape
    k = max(int(asset_count * top_fraction), 1)
    if period_count < 2:
        return 0.0

    turnover_sum = 0.0
    turnover_count = 0
    previous_selected: np.ndarray | None = None
    previous_usable = False

    # Bound temporary memory while batching ranking and membership overlap.
    for start in range(0, period_count, 64):
        block = signals[:, start : start + 64]
        valid = np.isfinite(block)
        usable = valid.sum(axis=0) >= k
        top_idx = np.argpartition(
            np.where(valid, block, -np.inf), -k, axis=0
        )[-k:, :]
        selected = np.zeros_like(valid)
        selected[top_idx, np.arange(block.shape[1])[None, :]] = True

        if previous_selected is not None and previous_usable and usable[0]:
            overlap = np.count_nonzero(previous_selected & selected[:, 0])
            turnover_sum += 1.0 - overlap / k
            turnover_count += 1

        consecutive = usable[1:] & usable[:-1]
        if np.any(consecutive):
            overlap = np.sum(selected[:, 1:] & selected[:, :-1], axis=0)
            turnover_sum += float(np.sum(1.0 - overlap[consecutive] / k))
            turnover_count += int(np.count_nonzero(consecutive))

        previous_selected = selected[:, -1].copy()
        previous_usable = bool(usable[-1])

    return turnover_sum / turnover_count if turnover_count else 0.0


# ---------------------------------------------------------------------------
# Comprehensive factor statistics
# ---------------------------------------------------------------------------


def compute_factor_stats(
    signals: np.ndarray,
    returns: np.ndarray,
) -> dict:
    """Compute comprehensive factor statistics.

    Parameters
    ----------
    signals : np.ndarray, shape (M, T)
    returns : np.ndarray, shape (M, T)

    Returns
    -------
    dict
        Historical ``ic_*`` keys remain Spearman RankIC for compatibility.
        Explicit ``rank_ic_*`` and ``pearson_ic_*`` keys prevent metric-name
        ambiguity in new benchmark artifacts.
    """
    rank_ic_series = compute_rank_ic(signals, returns)
    pearson_ic_series = compute_pearson_ic(signals, returns)
    valid_rank_ic = rank_ic_series[~np.isnan(rank_ic_series)]
    valid_pearson_ic = pearson_ic_series[~np.isnan(pearson_ic_series)]

    stats: dict = {
        "metric_version": METRIC_VERSION,
        "ic_definition": "spearman_rank",
        "ic_series": rank_ic_series,
        "ic_mean": compute_ic_mean(rank_ic_series),
        "ic_paper_mean": compute_ic_paper_mean(rank_ic_series),
        "ic_abs_mean": compute_ic_abs_mean(rank_ic_series),
        "icir": compute_icir(rank_ic_series),
        "ic_paper_icir": compute_ic_paper_icir(rank_ic_series),
        "ic_win_rate": compute_ic_win_rate(rank_ic_series),
        "ic_std": (float(np.std(valid_rank_ic, ddof=1)) if len(valid_rank_ic) > 2 else 0.0),
        "n_periods": int((~np.isnan(rank_ic_series)).sum()),
        "rank_ic_series": rank_ic_series,
        "rank_ic_mean": compute_ic_mean(rank_ic_series),
        "rank_ic_paper_mean": compute_ic_paper_mean(rank_ic_series),
        "rank_ic_abs_mean": compute_ic_abs_mean(rank_ic_series),
        "rank_icir": compute_icir(rank_ic_series),
        "rank_ic_paper_icir": compute_ic_paper_icir(rank_ic_series),
        "rank_ic_win_rate": compute_ic_win_rate(rank_ic_series),
        "rank_ic_std": (float(np.std(valid_rank_ic, ddof=1)) if len(valid_rank_ic) > 2 else 0.0),
        "rank_ic_n_periods": int((~np.isnan(rank_ic_series)).sum()),
        "pearson_ic_series": pearson_ic_series,
        "pearson_ic_mean": compute_ic_mean(pearson_ic_series),
        "pearson_ic_paper_mean": compute_ic_paper_mean(pearson_ic_series),
        "pearson_ic_abs_mean": compute_ic_abs_mean(pearson_ic_series),
        "pearson_icir": compute_icir(pearson_ic_series),
        "pearson_ic_paper_icir": compute_ic_paper_icir(pearson_ic_series),
        "pearson_ic_win_rate": compute_ic_win_rate(pearson_ic_series),
        "pearson_ic_std": (
            float(np.std(valid_pearson_ic, ddof=1)) if len(valid_pearson_ic) > 2 else 0.0
        ),
        "pearson_ic_n_periods": int((~np.isnan(pearson_ic_series)).sum()),
    }

    # Quintile analysis
    quintile = compute_quintile_returns(signals, returns)
    stats.update(quintile)

    # Turnover
    stats["turnover"] = compute_turnover(signals)

    return stats
