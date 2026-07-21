"""Core evaluation metrics for alpha factors.

Provides vectorized, production-quality implementations of Information
Coefficient (IC), ICIR, quintile analysis, turnover, and comprehensive
factor statistics used by the validation pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# Information Coefficient
# ---------------------------------------------------------------------------


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


def _compute_cross_sectional_correlation(
    signals: np.ndarray,
    returns: np.ndarray,
    *,
    rank: bool,
    min_assets: int = 5,
) -> np.ndarray:
    """Compute a Pearson correlation, optionally after ranking each cross-section."""
    signal_panel, return_panel = _validate_panel_pair(signals, returns)
    _, period_count = signal_panel.shape
    series = np.full(period_count, np.nan, dtype=np.float64)

    for period in range(period_count):
        signal = signal_panel[:, period]
        forward_return = return_panel[:, period]
        valid = np.isfinite(signal) & np.isfinite(forward_return)
        if int(valid.sum()) < min_assets:
            continue
        x = signal[valid]
        y = forward_return[valid]
        if rank:
            x = rankdata(x)
            y = rankdata(y)
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denominator = np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
        series[period] = (
            float(np.dot(x_centered, y_centered) / denominator) if denominator > 1e-12 else 0.0
        )
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
    signals, returns = _validate_panel_pair(signals, returns)
    M, T = signals.shape
    ic_series = np.full(T, np.nan, dtype=np.float64)

    # Mask invalid entries
    invalid = ~np.isfinite(signals) | ~np.isfinite(returns)

    # Rank each column independently (replace NaN with very large value to push to end)
    big = 1e18
    sig_filled = np.where(invalid, big, signals)
    ret_filled = np.where(invalid, big, returns)

    for t in range(T):
        valid = ~invalid[:, t]
        n = valid.sum()
        if n < 5:
            continue
        rs = rankdata(sig_filled[valid, t])
        rr = rankdata(ret_filled[valid, t])
        rs_m = rs - rs.mean()
        rr_m = rr - rr.mean()
        denom = np.sqrt((rs_m**2).sum() * (rr_m**2).sum())
        ic_series[t] = (rs_m * rr_m).sum() / denom if denom > 1e-12 else 0.0

    return ic_series


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
    M, T = signals_a.shape
    corrs = []

    for t in range(T):
        a = signals_a[:, t]
        b = signals_b[:, t]
        valid = np.isfinite(a) & np.isfinite(b)
        n = valid.sum()
        if n < 5:
            continue
        ra = rankdata(a[valid])
        rb = rankdata(b[valid])
        ra_m = ra - ra.mean()
        rb_m = rb - rb.mean()
        denom = np.sqrt((ra_m**2).sum() * (rb_m**2).sum())
        if denom < 1e-12:
            corrs.append(0.0)
        else:
            corrs.append(float((ra_m * rb_m).sum() / denom))

    if not corrs:
        return 0.0
    return float(np.mean(corrs))


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
    M, T = signals.shape
    # Accumulate per-quintile return sums
    quintile_returns: dict[int, list[float]] = {q: [] for q in range(1, n_quantiles + 1)}

    for t in range(T):
        s = signals[:, t]
        r = returns[:, t]
        eligible = np.isfinite(s)
        n = int(eligible.sum())
        if n < n_quantiles:
            continue
        s_valid = s[eligible]
        r_eligible = r[eligible]
        # Assign quintile labels via rank
        ranks = rankdata(s_valid)
        # Map to quintile: ceil(rank / n * n_quantiles), clamped
        q_labels = np.clip(
            np.ceil(ranks / n * n_quantiles).astype(int),
            1,
            n_quantiles,
        )
        for q in range(1, n_quantiles + 1):
            mask = q_labels == q
            bucket_returns = r_eligible[mask]
            if mask.any() and np.all(np.isfinite(bucket_returns)):
                quintile_returns[q].append(float(np.mean(bucket_returns)))

    result = {}
    means = {}
    for q in range(1, n_quantiles + 1):
        key = f"Q{q}"
        if quintile_returns[q]:
            means[q] = float(np.mean(quintile_returns[q]))
        else:
            means[q] = 0.0
        result[key] = means[q]

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
