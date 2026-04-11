"""Factor tear sheet generation for FactorMiner.

Produces comprehensive, multi-panel evaluation reports for individual
factors, following the style of Appendix O / Figure 10 from the paper.
Also provides summary table generation for the full factor library.
"""

from __future__ import annotations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from factorminer.evaluation.metrics import (
    compute_ic,
    compute_ic_mean,
    compute_ic_win_rate,
    compute_icir,
    compute_quintile_returns,
    compute_turnover,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute rolling mean with edge handling."""
    np.full_like(arr, np.nan, dtype=np.float64)
    clean = np.where(np.isnan(arr), 0.0, arr)
    kernel = np.ones(window) / window
    conv = np.convolve(clean, kernel, mode="same")
    # Fix edges using expanding window
    for i in range(window // 2):
        w = i + window // 2 + 1
        conv[i] = np.mean(clean[:w])
        conv[-(i + 1)] = np.mean(clean[-w:])
    return conv


def _compute_daily_turnover(signals: np.ndarray) -> np.ndarray:
    """Compute total daily turnover as fraction of positions changing.

    Parameters
    ----------
    signals : np.ndarray, shape (M, T)

    Returns
    -------
    np.ndarray, shape (T-1,)
        Turnover for each period transition.
    """
    M, T = signals.shape
    turnovers = np.full(T - 1, np.nan, dtype=np.float64)

    for t in range(1, T):
        prev = signals[:, t - 1]
        curr = signals[:, t]
        valid = ~(np.isnan(prev) | np.isnan(curr))
        n = valid.sum()
        if n < 5:
            continue
        # Rank-based positions
        prev_ranks = rankdata(prev[valid]) / n
        curr_ranks = rankdata(curr[valid]) / n
        turnovers[t - 1] = float(np.mean(np.abs(curr_ranks - prev_ranks)))

    return turnovers


# ---------------------------------------------------------------------------
# FactorTearSheet
# ---------------------------------------------------------------------------

class FactorTearSheet:
    """Generate comprehensive evaluation report for a single factor.

    Produces an 8-panel figure plus a metrics summary table.
    """

    # Panel colours
    IC_BAR_POS = "#4CAF50"
    IC_BAR_NEG = "#F44336"
    ROLLING_COLOR = "#1565C0"
    CUMULATIVE_COLOR = "#0D47A1"
    QUINTILE_CMAP = "RdYlGn"
    TURNOVER_COLOR = "#FF8F00"

    def generate(
        self,
        factor_id: int,
        factor_name: str,
        formula: str,
        signals: np.ndarray,
        returns: np.ndarray,
        dates: list[str],
        save_path: str | None = None,
    ) -> dict[str, float]:
        """Generate a multi-panel tear sheet.

        Panels:
            (a) IC time-series analysis -- daily mean rank IC with mean line
            (b) Rank IC distribution -- histogram with statistics
            (c) 21-day rolling IC -- aggregated daily rolling window
            (d) Cumulative IC composition
            (e) Quintile returns -- bar chart with Q1-Q5
            (f) Cumulative returns -- line chart for quintiles
            (g) Factor value distribution -- histogram
            (h) Turnover analysis -- daily total turnover

        Parameters
        ----------
        factor_id : int
            Unique identifier for the factor.
        factor_name : str
            Human-readable factor name.
        formula : str
            DSL expression string.
        signals : np.ndarray, shape (M, T)
            Factor signal values.
        returns : np.ndarray, shape (M, T)
            Forward returns.
        dates : List[str]
            Date strings of length T.
        save_path : Optional[str]
            If provided, saves the figure to this path.

        Returns
        -------
        Dict[str, float]
            Dictionary of computed metrics.
        """
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.dpi": 150,
        })

        M, T = signals.shape
        ic_series = compute_ic(signals, returns)
        ic_clean = np.where(np.isnan(ic_series), 0.0, ic_series)
        valid_ic = ic_series[~np.isnan(ic_series)]

        # Compute all metrics
        ic_mean = float(np.mean(valid_ic)) if len(valid_ic) > 0 else 0.0
        ic_abs_mean = compute_ic_mean(ic_series)
        icir = compute_icir(ic_series)
        win_rate = compute_ic_win_rate(ic_series)
        quintile = compute_quintile_returns(signals, returns)
        turnover = compute_turnover(signals)
        daily_turnover = _compute_daily_turnover(signals)

        metrics = {
            "ic_mean": ic_mean,
            "ic_abs_mean": ic_abs_mean,
            "icir": icir,
            "ic_win_rate": win_rate,
            "Q1_return": quintile.get("Q1", 0.0),
            "Q5_return": quintile.get("Q5", 0.0),
            "long_short": quintile.get("long_short", 0.0),
            "monotonicity": quintile.get("monotonicity", 0.0),
            "avg_turnover": turnover,
        }

        # Cumulative IC
        cumulative_ic = np.nancumsum(ic_clean)

        # Rolling IC (21-day)
        rolling_ic = _rolling_mean(ic_clean, 21)

        # Quintile cumulative returns (compute per-period Q returns)
        n_quantiles = 5
        quintile_ts = {q: [] for q in range(1, n_quantiles + 1)}
        for t in range(T):
            s = signals[:, t]
            r = returns[:, t]
            valid_mask = ~(np.isnan(s) | np.isnan(r))
            n = valid_mask.sum()
            if n < n_quantiles:
                for q in range(1, n_quantiles + 1):
                    quintile_ts[q].append(0.0)
                continue
            ranks = rankdata(s[valid_mask])
            q_labels = np.clip(
                np.ceil(ranks / n * n_quantiles).astype(int), 1, n_quantiles
            )
            r_valid = r[valid_mask]
            for q in range(1, n_quantiles + 1):
                mask_q = q_labels == q
                quintile_ts[q].append(float(np.mean(r_valid[mask_q])) if mask_q.any() else 0.0)

        quintile_cumulative = {}
        for q in range(1, n_quantiles + 1):
            quintile_cumulative[f"Q{q}"] = np.cumsum(quintile_ts[q])

        # ---- Build 4x2 panel figure ----
        fig = plt.figure(figsize=(16, 18))
        gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3)

        # Suptitle
        fig.suptitle(
            f"Factor #{factor_id}: {factor_name}\n{formula[:100]}{'...' if len(formula) > 100 else ''}",
            fontsize=13, fontweight="bold", y=0.98,
        )

        # (a) IC time-series
        ax_a = fig.add_subplot(gs[0, 0])
        x = np.arange(T)
        colors_ic = np.where(ic_clean >= 0, self.IC_BAR_POS, self.IC_BAR_NEG)
        ax_a.bar(x, ic_clean, color=colors_ic, alpha=0.5, width=1.0, edgecolor="none")
        ax_a.axhline(y=ic_mean, color="#FF6F00", linestyle="--", linewidth=1.0,
                     label=f"Mean = {ic_mean:.4f}")
        ax_a.axhline(y=0, color="black", linewidth=0.4)
        ax_a.set_title("(a) Daily Rank IC", fontsize=10)
        ax_a.set_ylabel("IC")
        ax_a.legend(fontsize=8, loc="upper left")
        self._set_date_ticks(ax_a, dates, T)

        # (b) IC distribution
        ax_b = fig.add_subplot(gs[0, 1])
        if len(valid_ic) > 0:
            ax_b.hist(valid_ic, bins=50, color=self.ROLLING_COLOR, alpha=0.7,
                      edgecolor="white", linewidth=0.5, density=True)
            ax_b.axvline(x=ic_mean, color="#FF6F00", linestyle="--", linewidth=1.2,
                         label=f"Mean = {ic_mean:.4f}")
            ax_b.axvline(x=0, color="black", linewidth=0.4)
        ax_b.set_title("(b) Rank IC Distribution", fontsize=10)
        ax_b.set_xlabel("IC")
        ax_b.set_ylabel("Density")
        stats_text = f"Mean={ic_mean:.4f}\nICIR={icir:.3f}\nWin={win_rate:.1%}"
        ax_b.text(0.97, 0.97, stats_text, transform=ax_b.transAxes,
                  ha="right", va="top", fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
        ax_b.legend(fontsize=8, loc="upper left")

        # (c) 21-day rolling IC
        ax_c = fig.add_subplot(gs[1, 0])
        ax_c.plot(x, rolling_ic, color=self.ROLLING_COLOR, linewidth=1.0)
        ax_c.fill_between(x, rolling_ic, alpha=0.15, color=self.ROLLING_COLOR)
        ax_c.axhline(y=0, color="black", linewidth=0.4)
        ax_c.axhline(y=ic_mean, color="#FF6F00", linestyle="--", linewidth=0.8,
                     label=f"Mean = {ic_mean:.4f}")
        ax_c.set_title("(c) 21-Day Rolling IC", fontsize=10)
        ax_c.set_ylabel("Rolling IC")
        ax_c.legend(fontsize=8, loc="upper left")
        self._set_date_ticks(ax_c, dates, T)

        # (d) Cumulative IC
        ax_d = fig.add_subplot(gs[1, 1])
        ax_d.fill_between(x, cumulative_ic, alpha=0.25, color=self.CUMULATIVE_COLOR)
        ax_d.plot(x, cumulative_ic, color=self.CUMULATIVE_COLOR, linewidth=1.0)
        ax_d.axhline(y=0, color="black", linewidth=0.4)
        ax_d.set_title("(d) Cumulative IC", fontsize=10)
        ax_d.set_ylabel("Cumulative IC")
        self._set_date_ticks(ax_d, dates, T)

        # (e) Quintile returns bar chart
        ax_e = fig.add_subplot(gs[2, 0])
        q_labels_list = [f"Q{q}" for q in range(1, n_quantiles + 1)]
        q_vals = [quintile.get(f"Q{q}", 0.0) for q in range(1, n_quantiles + 1)]
        cmap = plt.cm.RdYlGn
        q_colors = [cmap(i / max(n_quantiles - 1, 1)) for i in range(n_quantiles)]
        bars = ax_e.bar(q_labels_list, q_vals, color=q_colors, edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, q_vals):
            y_pos = bar.get_height()
            ax_e.text(bar.get_x() + bar.get_width() / 2, y_pos,
                      f"{val:.4f}", ha="center",
                      va="bottom" if y_pos >= 0 else "top", fontsize=8)
        ax_e.axhline(y=0, color="black", linewidth=0.4)
        ls = quintile.get("long_short", 0.0)
        mono = quintile.get("monotonicity", 0.0)
        ax_e.set_title(f"(e) Quintile Returns  |  L-S={ls:.4f}  Mono={mono:.2f}", fontsize=10)
        ax_e.set_ylabel("Mean Return")

        # (f) Cumulative quintile returns
        ax_f = fig.add_subplot(gs[2, 1])
        q_palette = plt.cm.RdYlGn(np.linspace(0.1, 0.9, n_quantiles))
        for i, q in enumerate(range(1, n_quantiles + 1)):
            key = f"Q{q}"
            ax_f.plot(quintile_cumulative[key], color=q_palette[i],
                      linewidth=1.1, label=key)
        ax_f.axhline(y=0, color="black", linewidth=0.4)
        ax_f.set_title("(f) Cumulative Quintile Returns", fontsize=10)
        ax_f.set_ylabel("Cumulative Return")
        ax_f.legend(loc="upper left", fontsize=8, ncol=n_quantiles, framealpha=0.9)
        self._set_date_ticks(ax_f, dates, T)

        # (g) Factor value distribution
        ax_g = fig.add_subplot(gs[3, 0])
        # Sample from signals for histogram (last period or flattened sample)
        flat_signals = signals[~np.isnan(signals)]
        if len(flat_signals) > 50000:
            rng = np.random.default_rng(42)
            flat_signals = rng.choice(flat_signals, 50000, replace=False)
        if len(flat_signals) > 0:
            # Clip to 1st/99th percentile for cleaner visualization
            lo, hi = np.percentile(flat_signals, [1, 99])
            clipped = flat_signals[(flat_signals >= lo) & (flat_signals <= hi)]
            ax_g.hist(clipped, bins=80, color="#7E57C2", alpha=0.7,
                      edgecolor="white", linewidth=0.3, density=True)
            mean_sig = float(np.mean(flat_signals))
            std_sig = float(np.std(flat_signals))
            ax_g.axvline(x=mean_sig, color="#FF6F00", linestyle="--", linewidth=1.0,
                         label=f"Mean={mean_sig:.4f}")
            stats_text_g = f"Std={std_sig:.4f}\nN={len(flat_signals):,}"
            ax_g.text(0.97, 0.97, stats_text_g, transform=ax_g.transAxes,
                      ha="right", va="top", fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
        ax_g.set_title("(g) Factor Value Distribution", fontsize=10)
        ax_g.set_xlabel("Factor Value")
        ax_g.set_ylabel("Density")
        ax_g.legend(fontsize=8, loc="upper left")

        # (h) Turnover analysis
        ax_h = fig.add_subplot(gs[3, 1])
        valid_turnover = daily_turnover[~np.isnan(daily_turnover)]
        if len(valid_turnover) > 0:
            t_x = np.arange(len(daily_turnover))
            ax_h.bar(t_x, np.where(np.isnan(daily_turnover), 0, daily_turnover),
                     color=self.TURNOVER_COLOR, alpha=0.5, width=1.0, edgecolor="none")
            avg_to = float(np.mean(valid_turnover))
            ax_h.axhline(y=avg_to, color="#D32F2F", linestyle="--", linewidth=1.0,
                         label=f"Avg = {avg_to:.4f}")
            ax_h.legend(fontsize=8, loc="upper right")
        ax_h.set_title("(h) Daily Turnover", fontsize=10)
        ax_h.set_ylabel("Turnover")
        ax_h.set_xlabel("Period")

        # Metrics table at the bottom
        metrics_ls_cum = float(np.sum([quintile_ts[n_quantiles][t] - quintile_ts[1][t]
                                       for t in range(T)]))
        metrics["long_short_cumulative"] = metrics_ls_cum

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", facecolor="white", dpi=200)
            plt.close(fig)
        else:
            plt.show()

        return metrics

    def generate_summary_table(self, factors: list[dict]) -> pd.DataFrame:
        """Generate summary table for all factors in the library.

        Parameters
        ----------
        factors : List[dict]
            Each dict should contain keys: 'id', 'name', 'formula',
            'ic_mean', 'icir', 'ic_win_rate', 'Q1_return', 'Q5_return',
            'long_short', 'monotonicity', 'avg_turnover'.

        Returns
        -------
        pd.DataFrame
            Summary table sorted by IC mean (descending).
        """
        if not factors:
            return pd.DataFrame()

        rows = []
        for f in factors:
            rows.append({
                "ID": f.get("id", ""),
                "Name": f.get("name", ""),
                "Formula": str(f.get("formula", ""))[:60],
                "IC Mean": f.get("ic_mean", 0.0),
                "ICIR": f.get("icir", 0.0),
                "IC Win Rate": f.get("ic_win_rate", 0.0),
                "Q1 Return": f.get("Q1_return", 0.0),
                "Q5 Return": f.get("Q5_return", 0.0),
                "L-S Return": f.get("long_short", 0.0),
                "Monotonicity": f.get("monotonicity", 0.0),
                "Avg Turnover": f.get("avg_turnover", 0.0),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("IC Mean", ascending=False).reset_index(drop=True)
        return df

    @staticmethod
    def _set_date_ticks(ax: plt.Axes, dates: list[str], T: int, n_ticks: int = 8) -> None:
        """Set evenly spaced date tick labels on the x-axis."""
        if T == 0:
            return
        n_ticks = min(n_ticks, T)
        step = max(1, T // n_ticks)
        positions = list(range(0, T, step))
        ax.set_xticks(positions)
        ax.set_xticklabels([dates[i] for i in positions], rotation=45, ha="right", fontsize=7)
