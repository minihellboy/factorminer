"""Core visualization functions for FactorMiner.

Provides publication-quality plots for factor analysis, mining diagnostics,
and performance reporting. Uses matplotlib and seaborn with a consistent
style inspired by the FactorMiner paper figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

_STYLE_APPLIED = False


def _apply_style() -> None:
    """Apply a clean, publication-quality matplotlib style once."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
    _STYLE_APPLIED = True


def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    """Save figure to disk or display interactively."""
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Correlation heatmap (Figure 2)
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    correlation_matrix: np.ndarray,
    factor_names: list[str],
    title: str = "Factor Library Correlation Heatmap",
    save_path: str | None = None,
) -> None:
    """Generate pairwise Spearman correlation heatmap.

    Displays the average off-diagonal |rho| in the title and uses a
    diverging colormap centred at zero.

    Parameters
    ----------
    correlation_matrix : np.ndarray, shape (N, N)
        Symmetric matrix of pairwise |rho| values.
    factor_names : List[str]
        Labels for each factor (length N).
    title : str
        Base title for the plot.
    save_path : Optional[str]
        If provided, saves the figure to this path instead of displaying.
    """
    _apply_style()
    n = correlation_matrix.shape[0]

    # Compute average off-diagonal correlation
    if n > 1:
        triu_idx = np.triu_indices(n, k=1)
        off_diag = correlation_matrix[triu_idx]
        avg_corr = float(np.nanmean(np.abs(off_diag)))
    else:
        avg_corr = 0.0

    # Scale figure size based on number of factors
    size = max(6, min(n * 0.35 + 2, 20))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))

    mask = np.zeros_like(correlation_matrix, dtype=bool)
    np.fill_diagonal(mask, True)

    sns.heatmap(
        correlation_matrix,
        mask=mask,
        xticklabels=factor_names,
        yticklabels=factor_names,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "Spearman |rho|"},
        ax=ax,
    )

    ax.set_title(f"{title}\nAvg off-diagonal |rho| = {avg_corr:.4f}", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=max(5, 10 - n // 20))
    ax.tick_params(axis="y", rotation=0, labelsize=max(5, 10 - n // 20))

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# IC time series (Figure 5)
# ---------------------------------------------------------------------------

def plot_ic_timeseries(
    ic_series: np.ndarray,
    dates: list[str],
    rolling_window: int = 21,
    title: str = "Daily Mean Rank IC",
    save_path: str | None = None,
) -> None:
    """Plot IC time series with rolling average and cumulative IC.

    Creates a two-panel figure: top panel shows daily IC bars with a
    rolling mean line; bottom panel shows cumulative IC.

    Parameters
    ----------
    ic_series : np.ndarray, shape (T,)
        Daily IC values (may contain NaN).
    dates : List[str]
        Date labels of length T.
    rolling_window : int
        Window for rolling mean (default 21 trading days).
    title : str
        Title for the figure.
    save_path : Optional[str]
        If provided, saves the figure to this path.
    """
    _apply_style()
    T = len(ic_series)
    x = np.arange(T)

    # Replace NaN with 0 for plotting
    ic_clean = np.where(np.isnan(ic_series), 0.0, ic_series)

    # Rolling mean
    kernel = np.ones(rolling_window) / rolling_window
    rolling_ic = np.convolve(ic_clean, kernel, mode="same")
    # Fix edges
    for i in range(rolling_window // 2):
        w = i + rolling_window // 2 + 1
        rolling_ic[i] = np.mean(ic_clean[:w])
        rolling_ic[-(i + 1)] = np.mean(ic_clean[-w:])

    # Cumulative IC
    cumulative_ic = np.nancumsum(ic_clean)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[2, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.08})

    # Top: daily IC bars + rolling mean
    colors = np.where(ic_clean >= 0, "#4CAF50", "#F44336")
    ax1.bar(x, ic_clean, color=colors, alpha=0.5, width=1.0, edgecolor="none")
    ax1.plot(x, rolling_ic, color="#1565C0", linewidth=1.5,
             label=f"{rolling_window}-day Rolling Mean")

    ic_mean = float(np.nanmean(ic_series))
    ax1.axhline(y=ic_mean, color="#FF6F00", linestyle="--", linewidth=1.0,
                label=f"Mean IC = {ic_mean:.4f}")
    ax1.axhline(y=0, color="black", linewidth=0.5)

    ax1.set_ylabel("Rank IC")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.9)

    # Bottom: cumulative IC
    ax2.fill_between(x, cumulative_ic, alpha=0.3, color="#1565C0")
    ax2.plot(x, cumulative_ic, color="#1565C0", linewidth=1.2)
    ax2.set_ylabel("Cumulative IC")
    ax2.set_xlabel("Date")
    ax2.axhline(y=0, color="black", linewidth=0.5)

    # X-axis tick labels
    if T > 0:
        n_ticks = min(10, T)
        step = max(1, T // n_ticks)
        tick_positions = list(range(0, T, step))
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([dates[i] for i in tick_positions], rotation=45, ha="right")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Quintile returns (Figure 6)
# ---------------------------------------------------------------------------

def plot_quintile_returns(
    quintile_returns: dict,
    title: str = "Quintile Returns",
    save_path: str | None = None,
) -> None:
    """Plot Q1-Q5 quintile bar chart and cumulative returns.

    Parameters
    ----------
    quintile_returns : dict
        Dictionary with keys Q1..Q5 (mean returns) and optionally
        'quintile_cumulative' mapping Qx -> array of cumulative returns.
        Also may contain 'long_short' and 'monotonicity'.
    title : str
        Title for the figure.
    save_path : Optional[str]
        If provided, saves the figure to this path.
    """
    _apply_style()

    # Extract quintile mean returns
    q_labels = [k for k in sorted(quintile_returns.keys()) if k.startswith("Q")]
    q_means = [quintile_returns[k] for k in q_labels]
    n_q = len(q_labels)

    has_cumulative = "quintile_cumulative" in quintile_returns
    n_panels = 2 if has_cumulative else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # Bar chart
    ax = axes[0]
    cmap = plt.cm.RdYlGn
    colors = [cmap(i / max(n_q - 1, 1)) for i in range(n_q)]
    bars = ax.bar(q_labels, q_means, color=colors, edgecolor="white", linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, q_means):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{val:.4f}", ha="center",
                va="bottom" if y >= 0 else "top", fontsize=9)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Mean Return")
    ax.set_xlabel("Quintile")

    # Subtitle with L-S return and monotonicity
    subtitle_parts = []
    if "long_short" in quintile_returns:
        subtitle_parts.append(f"L-S = {quintile_returns['long_short']:.4f}")
    if "monotonicity" in quintile_returns:
        subtitle_parts.append(f"Mono = {quintile_returns['monotonicity']:.2f}")
    subtitle = " | ".join(subtitle_parts) if subtitle_parts else ""
    ax.set_title(f"{title}\n{subtitle}" if subtitle else title, fontsize=12)

    # Cumulative returns panel
    if has_cumulative:
        ax2 = axes[1]
        cum_data = quintile_returns["quintile_cumulative"]
        for q_label in q_labels:
            if q_label in cum_data:
                ax2.plot(cum_data[q_label], label=q_label, linewidth=1.2)
        ax2.set_title("Cumulative Quintile Returns", fontsize=12)
        ax2.set_ylabel("Cumulative Return")
        ax2.set_xlabel("Period")
        ax2.legend(loc="upper left", framealpha=0.9)
        ax2.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Ablation comparison (Figure 3)
# ---------------------------------------------------------------------------

def plot_ablation_comparison(
    with_memory: dict,
    without_memory: dict,
    save_path: str | None = None,
) -> None:
    """Bar charts comparing Have Memory vs No Memory ablation.

    Shows side-by-side bars for: high-quality count, rejected count,
    admitted count, yield rate, and rejection rate.

    Parameters
    ----------
    with_memory : dict
        Keys: 'high_quality', 'rejected', 'admitted', 'yield_rate', 'rejection_rate'.
    without_memory : dict
        Same keys as with_memory.
    save_path : Optional[str]
        If provided, saves the figure to this path.
    """
    _apply_style()

    # Count metrics (left axis) and rate metrics (right axis)
    count_keys = ["high_quality", "rejected", "admitted"]
    rate_keys = ["yield_rate", "rejection_rate"]

    count_labels = ["High Quality", "Rejected", "Admitted"]
    rate_labels = ["Yield Rate", "Rejection Rate"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Count metrics
    x = np.arange(len(count_keys))
    w = 0.35
    vals_with = [with_memory.get(k, 0) for k in count_keys]
    vals_without = [without_memory.get(k, 0) for k in count_keys]

    bars1 = ax1.bar(x - w / 2, vals_with, w, label="With Memory",
                    color="#1565C0", edgecolor="white")
    bars2 = ax1.bar(x + w / 2, vals_without, w, label="Without Memory",
                    color="#E53935", edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h,
                     f"{int(h)}", ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(count_labels)
    ax1.set_ylabel("Count")
    ax1.set_title("Factor Counts: Memory Ablation", fontsize=12)
    ax1.legend(loc="upper right")

    # Panel 2: Rate metrics
    x2 = np.arange(len(rate_keys))
    vals_with_r = [with_memory.get(k, 0) * 100 for k in rate_keys]
    vals_without_r = [without_memory.get(k, 0) * 100 for k in rate_keys]

    bars3 = ax2.bar(x2 - w / 2, vals_with_r, w, label="With Memory",
                    color="#1565C0", edgecolor="white")
    bars4 = ax2.bar(x2 + w / 2, vals_without_r, w, label="Without Memory",
                    color="#E53935", edgecolor="white")

    for bars in [bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h,
                     f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(rate_labels)
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("Yield / Rejection Rates: Memory Ablation", fontsize=12)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Efficiency benchmark (Figure 4)
# ---------------------------------------------------------------------------

def plot_efficiency_benchmark(
    benchmarks: dict[str, dict[str, float]],
    save_path: str | None = None,
) -> None:
    """Grouped bar chart on log scale for computation time.

    Compares Python/C/GPU backends at operator and factor levels.

    Parameters
    ----------
    benchmarks : Dict[str, Dict[str, float]]
        Outer keys: backend names (e.g. "Python", "C", "GPU").
        Inner keys: operation names (e.g. "operator_eval", "factor_eval").
        Values: time in seconds.
    save_path : Optional[str]
        If provided, saves the figure to this path.
    """
    _apply_style()

    backends = list(benchmarks.keys())
    operations = sorted(
        {op for bm in benchmarks.values() for op in bm.keys()}
    )
    n_backends = len(backends)
    n_ops = len(operations)

    fig, ax = plt.subplots(figsize=(max(8, n_ops * 2), 5))

    x = np.arange(n_ops)
    total_width = 0.7
    w = total_width / n_backends

    palette = ["#1565C0", "#FF8F00", "#43A047", "#8E24AA", "#E53935"]

    for i, backend in enumerate(backends):
        vals = [benchmarks[backend].get(op, 0) for op in operations]
        offset = (i - (n_backends - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=backend,
                      color=palette[i % len(palette)], edgecolor="white")

        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.3g}s", ha="center", va="bottom", fontsize=7)

    ax.set_yscale("log")
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(operations, rotation=30, ha="right")
    ax.set_title("Computation Efficiency by Backend", fontsize=12)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Cost pressure (Figure 9)
# ---------------------------------------------------------------------------

def plot_cost_pressure(
    results: dict[float, dict],
    save_path: str | None = None,
) -> None:
    """Cumulative return plots under different transaction cost settings.

    Shows both linear and log-scale y-axis panels.

    Parameters
    ----------
    results : Dict[float, dict]
        Keys: transaction cost levels (e.g. 0.0, 0.001, 0.003).
        Values: dict with 'cumulative_returns' (np.ndarray) and
        optionally 'dates' (List[str]).
    save_path : Optional[str]
        If provided, saves the figure to this path.
    """
    _apply_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cost_levels = sorted(results.keys())
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(cost_levels)))

    for color, cost in zip(palette, cost_levels):
        data = results[cost]
        cum_ret = np.asarray(data["cumulative_returns"])
        label = f"TC = {cost*100:.1f}%" if cost > 0 else "No TC"

        ax1.plot(cum_ret, color=color, linewidth=1.3, label=label)
        # For log scale, shift to always positive
        shifted = cum_ret - cum_ret.min() + 1.0
        ax2.plot(shifted, color=color, linewidth=1.3, label=label)

    ax1.set_title("Cumulative Returns (Linear)", fontsize=12)
    ax1.set_ylabel("Cumulative Return")
    ax1.set_xlabel("Period")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.axhline(y=0, color="black", linewidth=0.5)

    ax2.set_yscale("log")
    ax2.set_title("Cumulative Returns (Log Scale)", fontsize=12)
    ax2.set_ylabel("Shifted Cumulative Return (log)")
    ax2.set_xlabel("Period")
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Mining funnel chart
# ---------------------------------------------------------------------------

def plot_mining_funnel(
    batch_stats: dict,
    save_path: str | None = None,
) -> None:
    """Funnel chart showing Stage 1 -> 2 -> 3 -> 4 filtering.

    Parameters
    ----------
    batch_stats : dict
        Keys: 'generated', 'ic_passed', 'corr_passed', 'admitted'.
        Each is an int count at the corresponding stage.
    save_path : Optional[str]
        If provided, saves the figure to this path.
    """
    _apply_style()

    stages = [
        ("Generated", batch_stats.get("generated", 0)),
        ("IC Screen Passed", batch_stats.get("ic_passed", 0)),
        ("Correlation Passed", batch_stats.get("corr_passed", 0)),
        ("Admitted", batch_stats.get("admitted", 0)),
    ]

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    max_val = max(values) if values else 1

    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw horizontal funnel bars centred on the y-axis
    y_positions = list(range(len(stages) - 1, -1, -1))
    bar_colors = ["#42A5F5", "#66BB6A", "#FFA726", "#EF5350"]

    for i, (y, val, label, color) in enumerate(
        zip(y_positions, values, labels, bar_colors)
    ):
        width = val / max_val if max_val > 0 else 0
        ax.barh(y, width, height=0.6, color=color, edgecolor="white",
                      linewidth=1.5, left=(1 - width) / 2)
        # Label inside the bar
        ax.text(0.5, y, f"{label}\n{val:,}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white" if width > 0.3 else "#333")

    # Draw connecting trapezoids
    for i in range(len(stages) - 1):
        y_top = y_positions[i]
        y_bot = y_positions[i + 1]
        w_top = values[i] / max_val if max_val > 0 else 0
        w_bot = values[i + 1] / max_val if max_val > 0 else 0

        (1 - w_top) / 2
        (1 + w_top) / 2
        (1 - w_bot) / 2
        (1 + w_bot) / 2

        # Drop rate annotation
        if values[i] > 0:
            drop = (1 - values[i + 1] / values[i]) * 100
            mid_y = (y_top + y_bot) / 2
            ax.text(1.02, mid_y, f"-{drop:.0f}%", ha="left", va="center",
                    fontsize=9, color="#E53935", fontweight="bold",
                    transform=ax.get_yaxis_transform())

    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.5, len(stages) - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Mining Pipeline Funnel", fontsize=13, fontweight="bold", pad=15)
    ax.spines[:].set_visible(False)

    fig.tight_layout()
    _save_or_show(fig, save_path)
