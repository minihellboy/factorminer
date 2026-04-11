"""Mining session reporting for FactorMiner.

Provides structured logging, text reports, JSON export, and progress
visualization for factor mining sessions. Designed to mirror the batch
reports shown in Appendix H of the paper.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data classes for structured logging
# ---------------------------------------------------------------------------

@dataclass
class FactorAdmissionRecord:
    """Record of a single factor admission."""

    factor_id: int
    name: str
    formula: str
    ic: float
    icir: float
    max_corr: float
    batch_number: int
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class BatchRecord:
    """Record of a single mining batch."""

    batch_num: int
    candidates: int = 0
    ic_passed: int = 0
    corr_passed: int = 0
    admitted: int = 0
    replaced: int = 0
    rejection_reasons: list[str] = field(default_factory=list)
    admitted_factors: list[FactorAdmissionRecord] = field(default_factory=list)
    library_size: int = 0
    elapsed_seconds: float = 0.0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @property
    def rejected(self) -> int:
        return self.candidates - self.admitted - self.replaced

    @property
    def yield_rate(self) -> float:
        if self.candidates == 0:
            return 0.0
        return self.admitted / self.candidates

    @property
    def rejection_rate(self) -> float:
        if self.candidates == 0:
            return 0.0
        return self.rejected / self.candidates

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rejected"] = self.rejected
        d["yield_rate"] = self.yield_rate
        d["rejection_rate"] = self.rejection_rate
        return d


# ---------------------------------------------------------------------------
# MiningReporter
# ---------------------------------------------------------------------------

class MiningReporter:
    """Track and report mining session progress.

    Collects batch-level and factor-level logs, generates text reports,
    JSON exports, and progress visualisations.

    Parameters
    ----------
    output_dir : str
        Directory for saving reports and plots.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batches: list[BatchRecord] = []
        self.factor_admissions: list[FactorAdmissionRecord] = []
        self._session_start: float = time.time()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_batch(
        self,
        batch_num: int,
        candidates: int,
        ic_passed: int,
        corr_passed: int,
        admitted: int,
        replaced: int,
        rejection_reasons: list[str],
        library_size: int = 0,
        elapsed_seconds: float = 0.0,
    ) -> None:
        """Log a batch's results.

        Parameters
        ----------
        batch_num : int
            Sequential batch number.
        candidates : int
            Number of candidates generated.
        ic_passed : int
            Number passing IC screening.
        corr_passed : int
            Number passing correlation screening.
        admitted : int
            Number admitted to the library.
        replaced : int
            Number that replaced existing library factors.
        rejection_reasons : List[str]
            List of rejection reason strings for this batch.
        library_size : int
            Current library size after this batch.
        elapsed_seconds : float
            Time taken for this batch.
        """
        record = BatchRecord(
            batch_num=batch_num,
            candidates=candidates,
            ic_passed=ic_passed,
            corr_passed=corr_passed,
            admitted=admitted,
            replaced=replaced,
            rejection_reasons=rejection_reasons,
            library_size=library_size,
            elapsed_seconds=elapsed_seconds,
        )
        self.batches.append(record)

    def log_factor_admission(
        self,
        factor_id: int,
        name: str,
        formula: str,
        ic: float,
        icir: float,
        max_corr: float,
    ) -> None:
        """Log an individual factor admission.

        Parameters
        ----------
        factor_id : int
            Unique factor identifier.
        name : str
            Human-readable factor name.
        formula : str
            DSL expression string.
        ic : float
            Mean IC of the admitted factor.
        icir : float
            ICIR of the admitted factor.
        max_corr : float
            Maximum pairwise correlation at admission time.
        """
        batch_num = self.batches[-1].batch_num if self.batches else 0
        record = FactorAdmissionRecord(
            factor_id=factor_id,
            name=name,
            formula=formula,
            ic=ic,
            icir=icir,
            max_corr=max_corr,
            batch_number=batch_num,
        )
        self.factor_admissions.append(record)
        if self.batches:
            self.batches[-1].admitted_factors.append(record)

    # ------------------------------------------------------------------
    # Text reports
    # ------------------------------------------------------------------

    def generate_batch_report(self, batch_num: int) -> str:
        """Generate text report for a specific batch.

        Parameters
        ----------
        batch_num : int
            The batch number to report on.

        Returns
        -------
        str
            Formatted text report.
        """
        batch = None
        for b in self.batches:
            if b.batch_num == batch_num:
                batch = b
                break

        if batch is None:
            return f"Batch {batch_num} not found."

        lines = [
            f"{'=' * 60}",
            f"  BATCH REPORT: Iteration {batch.batch_num}",
            f"  Timestamp: {batch.timestamp}",
            f"{'=' * 60}",
            "",
            f"  Candidates generated:    {batch.candidates:>6}",
            f"  IC screen passed:        {batch.ic_passed:>6}",
            f"  Correlation passed:      {batch.corr_passed:>6}",
            f"  Admitted to library:     {batch.admitted:>6}",
            f"  Replaced in library:     {batch.replaced:>6}",
            f"  Rejected:                {batch.rejected:>6}",
            "",
            f"  Yield rate:              {batch.yield_rate:>6.1%}",
            f"  Rejection rate:          {batch.rejection_rate:>6.1%}",
            f"  Library size (after):    {batch.library_size:>6}",
            f"  Elapsed:                 {batch.elapsed_seconds:>6.1f}s",
        ]

        # Admitted factors detail
        if batch.admitted_factors:
            lines.append("")
            lines.append("  Admitted Factors:")
            lines.append(f"  {'ID':>4}  {'IC':>8}  {'ICIR':>8}  {'MaxCorr':>8}  Name")
            lines.append(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}")
            for f in batch.admitted_factors:
                lines.append(
                    f"  {f.factor_id:>4}  {f.ic:>8.4f}  {f.icir:>8.3f}  "
                    f"{f.max_corr:>8.4f}  {f.name[:30]}"
                )

        # Rejection breakdown
        if batch.rejection_reasons:
            reason_counts: dict[str, int] = defaultdict(int)
            for reason in batch.rejection_reasons:
                # Normalise to short category
                if "IC" in reason.upper() or "ic" in reason.lower():
                    reason_counts["IC below threshold"] += 1
                elif "corr" in reason.lower():
                    reason_counts["Correlation too high"] += 1
                elif "parse" in reason.lower() or "invalid" in reason.lower():
                    reason_counts["Parse / invalid"] += 1
                else:
                    reason_counts["Other"] += 1

            lines.append("")
            lines.append("  Rejection Breakdown:")
            for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason:<30} {count:>5}")

        lines.append(f"\n{'=' * 60}")
        return "\n".join(lines)

    def generate_session_report(self) -> str:
        """Generate full session report with cumulative statistics.

        Returns
        -------
        str
            Formatted text report covering the entire mining session.
        """
        elapsed = time.time() - self._session_start

        total_candidates = sum(b.candidates for b in self.batches)
        total_ic_passed = sum(b.ic_passed for b in self.batches)
        total_corr_passed = sum(b.corr_passed for b in self.batches)
        total_admitted = sum(b.admitted for b in self.batches)
        total_replaced = sum(b.replaced for b in self.batches)
        total_rejected = total_candidates - total_admitted - total_replaced

        overall_yield = total_admitted / total_candidates if total_candidates > 0 else 0.0
        final_lib_size = self.batches[-1].library_size if self.batches else 0

        lines = [
            f"{'#' * 60}",
            "  FACTORMINER SESSION REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"{'#' * 60}",
            "",
            f"  Total batches:           {len(self.batches):>6}",
            f"  Total elapsed:           {elapsed:>6.0f}s ({elapsed/60:.1f}m)",
            "",
            "  --- Cumulative Pipeline ---",
            f"  Candidates generated:    {total_candidates:>6}",
            f"  IC screen passed:        {total_ic_passed:>6} ({total_ic_passed/total_candidates:.1%})" if total_candidates > 0 else f"  IC screen passed:        {total_ic_passed:>6}",
            f"  Correlation passed:      {total_corr_passed:>6} ({total_corr_passed/total_candidates:.1%})" if total_candidates > 0 else f"  Correlation passed:      {total_corr_passed:>6}",
            f"  Admitted:                {total_admitted:>6} ({overall_yield:.1%})",
            f"  Replaced:                {total_replaced:>6}",
            f"  Rejected:                {total_rejected:>6}",
            "",
            f"  Final library size:      {final_lib_size:>6}",
            f"  Overall yield rate:      {overall_yield:>6.1%}",
        ]

        # Per-batch summary table
        if self.batches:
            lines.append("")
            lines.append("  --- Per-Batch Summary ---")
            lines.append(
                f"  {'Batch':>5}  {'Cand':>5}  {'IC':>4}  {'Corr':>4}  "
                f"{'Adm':>4}  {'Rep':>4}  {'Lib':>4}  {'Yield':>6}  {'Time':>6}"
            )
            lines.append(f"  {'-'*5}  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*6}  {'-'*6}")
            for b in self.batches:
                lines.append(
                    f"  {b.batch_num:>5}  {b.candidates:>5}  {b.ic_passed:>4}  "
                    f"{b.corr_passed:>4}  {b.admitted:>4}  {b.replaced:>4}  "
                    f"{b.library_size:>4}  {b.yield_rate:>5.1%}  {b.elapsed_seconds:>5.1f}s"
                )

        # Top admitted factors
        if self.factor_admissions:
            top_factors = sorted(self.factor_admissions, key=lambda f: f.ic, reverse=True)[:10]
            lines.append("")
            lines.append("  --- Top 10 Factors by IC ---")
            lines.append(
                f"  {'ID':>4}  {'IC':>8}  {'ICIR':>8}  {'MaxCorr':>8}  Name"
            )
            lines.append(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*30}")
            for f in top_factors:
                lines.append(
                    f"  {f.factor_id:>4}  {f.ic:>8.4f}  {f.icir:>8.3f}  "
                    f"{f.max_corr:>8.4f}  {f.name[:30]}"
                )

        lines.append(f"\n{'#' * 60}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_to_json(self, path: str) -> None:
        """Export all mining logs to JSON.

        Parameters
        ----------
        path : str
            File path for the JSON output.
        """
        payload = {
            "session": {
                "start_time": datetime.fromtimestamp(self._session_start).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "elapsed_seconds": time.time() - self._session_start,
                "total_batches": len(self.batches),
                "total_admissions": len(self.factor_admissions),
            },
            "batches": [b.to_dict() for b in self.batches],
            "factor_admissions": [asdict(f) for f in self.factor_admissions],
            "summary": self._compute_summary(),
        }

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def save_session_report(self, filename: str = "session_report.txt") -> str:
        """Save the session report to a text file.

        Returns the path to the saved file.
        """
        report = self.generate_session_report()
        path = self.output_dir / filename
        with open(path, "w") as f:
            f.write(report)
        return str(path)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_mining_progress(self, save_path: str | None = None) -> None:
        """Plot library growth, yield rate, and rejection rate over batches.

        Produces a 3-panel figure:
            1. Library size growth
            2. Yield rate per batch
            3. Rejection breakdown stacked area

        Parameters
        ----------
        save_path : Optional[str]
            If provided, saves the figure to this path.
        """
        if not self.batches:
            return

        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "figure.dpi": 150,
        })

        batch_nums = [b.batch_num for b in self.batches]
        lib_sizes = [b.library_size for b in self.batches]
        yield_rates = [b.yield_rate * 100 for b in self.batches]
        admitted_counts = [b.admitted for b in self.batches]
        replaced_counts = [b.replaced for b in self.batches]
        rejected_counts = [b.rejected for b in self.batches]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10),
                                             sharex=True,
                                             gridspec_kw={"hspace": 0.15})

        # Panel 1: Library size growth
        ax1.plot(batch_nums, lib_sizes, color="#1565C0", linewidth=2.0,
                 marker="o", markersize=3)
        ax1.fill_between(batch_nums, lib_sizes, alpha=0.15, color="#1565C0")
        ax1.set_ylabel("Library Size")
        ax1.set_title("Mining Progress", fontsize=13, fontweight="bold")
        if lib_sizes:
            ax1.text(batch_nums[-1], lib_sizes[-1],
                     f"  {lib_sizes[-1]}", va="center", fontsize=9, color="#1565C0")

        # Panel 2: Yield rate
        ax2.bar(batch_nums, yield_rates, color="#43A047", alpha=0.7,
                edgecolor="white", linewidth=0.5)
        if yield_rates:
            avg_yield = sum(yield_rates) / len(yield_rates)
            ax2.axhline(y=avg_yield, color="#FF6F00", linestyle="--", linewidth=1.0,
                        label=f"Avg = {avg_yield:.1f}%")
            ax2.legend(fontsize=8, loc="upper right")
        ax2.set_ylabel("Yield Rate (%)")
        ax2.set_ylim(bottom=0)

        # Panel 3: Stacked bar of admitted / replaced / rejected
        ax3.bar(batch_nums, admitted_counts, label="Admitted",
                color="#43A047", edgecolor="white", linewidth=0.5)
        ax3.bar(batch_nums, replaced_counts, bottom=admitted_counts,
                label="Replaced", color="#FF8F00", edgecolor="white", linewidth=0.5)
        bottoms = [a + r for a, r in zip(admitted_counts, replaced_counts)]
        ax3.bar(batch_nums, rejected_counts, bottom=bottoms,
                label="Rejected", color="#E53935", alpha=0.6,
                edgecolor="white", linewidth=0.5)
        ax3.set_ylabel("Candidates")
        ax3.set_xlabel("Batch Number")
        ax3.legend(loc="upper right", fontsize=8)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", facecolor="white", dpi=200)
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_summary(self) -> dict:
        """Compute cumulative summary statistics."""
        total_candidates = sum(b.candidates for b in self.batches)
        total_admitted = sum(b.admitted for b in self.batches)
        total_replaced = sum(b.replaced for b in self.batches)

        return {
            "total_candidates": total_candidates,
            "total_admitted": total_admitted,
            "total_replaced": total_replaced,
            "total_rejected": total_candidates - total_admitted - total_replaced,
            "overall_yield_rate": total_admitted / total_candidates if total_candidates > 0 else 0.0,
            "final_library_size": self.batches[-1].library_size if self.batches else 0,
            "total_elapsed_seconds": time.time() - self._session_start,
        }
