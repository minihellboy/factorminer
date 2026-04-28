"""Structured logging system for FactorMiner mining sessions."""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TextIO

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Structured data records
# ---------------------------------------------------------------------------

@dataclass
class FactorRecord:
    """Log record for a single evaluated factor candidate."""

    expression: str
    ic: float | None = None
    icir: float | None = None
    max_correlation: float | None = None
    admitted: bool = False
    rejection_reason: str | None = None
    replaced_factor: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class IterationRecord:
    """Aggregated stats for a single mining iteration (batch)."""

    iteration: int
    candidates_generated: int = 0
    ic_passed: int = 0
    correlation_passed: int = 0
    admitted: int = 0
    rejected: int = 0
    replaced: int = 0
    library_size: int = 0
    best_ic: float = 0.0
    mean_ic: float = 0.0
    elapsed_seconds: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def yield_rate(self) -> float:
        """Fraction of candidates that were admitted to the library."""
        if self.candidates_generated == 0:
            return 0.0
        return self.admitted / self.candidates_generated

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["yield_rate"] = self.yield_rate
        return d


# ---------------------------------------------------------------------------
# JSON log exporter
# ---------------------------------------------------------------------------

class JSONLogExporter:
    """Collects structured records and exports them to a JSON file."""

    def __init__(self) -> None:
        self.iterations: list[dict[str, Any]] = []
        self.factors: list[dict[str, Any]] = []

    def add_iteration(self, record: IterationRecord) -> None:
        self.iterations.append(record.to_dict())

    def add_factor(self, record: FactorRecord) -> None:
        self.factors.append(record.to_dict())

    def export(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "iterations": self.iterations,
            "factors": self.factors,
            "summary": self._summary(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def _summary(self) -> dict[str, Any]:
        if not self.iterations:
            return {}
        total_candidates = sum(it["candidates_generated"] for it in self.iterations)
        total_admitted = sum(it["admitted"] for it in self.iterations)
        return {
            "total_iterations": len(self.iterations),
            "total_candidates": total_candidates,
            "total_admitted": total_admitted,
            "overall_yield_rate": total_admitted / total_candidates if total_candidates else 0.0,
            "final_library_size": self.iterations[-1].get("library_size", 0),
        }


# ---------------------------------------------------------------------------
# Console formatter
# ---------------------------------------------------------------------------

class _ConsoleFormatter(logging.Formatter):
    """Compact colored formatter for terminal output."""

    GREY = "\033[90m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED + BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, self.RESET)
        level = record.levelname[0]  # Single-char level
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        return f"{self.GREY}{ts}{self.RESET} {color}{level}{self.RESET} {record.getMessage()}"


def setup_logger(
    name: str = "factorminer",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    stream: TextIO = sys.stderr,
) -> logging.Logger:
    """Create and configure a FactorMiner logger.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional path for a plain-text log file.
        stream: Stream for console output (default stderr).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    # Console handler with colors
    console = logging.StreamHandler(stream)
    console.setFormatter(_ConsoleFormatter())
    logger.addHandler(console)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Mining session logger (high-level helper)
# ---------------------------------------------------------------------------

class MiningSessionLogger:
    """High-level logger for an entire mining session.

    Combines structured JSON export with pretty console output.
    """

    def __init__(
        self,
        output_dir: str | Path,
        verbose: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        level = logging.DEBUG if verbose else logging.INFO
        self.logger = setup_logger(
            level=level,
            log_file=self.output_dir / "mining.log",
        )
        self.exporter = JSONLogExporter()
        self._progress: tqdm | None = None

    # -- Progress bar ---------------------------------------------------

    def start_progress(self, total_iterations: int) -> None:
        self._progress = tqdm(
            total=total_iterations,
            desc="Mining",
            unit="iter",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

    def advance_progress(self) -> None:
        if self._progress is not None:
            self._progress.update(1)

    def close_progress(self) -> None:
        if self._progress is not None:
            self._progress.close()
            self._progress = None

    # -- Iteration-level ------------------------------------------------

    def log_iteration(self, record: IterationRecord) -> None:
        """Log a completed iteration to both console and structured store."""
        self.exporter.add_iteration(record)
        self.logger.info(
            "Iter %3d | gen=%d ic_ok=%d corr_ok=%d +%d -%d | "
            "lib=%d yield=%.1f%% best_ic=%.4f mean_ic=%.4f (%.1fs)",
            record.iteration,
            record.candidates_generated,
            record.ic_passed,
            record.correlation_passed,
            record.admitted,
            record.rejected,
            record.library_size,
            record.yield_rate * 100,
            record.best_ic,
            record.mean_ic,
            record.elapsed_seconds,
        )
        self.advance_progress()

    # -- Factor-level ---------------------------------------------------

    def log_factor(self, record: FactorRecord) -> None:
        """Log a single factor evaluation result."""
        self.exporter.add_factor(record)
        if record.admitted:
            self.logger.debug(
                "  + ADMIT  ic=%.4f icir=%.3f corr=%.3f  %s",
                record.ic or 0,
                record.icir or 0,
                record.max_correlation or 0,
                record.expression[:80],
            )
        else:
            self.logger.debug(
                "  - REJECT (%s)  %s",
                record.rejection_reason or "unknown",
                record.expression[:80],
            )

    # -- Session lifecycle ----------------------------------------------

    def log_session_start(self, config_summary: dict[str, Any]) -> None:
        self.logger.info("=" * 60)
        self.logger.info("FactorMiner session started")
        self.logger.info(
            "Target library: %d | Batch: %d | Max iters: %d",
            config_summary.get("target_library_size", "?"),
            config_summary.get("batch_size", "?"),
            config_summary.get("max_iterations", "?"),
        )
        self.logger.info("=" * 60)

    def log_session_end(self, library_size: int, total_time: float) -> None:
        summary = self.exporter._summary()
        self.close_progress()
        self.logger.info("=" * 60)
        self.logger.info("Session complete")
        self.logger.info(
            "Library: %d factors | %d iterations | %.0fs total",
            library_size,
            summary.get("total_iterations", 0),
            total_time,
        )
        self.logger.info(
            "Candidates: %d generated, %d admitted (%.1f%% yield)",
            summary.get("total_candidates", 0),
            summary.get("total_admitted", 0),
            summary.get("overall_yield_rate", 0) * 100,
        )
        self.logger.info("=" * 60)

        # Export structured log
        json_path = self.output_dir / "session_log.json"
        self.exporter.export(json_path)
        self.logger.info("Session log exported to %s", json_path)
