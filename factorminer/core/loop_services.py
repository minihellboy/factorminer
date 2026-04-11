"""Reusable orchestration helpers for Ralph and Helix loops.

This module keeps the loop classes focused on policy decisions while
centralizing the repeated stage-chain execution and iteration telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from factorminer.architecture import IterationPayload
from factorminer.utils.logging import FactorRecord, IterationRecord


@dataclass(slots=True)
class IterationTelemetry:
    """Structured snapshot of one completed iteration."""

    iteration: int
    candidates_generated: int
    stats: dict[str, Any]
    elapsed_seconds: float
    results: Sequence[Any] = field(default_factory=tuple)


class LoopExecutionService:
    """Shared orchestration helpers for factor-mining loops."""

    def __init__(self, loop: Any) -> None:
        self.loop = loop

    def new_payload(self, batch_size: int) -> IterationPayload:
        return IterationPayload(iteration=self.loop.iteration, batch_size=batch_size)

    def run_stage_chain(
        self,
        payload: IterationPayload,
        stage_names: Sequence[str],
    ) -> None:
        for stage_name in stage_names:
            self.loop.stages[stage_name].run(self.loop, payload)

    def candidate_count(self, payload: IterationPayload) -> int:
        if "candidates_before_canon" in payload.stage_metrics:
            return int(payload.stage_metrics["candidates_before_canon"])
        return len(payload.candidates)

    def empty_stats(self) -> dict[str, Any]:
        return self.loop._empty_stats()

    def build_stats(self, payload: IterationPayload, elapsed: float) -> dict[str, Any]:
        stats = self.loop._compute_stats(payload.results, payload.admitted_results, elapsed)
        stats.update(payload.stage_metrics)
        return stats

    def describe_empty_generation(self, payload: IterationPayload) -> str:
        if self.candidate_count(payload) > 0:
            return "all candidates removed by canonicalization"
        return "generator produced 0 candidates"

    def build_telemetry(
        self,
        payload: IterationPayload,
        stats: dict[str, Any],
        elapsed: float,
        *,
        candidates_generated: int | None = None,
    ) -> IterationTelemetry:
        return IterationTelemetry(
            iteration=self.loop.iteration,
            candidates_generated=candidates_generated
            if candidates_generated is not None
            else self.candidate_count(payload),
            stats=stats,
            elapsed_seconds=elapsed,
            results=tuple(payload.results),
        )

    def log_telemetry(self, telemetry: IterationTelemetry) -> None:
        """Emit the shared iteration telemetry to reporter and session logs."""
        self.loop.reporter.log_batch(**telemetry.stats)
        session_logger = getattr(self.loop, "_session_logger", None)
        if session_logger is None:
            return

        ic_values = [r.ic_mean for r in telemetry.results if getattr(r, "parse_ok", False)]
        record = IterationRecord(
            iteration=telemetry.iteration,
            candidates_generated=telemetry.candidates_generated,
            ic_passed=int(telemetry.stats.get("ic_passed", 0)),
            correlation_passed=int(telemetry.stats.get("corr_passed", 0)),
            admitted=int(telemetry.stats.get("admitted", 0)),
            rejected=max(telemetry.candidates_generated - int(telemetry.stats.get("admitted", 0)), 0),
            replaced=int(telemetry.stats.get("replaced", 0)),
            library_size=int(telemetry.stats.get("library_size", 0)),
            best_ic=max(ic_values) if ic_values else 0.0,
            mean_ic=float(np.mean(ic_values)) if ic_values else 0.0,
            elapsed_seconds=telemetry.elapsed_seconds,
        )
        session_logger.log_iteration(record)

        for result in telemetry.results:
            factor_rec = FactorRecord(
                expression=result.formula,
                ic=result.ic_mean if getattr(result, "parse_ok", False) else None,
                icir=result.icir if getattr(result, "parse_ok", False) else None,
                max_correlation=(
                    result.max_correlation if getattr(result, "parse_ok", False) else None
                ),
                admitted=bool(getattr(result, "admitted", False)),
                rejection_reason=result.rejection_reason or None,
                replaced_factor=str(result.replaced) if getattr(result, "replaced", None) else None,
            )
            session_logger.log_factor(factor_rec)

