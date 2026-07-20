"""Factor candidate lifecycle logging and trajectory reconstruction."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FactorLifecycleEvent:
    """One candidate event emitted during the mining loop."""

    iteration: int
    factor_name: str
    formula: str
    stage: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


class FactorLifecycleStore:
    """Structured event log for candidate trajectories across iterations."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.events: list[FactorLifecycleEvent] = []
        self._path: Path | None = None
        if output_dir is not None:
            self._path = Path(output_dir) / "factor_lifecycle.jsonl"
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, source: str | Path) -> FactorLifecycleStore:
        """Reconstruct a store from a persisted ``factor_lifecycle.jsonl`` log.

        Parameters
        ----------
        source : str | Path
            Either an output directory containing ``factor_lifecycle.jsonl``
            or a direct path to the JSONL log file. Missing files yield an
            empty store rather than raising, matching the tolerant style of
            other artifact readers in this codebase.
        """
        store = cls(output_dir=None)
        path = Path(source)
        if path.is_dir():
            path = path / "factor_lifecycle.jsonl"
        if not path.exists():
            return store
        with open(path) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                store.events.append(FactorLifecycleEvent(**payload))
        return store

    def record(
        self,
        iteration: int,
        factor_name: str,
        formula: str,
        *,
        stage: str,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = FactorLifecycleEvent(
            iteration=iteration,
            factor_name=factor_name,
            formula=formula,
            stage=stage,
            status=status,
            details=details or {},
        )
        self.events.append(event)
        if self._path is not None:
            with open(self._path, "a") as fp:
                fp.write(json.dumps(asdict(event), default=str) + "\n")

    def record_batch_results(self, iteration: int, results: Iterable[Any]) -> None:
        for result in results:
            self.record(
                iteration,
                str(getattr(result, "factor_name", "")),
                str(getattr(result, "formula", "")),
                stage="proposed",
                status="seen",
            )
            if getattr(result, "parse_ok", False):
                self.record(
                    iteration,
                    str(result.factor_name),
                    str(result.formula),
                    stage="parsed",
                    status="passed",
                )
            else:
                self.record(
                    iteration,
                    str(result.factor_name),
                    str(result.formula),
                    stage="parsed",
                    status="failed",
                    details={"reason": getattr(result, "rejection_reason", "")},
                )
                continue

            if getattr(result, "stage_passed", 0) >= 1:
                self.record(
                    iteration,
                    str(result.factor_name),
                    str(result.formula),
                    stage="fast_screened",
                    status="passed",
                    details={"ic_mean": float(getattr(result, "ic_mean", 0.0))},
                )

            if getattr(result, "admitted", False):
                stage = "replaced" if getattr(result, "replaced", None) is not None else "admitted"
                self.record(
                    iteration,
                    str(result.factor_name),
                    str(result.formula),
                    stage=stage,
                    status="passed",
                    details={
                        "ic_mean": float(getattr(result, "ic_mean", 0.0)),
                        "icir": float(getattr(result, "icir", 0.0)),
                        "replaced": getattr(result, "replaced", None),
                    },
                )
            elif getattr(result, "stage_passed", 0) >= 2:
                self.record(
                    iteration,
                    str(result.factor_name),
                    str(result.formula),
                    stage="correlation_rejected",
                    status="failed",
                    details={"reason": getattr(result, "rejection_reason", "")},
                )

    def record_memory_distillation(
        self,
        iteration: int,
        trajectory: Iterable[dict[str, Any]],
    ) -> None:
        for entry in trajectory:
            self.record(
                iteration,
                str(entry.get("factor_id", "")),
                str(entry.get("formula", "")),
                stage="memory_distilled",
                status="recorded",
                details={
                    "admitted": bool(entry.get("admitted", False)),
                    "rejection_reason": entry.get("rejection_reason", ""),
                },
            )

    def build_trajectory(self, iteration: int) -> list[dict[str, Any]]:
        by_factor: dict[tuple[str, str], dict[str, Any]] = {}
        for event in self.events:
            if event.iteration != iteration:
                continue
            key = (event.factor_name, event.formula)
            record = by_factor.setdefault(
                key,
                {
                    "factor_id": event.factor_name,
                    "formula": event.formula,
                    "ic": 0.0,
                    "icir": 0.0,
                    "max_correlation": 0.0,
                    "correlated_with": "",
                    "admitted": False,
                    "replaced": None,
                    "rejection_reason": "",
                },
            )
            if "ic_mean" in event.details:
                record["ic"] = event.details["ic_mean"]
            if "icir" in event.details:
                record["icir"] = event.details["icir"]
            if "replaced" in event.details:
                record["replaced"] = event.details["replaced"]
            if event.stage in {"admitted", "replaced"} and event.status == "passed":
                record["admitted"] = True
            if event.stage == "correlation_rejected":
                record["rejection_reason"] = event.details.get("reason", "")
        return list(by_factor.values())

    def telemetry_summary(self) -> dict[str, Any]:
        """Aggregate recorded events into per-round mining telemetry.

        Builds, for each iteration present in ``self.events``, a
        ``(stage, status)`` occurrence count table plus derived counters for
        the categories a mining round cares about most: parse errors,
        implicit IC-screen rejections (a candidate that parsed but never
        reached the ``fast_screened`` stage -- the pipeline does not emit an
        explicit failure event for this case today), intra-batch duplicate
        rejections (``correlation_rejected`` events whose reason mentions
        deduplication), and plain correlation rejections. Also returns an
        overall rejection-reason breakdown and a per-iteration rejection-rate
        trend.

        Returns
        -------
        dict with keys:
            iterations : list[int]
                Sorted iteration numbers present in the log.
            per_iteration : list[dict]
                One row per iteration with ``iteration``,
                ``stage_status_counts``, ``candidates_seen``,
                ``parse_errors``, ``ic_screen_rejected``,
                ``duplicate_rejected``, ``correlation_rejected``,
                ``admitted``, ``replaced``, ``total_rejected``,
                ``rejection_rate``.
            stage_status_totals : dict[str, int]
                ``"{stage}:{status}"`` -> count, summed across all iterations.
            rejection_reason_totals : dict[str, int]
                Counts keyed by ``parse_error`` / ``duplicate`` /
                ``correlation`` / ``ic_below_threshold``.
            rejection_rate_trend : list[dict]
                ``[{"iteration": i, "rejection_rate": r}, ...]`` in
                iteration order.
            total_candidates, total_rejected, overall_rejection_rate
        """
        iterations = sorted({event.iteration for event in self.events})
        stage_status_totals: dict[str, int] = defaultdict(int)
        rejection_reason_totals: dict[str, int] = defaultdict(int)
        per_iteration: list[dict[str, Any]] = []

        for iteration in iterations:
            events_at = [event for event in self.events if event.iteration == iteration]
            stage_status_counts: dict[str, int] = defaultdict(int)
            for event in events_at:
                stage_status_counts[f"{event.stage}:{event.status}"] += 1
                stage_status_totals[f"{event.stage}:{event.status}"] += 1

            candidates_seen = stage_status_counts.get("proposed:seen", 0)
            parse_errors = stage_status_counts.get("parsed:failed", 0)
            parsed_passed = stage_status_counts.get("parsed:passed", 0)
            fast_screened = stage_status_counts.get("fast_screened:passed", 0)
            ic_screen_rejected = max(0, parsed_passed - fast_screened)
            admitted = stage_status_counts.get("admitted:passed", 0)
            replaced = stage_status_counts.get("replaced:passed", 0)

            duplicate_rejected = 0
            correlation_rejected = 0
            for event in events_at:
                if event.stage != "correlation_rejected" or event.status != "failed":
                    continue
                reason = str(event.details.get("reason", "")).lower()
                if "dedup" in reason or "duplicate" in reason:
                    duplicate_rejected += 1
                    rejection_reason_totals["duplicate"] += 1
                else:
                    correlation_rejected += 1
                    rejection_reason_totals["correlation"] += 1
            rejection_reason_totals["parse_error"] += parse_errors
            rejection_reason_totals["ic_below_threshold"] += ic_screen_rejected

            total_rejected = parse_errors + ic_screen_rejected + duplicate_rejected + correlation_rejected
            rejection_rate = total_rejected / candidates_seen if candidates_seen else 0.0

            per_iteration.append({
                "iteration": iteration,
                "stage_status_counts": dict(stage_status_counts),
                "candidates_seen": candidates_seen,
                "parse_errors": parse_errors,
                "ic_screen_rejected": ic_screen_rejected,
                "duplicate_rejected": duplicate_rejected,
                "correlation_rejected": correlation_rejected,
                "admitted": admitted,
                "replaced": replaced,
                "total_rejected": total_rejected,
                "rejection_rate": rejection_rate,
            })

        total_candidates = sum(row["candidates_seen"] for row in per_iteration)
        total_rejected_all = sum(row["total_rejected"] for row in per_iteration)

        return {
            "iterations": iterations,
            "per_iteration": per_iteration,
            "stage_status_totals": dict(stage_status_totals),
            "rejection_reason_totals": dict(rejection_reason_totals),
            "rejection_rate_trend": [
                {"iteration": row["iteration"], "rejection_rate": row["rejection_rate"]}
                for row in per_iteration
            ],
            "total_candidates": total_candidates,
            "total_rejected": total_rejected_all,
            "overall_rejection_rate": (
                total_rejected_all / total_candidates if total_candidates else 0.0
            ),
        }
