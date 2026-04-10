"""Factor candidate lifecycle logging and trajectory reconstruction."""

from __future__ import annotations

import json
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
