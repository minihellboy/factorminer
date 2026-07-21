"""Persistent research knowledge, bounded retrieval, and outcome attribution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from factorminer.architecture.research_absorption import (
    ResearchAbsorptionService,
    ResearchNote,
)
from factorminer.domain.evidence import FrozenPayload


def _digest(payload: Mapping[str, Any]) -> str:
    normalized = FrozenPayload.from_mapping(payload).to_dict()
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = FrozenPayload.from_mapping(payload).to_dict()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise ValueError(f"Content-address collision at {path}")
        return
    path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class ResearchSourceRecord:
    """Immutable A-layer decision for one external research fragment."""

    source_id: str
    source: str
    text: str
    eligible: bool
    eligibility_reason: str
    eligibility_mode: str
    screening_provider: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResearchSourceRecord:
        return cls(
            source_id=str(payload["source_id"]),
            source=str(payload.get("source", "unspecified")),
            text=str(payload.get("text", "")),
            eligible=bool(payload.get("eligible", False)),
            eligibility_reason=str(payload.get("eligibility_reason", "")),
            eligibility_mode=str(payload.get("eligibility_mode", "ohlcv_only")),
            screening_provider=str(payload.get("screening_provider", "unknown")),
            created_at=str(payload.get("created_at", "")),
        )


@dataclass(frozen=True, slots=True)
class ResearchHypothesisRecord:
    """Immutable B/C-layer hypothesis linked to its screened source."""

    hypothesis_id: str
    source_id: str
    name: str
    mechanism_family: str
    fine_family: str
    mechanism_role: str
    research_paths: tuple[str, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["research_paths"] = list(self.research_paths)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResearchHypothesisRecord:
        return cls(
            hypothesis_id=str(payload["hypothesis_id"]),
            source_id=str(payload["source_id"]),
            name=str(payload["name"]),
            mechanism_family=str(payload["mechanism_family"]),
            fine_family=str(payload["fine_family"]),
            mechanism_role=str(payload["mechanism_role"]),
            research_paths=tuple(str(item) for item in payload.get("research_paths", ())),
            created_at=str(payload["created_at"]),
        )


@dataclass(frozen=True, slots=True)
class ResearchOutcomeRecord:
    """One factor evaluation attributed to retrieved research hypotheses."""

    outcome_id: str
    iteration: int
    source_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    factor_name: str
    formula: str
    admitted: bool
    rejection_reason: str
    metrics: Mapping[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "source_ids": list(self.source_ids),
            "hypothesis_ids": list(self.hypothesis_ids),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class ResearchRetrieval:
    """Bounded prompt context plus its exact attribution identifiers."""

    archetypes: tuple[Mapping[str, Any], ...] = ()
    source_ids: tuple[str, ...] = ()
    hypothesis_ids: tuple[str, ...] = ()

    def enrich(self, memory_signal: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(memory_signal)
        if not self.hypothesis_ids:
            return payload
        payload["research_archetypes"] = [dict(item) for item in self.archetypes]
        payload["source_ids"] = list(self.source_ids)
        payload["hypothesis_ids"] = list(self.hypothesis_ids)
        return payload


class ResearchKnowledgeStore:
    """Append-only source, hypothesis, and evaluation-outcome repository."""

    def __init__(self, output_dir: str | Path) -> None:
        self.root = Path(output_dir) / "research_knowledge"
        self.sources_dir = self.root / "sources"
        self.hypotheses_dir = self.root / "hypotheses"
        self.outcomes_dir = self.root / "outcomes"

    def ingest(
        self,
        note: ResearchNote,
        service: ResearchAbsorptionService,
    ) -> tuple[ResearchSourceRecord, ResearchHypothesisRecord | None]:
        """Screen, classify when eligible, and persist one source exactly once."""
        eligible, reason = service.screen_eligibility(note.text)
        identity = {
            "source": note.source,
            "text": note.text,
            "eligible": eligible,
            "eligibility_reason": reason,
            "eligibility_mode": service.eligibility_mode,
            "screening_provider": service.llm_provider.provider_name,
        }
        source_id = _digest(identity)
        source = ResearchSourceRecord(
            source_id=source_id,
            source=note.source,
            text=note.text,
            eligible=eligible,
            eligibility_reason=reason,
            eligibility_mode=service.eligibility_mode,
            screening_provider=service.llm_provider.provider_name,
            created_at=datetime.now(UTC).isoformat(),
        )
        source_path = self.sources_dir / f"{source_id}.json"
        if source_path.exists():
            source = ResearchSourceRecord.from_dict(
                json.loads(source_path.read_text(encoding="utf-8"))
            )
        else:
            _write_once(source_path, source.to_dict())
        if not eligible:
            return source, None

        archetype = service.classify_mechanism(note.text)
        hypothesis_identity = {
            "source_id": source_id,
            "name": archetype.name,
            "mechanism_family": archetype.mechanism_family,
            "fine_family": archetype.fine_family,
            "mechanism_role": archetype.mechanism_role,
            "research_paths": list(archetype.research_paths),
        }
        hypothesis_id = _digest(hypothesis_identity)
        hypothesis = ResearchHypothesisRecord(
            hypothesis_id=hypothesis_id,
            source_id=source_id,
            name=archetype.name,
            mechanism_family=archetype.mechanism_family,
            fine_family=archetype.fine_family,
            mechanism_role=archetype.mechanism_role,
            research_paths=tuple(archetype.research_paths),
            created_at=datetime.now(UTC).isoformat(),
        )
        hypothesis_path = self.hypotheses_dir / f"{hypothesis_id}.json"
        if hypothesis_path.exists():
            hypothesis = ResearchHypothesisRecord.from_dict(
                json.loads(hypothesis_path.read_text(encoding="utf-8"))
            )
        else:
            _write_once(hypothesis_path, hypothesis.to_dict())
        return source, hypothesis

    def retrieve(
        self,
        library_state: Mapping[str, Any],
        *,
        limit: int,
    ) -> ResearchRetrieval:
        """Rank hypotheses by observed yield and library coverage, capped by ``limit``."""
        if limit <= 0 or not self.hypotheses_dir.exists():
            return ResearchRetrieval()
        hypotheses = [
            ResearchHypothesisRecord.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            )
            for path in sorted(self.hypotheses_dir.glob("*.json"))
        ]
        outcomes = self._outcome_counts()
        category_counts = library_state.get("category_counts", {})
        if not isinstance(category_counts, Mapping):
            category_counts = {}

        def rank(record: ResearchHypothesisRecord) -> tuple[float, str]:
            attempts, admitted = outcomes.get(record.hypothesis_id, (0, 0))
            yield_score = admitted / attempts if attempts else 0.5
            coverage = float(category_counts.get(record.fine_family, 0) or 0)
            exploration_score = 1.0 / (1.0 + coverage)
            return (-(2.0 * yield_score + exploration_score), record.hypothesis_id)

        selected = sorted(hypotheses, key=rank)[:limit]
        return ResearchRetrieval(
            archetypes=tuple(
                {
                    "name": item.name,
                    "mechanism_family": item.mechanism_family,
                    "fine_family": item.fine_family,
                    "mechanism_role": item.mechanism_role,
                    "research_paths": list(item.research_paths),
                    "source_id": item.source_id,
                    "hypothesis_id": item.hypothesis_id,
                }
                for item in selected
            ),
            source_ids=tuple(sorted({item.source_id for item in selected})),
            hypothesis_ids=tuple(item.hypothesis_id for item in selected),
        )

    def record_results(
        self,
        results: Sequence[Any],
        memory_signal: Mapping[str, Any],
        *,
        iteration: int,
    ) -> tuple[ResearchOutcomeRecord, ...]:
        """Persist candidate outcomes against the research context that produced them."""
        source_ids = tuple(sorted(str(item) for item in memory_signal.get("source_ids", ())))
        hypothesis_ids = tuple(
            sorted(str(item) for item in memory_signal.get("hypothesis_ids", ()))
        )
        if not hypothesis_ids:
            return ()
        records: list[ResearchOutcomeRecord] = []
        for result in results:
            metrics = {
                "ic_mean": getattr(result, "ic_mean", None),
                "ic_paper_mean": getattr(result, "ic_paper_mean", None),
                "icir": getattr(result, "icir", None),
                "max_correlation": getattr(result, "max_correlation", None),
                "stage_passed": getattr(result, "stage_passed", None),
            }
            identity = {
                "iteration": int(iteration),
                "source_ids": source_ids,
                "hypothesis_ids": hypothesis_ids,
                "factor_name": str(getattr(result, "factor_name", "")),
                "formula": str(getattr(result, "formula", "")),
                "admitted": bool(getattr(result, "admitted", False)),
                "rejection_reason": str(getattr(result, "rejection_reason", "") or ""),
                "metrics": metrics,
            }
            outcome_id = _digest(identity)
            path = self.outcomes_dir / f"{outcome_id}.json"
            if path.exists():
                continue
            record = ResearchOutcomeRecord(
                outcome_id=outcome_id,
                iteration=int(iteration),
                source_ids=source_ids,
                hypothesis_ids=hypothesis_ids,
                factor_name=identity["factor_name"],
                formula=identity["formula"],
                admitted=identity["admitted"],
                rejection_reason=identity["rejection_reason"],
                metrics=metrics,
                created_at=datetime.now(UTC).isoformat(),
            )
            _write_once(path, record.to_dict())
            records.append(record)
        return tuple(records)

    def _outcome_counts(self) -> dict[str, tuple[int, int]]:
        counts: dict[str, list[int]] = {}
        if not self.outcomes_dir.exists():
            return {}
        for path in sorted(self.outcomes_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            admitted = bool(payload.get("admitted", False))
            for hypothesis_id in payload.get("hypothesis_ids", ()):
                bucket = counts.setdefault(str(hypothesis_id), [0, 0])
                bucket[0] += 1
                bucket[1] += int(admitted)
        return {key: (value[0], value[1]) for key, value in counts.items()}
