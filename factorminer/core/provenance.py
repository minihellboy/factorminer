"""Run and factor provenance helpers for mining sessions.

This module keeps provenance data compact, JSON-safe, and stable across
save/load boundaries.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    """Recursively convert common scientific Python objects into JSON-safe data."""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def stable_digest(payload: Any) -> str:
    """Compute a stable SHA256 digest for a JSON-serializable payload."""
    normalized = _json_safe(payload)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _compact_reference_list(entries: Any, limit: int = 8) -> list[str]:
    """Normalize a mixed list of factor references into readable strings."""
    if not entries:
        return []

    if isinstance(entries, (str, Mapping)):
        iterable: Sequence[Any] = [entries]
    else:
        iterable = list(entries)

    values: list[str] = []
    seen: set[str] = set()
    for entry in iterable[:limit]:
        text = ""
        if isinstance(entry, str):
            text = entry.strip()
        elif isinstance(entry, Mapping):
            name = str(entry.get("name", "")).strip()
            formula = str(entry.get("formula", "")).strip()
            category = str(entry.get("category", "")).strip()
            if name and formula:
                text = f"{name}: {formula}"
            elif name and category:
                text = f"{name} [{category}]"
            elif name:
                text = name
            elif formula:
                text = formula
        elif entry is not None:
            text = str(entry).strip()

        if text and text not in seen:
            values.append(text)
            seen.add(text)
    return values


def _compact_memory_signal(memory_signal: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only the most useful pieces of memory context."""
    if not memory_signal:
        return {}

    return {
        "library_state": _json_safe(memory_signal.get("library_state", {})),
        "recommended_directions": _compact_reference_list(
            memory_signal.get("recommended_directions", [])
        ),
        "forbidden_directions": _compact_reference_list(
            memory_signal.get("forbidden_directions", [])
        ),
        "insight_count": len(memory_signal.get("insights", []) or []),
        "semantic_neighbors": _compact_reference_list(
            memory_signal.get("semantic_neighbors", [])
        ),
        "semantic_duplicates": _compact_reference_list(
            memory_signal.get("semantic_duplicates", [])
        ),
        "semantic_gaps": _compact_reference_list(
            memory_signal.get("semantic_gaps", [])
        ),
        "complementary_patterns": _compact_reference_list(
            memory_signal.get("complementary_patterns", [])
        ),
    }


@dataclass
class RunManifest:
    """Serializable description of a mining run."""

    manifest_version: str = "1.0"
    run_id: str = ""
    session_id: str = ""
    loop_type: str = "ralph"
    benchmark_mode: str = "paper"
    created_at: str = ""
    updated_at: str = ""
    iteration: int = 0
    library_size: int = 0
    output_dir: str = ""
    config_digest: str = ""
    config_summary: dict[str, Any] = field(default_factory=dict)
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    phase2_features: list[str] = field(default_factory=list)
    target_stack: list[str] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class FactorProvenance:
    """Serializable provenance payload attached to an admitted factor."""

    manifest_version: str = "1.0"
    run_id: str = ""
    session_id: str = ""
    loop_type: str = "ralph"
    created_at: str = ""
    iteration: int = 0
    batch_number: int = 0
    candidate_rank: int = 0
    factor_name: str = ""
    formula: str = ""
    factor_category: str = ""
    factor_id: int = 0
    generator_family: str = ""
    memory_summary: dict[str, Any] = field(default_factory=dict)
    library_snapshot: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    admission: dict[str, Any] = field(default_factory=dict)
    phase2: dict[str, Any] = field(default_factory=dict)
    target_stack: list[str] = field(default_factory=list)
    research_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def build_run_manifest(
    *,
    run_id: str,
    session_id: str,
    loop_type: str,
    benchmark_mode: str,
    created_at: str,
    updated_at: str,
    iteration: int,
    library_size: int,
    output_dir: str,
    config_summary: Mapping[str, Any],
    dataset_summary: Mapping[str, Any],
    phase2_features: Sequence[str],
    target_stack: Sequence[str],
    artifact_paths: Mapping[str, str] | None = None,
    notes: Sequence[str] | None = None,
) -> RunManifest:
    """Build a run manifest from the live loop state."""
    return RunManifest(
        run_id=run_id,
        session_id=session_id,
        loop_type=loop_type,
        benchmark_mode=benchmark_mode,
        created_at=created_at,
        updated_at=updated_at,
        iteration=iteration,
        library_size=library_size,
        output_dir=output_dir,
        config_digest=stable_digest(config_summary),
        config_summary=_json_safe(dict(config_summary)),
        dataset_summary=_json_safe(dict(dataset_summary)),
        phase2_features=list(phase2_features),
        target_stack=list(target_stack),
        artifact_paths=_json_safe(dict(artifact_paths or {})),
        notes=list(notes or []),
    )


def build_factor_provenance(
    *,
    run_manifest: Mapping[str, Any],
    factor_name: str,
    formula: str,
    factor_category: str,
    factor_id: int,
    iteration: int,
    batch_number: int,
    candidate_rank: int,
    generator_family: str,
    memory_signal: Mapping[str, Any] | None,
    library_state: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any],
    admission: Mapping[str, Any],
    phase2: Mapping[str, Any] | None = None,
    target_stack: Sequence[str] | None = None,
    research_metrics: Mapping[str, Any] | None = None,
) -> FactorProvenance:
    """Build per-factor provenance from the current mining context."""
    manifest = dict(run_manifest)
    return FactorProvenance(
        run_id=str(manifest.get("run_id", "")),
        session_id=str(manifest.get("session_id", "")),
        loop_type=str(manifest.get("loop_type", "ralph")),
        created_at=str(datetime.now().isoformat()),
        iteration=iteration,
        batch_number=batch_number,
        candidate_rank=candidate_rank,
        factor_name=factor_name,
        formula=formula,
        factor_category=factor_category,
        factor_id=factor_id,
        generator_family=generator_family,
        memory_summary=_compact_memory_signal(memory_signal),
        library_snapshot=_json_safe(dict(library_state or {})),
        evaluation=_json_safe(dict(evaluation)),
        admission=_json_safe(dict(admission)),
        phase2=_json_safe(dict(phase2 or {})),
        target_stack=list(target_stack or manifest.get("target_stack", [])),
        research_metrics=_json_safe(dict(research_metrics or {})),
    )
