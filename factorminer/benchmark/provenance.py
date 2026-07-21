"""Benchmark provenance capture and JSON integrity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from factorminer.benchmark.datasets import _base_path
from factorminer.core.library_io import load_library
from factorminer.core.session import MiningSession
from factorminer.evaluation.metrics import METRIC_VERSION


def _json_safe(value: Any) -> Any:
    """Recursively convert NaN/inf values into JSON-safe nulls."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as fp:
            payload = json.load(fp)
    except Exception as exc:  # pragma: no cover - defensive provenance capture
        return {"path": str(path), "load_error": str(exc)}

    if isinstance(payload, dict):
        return payload
    return {"path": str(path), "payload_type": type(payload).__name__}


def _session_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return MiningSession.load(path).get_summary()
    except Exception as exc:  # pragma: no cover - defensive provenance capture
        return {"path": str(path), "load_error": str(exc)}


def _catalog_provenance(baseline: str, candidate_count: int, seed: int) -> dict[str, Any]:
    return {
        "kind": "catalog",
        "source": baseline,
        "baseline_kind": _baseline_kind(baseline),
        "candidate_count": candidate_count,
        "seed": seed,
        "metric_version": METRIC_VERSION,
    }


def _baseline_kind(baseline: str) -> str:
    if baseline in {"gplearn", "alphaforge_style", "alphaagent_style"}:
        return "catalog_proxy"
    if baseline in {"alpha101_classic", "alpha101_adapted", "random_exploration"}:
        return "catalog_baseline"
    if baseline == "factor_miner":
        return "builtin_paper_catalog"
    if baseline == "factor_miner_no_memory":
        return "synthetic_no_memory_proxy"
    return "unknown"


def _saved_library_provenance(
    requested_path: str,
    baseline: str,
) -> dict[str, Any]:
    base_path = Path(_base_path(requested_path)).expanduser()
    resolved_base = base_path.resolve() if base_path.exists() else base_path
    library_json = resolved_base.with_suffix(".json")
    signal_cache = Path(str(resolved_base) + "_signals.npz")
    parent = resolved_base.parent

    source_files: dict[str, dict[str, str]] = {}
    for label, path in {
        "library_json": library_json,
        "signal_cache": signal_cache,
        "session_json": parent / "session.json",
        "session_log_json": parent / "session_log.json",
        "checkpoint_session_json": parent / "checkpoint" / "session.json",
        "checkpoint_loop_state_json": parent / "checkpoint" / "loop_state.json",
        "checkpoint_memory_json": parent / "checkpoint" / "memory.json",
    }.items():
        if path.exists():
            source_files[label] = {
                "path": str(path),
                "sha256": _file_sha256(path),
            }

    provenance: dict[str, Any] = {
        "kind": "saved_library",
        "source": baseline,
        "baseline_kind": "saved_library",
        "metric_version": METRIC_VERSION,
        "requested_path": str(Path(requested_path)),
        "resolved_base_path": str(resolved_base),
        "source_files": source_files,
        "library_summary": {},
        "session_summary": _session_summary(parent / "session.json"),
        "session_log_summary": _json_summary(parent / "session_log.json"),
    }

    if library_json.exists():
        try:
            library = load_library(resolved_base)
        except Exception as exc:  # pragma: no cover - defensive provenance capture
            provenance["library_summary"] = {
                "path": str(library_json),
                "load_error": str(exc),
            }
        else:
            provenance["library_summary"] = {
                "path": str(library_json),
                "factor_count": library.size,
                "metric_version": getattr(library, "metric_version", "legacy_abs_ic"),
                "diagnostics": library.get_diagnostics(),
            }

    return provenance


def _baseline_provenance(
    baseline: str,
    *,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
    candidate_count: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    if baseline == "factor_miner" and factor_miner_library_path:
        return _saved_library_provenance(factor_miner_library_path, baseline)
    if baseline == "factor_miner_no_memory" and factor_miner_no_memory_library_path:
        return _saved_library_provenance(
            factor_miner_no_memory_library_path,
            baseline,
        )
    return _catalog_provenance(baseline, candidate_count, seed)
