"""Strict benchmark artifact serialization."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from factorminer.benchmark.contracts import BenchmarkManifest, json_safe


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_sha256(path: Path) -> str:
    """Return an artifact digest without loading the entire file into memory."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(json_safe(payload), fp, indent=2, sort_keys=False, allow_nan=False)


def _save_manifest(path: Path, manifest: BenchmarkManifest) -> None:
    _write_json(path, asdict(manifest))
