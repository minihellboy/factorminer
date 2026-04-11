"""Mining session management with persistence and resume support.

A ``MiningSession`` wraps the state that must survive across process
restarts: session metadata, per-iteration statistics, timing, and paths
to serialized artifacts (library, memory).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MiningSession:
    """Manages a complete mining session with persistence.

    Parameters
    ----------
    session_id : str
        Unique identifier for this session (e.g. timestamp or UUID).
    config : dict
        Serialized mining configuration (kept as dict for JSON compat).
    output_dir : str
        Directory for all session artifacts.
    """

    session_id: str
    config: dict[str, Any] = field(default_factory=dict)
    output_dir: str = "./output"
    start_time: str = ""
    end_time: str = ""
    iterations: list[dict[str, Any]] = field(default_factory=list)
    library_path: str = ""
    memory_path: str = ""
    run_manifest_path: str = ""
    run_manifest: dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # running | completed | interrupted

    def __post_init__(self) -> None:
        if not self.start_time:
            self.start_time = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Iteration tracking
    # ------------------------------------------------------------------

    def record_iteration(self, stats: dict[str, Any]) -> None:
        """Append iteration statistics to the session log."""
        stats = dict(stats)
        stats.setdefault("timestamp", datetime.now().isoformat())
        self.iterations.append(stats)

    @property
    def total_iterations(self) -> int:
        return len(self.iterations)

    @property
    def last_library_size(self) -> int:
        if not self.iterations:
            return 0
        return self.iterations[-1].get("library_size", 0)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state to a JSON-compatible dictionary."""
        return {
            "session_id": self.session_id,
            "config": self.config,
            "output_dir": self.output_dir,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "total_iterations": self.total_iterations,
            "last_library_size": self.last_library_size,
            "library_path": self.library_path,
            "memory_path": self.memory_path,
            "run_manifest_path": self.run_manifest_path,
            "run_manifest": self.run_manifest,
            "iterations": self.iterations,
        }

    def save(self, path: str | Path | None = None) -> str:
        """Save session state to a JSON file.

        Parameters
        ----------
        path : str or Path, optional
            Explicit save path.  Defaults to ``{output_dir}/session.json``.

        Returns
        -------
        str
            The path the session was saved to.
        """
        if path is None:
            save_dir = Path(self.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / "session.json"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> MiningSession:
        """Load session from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to a session JSON file.

        Returns
        -------
        MiningSession
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls(
            session_id=data["session_id"],
            config=data.get("config", {}),
            output_dir=data.get("output_dir", "./output"),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
            iterations=data.get("iterations", []),
            library_path=data.get("library_path", ""),
            memory_path=data.get("memory_path", ""),
            run_manifest_path=data.get("run_manifest_path", ""),
            run_manifest=data.get("run_manifest", {}),
            status=data.get("status", "interrupted"),
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Session summary statistics."""
        total_candidates = sum(
            it.get("candidates", 0) for it in self.iterations
        )
        total_admitted = sum(
            it.get("admitted", 0) for it in self.iterations
        )
        total_replaced = sum(
            it.get("replaced", 0) for it in self.iterations
        )

        # Compute elapsed time
        elapsed = 0.0
        if self.start_time:
            start = datetime.fromisoformat(self.start_time)
            end_str = self.end_time or datetime.now().isoformat()
            end = datetime.fromisoformat(end_str)
            elapsed = (end - start).total_seconds()

        return {
            "session_id": self.session_id,
            "status": self.status,
            "total_iterations": self.total_iterations,
            "total_candidates": total_candidates,
            "total_admitted": total_admitted,
            "total_replaced": total_replaced,
            "overall_yield_rate": (
                total_admitted / total_candidates if total_candidates > 0 else 0.0
            ),
            "final_library_size": self.last_library_size,
            "elapsed_seconds": elapsed,
        }

    def finalize(self) -> None:
        """Mark the session as completed and record end time."""
        self.end_time = datetime.now().isoformat()
        self.status = "completed"
