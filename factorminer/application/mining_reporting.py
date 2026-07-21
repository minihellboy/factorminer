"""Persistence-oriented reporting for mining iterations."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from factorminer.core.factor_library import FactorLibrary

# Mining Reporter
# ---------------------------------------------------------------------------


class MiningReporter:
    """Lightweight reporter that logs batch results to a JSONL file."""

    def __init__(self, output_dir: str = "./output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.output_dir / "mining_batches.jsonl"

    def log_batch(self, iteration: int, **stats: Any) -> None:
        """Append a batch record to the JSONL log."""
        record = {"iteration": iteration, "timestamp": time.time()}
        record.update(stats)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def export_library(self, library: FactorLibrary, path: str | None = None) -> str:
        """Export the factor library to JSON."""
        if path is None:
            path = str(self.output_dir / "factor_library.json")
        factors = [f.to_dict() for f in library.list_factors()]
        diagnostics = library.get_diagnostics()
        payload = {
            "factors": factors,
            "diagnostics": diagnostics,
            "exported_at": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return path


# ---------------------------------------------------------------------------
