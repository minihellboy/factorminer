"""Import boundary regressions for package-level exports."""

from __future__ import annotations

import subprocess
import sys


def test_evaluation_runtime_import_is_cycle_free_in_fresh_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import factorminer.evaluation.runtime as runtime; "
                "print(runtime.SignalComputationError.__name__)"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "SignalComputationError" in result.stdout

