#!/usr/bin/env python3
"""Report production mypy regressions against a staged diagnostic baseline.

This intentionally does not require the existing codebase to be type-clean.
The baseline ignores line numbers but retains path, error code, and message, so
ordinary edits do not churn it while new diagnostics remain visible.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

BASELINE_VERSION = 1
DEFAULT_BASELINE = Path(__file__).resolve().parents[1] / "mypy-baseline.json"
ERROR_PATTERN = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+)(?::\d+)?: error: "
    r"(?P<message>.*?)(?:  \[(?P<code>[^]]+)\])?$"
)


class Diagnostic(NamedTuple):
    path: str
    code: str
    message: str


def parse_diagnostics(output: str) -> Counter[Diagnostic]:
    """Parse stable diagnostic identities from normal mypy text output."""
    diagnostics: Counter[Diagnostic] = Counter()
    for line in output.splitlines():
        match = ERROR_PATTERN.match(line)
        if match is None:
            continue
        message = re.sub(r"\bline \d+\b", "line <N>", match.group("message"))
        diagnostics[
            Diagnostic(
                path=match.group("path").replace("\\", "/"),
                code=match.group("code") or "unknown",
                message=message,
            )
        ] += 1
    return diagnostics


def run_mypy() -> Counter[Diagnostic]:
    """Run mypy over production package code, excluding the test suite."""
    command = [
        sys.executable,
        "-m",
        "mypy",
        "factorminer",
        "--exclude",
        r"^factorminer/tests/",
        "--no-error-summary",
        "--no-pretty",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode not in (0, 1):
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(f"mypy invocation failed with exit code {completed.returncode}")
    return parse_diagnostics(completed.stdout + completed.stderr)


def load_baseline(path: Path) -> Counter[Diagnostic]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != BASELINE_VERSION:
        raise ValueError(f"Unsupported mypy baseline version in {path}")
    diagnostics: Counter[Diagnostic] = Counter()
    for row in payload.get("diagnostics", []):
        diagnostic = Diagnostic(
            path=str(row["path"]),
            code=str(row["code"]),
            message=str(row["message"]),
        )
        diagnostics[diagnostic] = int(row["count"])
    return diagnostics


def write_baseline(path: Path, diagnostics: Counter[Diagnostic]) -> None:
    rows = [
        {
            "path": diagnostic.path,
            "code": diagnostic.code,
            "message": diagnostic.message,
            "count": count,
        }
        for diagnostic, count in sorted(diagnostics.items())
    ]
    payload = {
        "version": BASELINE_VERSION,
        "scope": "factorminer production package; factorminer/tests excluded",
        "error_count": sum(diagnostics.values()),
        "diagnostics": rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize(diagnostics: Counter[Diagnostic]) -> str:
    by_code: Counter[str] = Counter()
    for diagnostic, count in diagnostics.items():
        by_code[diagnostic.code] += count
    codes = ", ".join(f"{code}={count}" for code, count in sorted(by_code.items()))
    return f"{sum(diagnostics.values())} production diagnostics ({codes})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    current = run_mypy()
    if args.write_baseline:
        write_baseline(args.baseline, current)
        print(f"Wrote {args.baseline}: {summarize(current)}")
        return 0
    if not args.baseline.exists():
        print(f"Missing baseline: {args.baseline}", file=sys.stderr)
        return 2

    baseline = load_baseline(args.baseline)
    regressions = current - baseline
    improvements = baseline - current
    print(f"Current: {summarize(current)}")
    print(f"Baseline: {summarize(baseline)}")
    if improvements:
        print(
            f"Improved by {sum(improvements.values())} diagnostic(s); "
            "refresh the baseline after review."
        )
    if not regressions:
        print("No new production mypy diagnostics.")
        return 0

    print(f"New production mypy diagnostics: {sum(regressions.values())}", file=sys.stderr)
    for diagnostic, count in sorted(regressions.items()):
        print(
            f"  {diagnostic.path}: [{diagnostic.code}] {diagnostic.message} (x{count})",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
