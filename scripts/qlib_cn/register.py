"""Freeze a trading procedure for prospective confirmation, or verify a frozen one.

    python scripts/qlib_cn/register.py freeze --spec spec.json --runs output/qlib_cn/csi300/live \\
        --out docs/registrations/<name>.json
    python scripts/qlib_cn/register.py verify docs/registrations/<name>.json

``spec.json`` names the procedure (``name``), its books (``books``: name ->
run globs under ``--runs``, the ``book_eval.py`` syntax), the ``baseline``
book, and the adoption rule (``rule``). ``freeze`` records the spec, the git
commit, the provider's last calendar date, and the SHA-256 of every matched
run's ``model.txt`` and ``metrics.json``; committing and pushing the file
timestamps it. The prospective test starts at the first session after the
provider date recorded here (``PROTOCOLS["live"]`` starts no earlier).
``verify`` recomputes the hashes, so predictions made later with
``run_qlib_baselines.py --load-model`` provably use the frozen boosters, and
warns about later runs that match a book's glob but were not registered.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hashes(runs: Path, books: dict[str, str]) -> dict[str, dict[str, str]]:
    out = {}
    for globs in books.values():
        for pattern in globs.split(","):
            matched = sorted(p for p in runs.iterdir() if p.is_dir() and fnmatch.fnmatch(p.name, pattern))
            if not matched:
                sys.exit(f"no runs match {runs}/{pattern}")
            for run in matched:
                if not (run / "model.txt").exists():
                    sys.exit(f"{run} has no model.txt")
                out[run.name] = {f: _sha(run / f) for f in ("model.txt", "metrics.json")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--spec", type=Path, required=True)
    freeze.add_argument("--runs", type=Path, required=True)
    freeze.add_argument("--provider", type=Path, default=Path("~/fm-data/qlib_cn"))
    freeze.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("registration", type=Path)
    args = parser.parse_args()

    if args.command == "freeze":
        if args.out.exists():
            sys.exit(f"{args.out} exists: a registration is never overwritten")
        spec = json.loads(args.spec.read_text())
        missing = {"name", "books", "baseline", "rule"} - set(spec)
        if missing:
            sys.exit(f"spec lacks {sorted(missing)}")
        calendar = (args.provider.expanduser() / "calendars" / "day.txt").read_text().split()
        record = {
            "spec": spec, "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
            "registered": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                         check=True).stdout.strip(),
            "provider_last_date": calendar[-1], "runs_root": str(args.runs),
            "runs": _hashes(args.runs, spec["books"]),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"frozen {len(record['runs'])} runs; test starts after {record['provider_last_date']}")
        return

    record = json.loads(args.registration.read_text())
    current = _hashes(Path(record["runs_root"]), record["spec"]["books"])
    changed = sorted(r for r in record["runs"] if current.get(r, {}).get("model.txt") != record["runs"][r]["model.txt"])
    if changed:
        sys.exit(f"boosters changed since registration: {changed}")
    added = sorted(set(current) - set(record["runs"]))
    if added:  # the registered runs define the books; later runs matching a glob are not part of them
        print(f"warning: unregistered runs match the books' globs; score only the registered runs, not {added}")
    print(f"ok: {len(record['runs'])} boosters match {args.registration}")


if __name__ == "__main__":
    main()
