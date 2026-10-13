"""Collect Qlib CN benchmark outputs into one results table (mean and std over seeds).

    python scripts/qlib_cn/aggregate.py output/qlib_cn > output/qlib_cn/results.json
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SEED = re.compile(r"_s(\d+)$")
KEYS = ("IC", "ICIR", "RankIC", "RankICIR", "excess_return_with_cost.annualized_return",
        "excess_return_with_cost.information_ratio", "excess_return_with_cost.max_drawdown",
        "excess_return_without_cost.annualized_return", "turnover_mean")


def main(root: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in sorted(root.glob("*/*/*/metrics.json")):
        universe, protocol, run = path.parts[-4:-1]
        match = SEED.search(run)
        name = SEED.sub("", run) if match else run
        grouped[(universe, protocol, name)].append(json.loads(path.read_text())["metrics"])
    models = []
    for (universe, protocol, name), runs in sorted(grouped.items()):
        row = {"universe": universe, "protocol": protocol, "model": name, "seeds": len(runs)}
        for key in KEYS:
            values = [run[key] for run in runs if key in run]
            if values:
                row[key] = statistics.fmean(values)
                row[f"{key}.std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        models.append(row)
    factor_sets = []
    for path in sorted(root.glob("*/*/*/summary.json")):
        universe, protocol, name = path.parts[-4:-1]
        summary = json.loads(path.read_text())
        factor_sets.append({"universe": universe, "protocol": protocol, "set": name, **summary})
    json.dump({"models": models, "factor_sets": factor_sets}, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
