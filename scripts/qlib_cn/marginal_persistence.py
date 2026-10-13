"""Does a factor's value beyond Alpha158 persist out of sample?

    uv run python scripts/qlib_cn/marginal_persistence.py output/qlib_cn/marginal_oos \\
        --out output/qlib_cn/marginal_persistence.json

Reads ``<root>/<protocol>/<set>/factors.csv`` written by ``evaluate_library.py
--marginal``. Partial ICs there are oriented by the train sign, so a factor
whose marginal value reverses scores negative on test. Per protocol and arm
(campaign names without the ``_cN`` replicate suffix and window prefix) it
reports the mean train and test partial IC, their ratio (*retention*), the
share of factors whose test partial IC keeps the train sign, and the
equal-weight composite's test partial IC. Pooled over every mined factor, it
bins train partial-IC t to show whether stronger in-sample evidence buys
more persistence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BINS = (0.0, 3.0, 4.0, 6.0, np.inf)


def _arm(protocol: str, name: str) -> str:
    name = re.sub(r"_c\d+$", "", name)
    return name.removeprefix(f"{protocol}_").removeprefix("classic_")


def _load(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/*/factors.csv")):
        if path.stat().st_size <= 1:  # an empty library
            continue
        frame = pd.read_csv(path)
        if "test_partial_ic" not in frame:
            continue
        summary = json.loads((path.parent / "summary.json").read_text())
        protocol, name = path.parent.parent.name, path.parent.name
        frame = frame.dropna(subset=["train_partial_ic", "test_partial_ic"]).assign(
            protocol=protocol, set=f"{protocol}/{name}", arm=_arm(protocol, name),
            composite_test_partial_ic=summary["marginal_vs_alpha158"]["composite"]["test_partial_ic"],
            composite_test_partial_t=summary["marginal_vs_alpha158"]["composite"]["test_partial_t"],
        )
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no factors.csv with marginal columns under {root}")
    return pd.concat(frames, ignore_index=True)


def _summary(group: pd.DataFrame) -> pd.Series:
    train, test = group["train_partial_ic"].mean(), group["test_partial_ic"].mean()
    sets = group.drop_duplicates("set")
    return pd.Series({
        "sets": len(sets), "factors": len(group),
        "train_partial_ic": train, "test_partial_ic": test,
        "retention": test / train if train > 0 else np.nan,
        "sign_held": float((group["test_partial_ic"] > 0).mean()),
        "test_t_gt_2": float((group["test_partial_t"] > 2).mean()),
        "train_residual_share": group["train_residual_share"].median(),
        "composite_test_partial_ic": sets["composite_test_partial_ic"].mean(),
        "composite_test_partial_t": sets["composite_test_partial_t"].mean(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    frame = _load(args.root)
    by_arm = frame.groupby(["protocol", "arm"]).apply(_summary, include_groups=False)
    mined = frame[~frame["arm"].isin(["fm_catalog", "alpha101"])]
    walk_forward = frame[frame["protocol"].str.startswith("wf")]
    pooled = walk_forward.groupby("arm").apply(_summary, include_groups=False)
    bins = pd.cut(mined["train_partial_t"].abs(), BINS, right=False)
    calibration = mined.groupby(bins, observed=True).apply(_summary, include_groups=False)
    calibration.index = calibration.index.astype(str)
    pd.set_option("display.width", 200)
    print("per protocol and arm\n", by_arm.round(4).to_string())
    print("\nwalk-forward windows pooled\n", pooled.round(4).to_string())
    print("\nmined factors by train partial-IC |t|\n", calibration.round(4).to_string())
    if args.out:
        args.out.write_text(json.dumps({
            "by_arm": {f"{p}/{a}": row.to_dict() for (p, a), row in by_arm.iterrows()},
            "walk_forward_pooled": {a: row.to_dict() for a, row in pooled.iterrows()},
            "calibration_by_train_t": {b: row.to_dict() for b, row in calibration.iterrows()},
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
