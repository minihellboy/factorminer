"""Paired comparison of two prediction sets on the same test rows.

    qlib-venv/bin/python -I scripts/qlib_cn/compare_predictions.py \\
        --label label.parquet --a run_a/predictions.parquet --b run_b/predictions.parquet

Reports the mean daily RankIC of each, their mean difference, and a
Newey-West (Bartlett) t-statistic for that difference. Pass several ``--a``
or ``--b`` files (seeds) to average predictions per side before scoring.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd


def _load(paths: list[str]) -> pd.Series:
    frames = [pd.read_parquet(p).set_index(["datetime", "instrument"])["score"] for p in paths]
    # Average seeds on the rank scale so each run carries equal weight.
    ranked = [f.groupby(level="datetime").rank(pct=True) for f in frames]
    return pd.concat(ranked, axis=1).mean(axis=1)


def _daily_rank_ic(score: pd.Series, label: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"s": score}).join(label.rename("y"), how="inner").dropna()
    return frame.groupby(level="datetime").apply(lambda g: g["s"].corr(g["y"], method="spearman"))


def _hac_t(series: np.ndarray) -> float:
    x = series[np.isfinite(series)]
    n = len(x)
    lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    variance = e @ e / n
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / math.sqrt(variance / n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--a", action="append", required=True)
    parser.add_argument("--b", action="append", required=True)
    args = parser.parse_args()
    label = pd.read_parquet(args.label)
    label["instrument"] = label["instrument"].str.upper()
    label = label.set_index(["datetime", "instrument"])["label"]
    ic_a = _daily_rank_ic(_load(args.a), label)
    ic_b = _daily_rank_ic(_load(args.b), label)
    both = pd.concat([ic_a, ic_b], axis=1, join="inner").dropna()
    diff = (both.iloc[:, 0] - both.iloc[:, 1]).to_numpy()
    print(json.dumps({
        "dates": int(len(both)), "rank_ic_a": float(both.iloc[:, 0].mean()),
        "rank_ic_b": float(both.iloc[:, 1].mean()), "diff": float(diff.mean()),
        "diff_hac_t": _hac_t(diff), "a_better_share": float((diff > 0).mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
