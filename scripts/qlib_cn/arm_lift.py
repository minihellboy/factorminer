"""Incremental lift of mining arms over a baseline model, across campaigns and seeds.

Run with the Qlib interpreter (it reads the parquet predictions)::

    qlib-venv/bin/python -I scripts/qlib_cn/arm_lift.py --label ~/fm-data/labels/csi300.parquet \\
        --root output/qlib_cn/csi300/classic --baseline alpha158_lgb \\
        --arm memory=classic_memory_c1,classic_memory_c2,classic_memory_c3 \\
        --arm marginal=classic_marginal_c1,classic_marginal_c2,classic_marginal_c3

For each campaign, predictions of its ``<campaign>_<handler>_lgb_s*`` seeds are
averaged on the rank scale and scored by daily RankIC; the baseline's seeds
are ensembled the same way. A campaign's *lift* is its daily RankIC minus the
baseline's on the same dates. An arm's lift is the mean of its campaigns'
lift series, tested with a Newey-West t over dates; ``campaign_sd`` is the
spread of campaign mean lifts, the luck of the draw one campaign cannot
show. Arm pairs are compared by the difference of their lift series. Backtest
metrics (``metrics.json``) are averaged over campaigns and seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

BACKTEST_KEYS = ("RankIC", "excess_return_with_cost.information_ratio",
                 "excess_return_with_cost.annualized_return", "turnover_mean")


def _hac_t(series: np.ndarray) -> float:
    x = series[np.isfinite(series)]
    n = len(x)
    if n < 3:
        return float("nan")
    lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    e = x - x.mean()
    variance = e @ e / n
    for lag in range(1, lags + 1):
        variance += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return float(x.mean() / math.sqrt(variance / n)) if variance > 0 else float("nan")


def _ensemble(runs: list[Path]) -> pd.Series:
    frames = [pd.read_parquet(run / "predictions.parquet").set_index(["datetime", "instrument"])["score"]
              for run in runs]
    ranked = [frame.groupby(level="datetime").rank(pct=True) for frame in frames]
    return pd.concat(ranked, axis=1).mean(axis=1)


def _daily_rank_ic(score: pd.Series, label: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"s": score}).join(label.rename("y"), how="inner").dropna()
    return frame.groupby(level="datetime").apply(lambda g: g["s"].corr(g["y"], method="spearman"))


def _runs(root: Path, prefix: str) -> list[Path]:
    return sorted(p for p in root.glob(f"{prefix}_s*") if (p / "predictions.parquet").exists())


def _backtest(runs: list[Path]) -> dict[str, float]:
    metrics = [json.loads((run / "metrics.json").read_text())["metrics"] for run in runs]
    out: dict[str, float] = {}
    for key in BACKTEST_KEYS:
        values = [m[key] for m in metrics if key in m]
        if values:
            out[key] = statistics.fmean(values)
            out[f"{key}.sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", default="alpha158_lgb")
    parser.add_argument("--handler", default="alpha158_plus")
    parser.add_argument("--arm", action="append", required=True, help="name=campaign,campaign,...")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    label = pd.read_parquet(args.label)
    label["instrument"] = label["instrument"].str.upper()
    label = label.set_index(["datetime", "instrument"])["label"]
    base_runs = _runs(args.root, args.baseline)
    base_ic = _daily_rank_ic(_ensemble(base_runs), label)
    report: dict = {"baseline": {"runs": len(base_runs), "rank_ic": float(base_ic.mean()),
                                 **_backtest(base_runs)}, "arms": {}, "pairs": {}}
    arm_lifts: dict[str, pd.Series] = {}
    for spec in args.arm:
        name, campaigns = spec.split("=", 1)
        lifts, rows = [], {}
        for campaign in campaigns.split(","):
            runs = _runs(args.root, f"{campaign}_{args.handler}_lgb")
            if not runs:
                rows[campaign] = {"missing": True}
                continue
            ic = _daily_rank_ic(_ensemble(runs), label)
            lift = (ic - base_ic).dropna()
            lifts.append(lift)
            alone = _runs(args.root, f"{campaign}_factors_lgb")
            rows[campaign] = {
                "seeds": len(runs), "rank_ic": float(ic.mean()), "lift": float(lift.mean()),
                "lift_t": _hac_t(lift.to_numpy()), **_backtest(runs),
                "alone": _backtest(alone) if alone else {},
            }
        if not lifts:
            report["arms"][name] = {"campaigns": rows}
            continue
        arm = pd.concat(lifts, axis=1).mean(axis=1)
        arm_lifts[name] = arm
        means = [rows[c]["lift"] for c in rows if "lift" in rows[c]]
        report["arms"][name] = {
            "campaigns_scored": len(means), "lift": float(arm.mean()),
            "lift_t": _hac_t(arm.to_numpy()),
            "campaign_sd": statistics.stdev(means) if len(means) > 1 else float("nan"),
            "campaigns": rows,
        }
    names = list(arm_lifts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = (arm_lifts[a] - arm_lifts[b]).dropna()
            report["pairs"][f"{a}-{b}"] = {"diff": float(diff.mean()), "diff_t": _hac_t(diff.to_numpy()),
                                           "a_better_share": float((diff > 0).mean())}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(f"baseline {args.baseline}: RankIC {report['baseline']['rank_ic']:.4f} "
          f"IR {report['baseline'].get('excess_return_with_cost.information_ratio', float('nan')):.2f}")
    for name, arm in report["arms"].items():
        if "lift" not in arm:
            print(f"{name:14s} (no scored campaigns)")
            continue
        ir = [c.get("excess_return_with_cost.information_ratio") for c in arm["campaigns"].values()
              if "lift" in c]
        print(f"{name:14s} lift {arm['lift']:+.4f} (t {arm['lift_t']:.2f}, campaign sd {arm['campaign_sd']:.4f})"
              f"  per-campaign {[round(c['lift'], 4) for c in arm['campaigns'].values() if 'lift' in c]}"
              f"  IR {[round(x, 2) for x in ir if x is not None]}")
    for pair, row in report["pairs"].items():
        print(f"{pair:28s} diff {row['diff']:+.4f} (t {row['diff_t']:.2f})")


if __name__ == "__main__":
    main()
