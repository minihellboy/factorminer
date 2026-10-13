"""Incremental lift of mining arms over a baseline model, across campaigns and seeds.

Run with the Qlib interpreter (it reads the parquet predictions)::

    qlib-venv/bin/python -I scripts/qlib_cn/arm_lift.py --label ~/fm-data/labels/csi300.parquet \\
        --root output/qlib_cn/csi300/classic --baseline alpha158_lgb \\
        --arm memory=classic_memory_c1,classic_memory_c2,classic_memory_c3 \\
        --arm marginal=classic_marginal_c1,classic_marginal_c2,classic_marginal_c3

For each campaign, predictions of its ``<campaign>_<handler>_lgb_s*`` seeds are
averaged on the rank scale and scored by daily RankIC; the baseline's seeds
are ensembled the same way. A campaign's *lift* is its daily RankIC minus the
baseline's on the same dates. Its *top lift* is the same paired difference
of the daily top-50 spread (mean label of the 50 best-scored names minus the
day's mean, bps): the part of the cross-section a TopkDropout book trades.
An arm's lift is the mean of its campaigns' lift series, tested with a Newey-West t over dates; ``campaign_sd`` is the
spread of campaign mean lifts, the luck of the draw one campaign cannot
show. Arm pairs are compared by the difference of their lift series. Backtest
metrics (``metrics.json``) are averaged over campaigns and seeds.

Walk-forward windows pool into one test: repeat ``--window name=root`` and
write ``{w}`` in campaign names (``--arm marginal={w}_marginal``). Each window
is scored against its own baseline, and the disjoint daily lift series are
concatenated, so the t covers every test year at once. With
``--campaigns campaigns/``, a campaign whose library admitted nothing has
the baseline's predictions and counts as zero lift rather than being skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

TOPK = 50  # Qlib's TopkDropout benchmark book
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


def _daily_top_spread(score: pd.Series, label: pd.Series, k: int = TOPK) -> pd.Series:
    """Mean label of the ``k`` best-scored names minus the day's mean, in bps.

    The book TopkDropout holds; RankIC weights every rank equally, the
    portfolio only the top.
    """
    frame = pd.DataFrame({"s": score}).join(label.rename("y"), how="inner").dropna()
    return frame.groupby(level="datetime").apply(
        lambda g: (g.nlargest(k, "s")["y"].mean() - g["y"].mean()) * 1e4)


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


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average backtest metrics over windows (each already a seed average)."""
    keys = {key for m in metrics for key in m}
    return {key: statistics.fmean(m[key] for m in metrics if key in m) for key in sorted(keys)}


def _empty_library(campaigns: Path | None, campaign: str) -> bool:
    if campaigns is None:
        return False
    path = campaigns / campaign / "factor_library.json"
    return path.exists() and not json.loads(path.read_text()).get("factors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True)
    root = parser.add_mutually_exclusive_group(required=True)
    root.add_argument("--root", type=Path)
    root.add_argument("--window", action="append", help="name=root; pools windows")
    parser.add_argument("--baseline", default="alpha158_lgb")
    parser.add_argument("--handler", default="alpha158_plus")
    parser.add_argument("--arm", action="append", required=True, help="name=campaign,campaign,...")
    parser.add_argument("--campaigns", type=Path, help="campaign dirs; empty library = zero lift")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    label = pd.read_parquet(args.label)
    label["instrument"] = label["instrument"].str.upper()
    label = label.set_index(["datetime", "instrument"])["label"]
    windows = ([tuple(spec.split("=", 1)) for spec in args.window] if args.window
               else [("", str(args.root))])
    base_ics, base_tops, base_metrics = [], [], []
    for _, root in windows:
        runs = _runs(Path(root), args.baseline)
        ensemble = _ensemble(runs)
        base_ics.append(_daily_rank_ic(ensemble, label))
        base_tops.append(_daily_top_spread(ensemble, label))
        base_metrics.append({"runs": len(runs), **_backtest(runs)})
    base_ic = pd.concat(base_ics)
    report: dict = {"baseline": {"rank_ic": float(base_ic.mean()),
                                 "top_spread_bps": float(pd.concat(base_tops).mean()),
                                 **_mean_metrics(base_metrics)},
                    "windows": [name for name, _ in windows], "arms": {}, "pairs": {}}
    arm_lifts: dict[str, pd.Series] = {}
    arm_tops: dict[str, pd.Series] = {}
    for spec in args.arm:
        name, campaigns = spec.split("=", 1)
        templates = campaigns.split(",")
        lifts, top_lifts, rows = [], [], {}
        for template in templates:
            pieces, top_pieces, metrics = [], [], []
            for (window, root), window_base, window_top in zip(windows, base_ics, base_tops,
                                                               strict=True):
                campaign = template.replace("{w}", window)
                runs = _runs(Path(root), f"{campaign}_{args.handler}_lgb")
                if runs:
                    ensemble = _ensemble(runs)
                    pieces.append((_daily_rank_ic(ensemble, label) - window_base).dropna())
                    top_pieces.append((_daily_top_spread(ensemble, label) - window_top).dropna())
                    metrics.append(_backtest(runs))
                elif _empty_library(args.campaigns, campaign):
                    pieces.append(pd.Series(0.0, index=window_base.index))
                    top_pieces.append(pd.Series(0.0, index=window_top.index))
                else:
                    pieces = []
                    break
            if not pieces:
                rows[template] = {"missing": True}
                continue
            lift, top_lift = pd.concat(pieces), pd.concat(top_pieces)
            lifts.append(lift)
            top_lifts.append(top_lift)
            rows[template] = {"lift": float(lift.mean()), "lift_t": _hac_t(lift.to_numpy()),
                              "top_lift_bps": float(top_lift.mean()),
                              "top_lift_t": _hac_t(top_lift.to_numpy()),
                              **_mean_metrics(metrics)}
            if len(windows) == 1:
                alone = _runs(Path(windows[0][1]), f"{template}_factors_lgb")
                rows[template]["alone"] = _backtest(alone) if alone else {}
        if not lifts:
            report["arms"][name] = {"campaigns": rows}
            continue
        arm = pd.concat(lifts, axis=1).mean(axis=1)
        arm_top = pd.concat(top_lifts, axis=1).mean(axis=1)
        arm_lifts[name] = arm
        arm_tops[name] = arm_top
        means = [rows[c]["lift"] for c in rows if "lift" in rows[c]]
        report["arms"][name] = {
            "campaigns_scored": len(means), "lift": float(arm.mean()),
            "lift_t": _hac_t(arm.to_numpy()),
            "top_lift_bps": float(arm_top.mean()), "top_lift_t": _hac_t(arm_top.to_numpy()),
            "campaign_sd": statistics.stdev(means) if len(means) > 1 else float("nan"),
            "campaigns": rows,
        }
    names = list(arm_lifts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff = (arm_lifts[a] - arm_lifts[b]).dropna()
            top = (arm_tops[a] - arm_tops[b]).dropna()
            report["pairs"][f"{a}-{b}"] = {"diff": float(diff.mean()), "diff_t": _hac_t(diff.to_numpy()),
                                           "a_better_share": float((diff > 0).mean()),
                                           "top_diff_bps": float(top.mean()),
                                           "top_diff_t": _hac_t(top.to_numpy())}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(f"baseline {args.baseline}: RankIC {report['baseline']['rank_ic']:.4f} "
          f"top{TOPK} spread {report['baseline']['top_spread_bps']:.1f} bps/day IR {report['baseline'].get('excess_return_with_cost.information_ratio', float('nan')):.2f}")
    for name, arm in report["arms"].items():
        if "lift" not in arm:
            print(f"{name:14s} (no scored campaigns)")
            continue
        ir = [c.get("excess_return_with_cost.information_ratio") for c in arm["campaigns"].values()
              if "lift" in c]
        print(f"{name:14s} lift {arm['lift']:+.4f} (t {arm['lift_t']:.2f}, campaign sd {arm['campaign_sd']:.4f})"
              f"  top{TOPK} {arm['top_lift_bps']:+.2f} bps (t {arm['top_lift_t']:.2f})"
              f"  per-campaign {[round(c['lift'], 4) for c in arm['campaigns'].values() if 'lift' in c]}"
              f"  IR {[round(x, 2) for x in ir if x is not None]}")
    for pair, row in report["pairs"].items():
        print(f"{pair:28s} diff {row['diff']:+.4f} (t {row['diff_t']:.2f})"
              f"  top{TOPK} {row['top_diff_bps']:+.2f} bps (t {row['top_diff_t']:.2f})")


if __name__ == "__main__":
    main()
