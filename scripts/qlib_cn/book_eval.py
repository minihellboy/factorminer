"""Score portfolio books built from saved predictions: seed/model ensembles, score smoothing, n_drop.

Run with the Qlib interpreter::

    qlib-venv/bin/python -I scripts/qlib_cn/book_eval.py --provider ~/fm-data/qlib_cn --universe csi300 \\
        --label ~/fm-data/labels/csi300.parquet --window wf2018=output/qlib_cn/csi300/wf2018 ... \\
        --book mtop='alpha158r_mtop_lgb_s0' --book mtop3='alpha158r_mtop_lgb_s*' \\
        --book blend='alpha158r_mtop_lgb_s*,alpha158r_alstm_s*' --smooth 0,3 --baseline mtop \\
        --out output/qlib_cn/book_mtop.json

A book is a comma-separated list of run-directory globs under each window's
root. Its runs' predictions are ranked per day and averaged, each *pattern*
weighing equally (seeds within a pattern share its weight), so ``a_s*,b_s*``
is a 50/50 blend of two models. ``--smooth h`` replaces the score with its
exponential moving average over trading days (half-life ``h``), which keeps
names near the top from being dropped and re-bought on noise. Each (book,
smoothing, n_drop) is backtested with the benchmark's TopkDropout book and
costs in every window; daily after-cost excess returns of the disjoint test
windows are concatenated. Against ``--baseline`` (unsmoothed), the paired
daily difference of after-cost excess return gets a Newey-West t, which
removes the market noise both books share and is far tighter than comparing
IRs across seeds.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_lift import _hac_t  # noqa: E402
from run_qlib_baselines import PROTOCOLS, TOPK, _backtest  # noqa: E402

DAYS = 238  # Qlib risk_analysis annualisation for daily returns


def _runs(root: Path, patterns: str) -> list[list[Path]]:
    groups = []
    for pattern in patterns.split(","):
        runs = sorted(p for p in root.iterdir() if fnmatch.fnmatch(p.name, pattern)
                      and (p / "predictions.parquet").exists())
        if not runs:
            raise SystemExit(f"no runs match {root}/{pattern}")
        groups.append(runs)
    return groups


def _rank(path: Path) -> pd.Series:
    score = pd.read_parquet(path / "predictions.parquet").set_index(["datetime", "instrument"])["score"]
    return score.groupby(level="datetime").rank(pct=True)


def _signal(groups: list[list[Path]], smooth: float) -> pd.Series:
    parts = [pd.concat([_rank(r) for r in runs], axis=1).mean(axis=1) for runs in groups]
    score = pd.concat(parts, axis=1).mean(axis=1).rename("score")
    if smooth > 0:
        wide = score.unstack("instrument")
        ema = wide.ewm(halflife=smooth, ignore_na=True).mean().where(wide.notna())
        score = ema.stack().rename("score")
    return score.sort_index()


def _daily_top(score: pd.Series, label: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"s": score}).join(label.rename("y"), how="inner").dropna()

    def top(day: pd.DataFrame) -> float:
        return (day.nlargest(TOPK, "s")["y"].mean() - day["y"].mean()) * 1e4
    return frame.groupby(level="datetime").apply(top)


def _daily_ric(score: pd.Series, label: pd.Series) -> pd.Series:
    frame = pd.DataFrame({"s": score}).join(label.rename("y"), how="inner").dropna()
    return frame.groupby(level="datetime").apply(lambda g: g["s"].corr(g["y"], method="spearman"))


def _summary(daily: pd.DataFrame) -> dict[str, float]:
    excess = daily["excess"].to_numpy()
    return {
        "dates": int(len(daily)),
        "annualized_excess": float(excess.mean() * DAYS),
        "information_ratio": float(excess.mean() / excess.std() * math.sqrt(DAYS)),
        "gross_information_ratio": float(daily["gross"].mean() / daily["gross"].std() * math.sqrt(DAYS)),
        "turnover_mean": float(daily["turnover"].mean()),
        "top_spread_bps": float(daily["top"].mean()),
        "RankIC": float(daily["ric"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--window", action="append", required=True, help="protocol=root")
    parser.add_argument("--book", action="append", required=True, help="name=glob[,glob...]")
    parser.add_argument("--smooth", default="0", help="comma-separated EMA half-lives (0 = none)")
    parser.add_argument("--n-drop", default="5", help="comma-separated TopkDropout n_drop values")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import qlib

    qlib.init(provider_uri=str(args.provider.expanduser()), region="cn")
    label = pd.read_parquet(args.label)
    label["instrument"] = label["instrument"].str.upper()
    label = label.set_index(["datetime", "instrument"])["label"]
    windows = dict(w.split("=", 1) for w in args.window)
    books = dict(b.split("=", 1) for b in args.book)
    smooths = [float(h) for h in args.smooth.split(",")]
    drops = [int(n) for n in args.n_drop.split(",")]

    daily: dict[str, pd.DataFrame] = {}
    for name, patterns in books.items():
        for h in smooths:
            for n_drop in drops:
                key = name + (f"@ema{h:g}" if h else "") + (f"@drop{n_drop}" if n_drop != 5 else "")
                per_window = []
                for protocol, root in windows.items():
                    split = PROTOCOLS[protocol]
                    score = _signal(_runs(Path(root), patterns), h)
                    _, report = _backtest(score, args.universe, split, n_drop=n_drop, return_report=True)
                    frame = pd.DataFrame({
                        "gross": report["return"] - report["bench"],
                        "excess": report["return"] - report["bench"] - report["cost"],
                        "turnover": report["turnover"]})
                    frame = frame.join(_daily_top(score, label).rename("top")).join(
                        _daily_ric(score, label).rename("ric"))
                    per_window.append(frame.assign(window=protocol))
                daily[key] = pd.concat(per_window)
                print(key, json.dumps(_summary(daily[key])), flush=True)

    result = {"windows": list(windows), "books": books, "results": {}}
    base = daily.get(args.baseline) if args.baseline else None
    for key, frame in daily.items():
        entry = _summary(frame)
        entry["by_window"] = {w: _summary(g) for w, g in frame.groupby("window")}
        if base is not None and key != args.baseline:
            both = frame.join(base, rsuffix="_base", how="inner")
            diff = (both["excess"] - both["excess_base"]).to_numpy()
            top = (both["top"] - both["top_base"]).to_numpy()
            entry["vs_baseline"] = {
                "annualized_excess_diff": float(diff.mean() * DAYS), "excess_diff_t": _hac_t(diff),
                "top_spread_diff_bps": float(np.nanmean(top)), "top_spread_diff_t": _hac_t(top),
                "RankIC_diff": float((both["ric"] - both["ric_base"]).mean()),
                "RankIC_diff_t": _hac_t((both["ric"] - both["ric_base"]).to_numpy()),
            }
        result["results"][key] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
