"""Float market caps from BaoStock daily bars, and how well they replicate the CSI indices.

    $QPY -I scripts/qlib_cn/float_cap.py build --cache ~/fm-data/baostock/cache --out ~/fm-data/float_cap.parquet
    $QPY -I scripts/qlib_cn/float_cap.py replicate --universe csi300

``build``: BaoStock's ``turn`` is the day's volume as a percentage of the
float (circulating A) shares, so ``volume / (turn / 100)`` is the float share
count. It is known at the close, carried forward over suspensions (no volume),
and smoothed with a rolling 20-day median against the rounding of ``turn`` on
low-turnover days. ``float_cap`` = unadjusted close x float shares (CNY).

``replicate``: weights proportional to the previous close's float cap over the
day's index members (Qlib instruments), the weighted member return against the
index's own return. CSI indices weight by free-float cap with banded
free-float ratios, which the circulating share count only approximates; the
tracking error says how far a benchmark-relative book built on these weights
can be trusted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qlib_baselines import BENCHMARK_INDEX  # noqa: E402

DAYS = 238


def build(args) -> None:
    frames = []
    for path in sorted(args.cache.glob("*.daily.parquet")):
        daily = pd.read_parquet(path, columns=["date", "close", "volume", "turn"])
        if daily.empty:
            continue
        daily = daily.apply(pd.to_numeric, errors="coerce").assign(date=pd.to_datetime(daily["date"]))
        shares = (daily["volume"] / (daily["turn"] / 100)).where((daily["volume"] > 0) & (daily["turn"] > 0))
        shares = shares.ffill().rolling(20, min_periods=1).median()
        frames.append(pd.DataFrame({"datetime": daily["date"], "instrument": path.name.split(".")[0],
                                    "float_shares": shares, "float_cap": daily["close"] * shares}))
    out = pd.concat(frames, ignore_index=True).dropna().set_index(["datetime", "instrument"]).sort_index()
    out.to_parquet(args.out)
    print(f"{args.out}: {len(out)} rows, {out.index.get_level_values('instrument').nunique()} instruments, "
          f"{out.index.get_level_values('datetime').min().date()} .. {out.index.get_level_values('datetime').max().date()}")


def weights(caps: pd.DataFrame, universe: str, start: str, end: str) -> pd.Series:
    """Index weight per (datetime, instrument): previous close's float cap over the day's members."""
    from qlib.data import D

    members = D.features(D.instruments(universe), ["$close"], start_time=start, end_time=end).index.swaplevel()
    cap = caps["float_cap"].unstack("instrument").shift(1).stack().rename("cap")
    cap = cap.reindex(members.sort_values()).dropna()
    return (cap / cap.groupby(level="datetime").transform("sum")).rename("weight")


def replicate(args) -> None:
    from qlib.data import D

    caps = pd.read_parquet(args.caps)
    start, end = args.start, args.end
    w = weights(caps, args.universe, start, end)
    ret = D.features(D.instruments(args.universe), ["$close/Ref($close, 1) - 1"], start_time=start,
                     end_time=end).iloc[:, 0].swaplevel().sort_index()
    joined = pd.concat([w, ret.rename("r")], axis=1, join="inner").dropna()
    coverage = joined["weight"].groupby(level="datetime").sum()
    port = (joined["weight"] * joined["r"]).groupby(level="datetime").sum() / coverage
    ew = joined["r"].groupby(level="datetime").mean()
    index = D.features([BENCHMARK_INDEX[args.universe]], ["$close/Ref($close, 1) - 1"], start_time=start,
                       end_time=end).iloc[:, 0].droplevel(0)
    names = w.groupby(level="datetime").size()
    member_count = D.features(D.instruments(args.universe), ["$close"], start_time=start, end_time=end) \
        .groupby(level="datetime").size()
    print(f"{args.universe} {start}..{end}: members with a float cap {names.mean():.0f} of "
          f"{member_count.mean():.0f} a day")
    for label, series in {"float-cap weights": port, "equal weights": ew}.items():
        diff = (series - index).dropna()
        by_year = diff.groupby(diff.index.year)
        te = by_year.std() * np.sqrt(DAYS)
        print(f"  {label:17s} tracking error {diff.std() * np.sqrt(DAYS):.2%}/yr, mean {diff.mean() * DAYS:+.2%}/yr, "
              f"corr {series.corr(index):.4f}; by year TE " + " ".join(f"{y}:{v:.1%}" for y, v in te.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build", "replicate"])
    parser.add_argument("--cache", type=Path, default=Path("~/fm-data/baostock/cache").expanduser())
    parser.add_argument("--out", type=Path, default=Path("~/fm-data/float_cap.parquet").expanduser())
    parser.add_argument("--caps", type=Path, default=Path("~/fm-data/float_cap.parquet").expanduser())
    parser.add_argument("--provider", type=Path, default=Path("~/fm-data/qlib_cn"))
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()
    if args.command == "replicate":
        import qlib

        qlib.init(provider_uri=str(args.provider.expanduser()), region="cn")
    {"build": build, "replicate": replicate}[args.command](args)


if __name__ == "__main__":
    main()
