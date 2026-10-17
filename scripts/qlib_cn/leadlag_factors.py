"""Industry lead-lag features: what the largest names of a stock's industry did recently.

    $QPY -I scripts/qlib_cn/leadlag_factors.py --universe csi300 --out output/qlib_cn/data_factors/csi300_leadlag.parquet

The industry is point in time from the Eastmoney quarterly table
(``fetch_em_fundamentals.py``): the industry of the latest report published
before the day. On each day, within each industry of the universe's members,
the three largest names by float capitalisation (``float_cap.py``) are the
leaders. Features for every member row, all from past closes (known at the
close of day t):

* ``ll_lead5``, ``ll_lead20``: the cap-weighted 5- and 20-day return of the
  industry leaders other than the stock itself;
* ``ll_gap5``: ``ll_lead5`` minus the stock's own 5-day return (the stock
  lags its leaders when positive);
* ``ll_ind5``: the equal-weight 5-day return of the other members of the
  industry (NaN with fewer than three peers);
* ``ll_size_rank``: the stock's float-cap rank within its industry that day
  (0 = largest), the follower side of the lead-lag hypothesis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_fit import _init, _load_index  # noqa: E402

LEADERS = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", type=Path, default=Path("~/fm-data/qlib_cn"))
    parser.add_argument("--cache", type=Path, default=Path("~/fm-data/cache"))
    parser.add_argument("--universe", required=True)
    parser.add_argument("--start", default="2009-01-01")
    parser.add_argument("--quarterly", type=Path, default=Path("~/fm-data/eastmoney/quarterly.parquet"))
    parser.add_argument("--float-cap", type=Path, default=Path("~/fm-data/float_cap.parquet"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from qlib.data import D

    _init(args.provider)
    rows = _load_index(args.cache, args.universe, args.start)
    dates = pd.DatetimeIndex(rows.get_level_values("datetime").unique()).sort_values()
    assets = pd.Index(rows.get_level_values("instrument").unique()).sort_values()
    first = str((dates[0] - pd.Timedelta(days=60)).date())
    close = D.features(list(assets), ["$close"], start_time=first, end_time=str(dates[-1].date()))
    close = close["$close"].swaplevel().sort_index().unstack("instrument").reindex(columns=assets)
    ret5 = (close / close.shift(5) - 1).reindex(dates)
    ret20 = (close / close.shift(20) - 1).reindex(dates)

    frame = pd.DataFrame(index=rows).sort_index()
    frame["ret5"] = ret5.stack(future_stack=True).reindex(rows)
    frame["ret20"] = ret20.stack(future_stack=True).reindex(rows)
    cap = pd.read_parquet(args.float_cap.expanduser(), columns=["float_cap"])["float_cap"]
    frame["cap"] = cap.reindex(rows)

    reports = pd.read_parquet(args.quarterly.expanduser(), columns=["asset_id", "pubDate", "industry"])
    reports = reports.dropna().rename(columns={"asset_id": "instrument"})
    reports["pubDate"] = pd.to_datetime(reports["pubDate"]).astype("datetime64[ns]")
    reports = reports.sort_values("pubDate")
    long = frame.reset_index().sort_values("datetime")
    long["datetime"] = long["datetime"].astype("datetime64[ns]")
    long = pd.merge_asof(long, reports, left_on="datetime", right_on="pubDate", by="instrument",
                         allow_exact_matches=False)  # a report counts from the day after publication
    long = long.set_index(["datetime", "instrument"]).sort_index()

    group = long.dropna(subset=["industry", "cap"]).groupby(["datetime", "industry"])
    rank = group["cap"].rank(ascending=False, method="first")
    leader = rank <= LEADERS
    lead = long.loc[leader.index[leader]].copy()
    lead["w5"] = lead["cap"] * lead["ret5"]
    lead["w20"] = lead["cap"] * lead["ret20"]
    sums = lead.groupby(["datetime", "industry"])[["cap", "w5", "w20"]].sum().rename(
        columns={"cap": "sum_cap", "w5": "sum_w5", "w20": "sum_w20"})
    long = long.join(sums, on=["datetime", "industry"])
    own_cap = np.where(leader.reindex(long.index, fill_value=False), long["cap"].fillna(0.0), 0.0)
    own5 = own_cap * long["ret5"].fillna(0.0)
    own20 = own_cap * long["ret20"].fillna(0.0)
    denom = long["sum_cap"] - own_cap
    out = pd.DataFrame(index=long.index)
    out["ll_lead5"] = (long["sum_w5"] - own5) / denom.where(denom > 0)
    out["ll_lead20"] = (long["sum_w20"] - own20) / denom.where(denom > 0)
    out["ll_gap5"] = out["ll_lead5"] - long["ret5"]
    peers = long.groupby(["datetime", "industry"])["ret5"]
    n = peers.transform("count")
    out["ll_ind5"] = ((peers.transform("sum") - long["ret5"].fillna(0.0)) / (n - 1)).where(n >= 4)
    out["ll_size_rank"] = (rank - 1).reindex(long.index)
    out = out.reindex(rows).astype(np.float32).replace([np.inf, -np.inf], np.nan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"{args.out}: {out.shape}; industries/day {long.dropna(subset=['industry']).groupby(level='datetime')['industry'].nunique().median():.0f}; "
          "coverage " + ", ".join(f"{c} {out[c].notna().mean():.2f}" for c in out))


if __name__ == "__main__":
    main()
