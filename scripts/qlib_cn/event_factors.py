"""Earnings-event features (post-announcement drift) from the Eastmoney quarterly reports.

    $QPY -I scripts/qlib_cn/event_factors.py --out ~/fm-data/event_factors.parquet

Per (datetime, instrument), for the latest report published by that day's
close and for ``LIVE`` trading days after it (NaN otherwise):

* ``sue``: standardised unexpected earnings, the single-quarter net income
  minus the same quarter a year earlier, over the standard deviation of the
  last eight such changes (at least four). Single quarters are differenced
  from Eastmoney's year-to-date figures.
* ``ear``: the earnings-announcement return, the name's two-day return after
  the announcement minus the same two days' mean over the instruments here;
  only for reports with an actual disclosure date (53%; the rest are dated on
  the statutory deadline, which is never early but can be late). Known at the
  close of the second day.
* ``age``: trading days since the report counted as known.

A report published on day d (Chinese reports are released outside trading
hours) is known at d's close, so the row dated d carries it and the book
trades it at d+1. Report values are Eastmoney's current table (restated
reports carry restated values from their first date; see fetch_em_fundamentals.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LIVE = 60


def _single_quarter(reports: pd.DataFrame) -> pd.Series:
    cum = reports["net_income"]
    year = reports["statDate"].dt.year
    quarter = reports["statDate"].dt.quarter
    prev = cum.groupby([reports["asset_id"], year]).shift(1)
    prev_q = quarter.groupby([reports["asset_id"], year]).shift(1)
    single = cum - prev
    single[quarter == 1] = cum[quarter == 1]
    single[(quarter > 1) & (prev_q != quarter - 1)] = np.nan
    return single


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, default=Path("~/fm-data/eastmoney/quarterly.parquet"))
    parser.add_argument("--provider", type=Path, default=Path("~/fm-data/qlib_cn"))
    parser.add_argument("--universes", default="csi300,csi500,csi1000")
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import qlib
    from qlib.data import D

    qlib.init(provider_uri=str(args.provider.expanduser()), region="cn")
    names = sorted({n for u in args.universes.split(",")
                    for n in D.list_instruments(D.instruments(u), start_time=args.start, end_time=args.end)})
    ret = D.features(names, ["$close/Ref($close, 1) - 1"], start_time=args.start, end_time=args.end)
    ret = ret.iloc[:, 0].unstack("instrument")
    calendar = ret.index
    abnormal = ret.sub(ret.mean(axis=1), axis=0)
    ear_2d = abnormal.shift(-1).add(abnormal.shift(-2))  # row d: days d+1 and d+2

    reports = pd.read_parquet(args.reports.expanduser())
    reports = reports[reports["asset_id"].isin(names)].sort_values(["asset_id", "statDate", "pubDate"])
    reports = reports.drop_duplicates(["asset_id", "statDate"], keep="first").reset_index(drop=True)
    reports["single"] = _single_quarter(reports)
    change = reports["single"] - reports.groupby("asset_id")["single"].shift(4)
    scale = change.groupby(reports["asset_id"]).transform(lambda c: c.shift(1).rolling(8, min_periods=4).std())
    reports["sue"] = (change / scale).clip(-10, 10)
    # the first trading day at or after pubDate whose close sees the report
    pos = calendar.searchsorted(reports["pubDate"].to_numpy())
    reports = reports[pos < len(calendar)].assign(known=lambda r: calendar[pos[pos < len(calendar)]])
    ear = []
    for asset, day, source in reports[["asset_id", "known", "pub_source"]].itertuples(index=False):
        ear.append(ear_2d.at[day, asset] if source == "actual" and asset in ear_2d.columns else np.nan)
    reports["ear"] = ear
    reports["ear_known"] = calendar[np.minimum(calendar.searchsorted(reports["known"].to_numpy()) + 2,
                                               len(calendar) - 1)]

    columns = {}
    for field, when in (("sue", "known"), ("ear", "ear_known")):
        events = reports.dropna(subset=[field]).sort_values(when).drop_duplicates(["asset_id", when], keep="last")
        wide = events.pivot(index=when, columns="asset_id", values=field).reindex(index=calendar, columns=names)
        stamp = pd.DataFrame(np.where(wide.notna(), np.arange(len(calendar))[:, None], np.nan),
                             index=calendar, columns=names)
        age = np.arange(len(calendar))[:, None] - stamp.ffill()
        columns[field] = wide.ffill().where(age <= LIVE)
        if field == "sue":
            columns["age"] = age.where(age <= LIVE)
    out = pd.concat({k: v.stack(future_stack=True) for k, v in columns.items()}, axis=1)
    out.index.names = ["datetime", "instrument"]
    out = out.dropna(how="all").astype(np.float32)
    out.to_parquet(args.out.expanduser())
    print(f"{args.out}: {len(out)} rows; non-null sue {out['sue'].notna().mean():.0%}, "
          f"ear {out['ear'].notna().mean():.0%}")


if __name__ == "__main__":
    main()
