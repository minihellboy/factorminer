"""Fetch the dragon-tiger list (龙虎榜, exchange disclosure of the day's most active seats) from Eastmoney.

Run in an environment with ``akshare``::

    bs-venv/bin/python -I scripts/qlib_cn/fetch_lhb.py --start 2010 --out ~/fm-data/lhb/lhb.parquet

``stock_lhb_detail_em(start_date, end_date)`` lists every (stock, day, reason)
entry: net seat buying, seat buy and sell amounts, the day's total turnover
and the reason the stock made the list. One call per calendar year is cached
under ``<out dir>/raw``.

Point in time: the exchanges publish the list after the close of day t, so a
feature on day t may use entries dated t (the book trades at the next close).
Consumers that want the conservative convention shift by one day.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

COLUMNS = {"代码": "code", "上榜日": "date", "龙虎榜净买额": "net_buy", "龙虎榜买入额": "buy", "龙虎榜卖出额": "sell",
           "龙虎榜成交额": "seat_turnover", "市场总成交额": "turnover", "净买额占总成交比": "net_buy_share",
           "成交额占总成交比": "seat_share", "换手率": "turnover_rate", "上榜原因": "reason"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.expanduser()
    raw = out.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    import akshare as ak

    frames = []
    for year in range(args.start, args.end + 1):
        path = raw / f"{year}.parquet"
        if not path.exists():
            try:
                table = ak.stock_lhb_detail_em(start_date=f"{year}0101", end_date=f"{year}1231")
            except Exception as exc:  # noqa: BLE001
                print(f"{year}: {exc!r}"[:200], flush=True)
                continue
            table = table[[c for c in COLUMNS if c in table]].rename(columns=COLUMNS)
            table.astype(str).to_parquet(path, index=False)
            print(f"{year}: {len(table)} rows", flush=True)
            time.sleep(2)
        frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True)
    frame["code"] = frame["code"].str.zfill(6)
    frame["asset_id"] = frame["code"].map(lambda c: ("SH" if c[0] in "69" else "BJ" if c[0] in "48" else "SZ") + c)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in COLUMNS.values():
        if column not in ("code", "date", "reason"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop(columns="code").dropna(subset=["date"])
    frame.sort_values(["asset_id", "date"]).to_parquet(out, index=False)
    print(f"{out}: {len(frame)} rows, {frame['asset_id'].nunique()} names, "
          f"{frame['date'].min().date()}..{frame['date'].max().date()}")


if __name__ == "__main__":
    main()
