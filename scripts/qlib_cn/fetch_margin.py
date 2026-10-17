"""Fetch per-stock margin financing and securities lending balances by trading day (SSE + SZSE).

Run in an environment with ``akshare``::

    bs-venv/bin/python -I scripts/qlib_cn/fetch_margin.py --start 2017-01-01 --out ~/fm-data/margin/margin.parquet

Each exchange publishes the previous session's per-security margin table
every morning: ``stock_margin_detail_sse(date)`` (融资余额, 融资买入额, 融资偿还额,
融券余量, 融券卖出量, 融券偿还量) and ``stock_margin_detail_szse(date)``
(融资买入额, 融资余额, 融券卖出量, 融券余量, 融券余额, 融资融券余额). Raw tables are
cached per exchange and day under ``<out dir>/raw``; the merged panel has one
row per (asset_id, date) with the common columns ``fin_balance``, ``fin_buy``,
``short_qty``, ``short_sell`` and the exchange-specific extras where present.

Point in time: a table for trading day t is published on the morning of t+1,
so a feature on day t may use tables up to t-1 (``data_factors``-style
consumers shift by one trading day). Trading days come from the Qlib
calendar file; a day the exchange has no table for (holiday, outage) is
recorded as an empty cache file and skipped.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

SSE = {"标的证券代码": "code", "融资余额": "fin_balance", "融资买入额": "fin_buy", "融资偿还额": "fin_repay",
       "融券余量": "short_qty", "融券卖出量": "short_sell", "融券偿还量": "short_repay"}
SZSE = {"证券代码": "code", "融资买入额": "fin_buy", "融资余额": "fin_balance", "融券卖出量": "short_sell",
        "融券余量": "short_qty", "融券余额": "short_balance", "融资融券余额": "margin_balance"}


class _TimeoutError(Exception):
    pass


def _call(fn, seconds: int = 90):
    """Run ``fn()`` with a SIGALRM deadline: AkShare requests carry no timeout and can hang."""
    import signal

    def _raise(signum, frame):
        raise _TimeoutError(f"no answer in {seconds}s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _fetch(ak, exchange: str, day: str) -> pd.DataFrame:
    if exchange == "sse":
        table = _call(lambda: ak.stock_margin_detail_sse(date=day))
        columns = SSE
    else:
        table = _call(lambda: ak.stock_margin_detail_szse(date=day))
        columns = SZSE
    if table is None or table.empty:
        return pd.DataFrame()
    table = table[[c for c in columns if c in table]].rename(columns=columns)
    table["code"] = table["code"].astype(str).str.zfill(6)
    for column in table.columns.drop("code"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--calendar", type=Path, default=Path("~/fm-data/qlib_cn/calendars/day.txt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pause", type=float, default=1.5)
    parser.add_argument("--exchanges", default="sse,szse", help="run one process per exchange to fetch in parallel")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    out = args.out.expanduser()
    raw = out.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    days = pd.to_datetime(args.calendar.expanduser().read_text().split())
    days = days[(days >= pd.Timestamp(args.start)) & (days <= pd.Timestamp(args.end or "2100-01-01"))]
    if not args.merge_only:
        import akshare as ak

        for day in days:
            for exchange in args.exchanges.split(","):
                path = raw / f"{exchange}_{day:%Y%m%d}.parquet"
                if path.exists():
                    continue
                for attempt in range(3):
                    try:
                        table = _fetch(ak, exchange, f"{day:%Y%m%d}")
                        break
                    except Exception as exc:  # noqa: BLE001 - network and endpoint failures
                        print(f"{exchange} {day.date()}: {exc!r}"[:200], flush=True)
                        table = None
                        time.sleep(10 * (attempt + 1))
                if table is None:
                    continue
                table.to_parquet(path, index=False)
                print(f"{exchange} {day.date()}: {len(table)} rows", flush=True)
                time.sleep(args.pause)
    frames = []
    for path in sorted(raw.glob("*.parquet")):
        table = pd.read_parquet(path)
        if table.empty:
            continue
        exchange, day = path.stem.split("_")
        table["date"] = pd.Timestamp(day)
        table["asset_id"] = ("SH" if exchange == "sse" else "SZ") + table["code"]
        frames.append(table.drop(columns="code"))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["asset_id"].str[2].isin(list("036"))]  # A-shares only: 60x, 00x, 30x (688 keeps STAR)
    frame.sort_values(["asset_id", "date"]).to_parquet(out, index=False)
    print(f"{out}: {len(frame)} rows, {frame['asset_id'].nunique()} names, "
          f"{frame['date'].min().date()}..{frame['date'].max().date()}")


if __name__ == "__main__":
    main()
