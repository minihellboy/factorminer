"""Fetch shareholder counts (股东户数) per report date for all A-shares from Eastmoney (AkShare).

Run in an environment with ``akshare``::

    bs-venv/bin/python -I scripts/qlib_cn/fetch_holders.py --start 2009 --out ~/fm-data/eastmoney/holders.parquet

``stock_zh_a_gdhs(symbol=<quarter end>)`` lists every company's shareholder
count at that date and at the previous one, the change, and the
announcement date (公告日期) of the figure. Eastmoney keeps the announcement
that carries the figure, which can be later than its first disclosure (a
2017-12-31 count dated 2018-08-31): late, so conservative, never early.
Raw tables are cached per quarter under ``<out dir>/holders_raw``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

COLUMNS = {"代码": "code", "股东户数-本次": "holders", "股东户数-上次": "holders_prev",
           "股东户数统计截止日-本次": "stat_date", "股东户数统计截止日-上次": "stat_date_prev",
           "公告日期": "pub_date"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=2009)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import akshare as ak

    raw = args.out.expanduser().parent / "holders_raw"
    raw.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(args.start, args.end + 1):
        for md in ("0331", "0630", "0930", "1231"):
            quarter = f"{year}{md}"
            path = raw / f"{quarter}.parquet"
            if not path.exists():
                try:
                    table = ak.stock_zh_a_gdhs(symbol=quarter)
                except Exception as exc:  # noqa: BLE001 - quarters not yet reported raise
                    print(f"{quarter}: {exc!r}"[:200], flush=True)
                    continue
                table[list(COLUMNS)].rename(columns=COLUMNS).astype(str).to_parquet(path, index=False)
                print(f"{quarter}: {len(table)} rows", flush=True)
                time.sleep(1)
            frames.append(pd.read_parquet(path))
    frame = pd.concat(frames, ignore_index=True)
    frame["code"] = frame["code"].str.zfill(6)
    frame["asset_id"] = frame["code"].map(lambda c: ("SH" if c[0] in "69" else "BJ" if c[0] in "48" else "SZ") + c)
    for column in ("holders", "holders_prev"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("stat_date", "stat_date_prev", "pub_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.drop(columns="code").drop_duplicates(["asset_id", "stat_date"])
    frame.to_parquet(args.out.expanduser(), index=False)
    print(f"{args.out}: {len(frame)} rows, {frame['asset_id'].nunique()} companies")


if __name__ == "__main__":
    main()
