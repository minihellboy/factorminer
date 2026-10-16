"""Fetch the history of institutional site visits (机构调研) for all A-shares from Eastmoney (AkShare).

Run in an environment with ``akshare``::

    bs-venv/bin/python -I scripts/qlib_cn/fetch_site_visits.py --since 20100101 --out ~/fm-data/eastmoney/site_visits.parquet

``stock_jgdy_tj_em(date)`` returns every visit announced from ``date`` on: the
company, the number of institutions received (接待机构数量), the reception
type, the visit date (接待日期) and the announcement date (公告日期). A visit
is public on its announcement date, which is the date a feature may use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = {"代码": "code", "接待机构数量": "institutions", "接待方式": "kind", "接待日期": "visit_date",
           "公告日期": "pub_date"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", default="20100101")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import akshare as ak

    raw = ak.stock_jgdy_tj_em(date=args.since)
    frame = raw[list(COLUMNS)].rename(columns=COLUMNS)
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    frame["asset_id"] = frame["code"].map(lambda c: ("SH" if c[0] in "69" else "BJ" if c[0] in "48" else "SZ") + c)
    for column in ("visit_date", "pub_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["institutions"] = pd.to_numeric(frame["institutions"], errors="coerce")
    frame.drop(columns="code").to_parquet(args.out.expanduser(), index=False)
    print(f"{args.out}: {len(frame)} visits, announced {frame['pub_date'].min()} .. {frame['pub_date'].max()}")


if __name__ == "__main__":
    main()
