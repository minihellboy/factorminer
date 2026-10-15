"""Fetch point-in-time quarterly report figures and industries for all A-shares from Eastmoney (AkShare).

Run in an environment with ``akshare``, ``pandas`` and ``pyarrow``::

    bs-venv/bin/python -I scripts/qlib_cn/fetch_em_fundamentals.py --out ~/fm-data/eastmoney \\
        --start 2006 --end 2026

Two whole-market tables per report quarter, so the fetch is about 80 paged
requests per year, run serially with a pause between quarters:

* ``stock_yjbb_em`` (业绩报表): EPS, revenue and net income with their
  year-on-year and quarter-on-quarter growth, book value and operating cash
  flow per share, ROE, gross margin, and the Eastmoney industry;
* ``stock_yysj_em`` (预约披露时间): each report's actual disclosure date.

A report counts as published on its actual disclosure date (53% of reports).
Otherwise: on the table's latest-announcement date when that falls by the
statutory deadline (it is at or after first publication), else on the
deadline (Q1 and annual reports by 30 April, half-year by 31 August, Q3 by 31
October; the 2019 annual and 2020 Q1 reports by 30 June 2020 after the
CSRC's COVID extension). Only a report filed after its deadline is dated too
early; using the latest-announcement date instead would delay 43% of reports
by about a year, since it usually marks a later announcement. The figures are
today's table, so a restated report carries its restated values from its
first date: values are not vintage-exact. Raw tables are cached per quarter under ``<out>/raw``. Writes
``<out>/quarterly.parquet`` with ``asset_id, pubDate, statDate, latest_notice`` and the
figures, the industry included (as of each report).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

FIELDS = {"每股收益": "eps", "营业总收入-营业总收入": "revenue", "营业总收入-同比增长": "yoy_rev",
          "净利润-净利润": "net_income", "净利润-同比增长": "yoy_ni", "净利润-季度环比增长": "qoq_ni",
          "每股净资产": "bps", "净资产收益率": "roe", "每股经营现金流量": "cfps", "销售毛利率": "gp_margin",
          "所处行业": "industry", "最新公告日期": "latest_notice"}


def _asset(code: pd.Series) -> pd.Series:
    code = code.astype(str).str.zfill(6)
    prefix = code.str[0].map({"6": "SH", "9": "SH", "0": "SZ", "2": "SZ", "3": "SZ"}).fillna("BJ")
    return prefix + code



def _deadline(stat: pd.Timestamp) -> pd.Timestamp:
    if (stat.year, stat.month) in {(2019, 12), (2020, 3)}:  # CSRC COVID extension
        return pd.Timestamp(2020, 6, 30)
    if stat.month == 12:
        return pd.Timestamp(stat.year + 1, 4, 30)
    return {3: pd.Timestamp(stat.year, 4, 30), 6: pd.Timestamp(stat.year, 8, 31),
            9: pd.Timestamp(stat.year, 10, 31)}[stat.month]


def _fetch(raw: Path, name: str, call, pause: float, optional: bool = False) -> pd.DataFrame:
    """Cached call; an ``optional`` table that keeps failing (no data that early) comes back empty, uncached."""
    target = raw / f"{name}.parquet"
    if target.exists():
        return pd.read_parquet(target)
    attempts = 2 if optional else 4
    for attempt in range(attempts):
        try:
            frame = call()
            break
        except Exception as exc:  # noqa: BLE001 - network; back off and retry
            print(f"{name}: {exc}; retry {attempt + 1}", flush=True)
            time.sleep(10 if optional else 30 * (attempt + 1))
    else:
        if optional:
            return pd.DataFrame()
        raise RuntimeError(f"{name} failed")
    frame = frame.astype({c: str for c in frame.columns if frame[c].dtype == object})
    frame.to_parquet(target, index=False)
    time.sleep(pause)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", type=int, default=2006)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument("--pause", type=float, default=5.0, help="seconds between requests")
    args = parser.parse_args()
    import akshare as ak

    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    frames = []
    today = pd.Timestamp.today()
    for year in range(args.start, args.end + 1):
        for month, day in ((3, 31), (6, 30), (9, 30), (12, 31)):
            stat = pd.Timestamp(year, month, day)
            if stat > today:
                continue
            key = stat.strftime("%Y%m%d")
            report = _fetch(raw, f"yjbb_{key}", lambda: ak.stock_yjbb_em(date=key), args.pause)
            if report.empty:
                continue
            dates = _fetch(raw, f"yysj_{key}", lambda: ak.stock_yysj_em(symbol="沪深A股", date=key), args.pause,
                           optional=True)
            report = report.rename(columns=FIELDS)[list(FIELDS.values()) + ["股票代码"]]
            report["asset_id"] = _asset(report.pop("股票代码"))
            if not dates.empty:
                actual = pd.Series(pd.to_datetime(dates["实际披露时间"], errors="coerce").to_numpy(),
                                   index=_asset(dates["股票代码"]).to_numpy())
                actual = actual[~actual.index.duplicated()]
                report["actual"] = report["asset_id"].map(actual)
            else:
                report["actual"] = pd.NaT
            notice = pd.to_datetime(report["latest_notice"], errors="coerce")
            report["latest_notice"] = notice
            deadline = _deadline(stat)
            report["pubDate"] = report["actual"].fillna(notice.where(notice <= deadline, deadline))
            report["statDate"] = stat
            report["pub_source"] = report["actual"].notna().map({True: "actual", False: "notice_or_deadline"})
            frames.append(report.drop(columns=["actual"]))
            print(f"{key}: {len(report)} rows, actual dates {float((report['pub_source'] == 'actual').mean()):.2f}",
                  flush=True)

    quarterly = pd.concat(frames, ignore_index=True)
    for column in ("eps", "revenue", "yoy_rev", "net_income", "yoy_ni", "qoq_ni", "bps", "roe", "cfps",
                   "gp_margin"):
        quarterly[column] = pd.to_numeric(quarterly[column], errors="coerce")
    quarterly = quarterly.dropna(subset=["pubDate"])
    quarterly.to_parquet(args.out / "quarterly.parquet", index=False)
    (args.out / "manifest.json").write_text(json.dumps({
        "source": "eastmoney via akshare (stock_yjbb_em, stock_yysj_em)",
        "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": int(len(quarterly)),
        "actual_date_share": float((quarterly["pub_source"] == "actual").mean()),
        "quarters": int(quarterly["statDate"].nunique()),
    }, indent=2) + "\n")
    print(f"done: {len(quarterly)} rows", flush=True)


if __name__ == "__main__":
    main()
