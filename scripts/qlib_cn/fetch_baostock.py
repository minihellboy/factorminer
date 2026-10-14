"""Fetch daily valuation and point-in-time quarterly fundamentals from BaoStock.

Run in an environment with ``baostock``, ``pandas`` and ``pyarrow`` (it is not
the FactorMiner environment)::

    bs-venv/bin/python -I scripts/qlib_cn/fetch_baostock.py \\
        --panel ~/fm-data/panels/csi300_2007_2026.parquet --out ~/fm-data/baostock \\
        --start 2007-01-01 --end 2026-10-13 --workers 6

BaoStock (www.baostock.com) is a free A-share data service. Per instrument of
the panel it downloads

* daily unadjusted ``close``, ``volume``, ``amount``, ``turn`` (turnover rate, % of float shares), ``peTTM``,
  ``pbMRQ``, ``psTTM``, ``pcfNcfTTM`` and ``isST``;
* quarterly reports from the profit, growth, balance, cash-flow and operation
  tables, each stamped with its publication date (``pubDate``), so a panel can
  take the figures point in time.

It also downloads the CSI300 and CSI500 index bars. Each instrument's result is
cached under ``<out>/cache`` so an interrupted run resumes. BaoStock sessions
are not thread-safe; ``--workers`` runs that many processes, each with one
login for its lifetime. The outputs are ``daily.parquet``, ``quarterly.parquet`` and
``index.parquet`` under ``--out``, plus ``manifest.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

DAILY_FIELDS = "date,code,close,volume,amount,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
QUARTERLY_TABLES = ("profit", "growth", "balance", "cash_flow", "operation")
INDEXES = {"sh.000300": "CSI300", "sh.000905": "CSI500"}


def _bs_code(asset_id: str) -> str:
    """``SH600000`` -> ``sh.600000``."""
    return f"{asset_id[:2].lower()}.{asset_id[2:]}"


def _rows(rs) -> pd.DataFrame:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"baostock error {rs.error_code}: {rs.error_msg}")
    return pd.DataFrame(rows, columns=rs.fields)


def _quarters(start: str, end: str):
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    for year in range(first.year, last.year + 1):
        for quarter in (1, 2, 3, 4):
            if pd.Timestamp(year=year, month=3 * quarter, day=1) <= last + pd.offsets.QuarterEnd():
                yield year, quarter


def _fetch_one(job: tuple[str, str, str, str, bool]) -> tuple[str, str]:
    """Download one instrument (daily + quarterly) into the cache; returns (asset, status)."""
    asset_id, start, end, cache, quarterly_tables = job
    import baostock as bs

    target = Path(cache) / f"{asset_id}.parquet"
    if target.exists():
        return asset_id, "cached"
    code = _bs_code(asset_id)
    for attempt in range(3):
        try:
            daily = _rows(bs.query_history_k_data_plus(
                code, DAILY_FIELDS, start_date=start, end_date=end, frequency="d", adjustflag="3"))
            frames = []
            for year, quarter in _quarters(start, end) if quarterly_tables else ():
                merged = None
                for table in QUARTERLY_TABLES:
                    part = _rows(getattr(bs, f"query_{table}_data")(code=code, year=year, quarter=quarter))
                    if part.empty:
                        continue
                    part = part.drop(columns=["code"])
                    merged = part if merged is None else merged.merge(
                        part, on=["pubDate", "statDate"], how="outer")
                if merged is not None:
                    frames.append(merged.assign(code=code))
            quarterly = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            daily.to_parquet(target.with_suffix(".daily.parquet"), index=False)
            quarterly.to_parquet(target.with_suffix(".quarterly.parquet"), index=False)
            target.write_bytes(b"")  # completion marker
            return asset_id, f"ok daily={len(daily)} quarterly={len(quarterly)}"
        except Exception as exc:  # noqa: BLE001 - network errors; re-login and retry
            error = str(exc)
            time.sleep(5 * (attempt + 1))
            _login()
    return asset_id, f"FAILED {error}"


def _login() -> None:
    """One BaoStock session per worker process (sessions are not thread-safe)."""
    import baostock as bs

    for _ in range(5):
        if bs.login().error_code == "0":
            return
        time.sleep(3)
    raise RuntimeError("baostock login failed")


def _fetch_indexes(start: str, end: str) -> pd.DataFrame:
    import baostock as bs

    bs.login()
    frames = []
    for code, name in INDEXES.items():
        frame = _rows(bs.query_history_k_data_plus(
            code, "date,code,open,high,low,close,volume,amount", start_date=start, end_date=end,
            frequency="d"))
        frames.append(frame.assign(index=name))
    bs.logout()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, action="append", required=True,
                        help="FactorMiner panel(s) whose instruments to fetch")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default=dt.date.today().isoformat())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quarterly", action="store_true",
                        help="also fetch the quarterly report tables (slow endpoint)")
    args = parser.parse_args()

    assets: list[str] = []
    for panel in args.panel:
        ids = pd.read_parquet(panel, columns=["asset_id"])["asset_id"].unique()
        assets += [a for a in ids if a not in assets]
    assets = assets  # panel order: first panel's instruments are fetched first
    cache = args.out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"{len(assets)} instruments, {args.workers} workers", flush=True)

    started = time.monotonic()
    jobs = [(a, args.start, args.end, str(cache), args.quarterly) for a in assets]
    failed = []
    with Pool(args.workers, initializer=_login) as pool:
        for n, (asset, status) in enumerate(pool.imap_unordered(_fetch_one, jobs), 1):
            if status.startswith("FAILED"):
                failed.append(asset)
            if n % 25 == 0 or status.startswith("FAILED"):
                print(f"[{n}/{len(assets)} {time.monotonic() - started:.0f}s] {asset}: {status}",
                      flush=True)

    daily = pd.concat([pd.read_parquet(cache / f"{a}.daily.parquet") for a in assets
                       if (cache / f"{a}.daily.parquet").exists()], ignore_index=True)
    quarterly = pd.concat([f for a in assets if (cache / f"{a}.quarterly.parquet").exists()
                           and not (f := pd.read_parquet(cache / f"{a}.quarterly.parquet")).empty],
                          ignore_index=True)
    for frame in (daily, quarterly):
        for column in frame.columns:
            if column not in {"date", "code", "pubDate", "statDate"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    daily.to_parquet(args.out / "daily.parquet", index=False)
    quarterly.to_parquet(args.out / "quarterly.parquet", index=False)
    index = _fetch_indexes(args.start, args.end)
    index.to_parquet(args.out / "index.parquet", index=False)
    (args.out / "manifest.json").write_text(json.dumps({
        "source": "baostock", "fetched": dt.datetime.now().isoformat(timespec="seconds"),
        "start": args.start, "end": args.end, "instruments": len(assets), "failed": failed,
        "daily_rows": int(len(daily)), "quarterly_rows": int(len(quarterly)),
        "daily_fields": DAILY_FIELDS.split(","),
        "quarterly_tables": list(QUARTERLY_TABLES) if args.quarterly else [],
    }, indent=2) + "\n")
    print(f"done: daily {len(daily)} rows, quarterly {len(quarterly)} rows, "
          f"{len(failed)} failed, {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
