"""Fetch daily Northbound (Stock Connect) holdings per A-share from Eastmoney (AkShare).

Run in an environment with ``akshare``::

    bs-venv/bin/python -I scripts/qlib_cn/fetch_northbound.py --universes csi300,csi500,csi1000 \\
        --out ~/fm-data/northbound/northbound.parquet

``stock_hsgt_individual_em(symbol)`` returns one stock's whole history of
Northbound holdings (持股日期, 持股数量, 持股市值, 持股数量占A股百分比, 今日增持股数,
今日增持资金, 今日持股市值变化) from 2017-03-17, when HKEX began publishing
per-stock Connect holdings, until 2024-08-16, when it stopped the daily
disclosure. Names are every member a Qlib universe file lists; raw tables are
cached per stock under ``<out dir>/raw`` so the fetch resumes.

Point in time: HKEX publishes day t's holdings after the close of t (CCASS
shareholding snapshot) and Eastmoney mirrors it the same evening or the next
morning; consumers shift the series by one trading day so a feature on day t
uses holdings up to t-1.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

COLUMNS = {"持股日期": "date", "持股数量": "hold_qty", "持股市值": "hold_value", "持股数量占A股百分比": "hold_pct",
           "今日增持股数": "chg_qty", "今日增持资金": "chg_value", "今日持股市值变化": "chg_mv"}


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


def _members(provider: Path, universes: list[str]) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for universe in universes:  # in the order given: the first universe's names come first
        for line in sorted((provider / "instruments" / f"{universe}.txt").read_text().splitlines()):
            code = line.split()[0] if line.strip() else ""
            if code and code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", type=Path, default=Path("~/fm-data/qlib_cn"))
    parser.add_argument("--universes", default="csi300,csi500,csi1000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--reverse", action="store_true", help="walk the names from the end (a second worker)")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    out = args.out.expanduser()
    raw = out.parent / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    members = _members(args.provider.expanduser(), args.universes.split(","))
    print(f"{len(members)} names", flush=True)
    if args.reverse:
        members = members[::-1]
    if not args.merge_only:
        import akshare as ak

        for n, asset in enumerate(members):
            path = raw / f"{asset}.parquet"
            if path.exists():
                continue
            table = None
            for attempt in range(3):
                try:
                    table = _call(lambda: ak.stock_hsgt_individual_em(symbol=asset[2:]))
                    break
                except TypeError:  # Eastmoney answers an unknown or delisted code with no data
                    table = pd.DataFrame()
                    break
                except Exception as exc:  # noqa: BLE001 - network failures
                    print(f"{asset}: {exc!r}"[:200], flush=True)
                    time.sleep(10 * (attempt + 1))
            if table is None:
                continue
            table = table[[c for c in COLUMNS if c in table]].rename(columns=COLUMNS) if len(table) else pd.DataFrame()
            table.to_parquet(path, index=False)
            if n % 50 == 0:
                print(f"{n}/{len(members)} {asset}: {len(table)} rows", flush=True)
            time.sleep(args.pause)
    frames = []
    for path in sorted(raw.glob("*.parquet")):
        table = pd.read_parquet(path)
        if table.empty:
            continue
        table["asset_id"] = path.stem
        frames.append(table)
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    for column in frame.columns.drop(["date", "asset_id"]):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.sort_values(["asset_id", "date"]).to_parquet(out, index=False)
    print(f"{out}: {len(frame)} rows, {frame['asset_id'].nunique()} names, "
          f"{frame['date'].min().date()}..{frame['date'].max().date()}")


if __name__ == "__main__":
    main()
