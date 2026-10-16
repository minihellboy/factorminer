"""Fill float-share inputs (close, volume, turnover rate) from Sina or Eastmoney daily bars (AkShare).

Run in an environment with ``akshare``::

    bs-venv/bin/python -I scripts/qlib_cn/fetch_float_inputs.py --assets missing.txt \\
        --cache ~/fm-data/baostock_csi1000/cache --start 20160601 --end 20261013

For instruments BaoStock could not serve, writes ``<asset>.daily.parquet`` in
the BaoStock cache layout (``date, code, close, volume, turn``; volume in
shares, turn in % of float shares) so ``float_cap.py build`` reads both.
``--source sina`` (default, ``stock_zh_a_daily``) reports the outstanding
float shares themselves, stored as turn = 100 x volume / shares so that
``volume / (turn / 100)`` returns them exactly. ``--source em``
(``stock_zh_a_hist``) has a two-decimal turnover rate (relative error under
1% above 0.5% turnover). Serial with a pause: the free endpoints blacklist
or throttle parallel and fast clients.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--assets", type=Path, required=True, help="file of Qlib codes, one per line")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--start", default="20160601")
    parser.add_argument("--end", default="20261013")
    parser.add_argument("--source", choices=["sina", "em"], default="sina")
    parser.add_argument("--pause", type=float, default=0.5)
    args = parser.parse_args()
    import akshare as ak

    cache = args.cache.expanduser()
    assets = [a.strip() for a in args.assets.read_text().split() if a.strip()]
    for n, asset in enumerate(assets, 1):
        target = cache / f"{asset}.daily.parquet"
        if target.exists():
            continue
        for attempt in range(3):
            try:
                if args.source == "sina":
                    bars = ak.stock_zh_a_daily(symbol=asset.lower(), start_date=args.start, end_date=args.end,
                                               adjust="")
                else:
                    bars = ak.stock_zh_a_hist(symbol=asset[2:], period="daily", start_date=args.start,
                                              end_date=args.end, adjust="")
                break
            except Exception as exc:  # noqa: BLE001 - network errors; back off and retry
                print(f"{asset}: {exc!r}"[:160], flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            continue
        if args.source == "sina":
            out = bars[["date", "close", "volume"]].assign(
                turn=100 * bars["volume"] / bars["outstanding_share"])
        else:
            out = bars.rename(columns={"日期": "date", "收盘": "close", "成交量": "volume", "换手率": "turn"})
            out = out[["date", "close", "volume", "turn"]].assign(volume=lambda f: f["volume"] * 100)
        out = out.assign(code=asset)
        out.astype(str).to_parquet(target, index=False)
        (cache / f"{asset}.parquet").write_bytes(b"")
        if n % 50 == 0:
            print(f"[{n}/{len(assets)}] {asset}: {len(out)} days", flush=True)
        time.sleep(args.pause)
    print("done", flush=True)


if __name__ == "__main__":
    main()
