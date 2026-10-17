"""Daily flow features from margin financing, Northbound holdings and the dragon-tiger list, point in time.

    $QPY -I scripts/qlib_cn/flow_factors.py --universe csi300 --margin ~/fm-data/margin/margin.parquet \\
        --northbound ~/fm-data/northbound/northbound.parquet --lhb ~/fm-data/lhb/lhb.parquet \\
        --out output/qlib_cn/data_factors/csi300_flow.parquet

Every source is shifted by one trading day before use: a figure for day t is
available to a signal on day t+1 at the earliest (the conservative reading of
each publication schedule, see the fetchers). Flows are scaled by the stock's
trailing 20-day mean daily turnover value (Qlib ``$amount``), so a change in a
balance is measured in days of normal trading; levels are ratios to the
float capitalisation (``float_cap.py``) or to the share count.

Features (``datetime, instrument`` rows of the universe's Alpha158 cache):

* margin (SSE and SZSE per-security tables): ``mg_fin_chg5`` and
  ``mg_fin_chg20``, the change in financing balance over 5 and 20 days in
  turnover days; ``mg_fin_level``, the financing balance over the float cap;
  ``mg_buy5``, five-day financing purchases over five-day turnover;
  ``mg_short_chg5``, the change in shares sold short (valued at the close)
  in turnover days;
* Northbound (HKEX per-stock Connect holdings, 2017-03 to 2024-08):
  ``nb_pct``, the held share of A-shares; ``nb_chg5``, ``nb_chg20``,
  ``nb_chg60``, its change in percentage points; ``nb_flow5``, five-day net
  purchase value in turnover days;
* dragon-tiger list: ``lhb_on20``, the number of listings in the last 20
  days; ``lhb_net20``, the summed seat net buying over 20 days in turnover
  days; ``lhb_last``, days since the last listing (capped at 60).

Rows outside a source's coverage are NaN; LightGBM treats them as missing
and the mining DSL leaves them to its own missing-value rules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_fit import _init, _load_index  # noqa: E402

FFILL = 5  # a balance missing for up to this many trading days carries forward


def _wide(frame: pd.DataFrame, column: str, dates: pd.DatetimeIndex, assets: pd.Index) -> pd.DataFrame:
    """``column`` as (trading day x asset), on the calendar, shifted one day (point in time)."""
    wide = frame.pivot_table(index="date", columns="asset_id", values=column, aggfunc="last")
    wide = wide.reindex(index=dates.union(wide.index)).sort_index().reindex(columns=assets)
    return wide.reindex(dates).shift(1)


def _stack(wide: pd.DataFrame, name: str) -> pd.Series:
    out = wide.stack(future_stack=True).rename(name)
    out.index.names = ["datetime", "instrument"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", type=Path, default=Path("~/fm-data/qlib_cn"))
    parser.add_argument("--cache", type=Path, default=Path("~/fm-data/cache"))
    parser.add_argument("--universe", required=True)
    parser.add_argument("--start", default="2009-01-01", help="Alpha158 cache start whose rows are written")
    parser.add_argument("--margin", type=Path, default=None)
    parser.add_argument("--northbound", type=Path, default=None)
    parser.add_argument("--lhb", type=Path, default=None)
    parser.add_argument("--float-cap", type=Path, default=Path("~/fm-data/float_cap.parquet"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from qlib.data import D

    _init(args.provider)
    rows = _load_index(args.cache, args.universe, args.start)
    dates = pd.DatetimeIndex(rows.get_level_values("datetime").unique()).sort_values()
    assets = pd.Index(rows.get_level_values("instrument").unique()).sort_values()
    first = str((dates[0] - pd.Timedelta(days=120)).date())
    market = D.features(list(assets), ["$amount", "$close"], start_time=first, end_time=str(dates[-1].date()))
    market = market.swaplevel().sort_index()
    amount = market["$amount"].unstack("instrument").reindex(columns=assets)
    close = market["$close"].unstack("instrument").reindex(columns=assets)
    amount20 = amount.rolling(20, min_periods=10).mean().reindex(dates)
    close = close.reindex(dates)
    features: list[pd.Series] = []

    if args.margin is not None and args.margin.expanduser().exists():
        margin = pd.read_parquet(args.margin.expanduser())
        fin = _wide(margin, "fin_balance", dates, assets).ffill(limit=FFILL)
        buy = _wide(margin, "fin_buy", dates, assets)
        short = _wide(margin, "short_qty", dates, assets).ffill(limit=FFILL)
        features += [_stack(fin.diff(5) / (5 * amount20), "mg_fin_chg5"),
                     _stack(fin.diff(20) / (20 * amount20), "mg_fin_chg20"),
                     _stack(buy.rolling(5, min_periods=3).sum() / amount.reindex(dates).rolling(5, min_periods=3).sum(),
                            "mg_buy5"),
                     _stack(short.diff(5) * close / (5 * amount20), "mg_short_chg5")]
        cap_path = args.float_cap.expanduser()
        if cap_path.exists():
            cap = pd.read_parquet(cap_path, columns=["float_cap"])["float_cap"].unstack("instrument")
            cap = cap.reindex(index=dates, columns=assets).ffill(limit=FFILL)
            features.append(_stack(fin / cap, "mg_fin_level"))
        print(f"margin: {margin['date'].min().date()}..{margin['date'].max().date()}, "
              f"{int(fin.notna().sum().sum())} covered rows", flush=True)

    if args.northbound is not None and args.northbound.expanduser().exists():
        north = pd.read_parquet(args.northbound.expanduser())
        pct = _wide(north, "hold_pct", dates, assets).ffill(limit=FFILL)
        flow = _wide(north, "chg_value", dates, assets)
        features += [_stack(pct, "nb_pct"), _stack(pct.diff(5), "nb_chg5"), _stack(pct.diff(20), "nb_chg20"),
                     _stack(pct.diff(60), "nb_chg60"),
                     _stack(flow.rolling(5, min_periods=3).sum() / (5 * amount20), "nb_flow5")]
        print(f"northbound: {north['date'].min().date()}..{north['date'].max().date()}, "
              f"{int(pct.notna().sum().sum())} covered rows", flush=True)

    if args.lhb is not None and args.lhb.expanduser().exists():
        lhb = pd.read_parquet(args.lhb.expanduser())
        daily = lhb.groupby(["date", "asset_id"]).agg(net_buy=("net_buy", "sum"), n=("net_buy", "size")).reset_index()
        on = _wide(daily, "n", dates, assets).fillna(0.0)
        net = _wide(daily, "net_buy", dates, assets).fillna(0.0)
        listed = on.gt(0)
        last = listed.cumsum()
        since = (np.arange(len(dates))[:, None] - pd.DataFrame(np.where(listed, np.arange(len(dates))[:, None], np.nan),
                                                              index=dates, columns=assets).ffill()).clip(upper=60)
        since = since.where(last > 0)
        features += [_stack(on.rolling(20, min_periods=1).sum(), "lhb_on20"),
                     _stack(net.rolling(20, min_periods=1).sum() / (20 * amount20), "lhb_net20"),
                     _stack(since, "lhb_last")]
        print(f"lhb: {lhb['date'].min().date()}..{lhb['date'].max().date()}, {len(daily)} listings", flush=True)

    frame = pd.concat(features, axis=1).reindex(rows).astype(np.float32)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out)
    print(f"{args.out}: {frame.shape}; coverage " + ", ".join(f"{c} {frame[c].notna().mean():.2f}" for c in frame))


if __name__ == "__main__":
    main()
