"""Add BaoStock valuation, size, liquidity and index columns to a FactorMiner panel.

Run in an environment with ``pandas`` and ``pyarrow``::

    python -I scripts/qlib_cn/merge_panel_features.py \\
        --panel ~/fm-data/panels/csi300_2007_2026.parquet --baostock ~/fm-data/baostock \\
        --out ~/fm-data/panels/csi300_2007_2026_fund

Writes ``<out>.parquet`` and ``<out>.csv`` with the panel's columns plus

* ``turnover``: daily turnover rate, % of float shares;
* ``pe_ttm``, ``pb``, ``ps_ttm``, ``pcf_ttm``: trailing P/E, P/B (most recent
  quarter), trailing P/S and P/CF, as BaoStock computes them from the latest
  published report on each date;
* ``mcap``: float market capitalisation in CNY, unadjusted close times float
  shares (``volume / turnover``); shares are carried forward over days with no
  turnover;
* ``st``: 1 when the stock carries an ST (special treatment) flag;
* ``index_close``: the universe's benchmark index close (CSI300 or CSI500),
  the same value for every instrument on a date.

Valuation ratios are NaN where BaoStock has no value (negative earnings give a
negative ``pe_ttm``, which is kept). Rows without a BaoStock bar keep NaN in
the new columns; FactorMiner's preprocessing forward-fills short gaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXTRA = ["turnover", "pe_ttm", "pb", "ps_ttm", "pcf_ttm", "mcap", "st", "index_close"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--baostock", type=Path, required=True)
    parser.add_argument("--index", default=None, help="CSI300 or CSI500 (default: from the panel name)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    index_name = args.index or ("CSI500" if "csi500" in args.panel.name.lower() else "CSI300")

    daily = pd.read_parquet(args.baostock / "daily.parquet")
    daily["asset_id"] = daily["code"].str.replace(".", "", regex=False).str.upper()
    daily["datetime"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["asset_id", "datetime"])
    turnover = daily["turn"].where(daily["turn"] > 0)
    shares = (daily["volume"] / (turnover / 100.0)).groupby(daily["asset_id"]).ffill()
    extra = pd.DataFrame({
        "asset_id": daily["asset_id"], "datetime": daily["datetime"],
        "turnover": daily["turn"].astype("float32"),
        "pe_ttm": daily["peTTM"].astype("float32"),
        "pb": daily["pbMRQ"].astype("float32"),
        "ps_ttm": daily["psTTM"].astype("float32"),
        "pcf_ttm": daily["pcfNcfTTM"].astype("float32"),
        "mcap": (daily["close"] * shares).astype("float32"),
        "st": daily["isST"].fillna(0).astype("int8"),
    })
    for column in ("pe_ttm", "pb", "ps_ttm", "pcf_ttm"):
        extra.loc[extra[column] == 0, column] = np.nan  # BaoStock reports 0 for "no value"

    index = pd.read_parquet(args.baostock / "index.parquet")
    index = index[index["index"] == index_name]
    index = pd.DataFrame({"datetime": pd.to_datetime(index["date"]),
                          "index_close": pd.to_numeric(index["close"]).astype("float32")})

    merged = panel.merge(extra, on=["asset_id", "datetime"], how="left")
    merged = merged.merge(index, on="datetime", how="left")
    merged = merged.sort_values(["datetime", "asset_id"]).reset_index(drop=True)
    assert len(merged) == len(panel)

    coverage = {c: float(merged[c].notna().mean()) for c in EXTRA}
    members = merged[merged["in_universe"] == 1]
    member_coverage = {c: float(members[c].notna().mean()) for c in EXTRA}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out.with_suffix(".parquet"), index=False)
    merged.to_csv(args.out.with_suffix(".csv"), index=False)
    manifest = {
        "panel": str(args.panel), "baostock": str(args.baostock), "index": index_name,
        "rows": int(len(merged)), "extra_columns": EXTRA,
        "coverage": coverage, "member_coverage": member_coverage,
    }
    source = args.baostock / "manifest.json"
    if source.exists():
        manifest["baostock_manifest"] = json.loads(source.read_text())
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(merged), "member_coverage": member_coverage}, indent=2))


if __name__ == "__main__":
    main()
