"""Export a Qlib CN provider into a FactorMiner long panel with PIT membership.

Run with an interpreter that has ``pyqlib`` installed (kept separate from the
FactorMiner environment)::

    qlib-venv/bin/python -I scripts/qlib_cn/export_panel.py \\
        --provider ~/fm-data/qlib_cn --universe csi300 \\
        --start 2007-01-01 --end 2020-12-31 --out ~/fm-data/panels/csi300_2007_2020.parquet

Every instrument that belonged to the universe at any date in the window is
exported for the whole window, so time-series operators see complete
histories. ``in_universe`` marks the dates on which the instrument was a
member; FactorMiner masks targets outside membership, matching Qlib's dynamic
instrument filter. Prices are the provider's adjusted values, the same ones
Qlib's Alpha158/Alpha360 handlers and label read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$amount", "$vwap", "$factor"]


def _membership(provider: Path, universe: str) -> pd.DataFrame:
    spans = pd.read_csv(
        provider / "instruments" / f"{universe}.txt", sep="\t", header=None,
        names=["instrument", "start", "end"], parse_dates=["start", "end"],
    )
    spans["instrument"] = spans["instrument"].str.upper()
    return spans


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=None,
                        help="Upstream release manifest recorded alongside the panel")
    args = parser.parse_args()

    import qlib
    from qlib.data import D

    provider = args.provider.expanduser().resolve()
    qlib.init(provider_uri=str(provider), region="cn")

    spans = _membership(provider, args.universe)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    spans = spans[(spans["end"] >= start) & (spans["start"] <= end)]
    instruments = sorted(spans["instrument"].unique())
    frame = D.features(instruments, FIELDS, start_time=start, end_time=end, freq="day")
    frame.columns = [column.lstrip("$") for column in frame.columns]
    frame = frame.reset_index().rename(columns={"instrument": "asset_id"})
    frame["asset_id"] = frame["asset_id"].str.upper()
    # Bars with no trade carry no price; keep them as gaps rather than rows.
    frame = frame.dropna(subset=["open", "high", "low", "close"], how="all")

    # One join of rows against membership spans, instead of a scan per instrument.
    keyed = frame[["asset_id", "datetime"]].reset_index().merge(
        spans.rename(columns={"instrument": "asset_id"}), on="asset_id", how="inner")
    inside = (keyed["datetime"] >= keyed["start"]) & (keyed["datetime"] <= keyed["end"])
    member_rows = keyed.loc[inside, "index"].unique()
    flags = np.zeros(len(frame), dtype=np.int8)
    flags[frame.index.get_indexer(member_rows)] = 1
    frame["in_universe"] = flags
    frame = frame.sort_values(["datetime", "asset_id"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)
    per_date = frame.groupby("datetime")["in_universe"].sum()
    manifest = {
        "schema_version": 1,
        "provider": str(provider),
        "universe": args.universe,
        "window": [args.start, args.end],
        "fields": [field.lstrip("$") for field in FIELDS],
        "price_adjustment": "qlib provider adjusted (price * factor)",
        "rows": int(len(frame)),
        "assets": int(frame["asset_id"].nunique()),
        "dates": int(frame["datetime"].nunique()),
        "first_date": str(frame["datetime"].min().date()),
        "last_date": str(frame["datetime"].max().date()),
        "members_per_date": {"min": int(per_date.min()), "median": float(per_date.median()),
                             "max": int(per_date.max())},
        "panel_sha256": _digest(args.out),
        "qlib_version": qlib.__version__,
    }
    if args.source_manifest is not None:
        manifest["source_release"] = json.loads(args.source_manifest.read_text())
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
