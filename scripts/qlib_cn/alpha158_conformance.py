"""Compare FactorMiner's Alpha158 transcription with Qlib's own feature values.

Two steps, one per environment::

    # Qlib environment: raw Alpha158 fields (no processors) for the panel's rows
    qlib-venv/bin/python -I scripts/qlib_cn/alpha158_conformance.py export \\
        --provider ~/fm-data/qlib_cn --universe csi300 --start 2008-01-01 --end 2014-12-31 \\
        --out ~/fm-data/alpha158/csi300_2008_2014.npz

    # FactorMiner environment: per-feature agreement on universe members
    uv run python scripts/qlib_cn/alpha158_conformance.py compare \\
        -c factorminer/configs/qlib_cn_csi300.yaml --data panel.csv \\
        --qlib ~/fm-data/alpha158/csi300_2008_2014.npz --out alpha158_conformance.json

``compare`` reports, per feature, the mean daily Spearman correlation between
the two value sets over point-in-time members, and the share of finite cells
on which they agree. Values are exchanged as a plain ``.npz`` (dates, asset ids,
a float matrix and feature names), which needs no parquet engine.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _export(args: argparse.Namespace) -> None:
    import qlib
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data import D

    provider = args.provider.expanduser().resolve()
    qlib.init(provider_uri=str(provider), region="cn")
    fields, names = Alpha158DL.get_feature_config()
    instruments = D.instruments(args.universe)
    frame = D.features(instruments, fields, start_time=args.start, end_time=args.end, freq="day")
    frame.columns = names
    frame = frame.reset_index().rename(columns={"instrument": "asset_id"})
    frame["asset_id"] = frame["asset_id"].str.upper()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        datetime=frame["datetime"].to_numpy(dtype="datetime64[ns]"),
        asset_id=frame["asset_id"].to_numpy(dtype=str),
        values=frame[names].to_numpy(dtype=np.float64),
        names=np.asarray(names, dtype=str),
    )
    print(json.dumps({"rows": len(frame), "features": len(names), "out": str(args.out)}))


def _daily_spearman(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    keep = mask & np.isfinite(a) & np.isfinite(b)
    values = []
    for t in range(a.shape[1]):
        rows = keep[:, t]
        if rows.sum() < 20:
            continue
        x = pd.Series(a[rows, t]).rank().to_numpy()
        y = pd.Series(b[rows, t]).rank().to_numpy()
        if x.std() > 0 and y.std() > 0:
            values.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(values)) if values else float("nan")


def _compare(args: argparse.Namespace) -> None:
    from factorminer.core.parser import parse
    from factorminer.core.reference_catalogs import ALPHA158
    from factorminer.data.loader import load_market_data
    from factorminer.evaluation.runtime import compute_batch_signals, load_runtime_dataset
    from factorminer.utils.config import load_config

    cfg = load_config(args.config)
    dataset = load_runtime_dataset(load_market_data(args.data), cfg)
    with np.load(args.qlib) as exported:
        qlib_frame = pd.DataFrame(exported["values"], columns=exported["names"].tolist())
        qlib_frame["datetime"] = pd.to_datetime(exported["datetime"])
        qlib_frame["asset_id"] = exported["asset_id"].astype(object)
    assets = pd.Index(np.asarray(dataset.asset_ids))
    stamps = pd.Index(pd.to_datetime(dataset.timestamps))
    rows = assets.get_indexer(qlib_frame["asset_id"])
    cols = stamps.get_indexer(qlib_frame["datetime"])
    keep = (rows >= 0) & (cols >= 0)
    members = np.isfinite(dataset.returns)

    trees = [parse(formula) for _, formula in ALPHA158]
    report = {}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        outcomes = compute_batch_signals(trees, dataset.data_dict, dataset.returns.shape,
                                         backend=args.backend)
        for (name, _), (_, ours, error) in zip(ALPHA158, outcomes, strict=True):
            if error is not None or ours is None:
                report[name] = {"error": str(error)}
                continue
            theirs = np.full(dataset.returns.shape, np.nan)
            theirs[rows[keep], cols[keep]] = qlib_frame.loc[keep, name].to_numpy(dtype=np.float64)
            window = np.isfinite(theirs).any(axis=0)
            mask = members & window[None, :]
            both = mask & np.isfinite(ours) & np.isfinite(theirs)
            close = np.isclose(ours[both], theirs[both], rtol=1e-6, atol=1e-9)
            report[name] = {
                "daily_spearman": _daily_spearman(ours, theirs, mask),
                "equal_share": float(close.mean()) if close.size else float("nan"),
                "cells": int(both.sum()),
            }
    spearman = np.array([v.get("daily_spearman", np.nan) for v in report.values()])
    summary = {
        "features": len(report),
        "median_daily_spearman": float(np.nanmedian(spearman)),
        "min_daily_spearman": float(np.nanmin(spearman)),
        "features_above_0.99": int(np.sum(spearman > 0.99)),
        "per_feature": report,
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    worst = sorted(report.items(), key=lambda item: item[1].get("daily_spearman", -1))[:8]
    print(json.dumps({k: v for k, v in summary.items() if k != "per_feature"}, indent=2))
    for name, row in worst:
        print(f"  {name:8s} {row}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--provider", type=Path, required=True)
    export.add_argument("--universe", required=True)
    export.add_argument("--start", required=True)
    export.add_argument("--end", required=True)
    export.add_argument("--out", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("-c", "--config", required=True)
    compare.add_argument("--data", required=True)
    compare.add_argument("--qlib", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--backend", default="numpy")
    args = parser.parse_args()
    _export(args) if args.command == "export" else _compare(args)


if __name__ == "__main__":
    main()
