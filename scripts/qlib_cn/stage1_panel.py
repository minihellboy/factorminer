"""Add the frozen stage-1 score (and optional flow features) to a FactorMiner panel for one window.

    $QPY -I scripts/qlib_cn/stage1_panel.py --panel ~/fm-data/panels/csi300_2007_2026.parquet \\
        --oof output/qlib_cn/csi300/wf2020/_stage1_alpha158_h5_lgb_s06_oof.parquet \\
        [--flow output/qlib_cn/data_factors/csi300_flow.parquet] --out ~/fm-data/panels/csi300_s1_wf2020

Writes ``<out>.parquet`` and ``<out>.csv``: the panel's columns plus ``s1``, the
stage-1 score percentile from ``two_stage.py``'s frozen panel (cross-fitted on
the window's train+valid span, the saved model's prediction on its test days,
NaN elsewhere), and the columns of ``--flow`` if given. A campaign on this
panel sets ``admission.zone_leaf: s1`` so the contention zone is the top of
s1 and s1 is residualised out of every candidate; the panel is specific to
the window whose OOF panel it carries and must not be used for another.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--flow", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    panel = pd.read_parquet(args.panel.expanduser()) if args.panel.suffix == ".parquet" else pd.read_csv(args.panel.expanduser())
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    oof = pd.read_parquet(args.oof)
    oof["datetime"] = pd.to_datetime(oof["datetime"])
    oof = oof.rename(columns={"instrument": "asset_id"})[["asset_id", "datetime", "s1"]]
    out = panel.merge(oof, on=["asset_id", "datetime"], how="left")
    if args.flow is not None:
        flow = pd.read_parquet(args.flow).reset_index().rename(columns={"instrument": "asset_id"})
        flow["datetime"] = pd.to_datetime(flow["datetime"])
        out = out.merge(flow, on=["asset_id", "datetime"], how="left")
    out = out.sort_values(["datetime", "asset_id"]).reset_index(drop=True)
    target = args.out.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target.with_suffix(".parquet"), index=False)
    out.to_csv(target.with_suffix(".csv"), index=False)
    covered = out["s1"].notna()
    print(f"{target}: {out.shape}; s1 on {covered.mean():.0%} of rows, "
          f"{out.loc[covered, 'datetime'].min().date()}..{out.loc[covered, 'datetime'].max().date()}")


if __name__ == "__main__":
    main()
