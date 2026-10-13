"""Score a formula set out of sample with frozen directions and export its values.

Run in the FactorMiner environment::

    uv run python scripts/qlib_cn/evaluate_library.py \\
        -c factorminer/configs/qlib_cn_csi300.yaml --data panel.csv \\
        --baseline factor_miner --out output/qlib_cn/csi300/classic/catalog_factor_miner

``--library factor_library.json`` scores a mined library instead of a built-in
catalog; ``--backend gpu`` evaluates the formulas on CUDA. Each formula's sign is fixed on the train split; validation and test
metrics are signed RankIC under that sign, so a factor that flips out of
sample scores negative rather than being credited by ``abs``. The equal-weight
composite averages cross-sectional ranks of the oriented factors. Factor
values on scored rows are written to ``factor_values.csv`` for
``run_qlib_baselines.py --handler factors``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from factorminer.benchmark.datasets import _get_baseline_entries
from factorminer.core.library_io import load_library
from factorminer.core.parser import try_parse
from factorminer.data.loader import load_market_data
from factorminer.evaluation.evidence.inference import compute_hac_mean_test
from factorminer.evaluation.metrics import compute_rank_ic
from factorminer.evaluation.runtime import compute_batch_signals, load_runtime_dataset
from factorminer.utils.config import load_config

SPLITS = ("train", "validation", "test")


def _formulas(args) -> list[tuple[str, str]]:
    if args.library:
        library = load_library(str(args.library).removesuffix(".json"))
        return [(f.name or f"factor_{f.id}", f.formula) for f in library.list_factors()]
    return [(entry.name, entry.formula) for entry in _get_baseline_entries(args.baseline, seed=42)]


def _column_ranks(signal: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile ranks per period, NaN preserved."""
    ranks = pd.DataFrame(signal).rank(axis=0, pct=True).to_numpy()
    return ranks - 0.5


def _summary(ic: np.ndarray) -> dict[str, float]:
    finite = ic[np.isfinite(ic)]
    if finite.size < 3:
        return {"rank_ic": float("nan"), "rank_icir": float("nan"), "hac_t": float("nan"), "periods": int(finite.size)}
    hac = compute_hac_mean_test(finite)
    return {"rank_ic": float(finite.mean()), "rank_icir": float(finite.mean() / finite.std(ddof=1)),
            "hac_t": float(hac.t_stat), "win_rate": float((finite > 0).mean()), "periods": int(finite.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--data", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--baseline")
    source.add_argument("--library", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--backend", default="numpy", help="evaluation backend: numpy or gpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = load_runtime_dataset(load_market_data(args.data), cfg)
    returns = dataset.returns
    split_idx = {name: dataset.get_split(name).indices for name in SPLITS}
    formulas = _formulas(args)

    rows, oriented, values = [], [], {}
    parsed = []
    for name, formula in formulas:
        tree = try_parse(formula)
        if tree is None:
            rows.append({"name": name, "formula": formula, "error": "parse"})
        else:
            parsed.append((name, formula, tree))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        outcomes = compute_batch_signals(
            [tree for _, _, tree in parsed], dataset.data_dict, returns.shape,
            signal_failure_policy="raise", backend=args.backend,
        )
        for (name, formula, _), (_, computed, error) in zip(parsed, outcomes, strict=True):
            if error is not None or computed is None:
                rows.append({"name": name, "formula": formula,
                             "error": f"{type(error).__name__}: {error}"})
                continue
            signal = np.array(computed, dtype=np.float64)
            signal[~np.isfinite(signal)] = np.nan
            ic = compute_rank_ic(signal, returns)
            train = _summary(ic[split_idx["train"]])
            direction = 1.0 if not np.isfinite(train["rank_ic"]) or train["rank_ic"] >= 0 else -1.0
            row = {"name": name, "formula": formula, "direction": direction}
            for split in SPLITS:
                row.update({f"{split}_{k}": v
                            for k, v in _summary(direction * ic[split_idx[split]]).items()})
            rows.append(row)
            oriented.append(direction * _column_ranks(signal))
            values[name] = signal

    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "factors.csv", index=False)
    if not oriented:
        errors = frame.get("error", pd.Series(dtype=object)).dropna()
        raise SystemExit(f"no formula evaluated; first errors: {errors.head(3).tolist()}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        composite = np.nanmean(np.stack(oriented), axis=0)
    composite_ic = compute_rank_ic(composite, returns)
    scored = frame.dropna(subset=["test_rank_ic"]) if "test_rank_ic" in frame else frame
    report = {
        "source": str(args.library) if args.library else args.baseline,
        "formulas": len(formulas), "evaluated": len(oriented),
        "failed": int(frame.get("error", pd.Series(dtype=object)).notna().sum()),
        "composite_equal_weight": {split: _summary(composite_ic[split_idx[split]]) for split in SPLITS},
        "single_factor_test": {
            "mean_rank_ic": float(scored["test_rank_ic"].mean()),
            "median_rank_ic": float(scored["test_rank_ic"].median()),
            "share_sign_held": float((scored["test_rank_ic"] > 0).mean()),
            "share_hac_t_gt_2": float((scored["test_hac_t"] > 2).mean()),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if not args.no_export and values:
        assets, stamps = np.asarray(dataset.asset_ids), pd.to_datetime(dataset.timestamps)
        keep = np.isfinite(returns)
        asset_pos, time_pos = np.nonzero(keep)
        export = pd.DataFrame({"datetime": stamps[time_pos], "instrument": assets[asset_pos]})
        for index, (name, signal) in enumerate(values.items()):
            export[f"f{index:03d}"] = signal[asset_pos, time_pos].astype(np.float32)
        export.to_csv(args.out / "factor_values.csv", index=False, float_format="%.6g")
        (args.out / "factor_columns.json").write_text(json.dumps(
            {f"f{index:03d}": name for index, name in enumerate(values)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
