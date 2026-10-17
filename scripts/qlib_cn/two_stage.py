"""Two-stage, winners-only model: re-rank the top of the h5 score with a model fitted on candidates only.

    $QPY -I scripts/qlib_cn/two_stage.py fit --universe csi300 --protocols wf2018,wf2020,wf2022,wf2024 \\
        --stage1 'alpha158_h5_lgb_s[0-6]' --cand 125 --seeds 0-6 --name ts125_h5_lgb_s{s}

Why. The book earns the mean of m(q) = E[r5 | rank(s) = q] over the top k/N of
the score, but every global statistic we fit or admit on (MSE on all rows,
RankIC, partial IC) is a slope over all q. After 2017 the curve is lopsided:
the bottom ventile is steep (losers keep losing) and the top three ventiles are
flat, so a model trained on every row spends its splits separating losers, and
a factor that sharpens the bottom raises RankIC without touching the book. The
relevant information coefficient is the one among the names we could buy:

    IC_top = corr(s, r5 | s in the top 2.5k)

This script changes n, not the loss: it keeps the stage-1 score s1 (the h5
7-seed rank average, the adopted baseline) to define the candidate set
C_t = the top ``--cand`` names by s1 on each day, and fits a second LightGBM
only on rows in C_t with r5 standardised within C_t, on Alpha158 plus s1 (plus
``--factors``). The final score orders C_t by the stage-2 prediction and the
rest of the universe by s1, so TopkDropout and every downstream judge are
unchanged.

Leakage control. On stage-2 training rows s1 must be out of fold, or stage 2
learns to trust an in-sample score and the test is void. The stage-1 model is
cross-fitted over ``--folds`` contiguous time blocks of the train+valid span
with the same LightGBM parameters and the tree count of the saved stage-1
booster; training rows whose 5-day label window could overlap a held-out label
window are purged (horizon + 1 days) and ``--embargo`` further days are
dropped on each side. On test days s1 is the saved stage-1 runs' prediction
(the same signal the baseline book trades). The s1 panel (OOF on the span,
saved model on the test) is frozen to ``_stage1_<tag>_oof.parquet`` per window
so later evaluators (retargeted mining) take it as a fixed input.

Outputs match ``fast_fit.py``: ``predictions.parquet``, ``valid_predictions.parquet``
(s1 is OOF on the validation segment too, so ``book_eval --select`` works),
``model.txt``, ``metrics.json``; the latter adds ``two_stage`` diagnostics on
the test days: IC_top of s1 and of the stage-2 score within C_t (daily Spearman,
HAC t), the within-C_t return curve by quintile of each score, and the mean
5-day excess of the top 50 by each score over the C_t mean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_lift import _hac_t  # noqa: E402
from fast_fit import _init, _load, _norm_label, _seeds, _segment  # noqa: E402
from run_qlib_baselines import (  # noqa: E402
    LGB_PARAMS,
    PROTOCOLS,
    TOPK,
    _ic_metrics,
    _label,
    _purged,
)

HORIZON = 5
STAGE2_CONFIGS = {  # name -> (LightGBM overrides, max rounds, early-stopping patience)
    # same parameters as stage 1 (Qlib's tuned LightGBM); the only change is the training rows
    "qlib": ({}, 1000, 50),
    # stronger regularisation for a 2.5k-name training set: fewer rows per day, correlated names
    "reg": ({"min_data_in_leaf": 300, "colsample_bytree": 0.7}, 1000, 50),
    # Qlib's L1 penalty (205) was tuned for a million rows; on a fifth of them it stops after one
    # or two trees. A slow, lightly penalised learner (fast_fit's slow31) can fit what is there.
    "slow": ({"learning_rate": 0.03, "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 1000,
              "bagging_freq": 1, "colsample_bytree": 0.7, "subsample": 0.8, "lambda_l1": 0.0,
              "lambda_l2": 10.0}, 4000, 300),
}


def _pct(score: pd.Series) -> pd.Series:
    """Per-day percentile rank in (0, 1]."""
    return score.groupby(level="datetime").rank(pct=True)


def _ensemble(preds: list[pd.Series]) -> pd.Series:
    """book_eval's signal: per-day percentile ranks averaged over runs, re-ranked to percentiles."""
    return _pct(pd.concat([_pct(p) for p in preds], axis=1).mean(axis=1)).rename("s1")


def _stage1_runs(root: Path, pattern: str) -> list[Path]:
    import fnmatch

    runs = sorted(p for p in root.iterdir() if fnmatch.fnmatch(p.name, pattern) and (p / "model.txt").exists())
    if not runs:
        raise SystemExit(f"no stage-1 runs match {root}/{pattern}")
    return runs


def _params(seed: int, threads: int, overrides: dict | None = None) -> dict:
    return {"objective": "mse", "verbosity": -1, **{k: v for k, v in LGB_PARAMS.items() if k != "loss"},
            **(overrides or {}), "num_threads": threads, "seed": seed}


def stage1_oof(features: pd.DataFrame, y: pd.Series, runs: list[Path], folds: int, embargo: int,
               threads: int) -> pd.Series:
    """Cross-fitted stage-1 score on ``features``' rows (one block held out at a time), ensembled over runs.

    Each run's saved booster gives its seed and tree count, so the OOF models have the capacity
    the test-time model has. Training rows within horizon + 1 + ``embargo`` trading days of the
    held-out block are dropped: a 5-day label on day t covers closes t+1..t+6, so rows closer than
    that share return days with the block's labels.
    """
    import lightgbm as lgb

    days = features.index.get_level_values("datetime")
    unique = days.unique().sort_values()
    pos = pd.Series(np.arange(len(unique)), index=unique).reindex(days).to_numpy()
    x = features.to_numpy(np.float32)
    yv = y.reindex(features.index).to_numpy(np.float64)
    keep = np.isfinite(yv)
    gap = HORIZON + 1 + embargo
    boosters = [lgb.Booster(model_file=str(r / "model.txt")) for r in runs]
    seeds = [json.loads((r / "metrics.json").read_text())["seed"] for r in runs]
    rounds = [b.num_trees() for b in boosters]
    out = {s: np.full(len(x), np.nan) for s in seeds}
    for block in np.array_split(np.arange(len(unique)), folds):
        lo, hi = int(block[0]), int(block[-1])
        train = keep & ((pos < lo - gap) | (pos > hi + gap))
        test = (pos >= lo) & (pos <= hi)
        data = lgb.Dataset(x[train], label=yv[train], free_raw_data=False)
        for seed, n in zip(seeds, rounds):
            booster = lgb.train(_params(seed, threads), data, num_boost_round=n)
            out[seed][test] = booster.predict(x[test])
        print(f"  stage-1 fold {unique[lo].date()}..{unique[hi].date()}: {int(train.sum())} train rows, "
              f"trees {rounds}", flush=True)
    return _ensemble([pd.Series(v, index=features.index) for v in out.values()])


def stage1_test(runs: list[Path]) -> pd.Series:
    preds = [pd.read_parquet(r / "predictions.parquet").set_index(["datetime", "instrument"])["score"] for r in runs]
    return _ensemble(preds)


def _curve(score: pd.Series, r5: pd.Series, bins: int = 5) -> list[float]:
    """Mean 5-day excess (bps, over the day's mean of the rows) by per-day quantile of ``score``."""
    frame = pd.DataFrame({"s": score, "r": r5}).dropna()
    frame["r"] -= frame.groupby(level="datetime")["r"].transform("mean")
    q = frame.groupby(level="datetime")["s"].rank(pct=True)
    frame["q"] = np.minimum((q * bins).astype(int), bins - 1)
    return [float(v) for v in (frame.groupby("q")["r"].mean() * 1e4).reindex(range(bins))]


def _top_excess(score: pd.Series, r5: pd.Series, k: int = TOPK) -> tuple[float, float]:
    """Mean 5-day excess (bps) of the top ``k`` by ``score`` over the rows' day mean, and its HAC t."""
    frame = pd.DataFrame({"s": score, "r": r5}).dropna()
    frame["r"] -= frame.groupby(level="datetime")["r"].transform("mean")
    rank = frame.groupby(level="datetime")["s"].rank(ascending=False, method="first")
    daily = frame[rank <= k].groupby(level="datetime")["r"].mean() * 1e4
    return float(daily.mean()), float(_hac_t(daily.to_numpy()))


def _ic(score: pd.Series, r5: pd.Series) -> tuple[float, float]:
    frame = pd.DataFrame({"s": score, "r": r5}).dropna()
    daily = frame.groupby(level="datetime").apply(lambda f: f["s"].corr(f["r"], method="spearman"))
    return float(daily.mean()), float(_hac_t(daily.to_numpy()))


def _final(s1: pd.Series, f2: pd.Series, cand: pd.Series, blend: bool) -> pd.Series:
    """C_t ordered by f2 (or the mean rank of f2 and s1) above everything else ordered by s1."""
    inside = _pct(f2)  # f2 is defined on the candidate rows only
    if blend:
        inside = _pct((inside + _pct(s1[cand])) / 2)
    outside = _pct(s1[~cand])
    return pd.concat([inside + 1.0, outside]).sort_index().rename("score")


def fit_window(features: pd.DataFrame, labels: pd.DataFrame, universe: str, protocol: str, args,
               root: Path) -> None:
    import lightgbm as lgb

    split = PROTOCOLS[protocol]
    runs = _stage1_runs(root, args.stage1)
    tag = args.stage1.replace("*", "").replace("[", "").replace("]", "").replace("-", "")
    oof_path = root / f"_stage1_{tag}_oof.parquet"
    span_end = _purged(split["valid"][1], HORIZON, args.purge)
    x_all = _segment(features, split["start"], split["test"][1])
    r5 = _segment(labels["h5"], split["start"], split["test"][1]).astype(np.float64)

    if oof_path.exists() and not args.force_stage1:
        s1 = pd.read_parquet(oof_path).set_index(["datetime", "instrument"])["s1"]
    else:
        started = time.monotonic()
        y1 = _norm_label(_segment(labels["h5"], split["start"], span_end), "zscore")
        span = _segment(x_all, split["start"], span_end)
        s1 = pd.concat([stage1_oof(span, y1, runs, args.folds, args.embargo, args.threads),
                        stage1_test(runs)]).sort_index()
        s1.rename("s1").reset_index().to_parquet(oof_path, index=False)
        print(f"{protocol}: stage-1 OOF panel {len(s1)} rows in {time.monotonic() - started:.0f}s -> {oof_path}",
              flush=True)
    s1 = s1.reindex(x_all.index)
    rank = s1.groupby(level="datetime").rank(ascending=False, method="first")
    cand = (rank <= args.cand).fillna(False)

    # stage-2 label: r5 standardised within the candidate set
    y2 = _norm_label(r5[cand], "zscore")
    x2 = x_all[cand].copy()
    x2["s1"] = s1[cand]
    segments = {"train": (split["start"], _purged(split["fit_end"], HORIZON, args.purge)),
                "valid": (split["valid"][0], span_end), "test": split["test"]}
    sets = []
    for key in ("train", "valid"):
        x = _segment(x2, *segments[key])
        y = y2.reindex(x.index)
        k = y.notna().to_numpy()
        sets.append(lgb.Dataset(x.to_numpy(np.float32)[k], label=y.to_numpy()[k], free_raw_data=False))

    test_rows = _segment(x_all, *split["test"]).index
    s1_test, cand_test, r5_test = s1[test_rows], cand[test_rows], r5[test_rows]
    ic_s1, t_s1 = _ic(s1_test[cand_test], r5_test[cand_test])
    top_s1 = _top_excess(s1_test[cand_test], r5_test[cand_test])
    for seed in _seeds(args.seeds):
        for blend in ([False, True] if args.blend else [False]):
            name = args.name.format(s=seed) + ("b" if blend else "")
            out = root / name
            if (out / "metrics.json").exists() and not args.force:
                continue
            started = time.monotonic()
            overrides, rounds, patience = STAGE2_CONFIGS[args.stage2_config]
            params = _params(seed, args.threads, overrides)
            booster = lgb.train(params, sets[0], num_boost_round=rounds, valid_sets=sets, valid_names=["train", "valid"],
                                callbacks=[lgb.early_stopping(patience, verbose=False)])

            def predict(seg):
                x = _segment(x2, *seg)
                return pd.Series(booster.predict(x.to_numpy(np.float32)), index=x.index)

            f2_test, f2_valid = predict(segments["test"]), predict(segments["valid"])
            pred = _final(s1_test, f2_test, cand_test, blend).dropna()
            valid_rows = _segment(x_all, *segments["valid"]).index
            valid_pred = _final(s1[valid_rows], f2_valid, cand[valid_rows], blend).dropna()
            ic_f2, t_f2 = _ic(f2_test, r5_test[cand_test])
            top_f2 = _top_excess(f2_test, r5_test[cand_test])
            metrics = _ic_metrics(pred, labels["h1"].reindex(pred.index))
            metrics.update(best_iteration=int(booster.best_iteration), fit_seconds=round(time.monotonic() - started, 1))
            two_stage = {
                "stage1": args.stage1, "stage1_runs": [r.name for r in runs], "cand": args.cand, "folds": args.folds,
                "embargo": args.embargo, "blend": blend, "stage2_config": args.stage2_config,
                "train_rows": int(sets[0].num_data()), "valid_rows": int(sets[1].num_data()),
                "ic_top_s1": ic_s1, "ic_top_s1_t": t_s1, "ic_top_f2": ic_f2, "ic_top_f2_t": t_f2,
                "top50_excess_bps_s1": top_s1[0], "top50_excess_t_s1": top_s1[1],
                "top50_excess_bps_f2": top_f2[0], "top50_excess_t_f2": top_f2[1],
                "curve_s1": _curve(s1_test[cand_test], r5_test[cand_test]),
                "curve_f2": _curve(f2_test, r5_test[cand_test]),
            }
            out.mkdir(parents=True, exist_ok=True)
            pred.to_frame().reset_index().to_parquet(out / "predictions.parquet", index=False)
            valid_pred.to_frame().reset_index().to_parquet(out / "valid_predictions.parquet", index=False)
            booster.save_model(str(out / "model.txt"))
            (out / "metrics.json").write_text(json.dumps({
                "universe": universe, "protocol": protocol, "split": split, "handler": "alpha158", "robust": False,
                "model": "lgb", "seed": seed, "objective": "mse", "early_stop": "loss", "patience": patience,
                "label": _label(1)[0], "train_label": _label(HORIZON)[0], "label_name": str(HORIZON),
                "horizon": HORIZON, "segments": segments, "purge": args.purge, "feature_norm": "raw",
                "label_norm": "zscore", "lgb_config": f"two_stage:{args.stage2_config}", "lgb_params": params,
                "metrics": metrics, "two_stage": two_stage, "n_features": int(x2.shape[1]),
                "factors": str(args.factors) if args.factors else None,
            }, indent=2) + "\n")
            print(f"{protocol} {name}: trees {booster.best_iteration} RankIC {metrics['RankIC']:.4f} "
                  f"IC_top s1 {ic_s1:.4f} (t {t_s1:.1f}) f2 {ic_f2:.4f} (t {t_f2:.1f}) "
                  f"top50 excess s1 {top_s1[0]:.1f} f2 {top_f2[0]:.1f} bps {metrics['fit_seconds']}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--provider", type=Path, default=Path("~/fm-data/qlib_cn"))
    fit.add_argument("--universe", required=True)
    fit.add_argument("--cache", type=Path, default=Path("~/fm-data/cache"))
    fit.add_argument("--protocols", default="wf2018,wf2020,wf2022,wf2024")
    fit.add_argument("--stage1", default="alpha158_h5_lgb_s[0-6]", help="glob of stage-1 runs under each window")
    fit.add_argument("--cand", type=int, default=int(2.5 * TOPK), help="candidate names per day (top by s1)")
    fit.add_argument("--folds", type=int, default=5, help="contiguous time blocks for the stage-1 cross-fit")
    fit.add_argument("--embargo", type=int, default=5, help="extra trading days dropped around a held-out block")
    fit.add_argument("--seeds", default="0-6")
    fit.add_argument("--name", default="ts125_h5_lgb_s{s}")
    fit.add_argument("--stage2-config", choices=sorted(STAGE2_CONFIGS), default="qlib")
    fit.add_argument("--blend", action="store_true",
                     help="also write <name>b: C_t ordered by the mean rank of the stage-2 score and s1")
    fit.add_argument("--factors", type=Path, default=None,
                     help="csv/parquet (datetime, instrument, columns...) joined as fm_<column>; {p} = protocol")
    fit.add_argument("--purge", choices=["qlib", "full"], default="qlib")
    fit.add_argument("--threads", type=int, default=int(os.environ.get("LGB_THREADS", "16")))
    fit.add_argument("--out", type=Path, default=Path("output/qlib_cn"))
    fit.add_argument("--force", action="store_true", help="refit stage-2 runs that already have metrics.json")
    fit.add_argument("--force-stage1", action="store_true", help="recompute the frozen stage-1 OOF panel")
    args = parser.parse_args()

    _init(args.provider)
    for protocol in args.protocols.split(","):
        start = PROTOCOLS[protocol]["start"]
        features, labels = _load(args.cache, args.universe, start)
        if args.factors is not None:
            path = Path(str(args.factors).format(p=protocol))
            extra = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
            extra["datetime"] = pd.to_datetime(extra["datetime"])
            extra["instrument"] = extra["instrument"].str.upper()
            extra = extra.set_index(["datetime", "instrument"]).sort_index().astype(np.float32)
            features = features.join(extra.add_prefix("fm_"), how="left")
        fit_window(features, labels, args.universe, protocol, args, args.out / args.universe / protocol)


if __name__ == "__main__":
    main()
