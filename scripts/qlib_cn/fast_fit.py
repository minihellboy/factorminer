"""Fit many Alpha158 LightGBM runs in one process from a cached feature matrix.

    $QPY -I scripts/qlib_cn/fast_fit.py cache --provider ~/fm-data/qlib_cn --universe csi300
    $QPY -I scripts/qlib_cn/fast_fit.py fit --provider ~/fm-data/qlib_cn --universe csi300 \\
        --protocols wf2018,wf2020,wf2022,wf2024 --horizons 1,5 --seeds 0-6 --name alpha158_h{h}_lgb_s{s}

``cache`` computes Qlib's Alpha158 features (no processors) and the
``horizon``-day training labels from each protocol's start date to the end of
the provider's calendar, to ``<cache>/<universe>_alpha158_<start>.parquet``.
Alpha158 uses fixed-length windows and Qlib extends every load to cover their
history, so a row's values depend on the span's start only through rounding:
Qlib's rolling std/corr accumulate from the first loaded day, and a few
near-constant windows (about 250 of 127M cells) differ in the last bits,
enough to move a LightGBM bin edge. One cache per start date makes every fit
bit-identical to the per-run handler.

``fit`` replays ``run_qlib_baselines.py --handler alpha158 --model lgb`` for
every (protocol, horizon, seed): the same segments and purge, the same
processors (raw or robust features, z-scored or ranked label, Qlib's own
functions), the same LightGBM call (train and valid sets, early stopping on
valid l2), and writes the same files (``predictions.parquet``,
``valid_predictions.parquet``, ``model.txt``, ``metrics.json``) to
``<out>/<universe>/<protocol>/<name>``. It skips Qlib's per-run data load,
MLflow recorder and in-fit backtest (``book_eval.py`` is the judge), so a fit
costs its LightGBM training: a 7-seed, 4-window, 2-horizon family takes
minutes instead of an hour. ``fit --check RUN`` refits one existing run and
asserts identical predictions.
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
from run_qlib_baselines import (  # noqa: E402
    LABEL,
    LGB_CONFIGS,
    LGB_PARAMS,
    PROTOCOLS,
    _ic_metrics,
    _label,
    _purged,
)

HORIZONS = (1, 2, 3, 5, 7, 10)


def _init(provider: Path) -> None:
    import qlib

    qlib.init(provider_uri=str(provider.expanduser()), region="cn")


def _cache_path(cache: Path, universe: str, start: str) -> Path:
    return cache.expanduser() / f"{universe}_alpha158_{start}.parquet"


def cache_features(provider: Path, universe: str, cache: Path, starts: list[str]) -> None:
    from qlib.contrib.data.handler import Alpha158
    from qlib.data import D

    _init(provider)
    end = str(D.calendar(freq="day")[-1].date())
    labels = [_label(h)[0] for h in HORIZONS]
    for start in starts:
        started = time.monotonic()
        handler = Alpha158(instruments=universe, start_time=start, end_time=end, infer_processors=[],
                           learn_processors=[], label=(labels, [f"h{h}" for h in HORIZONS]))
        frame = handler.fetch(col_set=["feature", "label"])
        frame.columns = [f"{group}/{name}" for group, name in frame.columns]
        path = _cache_path(cache, universe, start)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        print(f"cached {frame.shape} {universe} {start}..{end} in {time.monotonic() - started:.0f}s -> {path}",
              flush=True)


def _load(cache: Path, universe: str, start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(_cache_path(cache, universe, start))
    features = frame[[c for c in frame.columns if c.startswith("feature/")]]
    features.columns = [c.split("/", 1)[1] for c in features.columns]
    labels = frame[[c for c in frame.columns if c.startswith("label/")]]
    labels.columns = [c.split("/", 1)[1] for c in labels.columns]
    return features, labels


def _norm_label(label: pd.Series, how: str) -> pd.Series:
    """Qlib's DropnaLabel then CSZScoreNorm / CSRankNorm on the label."""
    from qlib.utils.data import zscore

    df = label.dropna().to_frame("LABEL0")
    if how == "zscore":
        df = df.groupby("datetime", group_keys=False).apply(zscore)
    else:  # CSRankNorm
        t = df.groupby("datetime", group_keys=False).rank(pct=True)
        t -= 0.5
        t *= 3.46
        df = t
    return df["LABEL0"]


def _robust(features: pd.DataFrame, fit_start: str, fit_end: str, clip: bool) -> pd.DataFrame:
    """Qlib's RobustZScoreNorm (fit on the train span) followed by Fillna(0)."""
    from qlib.data.dataset.processor import Fillna, RobustZScoreNorm

    proc = RobustZScoreNorm(fit_start_time=fit_start, fit_end_time=fit_end, fields_group=None, clip_outlier=clip)
    df = features.copy()
    proc.fit(df)
    df = proc(df)
    return Fillna(fields_group=None)(df)


def _segment(frame, start, end):
    dates = frame.index.get_level_values("datetime")
    return frame[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def _fit_one(features, labels, universe, protocol, horizon, seed, out: Path, feature_norm: str,
             label_norm: str, lgb_config: str, purge: str, threads: int) -> dict:
    import lightgbm as lgb

    started = time.monotonic()
    split = PROTOCOLS[protocol]
    segments = {"train": (split["start"], _purged(split["fit_end"], horizon, purge)),
                "valid": (split["valid"][0], _purged(split["valid"][1], horizon, purge)),
                "test": split["test"]}
    x_all = _segment(features, split["start"], split["test"][1])
    if feature_norm != "raw":
        x_all = _robust(x_all, split["start"], split["fit_end"], clip=feature_norm == "robust")
    y_all = _norm_label(_segment(labels[f"h{horizon}"], split["start"], split["test"][1]), label_norm)

    overrides, rounds, patience = LGB_CONFIGS[lgb_config]
    params = {"objective": "mse", "verbosity": -1,
              **{k: v for k, v in LGB_PARAMS.items() if k != "loss"}, **overrides,
              "num_threads": threads, "seed": seed}
    sets = []
    for key in ("train", "valid"):
        x = _segment(x_all, *segments[key])
        y = y_all.reindex(x.index)
        keep = y.notna().to_numpy()
        sets.append(lgb.Dataset(x.to_numpy()[keep], label=y.to_numpy()[keep]))
    booster = lgb.train(params, sets[0], num_boost_round=rounds, valid_sets=sets, valid_names=["train", "valid"],
                        callbacks=[lgb.early_stopping(patience, verbose=False)])

    def predict(seg):
        x = _segment(x_all, *seg)
        return pd.Series(booster.predict(x.to_numpy()), index=x.index, name="score")

    valid_pred = predict(segments["valid"])
    pred = predict(segments["test"]).dropna()
    metrics = {}
    if len(pred):
        metrics.update(_ic_metrics(pred, labels["h1"].reindex(pred.index)))
    else:
        metrics["test_dates"] = 0
    metrics["best_iteration"] = int(booster.best_iteration)
    metrics["fit_seconds"] = round(time.monotonic() - started, 1)

    out.mkdir(parents=True, exist_ok=True)
    pred.to_frame().reset_index().to_parquet(out / "predictions.parquet", index=False)
    valid_pred.dropna().to_frame().reset_index().to_parquet(out / "valid_predictions.parquet", index=False)
    booster.save_model(str(out / "model.txt"))
    (out / "metrics.json").write_text(json.dumps({
        "universe": universe, "protocol": protocol, "split": split, "handler": "alpha158", "robust": False,
        "model": "lgb", "seed": seed, "objective": "mse", "early_stop": "loss", "patience": patience,
        "label": LABEL[0], "train_label": _label(horizon)[0], "horizon": horizon, "segments": segments,
        "purge": purge, "feature_norm": feature_norm, "label_norm": label_norm, "lgb_config": lgb_config,
        "lgb_params": params, "metrics": metrics, "fast_fit": True,
    }, indent=2) + "\n")
    return metrics


def _seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("cache", "fit"):
        p = sub.add_parser(name)
        p.add_argument("--provider", type=Path, default=Path("~/fm-data/qlib_cn"))
        p.add_argument("--universe", required=True)
        p.add_argument("--cache", type=Path, default=Path("~/fm-data/cache"))
    sub.choices["cache"].add_argument("--starts", default=None,
                                      help="comma-separated start dates (default: every protocol's start)")
    fit = sub.choices["fit"]
    fit.add_argument("--protocols", default="wf2018,wf2020,wf2022,wf2024")
    fit.add_argument("--horizons", default="1")
    fit.add_argument("--seeds", default="0-6", help="e.g. 0-6 or 0,2,5")
    fit.add_argument("--name", default="alpha158_h{h}_lgb_s{s}", help="run name; {h} horizon, {s} seed")
    fit.add_argument("--feature-norm", choices=["raw", "robust", "robust_noclip"], default="raw")
    fit.add_argument("--label-norm", choices=["zscore", "rank"], default="zscore")
    fit.add_argument("--lgb-config", choices=sorted(LGB_CONFIGS), default="qlib")
    fit.add_argument("--purge", choices=["qlib", "full"], default="qlib")
    fit.add_argument("--threads", type=int, default=int(os.environ.get("LGB_THREADS", "16")))
    fit.add_argument("--out", type=Path, default=Path("output/qlib_cn"))
    fit.add_argument("--force", action="store_true", help="refit runs that already have metrics.json")
    fit.add_argument("--check", type=Path, default=None,
                     help="refit this existing run (same protocol/horizon/seed) and compare its predictions")
    args = parser.parse_args()

    if args.command == "cache":
        starts = args.starts.split(",") if args.starts else sorted({p["start"] for p in PROTOCOLS.values()})
        cache_features(args.provider, args.universe, args.cache, starts)
        return

    _init(args.provider)
    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def data(protocol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        start = PROTOCOLS[protocol]["start"]
        if start not in loaded:
            loaded.clear()
            loaded[start] = _load(args.cache, args.universe, start)
        return loaded[start]

    if args.check is not None:
        meta = json.loads((args.check / "metrics.json").read_text())
        scratch = args.out / "_check" / args.check.name
        _fit_one(*data(meta["protocol"]), args.universe, meta["protocol"], meta.get("horizon", 1), meta["seed"], scratch,
                 meta.get("feature_norm") or ("robust" if meta.get("robust") else "raw"),
                 meta.get("label_norm") or ("rank" if meta.get("robust") else "zscore"),
                 meta.get("lgb_config", "qlib"), meta.get("purge", "qlib"), meta["lgb_params"]["num_threads"])
        for file in ("predictions.parquet", "valid_predictions.parquet"):
            a, b = pd.read_parquet(args.check / file), pd.read_parquet(scratch / file)
            same = a.shape == b.shape and (a["score"].to_numpy() == b["score"].to_numpy()).all()
            gap = float(np.abs(a["score"].to_numpy() - b["score"].to_numpy()).max()) if a.shape == b.shape else None
            print(f"{file}: {'identical' if same else 'DIFFERENT'} shape {a.shape} vs {b.shape} max|diff| {gap}")
        return

    for protocol in args.protocols.split(","):
        for h in [int(x) for x in args.horizons.split(",")]:
            for s in _seeds(args.seeds):
                name = args.name.format(h=h, s=s)
                out = args.out / args.universe / protocol / name
                if (out / "metrics.json").exists() and not args.force:
                    continue
                m = _fit_one(*data(protocol), args.universe, protocol, h, s, out, args.feature_norm,
                             args.label_norm, args.lgb_config, args.purge, args.threads)
                print(f"{protocol} {name}: trees {m['best_iteration']} RankIC {m.get('RankIC', float('nan')):.4f} "
                      f"{m['fit_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
