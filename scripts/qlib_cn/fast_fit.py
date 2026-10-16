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
EW_DAYS = 20  # ``ew<d>`` labels sum this many forward daily returns


def _label_spec(name: str) -> tuple[str, int]:
    """Training label ``name`` -> (Qlib expression, days of lookahead for the purge).

    An integer ``h`` is the ``h``-day forward return (``run_qlib_baselines --horizon``).
    ``ew<d>`` is the survival-weighted forward return sum_k (d/100)^(k-1) r_{t+k},
    k = 1..``EW_DAYS``, with r_{t+k} the k-th daily return after entry at close t+1:
    what a name ranked on day t earns in a book that holds it with daily exit hazard
    1 - d/100 (TopkDropout top 50, n_drop 5 measures about 0.10).
    """
    if name.isdigit():
        return _label(int(name))[0], int(name)
    if name.startswith("ew"):
        decay = int(name[2:]) / 100
        terms = [f"{decay ** (k - 1):.6g}*(Ref($close, -{k + 1})/Ref($close, -{k}) - 1)" for k in range(1, EW_DAYS + 1)]
        return " + ".join(terms), EW_DAYS
    raise ValueError(f"unknown label {name}")


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


def _load_index(cache: Path, universe: str, start: str) -> pd.Index:
    return pd.read_parquet(_cache_path(cache, universe, start), columns=["label/h1"]).index


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


_NORMED: dict[tuple, pd.Series] = {}


def _ensure_label(labels: pd.DataFrame, name: str, universe: str, start: str) -> str:
    """Column of ``labels`` holding training label ``name``, computing a non-cached one with Qlib."""
    column = f"h{name}" if name.isdigit() else name
    if column not in labels:
        from qlib.data import D

        expr, _ = _label_spec(name)
        end = labels.index.get_level_values("datetime").max()
        values = D.features(D.instruments(universe), [expr], start_time=start, end_time=end).swaplevel().sort_index()
        labels[column] = values.iloc[:, 0].reindex(labels.index).astype(np.float32)
    return column


VOL_EXPR = "Std($close/Ref($close, 1) - 1, 20)"  # trailing 20-day daily volatility, known at close t
VOL_FLOOR = 0.005


def _weights(labels: pd.DataFrame, index: pd.MultiIndex, how: str, universe: str, start: str) -> np.ndarray | None:
    """Per-row training weights (mean 1), or None.

    ``invvar``: 1 / sigma^2 with sigma the name's trailing 20-day volatility (floored at
    0.5%/day; missing -> the day's median). Under squared loss a name's label noise scales
    with sigma^2, so unweighted MSE is dominated by volatile names; inverse-variance weights
    are the GLS (efficient) estimator of the same conditional mean.

    ``decay<H>``: 0.5^(age / H years), age measured back from the last training day. The
    return-generating process drifts (short-term reversal lost its long side in 2017); a
    half-life discounts the regime the model no longer trades in. Training rows only.
    """
    if how == "none":
        return None
    if how.startswith("decay"):
        dates = index.get_level_values("datetime")
        age = (dates.max() - dates).days.to_numpy() / 365.25
        w = 0.5 ** (age / float(how[5:]))
        return w / w.mean()
    if "_vol20" not in labels:
        from qlib.data import D

        end = labels.index.get_level_values("datetime").max()
        vol = D.features(D.instruments(universe), [VOL_EXPR], start_time=start, end_time=end).swaplevel().sort_index()
        labels["_vol20"] = vol.iloc[:, 0].reindex(labels.index).astype(np.float32)
    vol = labels["_vol20"].reindex(index)
    vol = vol.fillna(vol.groupby(level="datetime").transform("median")).fillna(vol.median())
    w = 1.0 / np.maximum(vol.to_numpy(np.float64), VOL_FLOOR) ** 2
    return w / w.mean()


def _fit_one(features, labels, universe, protocol, label, seed, out: Path, feature_norm: str,
             label_norm: str, lgb_config: str, purge: str, threads: int, weight: str = "none",
             outputs: dict[Path, pd.Index | None] | None = None, train_universes: list[str] | None = None) -> dict:
    import lightgbm as lgb

    started = time.monotonic()
    split = PROTOCOLS[protocol]
    label = str(label)
    expr, horizon = _label_spec(label)
    column = _ensure_label(labels, label, universe, split["start"])
    segments = {"train": (split["start"], _purged(split["fit_end"], horizon, purge)),
                "valid": (split["valid"][0], _purged(split["valid"][1], horizon, purge)),
                "test": split["test"]}
    x_all = _segment(features, split["start"], split["test"][1])
    if feature_norm != "raw":
        x_all = _robust(x_all, split["start"], split["fit_end"], clip=feature_norm == "robust")
    key = (protocol, column, label_norm)
    if key not in _NORMED:
        _NORMED.clear()
        _NORMED[key] = _norm_label(_segment(labels[column], split["start"], split["test"][1]), label_norm)
    y_all = _NORMED[key]

    overrides, rounds, patience = LGB_CONFIGS[lgb_config]
    params = {"objective": "mse", "verbosity": -1,
              **{k: v for k, v in LGB_PARAMS.items() if k != "loss"}, **overrides,
              "num_threads": threads, "seed": seed}
    sets = []
    for key in ("train", "valid"):
        x = _segment(x_all, *segments[key])
        y = y_all.reindex(x.index)
        keep = y.notna().to_numpy()
        w = (None if key == "valid" and weight.startswith("decay")
             else _weights(labels, x.index[keep], weight, universe, split["start"]))
        sets.append(lgb.Dataset(x.to_numpy()[keep], label=y.to_numpy()[keep], weight=w, free_raw_data=False))
    booster = lgb.train(params, sets[0], num_boost_round=rounds, valid_sets=sets, valid_names=["train", "valid"],
                        callbacks=[lgb.early_stopping(patience, verbose=False)])

    def predict(seg):
        x = _segment(x_all, *seg)
        return pd.Series(booster.predict(x.to_numpy()), index=x.index, name="score")

    valid_all = predict(segments["valid"])
    pred_all = predict(segments["test"]).dropna()
    # outputs: run dir -> rows (index) of the universe it serves; None keeps every row
    for run, rows in (outputs or {out: None}).items():
        valid_pred = valid_all if rows is None else valid_all[valid_all.index.isin(rows)]
        pred = pred_all if rows is None else pred_all[pred_all.index.isin(rows)]
        metrics = {}
        if len(pred):
            metrics.update(_ic_metrics(pred, labels["h1"].reindex(pred.index)))
        else:
            metrics["test_dates"] = 0
        metrics["best_iteration"] = int(booster.best_iteration)
        metrics["fit_seconds"] = round(time.monotonic() - started, 1)

        run.mkdir(parents=True, exist_ok=True)
        pred.to_frame().reset_index().to_parquet(run / "predictions.parquet", index=False)
        valid_pred.dropna().to_frame().reset_index().to_parquet(run / "valid_predictions.parquet", index=False)
        booster.save_model(str(run / "model.txt"))
        (run / "metrics.json").write_text(json.dumps({
            "universe": universe, "protocol": protocol, "split": split, "handler": "alpha158", "robust": False,
            "model": "lgb", "seed": seed, "objective": "mse", "early_stop": "loss", "patience": patience,
            "label": LABEL[0], "train_label": expr, "label_name": label, "horizon": horizon, "segments": segments,
            "purge": purge, "feature_norm": feature_norm, "label_norm": label_norm, "lgb_config": lgb_config,
            "lgb_params": params, "metrics": metrics, "fast_fit": True, "n_features": int(features.shape[1]),
            "weight": weight, "train_universes": train_universes, "train_rows": int(sets[0].num_data()),
        }, indent=2) + "\n")
    return metrics


def _weight_arg(text: str) -> str:
    if text in ("none", "invvar") or (text.startswith("decay") and text[5:].replace(".", "", 1).isdigit()):
        return text
    raise argparse.ArgumentTypeError(f"--weight: none, invvar or decay<years>, not {text!r}")


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
    fit.add_argument("--horizons", default="1", help="training labels: day counts and/or ew<d> (see _label_spec)")
    fit.add_argument("--seeds", default="0-6", help="e.g. 0-6 or 0,2,5")
    fit.add_argument("--name", default="alpha158_h{h}_lgb_s{s}", help="run name; {h} horizon, {s} seed")
    fit.add_argument("--train-universes", default=None,
                     help="comma-separated universes whose rows are pooled for training; the fit is written "
                          "to <out>/<u>/<protocol>/<name> for each, scored on that universe's rows")
    fit.add_argument("--weight", default="none", type=_weight_arg,
                     help="training weights: invvar = 1/sigma_20d^2 (GLS under heteroskedastic noise); "
                          "decay<H> = half-life of H years back from the last training day")
    fit.add_argument("--factors", type=Path, default=None,
                     help="csv/parquet (datetime, instrument, columns...) joined to Alpha158 as fm_<column>; "
                          "{p} in the path is replaced by the protocol (per-window libraries)")
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
        factors = Path(str(args.factors).format(p=protocol)) if args.factors is not None else None
        if (start, factors) not in loaded:
            loaded.clear()
            features, labels = _load(args.cache, args.universe, start)
            if args.train_universes:  # union of the universes' rows; the label is normalised over the union
                parts = [_load(args.cache, u, start) for u in args.train_universes.split(",")]
                features = pd.concat([f for f, _ in parts]).sort_index()
                labels = pd.concat([lab for _, lab in parts]).sort_index()
                keep = ~features.index.duplicated()
                features, labels = features[keep], labels[~labels.index.duplicated()]
            if factors is not None:  # extra columns, as run_qlib_baselines --handler alpha158_plus
                extra = pd.read_csv(factors) if factors.suffix == ".csv" else pd.read_parquet(factors)
                extra["datetime"] = pd.to_datetime(extra["datetime"])
                extra["instrument"] = extra["instrument"].str.upper()
                extra = extra.set_index(["datetime", "instrument"]).sort_index().astype(np.float32)
                features = features.join(extra.add_prefix("fm_"), how="left")
            loaded[start, factors] = features, labels
        return loaded[start, factors]

    def _universe_rows(universe: str, protocol: str) -> pd.Index:
        return _load_index(args.cache, universe, PROTOCOLS[protocol]["start"])

    if args.check is not None:
        meta = json.loads((args.check / "metrics.json").read_text())
        scratch = args.out / "_check" / args.check.name
        _fit_one(*data(meta["protocol"]), args.universe, meta["protocol"], meta.get("label_name", meta.get("horizon", 1)), meta["seed"], scratch,
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
        for h in args.horizons.split(","):
            for s in _seeds(args.seeds):
                name = args.name.format(h=h, s=s)
                out = args.out / args.universe / protocol / name
                if (out / "metrics.json").exists() and not args.force:
                    continue
                outputs = None
                if args.train_universes:  # one fit, scored in every universe it was trained on
                    outputs = {args.out / u / protocol / name: _universe_rows(u, protocol)
                               for u in args.train_universes.split(",")}
                m = _fit_one(*data(protocol), args.universe, protocol, h, s, out, args.feature_norm,
                             args.label_norm, args.lgb_config, args.purge, args.threads, args.weight,
                             outputs, args.train_universes.split(",") if args.train_universes else None)
                print(f"{protocol} {name}: trees {m['best_iteration']} RankIC {m.get('RankIC', float('nan')):.4f} "
                      f"{m['fit_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
