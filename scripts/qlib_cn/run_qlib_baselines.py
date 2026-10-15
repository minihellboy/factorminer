"""Run Qlib reference workflows on a Qlib CN provider and record auditable outputs.

Run with an interpreter that has ``pyqlib`` and ``lightgbm`` installed::

    qlib-venv/bin/python -I scripts/qlib_cn/run_qlib_baselines.py \\
        --provider ~/fm-data/qlib_cn --universe csi300 --protocol classic \\
        --model lgb --handler alpha158 --out output/qlib_cn/csi300/classic/alpha158_lgb

``--handler factors --factors <csv|parquet>`` trains the same model on
externally computed factor values (``datetime, instrument, <factor columns>``)
with the same label, processors, splits, and backtest, so a FactorMiner library
is compared with Alpha158/Alpha360 under one protocol. ``--handler
alpha158_plus`` appends those factors to Alpha158's features to measure their
incremental value. ``--robust`` gives the Alpha158 handlers the factor
handler's processors (train-fit robust z-score and zero fill of features,
cross-sectional rank of the label) instead of Qlib's benchmark default (raw
features, z-scored label): a stronger Alpha158 + LightGBM baseline to measure
lift against.

``--objective lambdarank`` trains LightGBM to rank each day's cross-section
with LambdaRank (NDCG@50 over sextile relevance grades) instead of regressing
the label, so the loss concentrates on the names a top-50 book would buy. Its
gradients are a few orders of magnitude smaller than regression residuals, so
Qlib's L1/L2 penalties (tuned for MSE) would zero every leaf; LambdaRank runs
without them. ``--objective topmse`` keeps the regression loss but weights
each row ``1 + TOP_WEIGHT`` when its label is in the day's top sextile, so
the fit is judged mostly on the names a book would hold. ``--early-stop top``
stops on the validation top-50 spread (mean raw label of the 50 best-scored
names minus the day's mean) rather than on the loss. ``--objective xendcg``
is LightGBM's rank_xendcg (listwise cross-entropy over the same grades, also
without L1/L2); ``topclass`` is binary logloss on the top-sextile indicator;
``listnet`` is a custom per-day softmax cross-entropy between the scores and
``softmax(label / --temperature)``, so a low temperature puts the target mass
on the day's best names.

``--horizon H`` trains on the H-day forward return from the entry close
(``Ref($close, -H-1) / Ref($close, -1) - 1``) instead of the next day's,
closer to how long a TopkDropout book holds a name (about ten days at
``n_drop`` 5 of 50). Segment ends are moved so every horizon's labels reach
the same two days past a segment as Qlib's 1-day label (``--purge qlib``),
or none at all (``--purge full``). Top-50 early stopping then uses that label too; every
reported metric and the backtest stay on the 1-day label.

``--feature-norm raw|robust|robust_noclip`` and ``--label-norm zscore|rank``
set the two halves of ``--robust`` separately. ``--refit`` refits LightGBM on
train + valid with the early-stopped tree count before predicting test.

Every run also saves ``valid_predictions.parquet``, the validation-segment
scores of the model as fitted on train only (before any refit), so books can
be compared and selected on validation without touching test, and LightGBM
runs save the booster that predicted test as ``model.txt``. ``--load-model``
predicts with a saved booster instead of fitting, for a frozen procedure.
Protocol ``live`` is the prospective holdout: it trains through 2026-09-30
and tests from 2026-10-16, the first session after its registration
(``docs/qlib-cn-protocol.md``); its test extends as the provider is updated.

``--checkpoints 16,64,256`` fits LightGBM to the largest count without early
stopping and also writes each count as a sibling run ``<out>@t<k>`` (test and
validation scores of the first k trees), so ``book_eval.py --select`` can
choose the tree count on the validation book.

``--lgb-config`` picks a LightGBM preset: ``qlib`` (benchmark), ``qlib_bag``
(row bagging actually on), ``slow63`` / ``slow31`` (lower learning rate, more
trees, larger leaves' minimum size, lighter L1/L2).

``--model ridge`` is an L2 linear model (``RIDGE_ALPHA``) for blends with the
trees; it needs filled features (``--feature-norm robust``).

``--model dens`` / ``xgb`` / ``cat`` fit Qlib's DoubleEnsemble, XGBoost and
CatBoost with their Alpha158 benchmark-config hyperparameters.

``--model mlp`` / ``gru`` / ``alstm`` fit Qlib's PyTorch reference models
(``pytorch_nn.DNNModelPytorch``, ``pytorch_gru_ts.GRU``, ``pytorch_alstm_ts.ALSTM``)
with the hyperparameters of their Alpha158 benchmark configs. The sequence
models read 20-day windows of the 20 Alpha158 columns Qlib's configs keep
(``QLIB_TS_COLUMNS``) plus any appended factors, and run on the GPU.

Model hyperparameters, processors, and the TopK-dropout backtest follow Qlib's
``examples/benchmarks`` workflow configs. Signal metrics are computed against
the raw label (``Ref($close, -2) / Ref($close, -1) - 1``), as Qlib's
``SigAnaRecord`` does.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOLS = {
    # Qlib examples/benchmarks splits.
    "classic": {"start": "2008-01-01", "fit_end": "2014-12-31", "valid": ("2015-01-01", "2016-12-31"),
                "test": ("2017-01-01", "2020-08-01")},
    # Same lengths shifted forward; the test window postdates the classic study.
    "recent": {"start": "2013-01-01", "fit_end": "2019-12-31", "valid": ("2020-01-01", "2021-12-31"),
               "test": ("2022-01-01", "2026-09-30")},
}
# Walk-forward: disjoint two-year tests, each after 7 training and 2 validation years.
WALK_FORWARD_YEARS = (2018, 2020, 2022, 2024)
for _year in WALK_FORWARD_YEARS:
    PROTOCOLS[f"wf{_year}"] = {
        "start": f"{_year - 9}-01-01", "fit_end": f"{_year - 3}-12-31",
        "valid": (f"{_year - 2}-01-01", f"{_year - 1}-12-31"),
        "test": (f"{_year}-01-01", f"{_year + 1}-12-31"),
    }
# Prospective holdout registered 2026-10-15: no decision may use its test days (docs/qlib-cn-protocol.md).
PROTOCOLS["live"] = {"start": "2017-10-01", "fit_end": "2024-09-30", "valid": ("2024-10-01", "2026-09-30"),
                     "test": ("2026-10-16", "2030-12-31")}
BENCHMARK_INDEX = {"csi300": "SH000300", "csi500": "SH000905", "csi1000": "SH000852"}
LABEL = ["Ref($close, -2)/Ref($close, -1) - 1"]
LGB_PARAMS = {
    "loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.2, "subsample": 0.8789,
    "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8, "num_leaves": 210,
    "num_threads": int(os.environ.get("LGB_THREADS", "20")),  # LGB_THREADS shares a machine
}

# --lgb-config presets: (parameter overrides, boosting rounds, early-stopping patience).
# "qlib" is the benchmark config; its subsample is inert without bagging_freq.
LGB_CONFIGS = {
    "qlib": ({}, 1000, 50),
    "qlib_bag": ({"bagging_freq": 1}, 1000, 50),
    "slow63": ({"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500, "bagging_freq": 1,
                "colsample_bytree": 0.8, "subsample": 0.8, "lambda_l1": 20.0, "lambda_l2": 60.0},
               3000, 200),
    "slow31": ({"learning_rate": 0.03, "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 1000,
                "bagging_freq": 1, "colsample_bytree": 0.7, "subsample": 0.8, "lambda_l1": 0.0,
                "lambda_l2": 10.0}, 4000, 300),
}


def _label(horizon: int) -> list[str]:
    """Training label: the ``horizon``-day forward return from the entry close (1 = ``LABEL``)."""
    return LABEL if horizon == 1 else [f"Ref($close, -{horizon + 1})/Ref($close, -1) - 1"]


def _purged(end: str, horizon: int, purge: str = "qlib") -> str:
    """Segment end for an ``horizon``-day label.

    A label on day t exits at close t+H+1. ``qlib`` lets the last labels reach
    two days past ``end``, as Qlib's own 1-day label does, so every horizon has
    the same overreach; ``full`` ends the segment at T-H-1 so no label leaves it.
    """
    from qlib.data import D

    calendar = D.calendar(end_time=end)
    offset = horizon + 2 if purge == "full" else (horizon if horizon > 1 else 1)
    return str(calendar[-offset].date())


def _processors(handler: str, robust_features: bool = False, feature_norm: str | None = None,
                label_norm: str | None = None) -> tuple[list, list]:
    """Processors (infer, learn). Defaults: Qlib's benchmark ones for ``alpha158`` (raw features,
    z-scored label), robust features and a ranked label otherwise or with ``robust_features``;
    ``feature_norm`` (raw | robust | robust_noclip) and ``label_norm`` (zscore | rank) override."""
    tuned = handler != "alpha158" or robust_features
    feature_norm = feature_norm or ("robust" if tuned else "raw")
    label_norm = label_norm or ("rank" if tuned else "zscore")
    infer = [] if feature_norm == "raw" else [
        {"class": "RobustZScoreNorm",
         "kwargs": {"fields_group": "feature", "clip_outlier": feature_norm == "robust"}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}}]
    norm = {"zscore": "CSZScoreNorm", "rank": "CSRankNorm"}[label_norm]
    return infer, [{"class": "DropnaLabel"}, {"class": norm, "kwargs": {"fields_group": "label"}}]


def _read_factors(path: Path) -> pd.DataFrame:
    factors = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    factors["datetime"] = pd.to_datetime(factors["datetime"])
    factors["instrument"] = factors["instrument"].str.upper()
    return factors.set_index(["datetime", "instrument"]).sort_index().astype(np.float32)


def _static_handler(args, split, features: pd.DataFrame, labels: pd.DataFrame, infer, learn):
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    joined = labels.join(features, how="left")
    joined.columns = pd.MultiIndex.from_tuples(
        [("label", "LABEL0")] + [("feature", c) for c in features.columns])
    # Rows are already the universe's point-in-time members (the label index).
    return DataHandlerLP(
        instruments=None, start_time=split["start"], end_time=split["test"][1],
        data_loader=StaticDataLoader(config=joined),
        infer_processors=infer, learn_processors=learn, process_type=DataHandlerLP.PTYPE_A,
    )


def _handler(args, split) -> object:
    from qlib.contrib.data.handler import Alpha158, Alpha360
    from qlib.data import D

    infer, learn = _processors("alpha158" if args.handler == "alpha158_plus" else args.handler,
                               robust_features=args.robust, feature_norm=args.feature_norm,
                               label_norm=args.label_norm)
    if args.model in SEQUENCE_MODELS:
        keep = list(QLIB_TS_COLUMNS)
        if args.factors is not None:
            keep += [f"fm_{c}" for c in _read_factors(args.factors).columns]
        infer = [{"class": "FilterCol", "kwargs": {"fields_group": "feature", "col_list": keep}}] + infer
    fit = {"fit_start_time": split["start"], "fit_end_time": split["fit_end"]}
    infer = [dict(p, kwargs={**p.get("kwargs", {}), **fit}) if p["class"] == "RobustZScoreNorm" else p
             for p in infer]
    common = {"instruments": args.universe, "start_time": split["start"], "end_time": split["test"][1], **fit}
    label = (_label(args.horizon), ["LABEL0"])
    if args.handler == "alpha158":
        return Alpha158(**common, infer_processors=infer, learn_processors=learn, label=label)
    if args.handler == "alpha360":
        return Alpha360(**common, infer_processors=infer, learn_processors=learn, label=label)

    factors = _read_factors(args.factors)
    labels = D.features(D.instruments(args.universe), _label(args.horizon), start_time=split["start"],
                        end_time=split["test"][1]).swaplevel().sort_index()
    labels.columns = ["LABEL0"]
    if args.handler == "alpha158_plus":
        base = Alpha158(**common, infer_processors=[], learn_processors=[])
        alpha = base.fetch(col_set="feature")
        factors = alpha.join(factors.add_prefix("fm_"), how="left")
    return _static_handler(args, split, factors, labels, infer, learn)


TOPK = 50  # TopkDropout book size
RIDGE_ALPHA = 1e5  # --model ridge: ~7% of a robust-scaled feature's sum of squares over ~1.4M train rows
RANK_GRADES = 6  # LambdaRank relevance: per-day sextile of the label (top grade ~ the book)
TOP_WEIGHT = 4.0  # topmse: extra weight on rows whose label is in the day's top sextile


def _model(name: str, objective: str = "mse", early_stop: str = "loss", seed: int = 0,
           top_weight: float = TOP_WEIGHT, patience: int = 50, temperature: float = 1.0,
           rounds: int = 1000):
    if name == "lgb" and (objective, early_stop) != ("mse", "loss"):
        return RankLGB(objective, early_stop, seed, top_weight=top_weight, num_boost_round=rounds,
                       early_stopping_rounds=patience, temperature=temperature)
    if name == "lgb":
        from qlib.contrib.model.gbdt import LGBModel

        return LGBModel(**LGB_PARAMS, num_boost_round=rounds, early_stopping_rounds=patience)
    if name == "linear":
        from qlib.contrib.model.linear import LinearModel

        return LinearModel(estimator="ols")
    if name == "ridge":
        from qlib.contrib.model.linear import LinearModel

        return LinearModel(estimator="ridge", alpha=RIDGE_ALPHA)
    threads = LGB_PARAMS["num_threads"]
    if name == "dens":  # workflow_config_doubleensemble_Alpha158.yaml
        from qlib.contrib.model.double_ensemble import DEnsembleModel

        lgb = {k: v for k, v in LGB_PARAMS.items() if k != "loss"}
        return DEnsembleModel(base_model="gbm", loss="mse", num_models=6, enable_sr=True, enable_fs=True,
                              alpha1=1, alpha2=1, bins_sr=10, bins_fs=5, decay=0.5,
                              sample_ratios=[0.8, 0.7, 0.6, 0.5, 0.4], sub_weights=[1] * 6, epochs=136,
                              verbosity=-1, seed=seed, **lgb)
    if name == "xgb":  # workflow_config_xgboost_Alpha158.yaml
        from qlib.contrib.model.xgboost import XGBModel

        # Qlib's wrapper ignores n_estimators (rounds are a fit() argument) and predicts with
        # every tree; fit for the config's 647 rounds without early stopping instead.
        model = XGBModel(eval_metric="rmse", colsample_bytree=0.8879, eta=0.0421, max_depth=8,
                         subsample=0.8789, nthread=threads, seed=seed)
        fit = model.fit
        model.fit = lambda dataset, **kw: fit(dataset, num_boost_round=647, early_stopping_rounds=10**6, **kw)
        return model
    if name == "cat":  # workflow_config_catboost_Alpha158.yaml
        from qlib.contrib.model.catboost_model import CatBoostModel

        return CatBoostModel(loss="RMSE", learning_rate=0.0421, subsample=0.8789, max_depth=6,
                             num_leaves=100, thread_count=threads, grow_policy="Lossguide",
                             bootstrap_type="Bernoulli", random_seed=seed)
    raise ValueError(name)


SEQUENCE_MODELS = {"gru", "alstm"}
STEP_LEN = 20  # Qlib's TSDatasetH window for the RNN benchmarks
QLIB_TS_COLUMNS = ["RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10", "ROC60",
                   "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5", "RSQR20", "CORD60",
                   "CORD10", "CORR20", "KLOW"]  # Qlib's Alpha158 subset for GRU/ALSTM/Transformer


def _torch_model(name: str, n_features: int, seed: int):
    """Qlib PyTorch reference models with their Alpha158 benchmark hyperparameters."""
    import torch

    torch.manual_seed(seed)
    common = dict(n_epochs=200, lr=1e-3, early_stop=10, batch_size=800, metric="loss",
                  loss="mse", n_jobs=20, GPU=0, seed=seed)
    if name == "mlp":
        from qlib.contrib.model.pytorch_nn import DNNModelPytorch

        return DNNModelPytorch(loss="mse", lr=0.002, optimizer="adam", max_steps=8000,
                               batch_size=8192, GPU=0, weight_decay=0.0002, seed=seed,
                               pt_model_kwargs={"input_dim": n_features})
    if name == "gru":
        from qlib.contrib.model.pytorch_gru_ts import GRU

        return GRU(d_feat=n_features, hidden_size=64, num_layers=2, dropout=0.0, **{**common, "lr": 2e-4})
    if name == "alstm":
        from qlib.contrib.model.pytorch_alstm_ts import ALSTM

        return ALSTM(d_feat=n_features, hidden_size=64, num_layers=2, dropout=0.0, rnn_type="GRU",
                     **common)
    raise ValueError(name)




def _day_groups(index: pd.MultiIndex) -> np.ndarray:
    """Row counts per datetime; rows must be sorted by datetime."""
    days = index.get_level_values("datetime")
    if not days.is_monotonic_increasing:
        raise ValueError("rows must be sorted by datetime")
    return np.diff(np.flatnonzero(np.r_[True, days[1:] != days[:-1], True]))


def _top_spread(pred: np.ndarray, raw: np.ndarray, groups: np.ndarray, k: int = TOPK) -> float:
    """Mean over days of the ``k`` best-scored names' label minus the day's mean (bps)."""
    day = np.repeat(np.arange(groups.size), groups)
    finite = np.isfinite(raw)
    order = np.lexsort((-pred, day))
    position = np.arange(order.size) - np.repeat(np.cumsum(groups) - groups, groups)
    top = np.zeros(order.size, dtype=bool)
    top[order[position < k]] = True
    top &= finite
    day_sum = np.bincount(day, np.where(finite, raw, 0.0), groups.size)
    day_n = np.bincount(day, finite, groups.size)
    top_sum = np.bincount(day, np.where(top, raw, 0.0), groups.size)
    top_n = np.bincount(day, top, groups.size)
    usable = (top_n > 0) & (day_n > 0)
    return float(np.mean(top_sum[usable] / top_n[usable] - day_sum[usable] / day_n[usable]) * 1e4)


def _grades(label: pd.Series) -> np.ndarray:
    """Per-day relevance grades 0..RANK_GRADES-1 from the label's percentile rank."""
    pct = label.groupby(level="datetime").rank(pct=True, method="first")
    return np.minimum((pct.to_numpy() * RANK_GRADES).astype(int), RANK_GRADES - 1)


class RankLGB:
    """LightGBM with a LambdaRank or MSE objective and top-50 early stopping.

    Matches Qlib's ``LGBModel`` interface (``fit(dataset)``, ``predict``) and
    its hyperparameters; only the objective and the stopping metric change.
    """

    def __init__(self, objective: str, early_stop: str, seed: int,
                 early_stopping_rounds: int = 50, num_boost_round: int = 1000,
                 top_weight: float = TOP_WEIGHT, temperature: float = 1.0) -> None:
        self.objective, self.early_stop, self.top_weight = objective, early_stop, top_weight
        self.params = {k: v for k, v in LGB_PARAMS.items() if k != "loss"}
        self.params.update(seed=seed, verbosity=-1)
        if objective in {"lambdarank", "xendcg"}:
            self.params.update(objective="lambdarank" if objective == "lambdarank" else "rank_xendcg",
                               lambda_l1=0.0, lambda_l2=0.0)
            if objective == "lambdarank":
                self.params.update(eval_at=[TOPK], lambdarank_truncation_level=TOPK + 10)
        elif objective == "topclass":
            self.params.update(objective="binary")
        elif objective == "listnet":
            self.params.update(objective=self._listnet)
        else:
            self.params.update(objective="regression")
        if early_stop == "top":
            self.params["metric"] = "None"
        self.rounds, self.patience = num_boost_round, early_stopping_rounds
        self.best_iteration = 0
        self.temperature = temperature
        self._groups: dict[int, np.ndarray] = {}

    def _listnet(self, pred: np.ndarray, data) -> tuple[np.ndarray, np.ndarray]:
        """ListNet: per-day softmax cross-entropy between scores and softmax(label / T).

        Gradients are scaled by the day's size so a row's step matches a
        regression residual and Qlib's L1/L2 penalties keep their meaning.
        """
        starts = self._groups[id(data)]
        sizes = np.diff(np.r_[starts, pred.size])
        day = np.repeat(np.arange(starts.size), sizes)
        target = data.get_label() / self.temperature

        def softmax(x: np.ndarray) -> np.ndarray:
            e = np.exp(x - np.maximum.reduceat(x, starts)[day])
            return e / np.add.reduceat(e, starts)[day]

        p, q, n = softmax(pred), softmax(target), sizes[day]
        return (p - q) * n, np.maximum(p * (1 - p) * n, 1e-6)

    def _segment(self, dataset, key: str):
        import lightgbm as lgb
        from qlib.data.dataset.handler import DataHandlerLP

        frame = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        frame = frame.sort_index(level="datetime", sort_remaining=True)
        label = frame["label"].iloc[:, 0]
        groups = _day_groups(frame.index)
        grades = _grades(label)
        ranked = self.objective in {"lambdarank", "xendcg"}
        if ranked:
            target = grades
        elif self.objective == "topclass":
            target = (grades == RANK_GRADES - 1).astype(np.float64)
        else:
            target = label.to_numpy()
        weight = 1.0 + self.top_weight * (grades == RANK_GRADES - 1) if self.objective == "topmse" else None
        data = lgb.Dataset(frame["feature"].to_numpy(), label=target, weight=weight,
                           group=groups if ranked else None, free_raw_data=False)
        self._groups[id(data)] = np.r_[0, np.cumsum(groups)[:-1]]
        return frame, data, groups

    def fit(self, dataset) -> None:
        import lightgbm as lgb
        from qlib.data.dataset.handler import DataHandlerLP

        _, train, _ = self._segment(dataset, "train")
        valid_frame, valid, valid_groups = self._segment(dataset, "valid")
        feval = None
        if self.early_stop == "top":
            raw = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
            raw = raw.reindex(valid_frame.index).to_numpy(dtype=np.float64)

            def feval(pred, _data):
                return f"top{TOPK}_bps", _top_spread(pred, raw, valid_groups), True
        booster = lgb.train(
            self.params, train, num_boost_round=self.rounds, valid_sets=[valid],
            valid_names=["valid"], feval=feval,
            callbacks=[lgb.early_stopping(self.patience, verbose=False), lgb.log_evaluation(50)])
        self.model, self.best_iteration = booster, booster.best_iteration
        self.best_valid = {name: float(value) for metric in booster.best_score.values()
                           for name, value in metric.items()}

    def predict(self, dataset, segment: str = "test") -> pd.Series:
        from qlib.data.dataset.handler import DataHandlerLP

        x = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self.model.predict(x.to_numpy(), num_iteration=self.best_iteration),
                         index=x.index)


def _refit(model, dataset, start: str, end: str) -> int:
    """Refit a fitted LightGBM on ``start..end`` (train + valid) with its selected tree count."""
    import lightgbm as lgb
    from qlib.data.dataset.handler import DataHandlerLP

    if isinstance(model, RankLGB):
        if model.objective not in {"mse", "topmse"}:
            raise ValueError("--refit supports the mse and topmse objectives")
        rounds = max(model.best_iteration, 1)
        _, data, _ = model._segment(dataset, slice(start, end))
        model.model = lgb.train(model.params, data, num_boost_round=rounds)
        model.best_iteration = rounds
        return rounds
    rounds = max(model.model.best_iteration, 1)
    frame = dataset.prepare(slice(start, end), col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    data = lgb.Dataset(frame["feature"].to_numpy(), label=frame["label"].iloc[:, 0].to_numpy())
    model.model = lgb.train(model.params, data, num_boost_round=rounds)
    return rounds


def _checkpoints(booster, dataset, args, counts: list[int]) -> None:
    """Write the first-k-trees scores of ``booster`` as sibling runs ``<out>@t<k>``."""
    from qlib.data.dataset.handler import DataHandlerLP

    x = {seg: dataset.prepare(seg, col_set="feature", data_key=DataHandlerLP.DK_I) for seg in ("valid", "test")}
    for k in counts:
        out = args.out.parent / f"{args.out.name}@t{k}"
        out.mkdir(parents=True, exist_ok=True)
        for seg, file in (("valid", "valid_predictions.parquet"), ("test", "predictions.parquet")):
            score = pd.Series(booster.predict(x[seg].to_numpy(), num_iteration=k), index=x[seg].index, name="score")
            score.to_frame().reset_index().to_parquet(out / file, index=False)
        (out / "metrics.json").write_text(json.dumps({"checkpoint_of": str(args.out), "trees": k}) + "\n")


def _load(model, path: Path) -> None:
    """Install a saved LightGBM booster in place of fitting."""
    import lightgbm as lgb

    if not hasattr(model, "model") or not (isinstance(model, RankLGB) or type(model).__name__ == "LGBModel"):
        raise ValueError("--load-model applies to --model lgb only")
    model.model = lgb.Booster(model_file=str(path))
    if isinstance(model, RankLGB):
        model.best_iteration = model.model.current_iteration()
        model.best_valid = {}


def _ic_metrics(pred: pd.Series, label: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"pred": pred, "label": label}).dropna()
    grouped = frame.groupby(level="datetime")
    ic = grouped.apply(lambda f: f["pred"].corr(f["label"]))
    ric = grouped.apply(lambda f: f["pred"].corr(f["label"], method="spearman"))
    return {"IC": float(ic.mean()), "ICIR": float(ic.mean() / ic.std()),
            "RankIC": float(ric.mean()), "RankICIR": float(ric.mean() / ric.std()),
            "test_dates": int(ic.notna().sum()), "test_rows": int(len(frame))}


def _untie(pred: pd.Series, salt: int = 0) -> pd.Series:
    """Same daily order, ties broken by a stable pseudo-random key per (day, instrument, salt).

    Shallow, heavily regularised trees emit few distinct scores (up to ~90% of a
    day's names tied). TopkDropout then picks among tied names in an order set by
    Python's per-process string hashing, so the same predictions backtest
    differently from run to run.
    """
    dense = pred.groupby(level="datetime").rank(method="dense")
    key = pd.util.hash_pandas_object(pred.index.to_frame(index=False).assign(salt=salt), index=False)
    return dense + 0.5 * (key.to_numpy() >> np.uint64(11)).astype(np.float64) / 2.0**53


EXCHANGE = {"freq": "day", "limit_threshold": 0.095, "deal_price": "close",
            "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5}
_EXCHANGES: dict = {}


def _exchange(universe: str):
    """One Exchange per universe over the provider's whole calendar, built once per process.

    Qlib builds an Exchange (every quote between start and end, all instruments) for each
    backtest, about a minute for an eight-year book. The book only ever trades names that
    held membership, so one exchange over those codes and the full calendar answers every
    backtest of the process with the same fills.
    """
    if universe not in _EXCHANGES:
        from qlib.backtest.exchange import Exchange
        from qlib.data import D

        calendar = D.calendar(freq="day")
        codes = D.list_instruments(D.instruments(universe), start_time=calendar[0], end_time=calendar[-1],
                                   as_list=True)
        _EXCHANGES[universe] = Exchange(start_time=calendar[0], end_time=calendar[-1], codes=sorted(codes),
                                        **EXCHANGE)
    return _EXCHANGES[universe]


def _backtest(pred: pd.Series, universe: str, split, n_drop: int = 5,
              return_report: bool = False, tie_salt: int = 0, cache_exchange: bool = True):
    from qlib.backtest import backtest, executor
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy

    strategy = TopkDropoutStrategy(signal=_untie(pred, tie_salt).to_frame("score"), topk=TOPK, n_drop=n_drop)
    # The test window can run past the provider's calendar (protocol live): stop at the last signal.
    end = min(pd.Timestamp(split["test"][1]), pred.index.get_level_values("datetime").max())
    exec_obj = executor.SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    portfolio, _ = backtest(
        start_time=split["test"][0], end_time=end, strategy=strategy, executor=exec_obj,
        benchmark=BENCHMARK_INDEX[universe], account=100_000_000,
        exchange_kwargs={"exchange": _exchange(universe)} if cache_exchange else dict(EXCHANGE),
    )
    report, _ = portfolio["1day"]
    out: dict[str, float] = {}
    for key, series in {"excess_return_without_cost": report["return"] - report["bench"],
                        "excess_return_with_cost": report["return"] - report["bench"] - report["cost"]}.items():
        risk = risk_analysis(series, freq="day")["risk"]
        out[f"{key}.annualized_return"] = float(risk["annualized_return"])
        out[f"{key}.information_ratio"] = float(risk["information_ratio"])
        out[f"{key}.max_drawdown"] = float(risk["max_drawdown"])
    out["turnover_mean"] = float(report["turnover"].mean())
    return (out, report) if return_report else out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--universe", choices=sorted(BENCHMARK_INDEX), required=True)
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--handler", choices=["alpha158", "alpha360", "factors", "alpha158_plus"],
                        required=True)
    parser.add_argument("--factors", type=Path, default=None)
    parser.add_argument("--robust", action="store_true",
                        help="robust feature/rank label processors for the Alpha158 handlers")
    parser.add_argument("--model", choices=["lgb", "linear", "ridge", "dens", "xgb", "cat", "mlp", "gru", "alstm"], default="lgb")
    parser.add_argument("--objective", choices=["mse", "topmse", "lambdarank", "xendcg", "listnet",
                                                "topclass"], default="mse",
                        help="LightGBM objective: mse, topmse (top-sextile-weighted mse), lambdarank "
                             "or xendcg (rank each day on sextile grades), listnet (per-day softmax "
                             "cross-entropy), topclass (logloss on the top-sextile indicator)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="listnet: target softmax(label / T); lower T concentrates on the top")
    parser.add_argument("--early-stop", choices=["loss", "top"], default=None,
                        help="validation stopping metric (default: loss for mse, top otherwise)")
    parser.add_argument("--top-weight", type=float, default=TOP_WEIGHT,
                        help="topmse: extra weight on top-sextile rows")
    parser.add_argument("--patience", type=int, default=50,
                        help="early-stopping rounds for the non-default objectives (Qlib: 50)")
    parser.add_argument("--feature-norm", choices=["raw", "robust", "robust_noclip"], default=None,
                        help="feature processing (default: raw for alpha158, robust with --robust)")
    parser.add_argument("--label-norm", choices=["zscore", "rank"], default=None,
                        help="label normalisation (default: zscore for alpha158, rank with --robust)")
    parser.add_argument("--purge", choices=["qlib", "full"], default="qlib",
                        help="segment ends: qlib (labels overreach 2 days, as Qlib's) or full")
    parser.add_argument("--lgb-config", choices=sorted(LGB_CONFIGS), default="qlib",
                        help="LightGBM hyperparameter preset (qlib: the benchmark config)")
    parser.add_argument("--refit", action="store_true",
                        help="LightGBM: after early stopping, refit on train+valid with that tree count")
    parser.add_argument("--load-model", type=Path, default=None,
                        help="LightGBM: predict with this saved booster (model.txt) instead of fitting")
    parser.add_argument("--checkpoints", default=None,
                        help="LightGBM: comma-separated tree counts; no early stopping, one sibling run each")
    parser.add_argument("--horizon", type=int, default=1,
                        help="training label horizon in days (metrics and backtest stay 1-day)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.handler in {"factors", "alpha158_plus"}) != (args.factors is not None):
        parser.error("--factors is required exactly for --handler factors/alpha158_plus")
    if args.early_stop is None:
        args.early_stop = "loss" if args.objective == "mse" else "top"
    if args.model != "lgb" and (args.objective, args.early_stop) != ("mse", "loss"):
        parser.error("--objective/--early-stop apply to --model lgb only")

    import qlib
    from qlib.data.dataset import DatasetH

    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    # Qlib's recorder uses MLflow's file store, which newer MLflow gates behind an opt-in.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    qlib.init(provider_uri=str(args.provider.expanduser()), region="cn",
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": str((args.out / "mlruns").resolve()),
                                      "default_exp_name": "factorminer_benchmark"}})
    split = PROTOCOLS[args.protocol]
    started = time.monotonic()
    handler = _handler(args, split)
    segments = {"train": (split["start"], _purged(split["fit_end"], args.horizon, args.purge)),
                "valid": (split["valid"][0], _purged(split["valid"][1], args.horizon, args.purge)),
                "test": split["test"]}
    dataset = DatasetH(handler=handler, segments=segments)
    from qlib.data.dataset.handler import DataHandlerLP

    if args.model in SEQUENCE_MODELS or args.model == "mlp":
        n_features = dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L).shape[1]
        model = _torch_model(args.model, n_features, args.seed)
        if args.model in SEQUENCE_MODELS:
            from qlib.data.dataset import TSDatasetH

            fit_dataset = TSDatasetH(handler=handler, segments=segments, step_len=STEP_LEN)
        else:
            fit_dataset = dataset
    else:
        fit_dataset = dataset
        overrides, rounds, patience = LGB_CONFIGS[args.lgb_config]
        LGB_PARAMS.update(overrides)
        if args.lgb_config == "qlib":
            patience = args.patience
        if args.checkpoints:
            if args.model != "lgb" or args.refit:
                parser.error("--checkpoints applies to --model lgb without --refit")
            rounds = max(int(k) for k in args.checkpoints.split(","))
            patience = rounds + 1
        model = _model(args.model, args.objective, args.early_stop, args.seed, args.top_weight,
                       patience, args.temperature, rounds)
        if isinstance(model, RankLGB):
            LGB_PARAMS["seed"] = args.seed
        elif args.model == "lgb":
            LGB_PARAMS["seed"] = args.seed
            model.params["seed"] = args.seed
    if args.load_model is not None:
        _load(model, args.load_model)
    else:
        model.fit(fit_dataset)
    valid_pred = model.predict(fit_dataset, segment="valid")
    if isinstance(valid_pred, pd.DataFrame):
        valid_pred = valid_pred.iloc[:, 0]
    refit_rounds = None
    if args.refit and args.load_model is None:
        if args.model != "lgb":
            parser.error("--refit applies to --model lgb only")
        refit_rounds = _refit(model, dataset, segments["train"][0], segments["valid"][1])
    from qlib.data import D

    if len(D.calendar(start_time=split["test"][0], end_time=split["test"][1])):
        pred = model.predict(fit_dataset) if args.model in SEQUENCE_MODELS else model.predict(fit_dataset, segment="test")
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        pred = pred.dropna()
    else:  # protocol live before its first test session: save the model and validation scores only
        pred = pd.Series(dtype=np.float64, name="score",
                         index=pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), []], names=["datetime", "instrument"]))

    if pred.empty:
        metrics = {"test_dates": 0}
    else:
        if args.horizon == 1:
            raw_label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
        else:
            raw_label = D.features(D.instruments(args.universe), LABEL, start_time=split["test"][0],
                                   end_time=split["test"][1]).swaplevel().sort_index().iloc[:, 0]
            raw_label = raw_label.reindex(pred.index)
        metrics = {**_ic_metrics(pred, raw_label), **_backtest(pred, args.universe, split)}
    metrics["fit_seconds"] = round(time.monotonic() - started, 1)
    if isinstance(model, RankLGB):
        metrics["best_iteration"] = int(model.best_iteration)
        metrics.update({f"valid_best_{k}": v for k, v in model.best_valid.items()})

    args.out.mkdir(parents=True, exist_ok=True)
    pred.rename("score").to_frame().reset_index().to_parquet(args.out / "predictions.parquet", index=False)
    valid_pred.dropna().rename("score").to_frame().reset_index().to_parquet(
        args.out / "valid_predictions.parquet", index=False)
    if args.checkpoints:
        _checkpoints(model.model, dataset, args, [int(k) for k in args.checkpoints.split(",")])
    if args.model == "lgb":
        model.model.save_model(str(args.out / "model.txt"))
    (args.out / "metrics.json").write_text(json.dumps({
        "universe": args.universe, "protocol": args.protocol, "split": split,
        "handler": args.handler, "robust": args.robust, "model": args.model, "seed": args.seed,
        "objective": args.objective, "early_stop": args.early_stop,
        "top_weight": args.top_weight if args.objective == "topmse" else None,
        "patience": args.patience,
        "temperature": args.temperature if args.objective == "listnet" else None,
        "label": LABEL[0], "train_label": _label(args.horizon)[0], "horizon": args.horizon,
        "segments": segments, "purge": args.purge, "feature_norm": args.feature_norm,
        "label_norm": args.label_norm, "refit_rounds": refit_rounds, "lgb_config": args.lgb_config, "lgb_params": LGB_PARAMS if args.model == "lgb" else None,
        "qlib_version": qlib.__version__, "metrics": metrics,
        "factors_file": str(args.factors) if args.factors else None,
        "loaded_model": str(args.load_model) if args.load_model else None,
    }, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
