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
names minus the day's mean) rather than on the loss.

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
BENCHMARK_INDEX = {"csi300": "SH000300", "csi500": "SH000905"}
LABEL = ["Ref($close, -2)/Ref($close, -1) - 1"]
LGB_PARAMS = {
    "loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.2, "subsample": 0.8789,
    "lambda_l1": 205.6999, "lambda_l2": 580.9768, "max_depth": 8, "num_leaves": 210,
    "num_threads": int(os.environ.get("LGB_THREADS", "20")),  # LGB_THREADS shares a machine
}


def _processors(handler: str, robust_features: bool = False) -> tuple[list, list]:
    """Qlib benchmark processors: (infer, learn)."""
    label_norm = {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
    if handler == "alpha158" and not robust_features:
        return [], [{"class": "DropnaLabel"}, {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}}]
    robust = [{"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
              {"class": "Fillna", "kwargs": {"fields_group": "feature"}}]
    return robust, [{"class": "DropnaLabel"}, label_norm]


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
                               robust_features=args.robust)
    if args.model in SEQUENCE_MODELS:
        keep = list(QLIB_TS_COLUMNS)
        if args.factors is not None:
            keep += [f"fm_{c}" for c in _read_factors(args.factors).columns]
        infer = [{"class": "FilterCol", "kwargs": {"fields_group": "feature", "col_list": keep}}] + infer
    fit = {"fit_start_time": split["start"], "fit_end_time": split["fit_end"]}
    infer = [dict(p, kwargs={**p.get("kwargs", {}), **fit}) if p["class"] == "RobustZScoreNorm" else p
             for p in infer]
    common = {"instruments": args.universe, "start_time": split["start"], "end_time": split["test"][1], **fit}
    if args.handler == "alpha158":
        return Alpha158(**common, infer_processors=infer, learn_processors=learn)
    if args.handler == "alpha360":
        return Alpha360(**common, infer_processors=infer, learn_processors=learn)

    factors = _read_factors(args.factors)
    labels = D.features(D.instruments(args.universe), LABEL, start_time=split["start"],
                        end_time=split["test"][1]).swaplevel().sort_index()
    labels.columns = ["LABEL0"]
    if args.handler == "alpha158_plus":
        base = Alpha158(**common, infer_processors=[], learn_processors=[])
        alpha = base.fetch(col_set="feature")
        factors = alpha.join(factors.add_prefix("fm_"), how="left")
    return _static_handler(args, split, factors, labels, infer, learn)


TOPK = 50  # TopkDropout book size
RANK_GRADES = 6  # LambdaRank relevance: per-day sextile of the label (top grade ~ the book)
TOP_WEIGHT = 4.0  # topmse: extra weight on rows whose label is in the day's top sextile


def _model(name: str, objective: str = "mse", early_stop: str = "loss", seed: int = 0,
           top_weight: float = TOP_WEIGHT):
    if name == "lgb" and (objective, early_stop) != ("mse", "loss"):
        return RankLGB(objective, early_stop, seed, top_weight=top_weight)
    if name == "lgb":
        from qlib.contrib.model.gbdt import LGBModel

        return LGBModel(**LGB_PARAMS)
    if name == "linear":
        from qlib.contrib.model.linear import LinearModel

        return LinearModel(estimator="ols")
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
                 top_weight: float = TOP_WEIGHT) -> None:
        self.objective, self.early_stop, self.top_weight = objective, early_stop, top_weight
        self.params = {k: v for k, v in LGB_PARAMS.items() if k != "loss"}
        self.params.update(seed=seed, verbosity=-1)
        if objective == "lambdarank":
            self.params.update(objective="lambdarank", eval_at=[TOPK],
                               lambdarank_truncation_level=TOPK + 10,
                               lambda_l1=0.0, lambda_l2=0.0)
        else:
            self.params.update(objective="regression")
        if early_stop == "top":
            self.params["metric"] = "None"
        self.rounds, self.patience = num_boost_round, early_stopping_rounds
        self.best_iteration = 0

    def _segment(self, dataset, key: str):
        import lightgbm as lgb
        from qlib.data.dataset.handler import DataHandlerLP

        frame = dataset.prepare(key, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        frame = frame.sort_index(level="datetime", sort_remaining=True)
        label = frame["label"].iloc[:, 0]
        groups = _day_groups(frame.index)
        grades = _grades(label)
        target = grades if self.objective == "lambdarank" else label.to_numpy()
        weight = 1.0 + self.top_weight * (grades == RANK_GRADES - 1) if self.objective == "topmse" else None
        data = lgb.Dataset(frame["feature"].to_numpy(), label=target, weight=weight,
                           group=groups if self.objective == "lambdarank" else None)
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


def _ic_metrics(pred: pd.Series, label: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"pred": pred, "label": label}).dropna()
    grouped = frame.groupby(level="datetime")
    ic = grouped.apply(lambda f: f["pred"].corr(f["label"]))
    ric = grouped.apply(lambda f: f["pred"].corr(f["label"], method="spearman"))
    return {"IC": float(ic.mean()), "ICIR": float(ic.mean() / ic.std()),
            "RankIC": float(ric.mean()), "RankICIR": float(ric.mean() / ric.std()),
            "test_dates": int(ic.notna().sum()), "test_rows": int(len(frame))}


def _backtest(pred: pd.Series, universe: str, split) -> dict[str, float]:
    from qlib.backtest import backtest, executor
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy

    strategy = TopkDropoutStrategy(signal=pred.to_frame("score"), topk=50, n_drop=5)
    exec_obj = executor.SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True)
    portfolio, _ = backtest(
        start_time=split["test"][0], end_time=split["test"][1], strategy=strategy, executor=exec_obj,
        benchmark=BENCHMARK_INDEX[universe], account=100_000_000,
        exchange_kwargs={"freq": "day", "limit_threshold": 0.095, "deal_price": "close",
                         "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5},
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
    return out


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
    parser.add_argument("--model", choices=["lgb", "linear", "mlp", "gru", "alstm"], default="lgb")
    parser.add_argument("--objective", choices=["mse", "topmse", "lambdarank"], default="mse",
                        help="LightGBM objective: mse, topmse (top-sextile-weighted mse), or "
                             "lambdarank (ranks each day for a top-50 book)")
    parser.add_argument("--early-stop", choices=["loss", "top"], default=None,
                        help="validation stopping metric (default: loss for mse, top otherwise)")
    parser.add_argument("--top-weight", type=float, default=TOP_WEIGHT,
                        help="topmse: extra weight on top-sextile rows")
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
    segments = {"train": (split["start"], split["fit_end"]), "valid": split["valid"], "test": split["test"]}
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
        model = _model(args.model, args.objective, args.early_stop, args.seed, args.top_weight)
        if isinstance(model, RankLGB):
            LGB_PARAMS["seed"] = args.seed
        elif args.model == "lgb":
            LGB_PARAMS["seed"] = args.seed
            model.params["seed"] = args.seed
    model.fit(fit_dataset)
    pred = model.predict(fit_dataset) if args.model in SEQUENCE_MODELS else model.predict(fit_dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred = pred.dropna()

    raw_label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
    metrics = {**_ic_metrics(pred, raw_label), **_backtest(pred, args.universe, split)}
    metrics["fit_seconds"] = round(time.monotonic() - started, 1)
    if isinstance(model, RankLGB):
        metrics["best_iteration"] = int(model.best_iteration)
        metrics.update({f"valid_best_{k}": v for k, v in model.best_valid.items()})

    args.out.mkdir(parents=True, exist_ok=True)
    pred.rename("score").to_frame().reset_index().to_parquet(args.out / "predictions.parquet", index=False)
    (args.out / "metrics.json").write_text(json.dumps({
        "universe": args.universe, "protocol": args.protocol, "split": split,
        "handler": args.handler, "robust": args.robust, "model": args.model, "seed": args.seed,
        "objective": args.objective, "early_stop": args.early_stop,
        "top_weight": args.top_weight if args.objective == "topmse" else None,
        "label": LABEL[0], "lgb_params": LGB_PARAMS if args.model == "lgb" else None,
        "qlib_version": qlib.__version__, "metrics": metrics,
        "factors_file": str(args.factors) if args.factors else None,
    }, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
