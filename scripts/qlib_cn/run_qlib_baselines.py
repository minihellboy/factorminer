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
    "num_threads": 20,
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


def _model(name: str):
    if name == "lgb":
        from qlib.contrib.model.gbdt import LGBModel

        return LGBModel(**LGB_PARAMS)
    if name == "linear":
        from qlib.contrib.model.linear import LinearModel

        return LinearModel(estimator="ols")
    raise ValueError(name)


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
    parser.add_argument("--model", choices=["lgb", "linear"], default="lgb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.handler in {"factors", "alpha158_plus"}) != (args.factors is not None):
        parser.error("--factors is required exactly for --handler factors/alpha158_plus")

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
    dataset = DatasetH(handler=handler, segments={
        "train": (split["start"], split["fit_end"]), "valid": split["valid"], "test": split["test"]})
    model = _model(args.model)
    if args.model == "lgb":
        LGB_PARAMS["seed"] = args.seed
        model.params["seed"] = args.seed
    model.fit(dataset)
    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    from qlib.data.dataset.handler import DataHandlerLP

    raw_label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R).iloc[:, 0]
    metrics = {**_ic_metrics(pred, raw_label), **_backtest(pred, args.universe, split)}
    metrics["fit_seconds"] = round(time.monotonic() - started, 1)

    args.out.mkdir(parents=True, exist_ok=True)
    pred.rename("score").to_frame().reset_index().to_parquet(args.out / "predictions.parquet", index=False)
    (args.out / "metrics.json").write_text(json.dumps({
        "universe": args.universe, "protocol": args.protocol, "split": split,
        "handler": args.handler, "robust": args.robust, "model": args.model, "seed": args.seed,
        "label": LABEL[0], "lgb_params": LGB_PARAMS if args.model == "lgb" else None,
        "qlib_version": qlib.__version__, "metrics": metrics,
        "factors_file": str(args.factors) if args.factors else None,
    }, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
