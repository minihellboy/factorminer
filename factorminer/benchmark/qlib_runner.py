"""Isolated Qlib baseline over the exact frozen FactorMiner panel."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _selectors(dataset: Any) -> tuple[np.ndarray, np.ndarray, int]:
    train = np.asarray(dataset.get_split("train").indices, dtype=np.int64)
    test = np.asarray(dataset.get_split("test").indices, dtype=np.int64)
    if not len(train) or not len(test) or np.any(np.diff(train) <= 0) or np.any(np.diff(test) <= 0):
        raise ValueError("Qlib requires nonempty, strictly ordered train/test selectors")
    if train[-1] >= test[0]:
        raise ValueError("Qlib requires chronological, disjoint train/test periods")
    target = dataset.target_specs[dataset.default_target]
    maturity = int(target.entry_delay_bars) + int(target.holding_bars)
    train = train[train + maturity < test[0]]
    if not len(train):
        raise ValueError("Qlib has no mature training labels before the test period")
    return train, test, maturity


def _panel_arrays(dataset: Any, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Qlib features must be unique and nonempty")
    features = np.stack([np.asarray(dataset.data_dict[name], dtype=np.float64).T
                         for name in feature_names], axis=2)
    labels = np.asarray(dataset.get_target(), dtype=np.float64).T
    times = pd.to_datetime(dataset.timestamps).to_numpy(dtype="datetime64[ns]").view("int64")
    assets = np.asarray(dataset.asset_ids, dtype=str)
    if (features.shape[:2] != labels.shape or features.shape[0] != len(times)
            or features.shape[1] != len(assets) or np.any(np.diff(times) <= 0)
            or len(set(assets)) != len(assets)):
        raise ValueError("Qlib frozen panel has inconsistent axes")
    return features, labels, times, assets


def _metric_payload(prediction: np.ndarray, labels: np.ndarray, test: np.ndarray,
                    *, qlib: bool) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    if qlib:
        from qlib.contrib.eva.alpha import calc_ic  # type: ignore[import-not-found]

    n_assets = labels.shape[1]
    dates = np.repeat(pd.to_datetime(test), n_assets)
    instruments = np.tile(np.arange(n_assets), len(test))
    index = pd.MultiIndex.from_arrays([dates, instruments], names=["datetime", "instrument"])
    pred_series = pd.Series(prediction, index=index)
    label_series = pd.Series(labels.ravel(), index=index)
    if qlib:
        ic, rank_ic = calc_ic(pred_series, label_series)
    else:
        grouped = pd.DataFrame({"pred": pred_series, "label": label_series}).groupby(level="datetime")
        ic = grouped.apply(lambda frame: frame["pred"].corr(frame["label"]))
        rank_ic = grouped.apply(lambda frame: frame["pred"].corr(frame["label"], method="spearman"))
    ic_values = ic.to_numpy(dtype=np.float64)
    rank_values = rank_ic.to_numpy(dtype=np.float64)
    valid_ic = ic_values[np.isfinite(ic_values)]
    valid_rank = rank_values[np.isfinite(rank_values)]
    if len(valid_ic) < 2 or len(valid_rank) < 2:
        raise ValueError("Qlib needs at least two scored test dates")
    metrics: dict[str, float | int] = {
        "IC": float(valid_ic.mean()),
        "ICIR": float(valid_ic.mean() / valid_ic.std(ddof=1)),
        "Rank IC": float(valid_rank.mean()),
        "Rank ICIR": float(valid_rank.mean() / valid_rank.std(ddof=1)),
        "ic_days": int(len(valid_ic)),
        "rank_ic_days": int(len(valid_rank)),
        "prediction_rows": int(prediction.size),
        "scored_rows": int(np.count_nonzero(np.isfinite(prediction) & np.isfinite(labels.ravel()))),
    }
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError("Qlib returned nonfinite aggregate metrics")
    return metrics, ic_values, rank_values


def _worker(job_path: Path, panel_path: Path, output_dir: Path) -> None:
    import qlib  # type: ignore[import-not-found]
    from qlib.contrib.model.linear import LinearModel  # type: ignore[import-not-found]
    from qlib.data.dataset import DatasetH  # type: ignore[import-not-found]
    from qlib.data.dataset.handler import DataHandlerLP  # type: ignore[import-not-found]

    job = json.loads(job_path.read_text())
    if job["panel_sha256"] != _sha256(panel_path):
        raise ValueError("Qlib frozen panel digest changed")
    with np.load(panel_path, allow_pickle=False) as frozen:
        features = frozen["features"]
        labels = frozen["labels"]
        times = frozen["times"]
        assets = frozen["assets"].astype(str)
    train = np.asarray(job["train_indices"], dtype=np.int64)
    test = np.asarray(job["test_indices"], dtype=np.int64)
    if (features.ndim != 3 or labels.shape != features.shape[:2]
            or len(times) != features.shape[0] or len(assets) != features.shape[1]
            or features.shape[2] != len(job["features"])
            or not len(train) or not len(test) or train[-1] >= test[0]
            or np.any(np.diff(train) <= 0) or np.any(np.diff(test) <= 0)
            or train[0] < 0 or test[-1] >= len(times)):
        raise ValueError("Qlib job selectors or panel shape are invalid")
    chosen = np.concatenate([train, test])
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(times[chosen]), assets], names=["datetime", "instrument"]
    )
    feature_frame = pd.DataFrame(features[chosen].reshape(-1, features.shape[2]),
                                 index=index, columns=job["features"])
    label_frame = pd.DataFrame({job["target"]: labels[chosen].ravel()}, index=index)
    handler = DataHandlerLP.from_df(pd.concat({"feature": feature_frame, "label": label_frame}, axis=1))
    segments = {
        "train": (pd.Timestamp(times[train[0]]), pd.Timestamp(times[train[-1]])),
        "test": (pd.Timestamp(times[test[0]]), pd.Timestamp(times[test[-1]])),
    }
    qlib_dataset = DatasetH(handler=handler, segments=segments)
    train_frame = qlib_dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    test_frame = qlib_dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    if not train_frame.index.equals(index[:len(train) * len(assets)]) or not test_frame.index.equals(index[len(train) * len(assets):]):
        raise ValueError("Qlib handler changed frozen train/test row membership")
    model = LinearModel(estimator="ridge", alpha=float(job["alpha"]),
                        fit_intercept=True, include_valid=False)
    model.fit(qlib_dataset)
    predicted = model.predict(qlib_dataset, segment="test")
    if not predicted.index.equals(test_frame.index):
        raise ValueError("Qlib predictions are not aligned with the frozen test rows")
    prediction = predicted.to_numpy(dtype=np.float64)
    metrics, ic, rank_ic = _metric_payload(prediction, labels[test], times[test], qlib=True)
    train_rows = int(train_frame.dropna().shape[0])
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez(output_dir / "predictions.npz", prediction=prediction, ic=ic, rank_ic=rank_ic)
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(output_dir / "model.json", {
        "class": "qlib.contrib.model.linear.LinearModel", "estimator": "ridge",
        "alpha": float(job["alpha"]), "fit_intercept": True,
        "coefficients": np.asarray(model.coef_, dtype=np.float64).tolist(),
        "intercept": float(model.intercept_), "train_rows": train_rows,
        "test_rows": int(len(prediction)), "qlib_version": qlib.__version__,
        "panel_sha256": job["panel_sha256"], "job_sha256": _sha256(job_path),
    })


def audit_qlib_against_dataset(dataset: Any, cfg: Any, job_path: Path,
                               predictions_path: Path, metrics_path: Path,
                               model_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refit and rescore a Qlib result from the committed FactorMiner input."""
    from sklearn.linear_model import Ridge  # type: ignore[import-untyped]

    job = json.loads(job_path.read_text())
    feature_names = list(cfg.data.features)
    train, test, maturity = _selectors(dataset)
    features, labels, times, assets = _panel_arrays(dataset, feature_names)
    if (job["schema_version"] != 1
            or job["features"] != feature_names or job["target"] != dataset.default_target
            or job["train_indices"] != train.tolist() or job["test_indices"] != test.tolist()
            or job["label_maturity_bars"] != maturity
            or job["asset_count"] != len(assets) or job["timestamp_count"] != len(times)
            or job["protocol"] != "frozen_panel_qlib_ridge"):
        raise ValueError("Qlib job differs from committed frozen dataset selectors")
    with tempfile.TemporaryDirectory() as temporary:
        recreated = Path(temporary) / "panel.npz"
        np.savez(recreated, features=features, labels=labels, times=times, assets=assets)
        if _sha256(recreated) != job["panel_sha256"]:
            raise ValueError("Qlib panel differs from committed frozen dataset")
    with np.load(predictions_path, allow_pickle=False) as result:
        prediction = np.asarray(result["prediction"], dtype=np.float64)
        worker_ic = np.asarray(result["ic"], dtype=np.float64)
        worker_rank_ic = np.asarray(result["rank_ic"], dtype=np.float64)
    metrics = json.loads(metrics_path.read_text())
    model = json.loads(model_path.read_text())
    train_x = features[train].reshape(-1, len(feature_names))
    train_y = labels[train].ravel()
    valid_train = np.isfinite(train_x).all(axis=1) & np.isfinite(train_y)
    independent = Ridge(alpha=float(job["alpha"]), fit_intercept=True, copy_X=True).fit(
        train_x[valid_train], train_y[valid_train]
    )
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if (model["class"] != "qlib.contrib.model.linear.LinearModel"
            or model["estimator"] != "ridge" or model["alpha"] != job["alpha"]
            or model["fit_intercept"] is not True
            or model["panel_sha256"] != job["panel_sha256"]
            or model["job_sha256"] != _sha256(job_path)
            or model["train_rows"] != int(valid_train.sum())
            or model["test_rows"] != len(test) * len(assets)
            or not np.allclose(coefficients, independent.coef_, rtol=1e-8, atol=1e-10)
            or not np.isclose(model["intercept"], independent.intercept_, rtol=1e-8, atol=1e-10)):
        raise ValueError("Qlib model failed independent frozen-training audit")
    test_x = features[test].reshape(-1, len(feature_names))
    expected = test_x @ independent.coef_ + independent.intercept_
    if prediction.shape != expected.shape or not np.allclose(
        prediction, expected, rtol=1e-8, atol=1e-10, equal_nan=True
    ):
        raise ValueError("Qlib predictions failed independent frozen-test audit")
    expected_metrics, ic, rank_ic = _metric_payload(prediction, labels[test], times[test], qlib=False)
    if (metrics.keys() != expected_metrics.keys()
            or any(not np.isclose(metrics[key], value, rtol=1e-8, atol=1e-10)
                   for key, value in expected_metrics.items())
            or not np.allclose(worker_ic, ic, rtol=1e-8, atol=1e-10, equal_nan=True)
            or not np.allclose(worker_rank_ic, rank_ic, rtol=1e-8, atol=1e-10, equal_nan=True)):
        raise ValueError("Qlib metrics failed independent frozen-label audit")
    return metrics, model


def run_isolated_qlib(dataset: Any, cfg: Any, output_dir: Path, *,
                      python: str | None = None, alpha: float = 1.0,
                      timeout_seconds: int = 300) -> tuple[dict[str, Any], dict[str, Path]]:
    """Run Qlib in a separate interpreter and audit its result before release."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Qlib ridge alpha must be finite and positive")
    output_dir = output_dir.resolve()
    train, test, maturity = _selectors(dataset)
    feature_names = list(cfg.data.features)
    features, labels, times, assets = _panel_arrays(dataset, feature_names)
    if not np.any(np.isfinite(features[train]).all(axis=2) & np.isfinite(labels[train])):
        raise ValueError("Qlib frozen training panel has no complete rows")
    output_dir.mkdir(parents=True, exist_ok=False)
    panel_path = output_dir / "panel.npz"
    np.savez(panel_path, features=features, labels=labels, times=times, assets=assets)
    job_path = output_dir / "job.json"
    job = {
        "schema_version": 1, "protocol": "frozen_panel_qlib_ridge",
        "panel_sha256": _sha256(panel_path), "features": feature_names,
        "target": dataset.default_target, "train_indices": train.tolist(),
        "test_indices": test.tolist(), "label_maturity_bars": maturity,
        "asset_count": int(len(assets)), "timestamp_count": int(len(times)),
        "alpha": float(alpha),
    }
    _write_json(job_path, job)
    result_dir = output_dir / "result"
    interpreter = os.path.abspath(python) if python else sys.executable
    # Isolated Python ignores user site packages and Python path; only the
    # explicitly chosen interpreter and this fixed worker script are executed.
    env = {name: value for name, value in os.environ.items()
           if name in {"PATH", "SYSTEMROOT", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
                       "LANG", "LC_ALL"}}
    env.update({"HOME": str(output_dir), "TMPDIR": str(output_dir),
                "MLFLOW_DISABLE_AGENT_HINT": "1", "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    try:
        process = subprocess.run(
            [interpreter, "-I", str(Path(__file__).resolve()), str(job_path),
             str(panel_path), str(result_dir)],
            capture_output=True, text=True, timeout=timeout_seconds, check=False, env=env,
            cwd=output_dir,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("isolated Qlib worker timed out") from exc
    except OSError as exc:
        raise RuntimeError(f"isolated Qlib interpreter could not start: {exc}") from exc
    if process.returncode != 0:
        detail = process.stderr[-2000:] or process.stdout[-2000:]
        raise RuntimeError(f"isolated Qlib worker failed ({process.returncode}): {detail.strip()}")
    predictions_path = result_dir / "predictions.npz"
    metrics_path = result_dir / "metrics.json"
    model_path = result_dir / "model.json"
    metrics, model = audit_qlib_against_dataset(
        dataset, cfg, job_path, predictions_path, metrics_path, model_path
    )
    audit_path = output_dir / "audit.json"
    _write_json(audit_path, {"passed": True, "panel_sha256": job["panel_sha256"],
                             "job_sha256": _sha256(job_path),
                             "predictions_sha256": _sha256(predictions_path),
                             "metrics_sha256": _sha256(metrics_path),
                             "model_sha256": _sha256(model_path),
                             "checks": ["split_membership", "label_maturity", "model_refit",
                                        "test_predictions", "per_date_ic", "aggregate_metrics"]})
    result = {
        "comparable": True, "protocol": "frozen_panel_qlib_ridge", "metrics": metrics,
        "model": {"class": model["class"], "qlib_version": model["qlib_version"],
                  "alpha": alpha, "train_rows": model["train_rows"],
                  "test_rows": model["test_rows"]},
        "panel_sha256": job["panel_sha256"], "independent_audit": "passed",
        "scope": {"universe": cfg.benchmark.freeze_universe,
                  "target": dataset.default_target, "feature_names": feature_names},
    }
    # Frozen market values stay local. The portable receipt carries their digest,
    # selectors, predictions, and derivation audit without redistributing source data.
    artifacts = {"qlib_job": job_path, "qlib_predictions": predictions_path,
                 "qlib_metrics": metrics_path, "qlib_model": model_path,
                 "qlib_audit": audit_path}
    return result, artifacts


def verify_qlib_receipt_artifacts(manifest: dict[str, Any],
                                  inventory: dict[str, Path]) -> list[str]:
    """Check Qlib derivation links and summaries without redistributing market values."""
    qlib = manifest.get("qlib") or {}
    if not isinstance(qlib, dict):
        return ["Qlib receipt result must be an object"]
    if qlib.get("protocol") != "frozen_panel_qlib_ridge":
        return []
    names = ("qlib_job", "qlib_predictions", "qlib_metrics", "qlib_model", "qlib_audit")
    paths: dict[str, Path] = {}
    for name in names:
        path = inventory.get(f"phase2/{name}")
        if path is None or not path.is_file():
            return ["Qlib run is missing a required receipt artifact"]
        paths[name] = path
    try:
        job = json.loads(paths["qlib_job"].read_text())
        metrics = json.loads(paths["qlib_metrics"].read_text())
        model = json.loads(paths["qlib_model"].read_text())
        audit = json.loads(paths["qlib_audit"].read_text())
        with np.load(paths["qlib_predictions"], allow_pickle=False) as result:
            prediction = np.asarray(result["prediction"], dtype=np.float64)
            ic = np.asarray(result["ic"], dtype=np.float64)
            rank_ic = np.asarray(result["rank_ic"], dtype=np.float64)
        train = np.asarray(job["train_indices"], dtype=np.int64)
        test = np.asarray(job["test_indices"], dtype=np.int64)
        if (not len(train) or not len(test) or np.any(np.diff(train) <= 0)
                or np.any(np.diff(test) <= 0) or train[-1] >= test[0]
                or train[-1] + int(job["label_maturity_bars"]) >= test[0]
                or prediction.size != len(test) * int(job["asset_count"])
                or ic.shape != (len(test),) or rank_ic.shape != (len(test),)
                or model["train_rows"] > len(train) * int(job["asset_count"])):
            raise ValueError("frozen Qlib selectors and output shapes disagree")
        if (audit["passed"] is not True or job["panel_sha256"] != qlib["panel_sha256"]
                or audit["panel_sha256"] != job["panel_sha256"]
                or model["panel_sha256"] != job["panel_sha256"]
                or model["job_sha256"] != _sha256(paths["qlib_job"])
                or audit["job_sha256"] != _sha256(paths["qlib_job"])
                or audit["predictions_sha256"] != _sha256(paths["qlib_predictions"])
                or audit["metrics_sha256"] != _sha256(paths["qlib_metrics"])
                or audit["model_sha256"] != _sha256(paths["qlib_model"])):
            raise ValueError("frozen Qlib derivation digests disagree")
        if (qlib["metrics"] != metrics or model["test_rows"] != prediction.size
                or qlib["model"]["qlib_version"] != model["qlib_version"]
                or qlib["dataset_replay_identity"] != manifest["freeze_dataset_contract"]["replay_identity"]):
            raise ValueError("Qlib result does not match its receipt manifest")
        valid_ic = ic[np.isfinite(ic)]
        valid_rank = rank_ic[np.isfinite(rank_ic)]
        summaries = {
            "IC": valid_ic.mean(), "ICIR": valid_ic.mean() / valid_ic.std(ddof=1),
            "Rank IC": valid_rank.mean(),
            "Rank ICIR": valid_rank.mean() / valid_rank.std(ddof=1),
            "ic_days": len(valid_ic), "rank_ic_days": len(valid_rank),
            "prediction_rows": prediction.size,
        }
        if any(not np.isclose(metrics[key], value, rtol=1e-8, atol=1e-10)
               for key, value in summaries.items()):
            raise ValueError("Qlib metrics do not match per-date receipt predictions")
    except Exception as exc:  # noqa: BLE001 - malformed external artifacts must not crash verification
        return [f"Qlib receipt derivation is invalid: {exc}"]
    return []


def verify_qlib_against_committed_input(manifest: dict[str, Any],
                                        inventory: dict[str, Path],
                                        data_path: Path | None) -> list[str]:
    """Replay the frozen panel and audit model outputs without importing Qlib."""
    qlib_result = manifest.get("qlib") or {}
    if not isinstance(qlib_result, dict) or qlib_result.get("protocol") != "frozen_panel_qlib_ridge":
        return []
    try:
        from factorminer.architecture.dataset_contract import DatasetContract
        from factorminer.benchmark.datasets import _cfg_with_overrides, load_benchmark_dataset
        from factorminer.utils.config import load_config

        cfg = load_config(overrides=json.loads(
            inventory["phase2/effective_config"].read_text()
        ))
        universe = cfg.benchmark.freeze_universe
        freeze_cfg = _cfg_with_overrides(cfg, universe)
        dataset, dataset_hash = load_benchmark_dataset(
            freeze_cfg, data_path=str(data_path) if data_path else None,
            universe=universe, mock=data_path is None,
        )
        replay_identity = DatasetContract.from_runtime_dataset(freeze_cfg, dataset).replay_identity()
        if replay_identity != manifest["dataset_contracts"][universe]:
            raise ValueError("committed input produced a different frozen dataset contract")
        if any(hashes[universe] != dataset_hash for hashes in manifest["dataset_hashes"].values()):
            raise ValueError("committed input produced a different frozen dataset hash")
        metrics, model = audit_qlib_against_dataset(
            dataset, cfg, inventory["phase2/qlib_job"],
            inventory["phase2/qlib_predictions"], inventory["phase2/qlib_metrics"],
            inventory["phase2/qlib_model"],
        )
        if metrics != manifest["qlib"]["metrics"] or model["qlib_version"] != manifest["qlib"]["model"]["qlib_version"]:
            raise ValueError("replayed Qlib result differs from the receipt")
    except Exception as exc:  # noqa: BLE001 - verifier reports malformed external evidence
        return [f"Qlib committed-input replay failed: {exc}"]
    return []


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: qlib_runner JOB PANEL OUTPUT_DIR")
    _worker(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
