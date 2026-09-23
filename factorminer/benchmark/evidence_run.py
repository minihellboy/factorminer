"""One benchmark run with checked inputs, frozen selections, and a verified receipt."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from factorminer.architecture.research_receipt import EvidenceTier, RunStatus
from factorminer.benchmark.datasets import _cfg_with_overrides, load_benchmark_dataset
from factorminer.benchmark.qlib_conformance import (
    QlibHandlerSpec,
    check_qlib_conformance,
    require_conformance,
)
from factorminer.benchmark.receipt import (
    build_research_receipt,
    publish_portable_bundle,
    verify_research_receipt,
)
from factorminer.benchmark.reporting import _write_json, file_sha256
from factorminer.benchmark.runtime import run_table1_benchmark
from factorminer.benchmark.statistics import unavailable_selections


def _input_path(value: str, *, relative_to: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = relative_to / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _qlib_panel(dataset: Any, columns: dict[str, str]) -> pd.DataFrame:
    timestamps = pd.to_datetime(dataset.timestamps)
    instruments = np.asarray(dataset.asset_ids).astype(str)
    index = pd.MultiIndex.from_product(
        [timestamps, instruments], names=["datetime", "instrument"]
    )
    values: dict[str, np.ndarray] = {}
    for name in columns.values():
        kind, separator, field = name.partition(":")
        if not separator or not field:
            raise ValueError(f"invalid FactorMiner Qlib value column: {name}")
        if kind == "feature":
            panel = dataset.data_dict.get(field)
        elif kind == "target":
            panel = dataset.target_panels.get(field)
        else:
            raise ValueError(f"invalid FactorMiner Qlib value column: {name}")
        if panel is None:
            raise ValueError(f"FactorMiner panel is missing {name}")
        matrix = np.asarray(panel, dtype=np.float64)
        if matrix.shape != (len(instruments), len(timestamps)):
            raise ValueError(f"FactorMiner panel {name} has an unexpected shape")
        values[name] = matrix.T.ravel()
    return pd.DataFrame(values, index=index)


def _qlib_preflight(cfg: Any, evidence_path: Path, data_path: Path | None,
                    mock: bool, evidence_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    bundle = _json_object(evidence_path)
    for field in ("handler", "dataset_contract", "columns", "values_path", "metrics_path"):
        if field not in bundle:
            raise ValueError(f"Qlib evidence is missing {field}")
    columns = bundle["columns"]
    if not isinstance(columns, dict) or not columns or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in columns.items()
    ):
        raise ValueError("Qlib columns must map exported names to FactorMiner panel names")
    if not any(value.startswith("feature:") for value in columns.values()) or not any(
        value.startswith("target:") for value in columns.values()
    ):
        raise ValueError("Qlib conformance requires a feature and a target value check")
    values_path = _input_path(str(bundle["values_path"]), relative_to=evidence_path.parent)
    metrics_path = _input_path(str(bundle["metrics_path"]), relative_to=evidence_path.parent)
    metrics = _json_object(metrics_path)
    if not metrics or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in metrics.values()
    ):
        raise ValueError("Qlib metrics must be a non-empty map of finite numbers")
    exported = pd.read_csv(values_path)
    if not {"datetime", "instrument", *columns}.issubset(exported.columns):
        raise ValueError("Qlib value export is missing row keys or checked columns")
    exported["datetime"] = pd.to_datetime(exported["datetime"], errors="raise")
    exported["instrument"] = exported["instrument"].astype(str)
    exported = exported.set_index(["datetime", "instrument"]).sort_index()
    freeze_cfg = _cfg_with_overrides(cfg, cfg.benchmark.freeze_universe)
    dataset, _ = load_benchmark_dataset(
        freeze_cfg, data_path=str(data_path) if data_path else None, mock=mock,
        universe=cfg.benchmark.freeze_universe,
    )
    from factorminer.architecture.dataset_contract import DatasetContract

    contract = DatasetContract.from_runtime_dataset(freeze_cfg, dataset).replay_identity()
    if not isinstance(bundle["handler"], dict) or not isinstance(bundle["dataset_contract"], dict):
        raise ValueError("Qlib handler and dataset_contract must be objects")
    report = check_qlib_conformance(
        QlibHandlerSpec.from_handler_config(bundle["handler"]), freeze_cfg,
        dataset_contract=contract,
        qlib_dataset_contract=bundle["dataset_contract"],
        qlib_values=exported,
        factorminer_values=_qlib_panel(dataset, columns),
        columns=columns,
    )
    report_path = evidence_dir / "qlib_conformance.json"
    require_conformance(report)
    _write_json(report_path, report.to_dict())
    return {
        "comparable": True, "metrics": metrics, "handler": report.handler,
        "conformance": report.to_dict(), "dataset_replay_identity": contract,
    }, {
        "qlib_evidence": evidence_path,
        "qlib_values": values_path,
        "qlib_metrics": metrics_path,
        "qlib_conformance": report_path,
    }


def _require_complete(summary: dict[str, Any], baselines: list[str], universes: list[str]) -> None:
    if set(summary) != set(baselines):
        raise ValueError("benchmark did not return exactly the requested baselines")
    for baseline in baselines:
        result = summary[baseline]
        if not result.get("frozen_top_k"):
            raise ValueError(f"{baseline} selected no frozen factors")
        if set(result.get("universes", {})) != set(universes):
            raise ValueError(f"{baseline} did not evaluate every report universe")
        for universe, payload in result["universes"].items():
            if int(payload.get("factor_count", 0)) <= 0:
                raise ValueError(f"{baseline}/{universe} recomputed no frozen factors")
            unavailable = unavailable_selections(payload.get("selections", {}))
            if unavailable:
                raise ValueError(
                    f"{baseline}/{universe} has unavailable selections: {', '.join(unavailable)}"
                )
            missing = {"lasso", "forward_stepwise", "xgboost"} - set(payload.get("selections", {}))
            if missing:
                raise ValueError(f"{baseline}/{universe} omitted selections: {', '.join(sorted(missing))}")
            library = payload.get("library", {})
            for metric in ("ic", "icir"):
                value = library.get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{baseline}/{universe} is missing finite library {metric}")


def run_evidence_benchmark(
    cfg: Any,
    output_dir: Path,
    *,
    data_path: str | None = None,
    mock: bool = False,
    baseline_names: list[str] | None = None,
    qlib_evidence_path: str | None = None,
    qlib_model: str | None = None,
    qlib_python: str | None = None,
    qlib_alpha: float = 1.0,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
) -> dict[str, Any]:
    """Publish one completed benchmark only after every declared comparison passes."""
    baselines = list(baseline_names) if baseline_names is not None else list(cfg.benchmark.baselines)
    if not baselines or len(set(baselines)) != len(baselines) or any(
        not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in baselines
    ):
        raise ValueError("baseline ids must be unique, non-empty, lower-case identifiers")
    if mock and data_path:
        raise ValueError("choose either mock data or a data file")
    if qlib_evidence_path and qlib_model:
        raise ValueError("choose either external Qlib evidence or an isolated Qlib model")
    if qlib_model not in (None, "ridge"):
        raise ValueError("isolated Qlib model must be ridge")
    raw_path = None if mock else _input_path(
        str(data_path or getattr(cfg, "_raw", {}).get("data_path") or ""),
        relative_to=Path.cwd(),
    )
    evidence_dir = output_dir / "benchmark" / "evidence"
    table1_dir = output_dir / "benchmark" / "table1"
    if evidence_dir.exists() or (table1_dir.exists() and any(table1_dir.iterdir())):
        raise FileExistsError("evidence output already exists; use a fresh output directory")
    qlib_result = {"comparable": False}
    qlib_artifacts: dict[str, Path] = {}
    if qlib_evidence_path:
        evidence_path = _input_path(qlib_evidence_path, relative_to=Path.cwd())
        qlib_result, qlib_artifacts = _qlib_preflight(
            cfg, evidence_path, raw_path, mock, evidence_dir
        )

    evidence_dir.mkdir(parents=True, exist_ok=True)

    summary = run_table1_benchmark(
        cfg, output_dir, data_path=str(raw_path) if raw_path else None, mock=mock,
        baseline_names=baselines,
        factor_miner_library_path=factor_miner_library_path,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
    )
    _require_complete(summary, baselines, list(cfg.benchmark.report_universes))

    effective_config = evidence_dir / "effective_config.json"
    _write_json(effective_config, cfg.to_dict())
    artifact_paths: dict[str, str] = {"effective_config": str(effective_config.resolve())}
    artifact_paths.update({name: str(path) for name, path in qlib_artifacts.items()})
    dataset_hashes: dict[str, dict[str, str]] = {}
    dataset_contracts: dict[str, Any] = {}
    selected_factors: dict[str, Any] = {}
    baseline_provenance: dict[str, Any] = {}
    runtime_contracts: dict[str, Any] = {}
    for baseline in baselines:
        result_path = output_dir / "benchmark" / "table1" / f"{baseline}.json"
        baseline_manifest_path = output_dir / "benchmark" / "table1" / f"{baseline}_manifest.json"
        result = _json_object(result_path)
        baseline_manifest = _json_object(baseline_manifest_path)
        if result.get("baseline") != baseline or baseline_manifest.get("baseline") != baseline:
            raise ValueError(f"{baseline} artifact identity does not match its baseline")
        if result.get("frozen_top_k") != summary[baseline]["frozen_top_k"]:
            raise ValueError(f"{baseline} serialized selection differs from the run result")
        if result.get("runtime_contract") != baseline_manifest.get("runtime_contract"):
            raise ValueError(f"{baseline} runtime contract differs between artifacts")
        declared_paths = baseline_manifest.get("artifact_paths", {})
        if (Path(declared_paths.get("result", "")).resolve() != result_path.resolve()
                or Path(declared_paths.get("manifest", "")).resolve() != baseline_manifest_path.resolve()):
            raise ValueError(f"{baseline} manifest points to different artifacts")
        artifact_paths[f"{baseline}_result"] = str(result_path.resolve())
        artifact_paths[f"{baseline}_manifest"] = str(baseline_manifest_path.resolve())
        dataset_hashes[baseline] = dict(baseline_manifest["dataset_hashes"])
        selected_factors[baseline] = result["frozen_top_k"]
        baseline_provenance[baseline] = result["provenance"]
        runtime_contracts[baseline] = result["runtime_contract"]
        for label, source in result["provenance"].get("source_files", {}).items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", label):
                raise ValueError(f"{baseline} has an invalid provenance artifact name")
            source_path = _input_path(str(source["path"]), relative_to=Path.cwd())
            if file_sha256(source_path) != source.get("sha256"):
                raise ValueError(f"{baseline} provenance artifact changed: {label}")
            artifact_paths[f"{baseline}_source_{label}"] = str(source_path)
    freeze_contract = summary[baselines[0]]["freeze_dataset_contract"]
    if qlib_result["comparable"] and qlib_result["dataset_replay_identity"] != freeze_contract.get("replay_identity"):
        raise ValueError("Qlib preflight and benchmark used different frozen datasets")
    all_universes = list(dict.fromkeys([
        cfg.benchmark.freeze_universe, *cfg.benchmark.report_universes,
    ]))
    freeze_dataset = None
    for universe in all_universes:
        universe_cfg = _cfg_with_overrides(cfg, universe)
        dataset, dataset_hash = load_benchmark_dataset(
            universe_cfg, data_path=str(raw_path) if raw_path else None,
            universe=universe, mock=mock,
        )
        from factorminer.architecture.dataset_contract import DatasetContract

        dataset_contracts[universe] = DatasetContract.from_runtime_dataset(
            universe_cfg, dataset
        ).replay_identity()
        if universe == cfg.benchmark.freeze_universe:
            freeze_dataset = dataset
        for baseline in baselines:
            if dataset_hashes[baseline].get(universe) != dataset_hash:
                raise ValueError(f"{baseline}/{universe} input changed during the evidence run")
    if dataset_contracts[cfg.benchmark.freeze_universe] != freeze_contract.get("replay_identity"):
        raise ValueError("frozen dataset changed during the evidence run")
    if qlib_model:
        from factorminer.benchmark.qlib_runner import run_isolated_qlib

        if freeze_dataset is None:
            raise ValueError("frozen dataset was not loaded for Qlib")
        qlib_result, qlib_artifacts = run_isolated_qlib(
            freeze_dataset, cfg, evidence_dir / "qlib_run",
            python=qlib_python, alpha=qlib_alpha,
        )
        qlib_result["dataset_replay_identity"] = dataset_contracts[cfg.benchmark.freeze_universe]
        artifact_paths.update({name: str(path.resolve()) for name, path in qlib_artifacts.items()})
    manifest = {
        "schema_version": 1,
        "kind": "benchmark_evidence",
        "status": "completed",
        "seed": int(cfg.benchmark.seed),
        "mode": cfg.benchmark.mode,
        "metric_version": summary[baselines[0]]["metric_version"],
        "baselines": baselines,
        "report_universes": list(cfg.benchmark.report_universes),
        "freeze_dataset_contract": freeze_contract,
        "dataset_contracts": dataset_contracts,
        "dataset_hashes": dataset_hashes,
        "selected_factors": selected_factors,
        "baseline_provenance": baseline_provenance,
        "runtime_contracts": runtime_contracts,
        "qlib": qlib_result,
        "artifact_paths": artifact_paths,
    }
    manifest_path = evidence_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    first = summary[baselines[0]]
    receipt = build_research_receipt(
        phase2_manifest=manifest,
        phase2_manifest_path=manifest_path,
        evidence_tier=EvidenceTier.SIMULATED if mock else EvidenceTier.UNVERIFIED,
        run_status=RunStatus.COMPLETED,
        seed=int(cfg.benchmark.seed),
        config_path=effective_config,
        data_path=raw_path,
        dataset_descriptor={
            "source": raw_path.name if raw_path is not None else "mock",
            "freeze_universe": cfg.benchmark.freeze_universe,
            "report_universes": list(cfg.benchmark.report_universes),
            "dataset_hashes": dataset_hashes,
            "replay_identity": dataset_contracts,
        },
        protocol_admission_contract={
            "ic_threshold": float(cfg.mining.ic_threshold),
            "correlation_threshold": float(cfg.mining.correlation_threshold),
        },
        protocol_replacement_contract={
            "replacement_ic_min": float(cfg.mining.replacement_ic_min),
            "replacement_ic_ratio": float(cfg.mining.replacement_ic_ratio),
        },
        memory_policy_schema={"baseline_runtime_contracts": runtime_contracts},
        ic_metric="spearman_rank",
        metric_version=first["metric_version"],
        walk_forward_contract=first["walk_forward_contract"],
        stress_contract=first["stress_contract"],
        data_license_class="synthetic" if mock else "unknown",
    )
    receipt_path = publish_portable_bundle(
        receipt, phase2_manifest=manifest, releases_root=output_dir / "releases"
    )
    verification = verify_research_receipt(receipt_path.parent, commitment_input=raw_path)
    if not verification.passed:
        raise RuntimeError("published receipt failed verification: " + "; ".join(verification.mismatches))
    return {
        "manifest_path": str(manifest_path),
        "receipt_path": str(receipt_path),
        "release_id": verification.release_id,
        "baselines": baselines,
        "qlib_comparable": bool(qlib_result["comparable"]),
    }
