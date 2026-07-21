"""Strict paper/research benchmark runners built on runtime recomputation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factorminer.architecture import PaperProtocol
from factorminer.benchmark.contracts import (
    BenchmarkManifest,
    BenchmarkRuntimeContract,
    StrategyGridBenchmarkContract,
    StressBenchmarkContract,
    WalkForwardBenchmarkContract,
)
from factorminer.benchmark.datasets import (
    _base_path,
    _cfg_with_overrides,
    _clone_cfg,
    _factors_from_entries,
    _get_baseline_entries,
    build_benchmark_library,
    load_benchmark_dataset,
)
from factorminer.benchmark.frozen_evaluation import (
    _default_capacity_levels,
    _extract_volume_panel,
    evaluate_frozen_set,
    select_frozen_top_k,
)
from factorminer.benchmark.reporting import _ensure_dir, _save_manifest, _write_json
from factorminer.benchmark.speed import (
    SpeedBenchmark as SpeedBenchmark,
)
from factorminer.benchmark.speed import (
    _build_mock_data_dict as _build_mock_data_dict,
)
from factorminer.benchmark.statistics import (
    AblationResult,
    BenchmarkResult,
    MethodResult,
    StatisticalComparisonTests,
)
from factorminer.benchmark.statistics import (
    DMTestResult as DMTestResult,
)
from factorminer.benchmark.statistics import (
    OperatorSpeedResult as OperatorSpeedResult,
)
from factorminer.benchmark.statistics import (
    PipelineSpeedResult as PipelineSpeedResult,
)
from factorminer.core.library_io import load_library
from factorminer.core.session import MiningSession
from factorminer.evaluation.metrics import METRIC_VERSION
from factorminer.evaluation.runtime import (
    EvaluationDataset,
    evaluate_factors,
)
from factorminer.operators.c_backend import backend_available as c_backend_available

logger = logging.getLogger(__name__)

RUNTIME_LOOP_BASELINES = {
    "ralph_loop",
    "helix_phase2",
    "helix_no_memory",
    "helix_no_debate",
    "helix_no_significance",
    "helix_no_capacity",
    "helix_no_regime",
    "helix_no_causal",
    "helix_no_canonicalize",
}


def _json_safe(value: Any) -> Any:
    """Recursively convert NaN/inf values into JSON-safe nulls."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as fp:
            payload = json.load(fp)
    except Exception as exc:  # pragma: no cover - defensive provenance capture
        return {"path": str(path), "load_error": str(exc)}

    if isinstance(payload, dict):
        return payload
    return {"path": str(path), "payload_type": type(payload).__name__}


def _session_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return MiningSession.load(path).get_summary()
    except Exception as exc:  # pragma: no cover - defensive provenance capture
        return {"path": str(path), "load_error": str(exc)}


def _catalog_provenance(baseline: str, candidate_count: int, seed: int) -> dict[str, Any]:
    return {
        "kind": "catalog",
        "source": baseline,
        "baseline_kind": _baseline_kind(baseline),
        "candidate_count": candidate_count,
        "seed": seed,
        "metric_version": METRIC_VERSION,
    }


def _baseline_kind(baseline: str) -> str:
    if baseline in {"gplearn", "alphaforge_style", "alphaagent_style"}:
        return "catalog_proxy"
    if baseline in {"alpha101_classic", "alpha101_adapted", "random_exploration"}:
        return "catalog_baseline"
    if baseline == "factor_miner":
        return "builtin_paper_catalog"
    if baseline == "factor_miner_no_memory":
        return "synthetic_no_memory_proxy"
    return "unknown"


def _saved_library_provenance(
    requested_path: str,
    baseline: str,
) -> dict[str, Any]:
    base_path = Path(_base_path(requested_path)).expanduser()
    resolved_base = base_path.resolve() if base_path.exists() else base_path
    library_json = resolved_base.with_suffix(".json")
    signal_cache = Path(str(resolved_base) + "_signals.npz")
    parent = resolved_base.parent

    source_files: dict[str, dict[str, str]] = {}
    for label, path in {
        "library_json": library_json,
        "signal_cache": signal_cache,
        "session_json": parent / "session.json",
        "session_log_json": parent / "session_log.json",
        "checkpoint_session_json": parent / "checkpoint" / "session.json",
        "checkpoint_loop_state_json": parent / "checkpoint" / "loop_state.json",
        "checkpoint_memory_json": parent / "checkpoint" / "memory.json",
    }.items():
        if path.exists():
            source_files[label] = {
                "path": str(path),
                "sha256": _file_sha256(path),
            }

    provenance: dict[str, Any] = {
        "kind": "saved_library",
        "source": baseline,
        "baseline_kind": "saved_library",
        "metric_version": METRIC_VERSION,
        "requested_path": str(Path(requested_path)),
        "resolved_base_path": str(resolved_base),
        "source_files": source_files,
        "library_summary": {},
        "session_summary": _session_summary(parent / "session.json"),
        "session_log_summary": _json_summary(parent / "session_log.json"),
    }

    if library_json.exists():
        try:
            library = load_library(resolved_base)
        except Exception as exc:  # pragma: no cover - defensive provenance capture
            provenance["library_summary"] = {
                "path": str(library_json),
                "load_error": str(exc),
            }
        else:
            provenance["library_summary"] = {
                "path": str(library_json),
                "factor_count": library.size,
                "metric_version": getattr(library, "metric_version", "legacy_abs_ic"),
                "diagnostics": library.get_diagnostics(),
            }

    return provenance


def _baseline_provenance(
    baseline: str,
    *,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
    candidate_count: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    if baseline == "factor_miner" and factor_miner_library_path:
        return _saved_library_provenance(factor_miner_library_path, baseline)
    if baseline == "factor_miner_no_memory" and factor_miner_no_memory_library_path:
        return _saved_library_provenance(
            factor_miner_no_memory_library_path,
            baseline,
        )
    return _catalog_provenance(baseline, candidate_count, seed)


def _runtime_manifest_value(
    runtime_manifests: dict[str, dict[str, Any]] | None,
    baseline: str,
) -> dict[str, Any]:
    """Return the runtime manifest for one baseline if supplied."""
    if not runtime_manifests:
        return {}
    value = runtime_manifests.get(baseline, {})
    return dict(value) if isinstance(value, dict) else {}


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _build_walk_forward_contract(
    cfg,
    freeze_dataset_contract: dict[str, Any],
) -> WalkForwardBenchmarkContract:
    """Build the canonical walk-forward benchmark contract."""
    return WalkForwardBenchmarkContract(
        freeze_universe=str(cfg.benchmark.freeze_universe),
        report_universes=list(cfg.benchmark.report_universes),
        train_period=list(cfg.data.train_period),
        test_period=list(cfg.data.test_period),
        freeze_top_k=int(cfg.benchmark.freeze_top_k),
        fit_split="train",
        eval_split="test",
        default_target=str(cfg.data.default_target),
        signal_failure_policy="reject",
        dataset_contract=dict(freeze_dataset_contract),
    )


def _build_stress_contract(cfg, runtime_manifest: dict[str, Any] | None = None) -> StressBenchmarkContract:
    """Build the canonical cost/capacity stress contract."""
    runtime_manifest = dict(runtime_manifest or {})
    cost_bps = runtime_manifest.get("cost_bps", getattr(cfg.benchmark, "cost_bps", [1.0]))
    capacity_levels = runtime_manifest.get("capacity_levels", _default_capacity_levels())
    from factorminer.evaluation.capacity import CapacityConfig as RuntimeCapacityConfig

    capacity_cfg = RuntimeCapacityConfig()
    return StressBenchmarkContract(
        cost_bps=[float(value) for value in cost_bps],
        capacity_levels=[float(value) for value in capacity_levels],
        base_capacity_usd=float(
            runtime_manifest.get("base_capacity_usd", capacity_cfg.base_capital_usd)
        ),
        net_icir_threshold=float(
            runtime_manifest.get("net_icir_threshold", capacity_cfg.net_icir_threshold)
        ),
        ic_degradation_limit=float(
            runtime_manifest.get("ic_degradation_limit", capacity_cfg.ic_degradation_limit)
        ),
    )


def _build_strategy_grid_contract(
    cfg,
    *,
    baseline: str,
    runtime_manifest: dict[str, Any] | None = None,
) -> StrategyGridBenchmarkContract:
    """Build the canonical memory-policy / dependence / backend strategy grid."""
    runtime_manifest = dict(runtime_manifest or {})
    raw_strategy = _strategy_ablation_raw_config(cfg)
    return StrategyGridBenchmarkContract(
        baseline=baseline,
        enabled=bool(raw_strategy.get("enabled", False)),
        memory_policies=[
            str(value)
            for value in runtime_manifest.get(
                "memory_policies",
                raw_strategy.get(
                    "memory_policies",
                    ["paper", "none", "kg", "family_aware", "regime_aware"],
                ),
            )
        ],
        dependence_metrics=[
            str(value)
            for value in runtime_manifest.get(
                "dependence_metrics",
                raw_strategy.get(
                    "dependence_metrics",
                    ["spearman", "pearson", "distance_correlation"],
                ),
            )
        ],
        backends=[
            str(value)
            for value in runtime_manifest.get(
                "backends",
                raw_strategy.get("backends", ["numpy", "c", "gpu"]),
            )
        ],
        selected_memory_policy=(
            str(runtime_manifest["memory_policy"])
            if runtime_manifest.get("memory_policy") is not None
            else None
        ),
        selected_dependence_metric=(
            str(runtime_manifest["redundancy_metric"])
            if runtime_manifest.get("redundancy_metric") is not None
            else None
        ),
        selected_backend=(
            str(runtime_manifest["backend"]) if runtime_manifest.get("backend") is not None else None
        ),
    )


def _benchmark_dataset_contract(cfg, dataset: Any) -> dict[str, Any]:
    """Build a safe, benchmark-local summary of the frozen dataset."""
    data_dict = getattr(dataset, "data_dict", {}) or {}
    target_panels = getattr(dataset, "target_panels", {}) or {}
    target_specs = getattr(dataset, "target_specs", {}) or {}
    splits = getattr(dataset, "splits", {}) or {}

    if isinstance(target_panels, dict):
        target_names = list(target_panels.keys())
    else:
        target_names = [str(getattr(cfg.data, "default_target", "paper"))]

    split_sizes: dict[str, int] = {}
    if isinstance(splits, dict):
        for name, split in splits.items():
            split_sizes[name] = int(getattr(split, "size", 0))

    return {
        "feature_names": list(getattr(data_dict, "keys", lambda: [])()),
        "data_shape": tuple(np.shape(getattr(dataset, "data_tensor", ()))),
        "returns_shape": tuple(np.shape(getattr(dataset, "returns", ()))),
        "default_target": str(
            getattr(dataset, "default_target", getattr(cfg.data, "default_target", "paper"))
        ),
        "target_names": target_names,
        "target_horizons": {
            name: max(int(getattr(spec, "holding_bars", 1)), 1)
            for name, spec in target_specs.items()
        },
        "train_period": list(getattr(cfg.data, "train_period", [])),
        "test_period": list(getattr(cfg.data, "test_period", [])),
        "asset_count": int(_safe_len(getattr(dataset, "asset_ids", None))),
        "period_count": int(_safe_len(getattr(dataset, "timestamps", None))),
        "split_sizes": split_sizes,
    }


def build_benchmark_runtime_contract(
    cfg,
    freeze_dataset_contract: dict[str, Any],
    *,
    baseline: str,
    runtime_manifest: dict[str, Any] | None = None,
) -> BenchmarkRuntimeContract:
    """Build the canonical benchmark runtime contract for one baseline."""
    paper_protocol = PaperProtocol.from_config(cfg).runtime_contract()
    return BenchmarkRuntimeContract(
        paper_protocol=paper_protocol,
        walk_forward=_build_walk_forward_contract(cfg, freeze_dataset_contract),
        stress=_build_stress_contract(cfg, runtime_manifest),
        strategy_grid=_build_strategy_grid_contract(
            cfg,
            baseline=baseline,
            runtime_manifest=runtime_manifest,
        ),
        runtime_manifest=dict(runtime_manifest or {}),
    )


def _build_runtime_provider(cfg, *, mock: bool):
    """Create the benchmark-time LLM provider."""
    from factorminer.agent.llm_interface import MockProvider, create_provider

    if mock or getattr(cfg.llm, "provider", "mock") == "mock":
        return MockProvider()

    provider_cfg = {
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
    }
    raw_llm_cfg = getattr(cfg, "_raw", {}).get("llm", {})
    if raw_llm_cfg.get("api_key"):
        provider_cfg["api_key"] = raw_llm_cfg["api_key"]
    return create_provider(provider_cfg)


def _filter_dataclass_kwargs(source, target_cls):
    """Copy shared dataclass fields from one config object to another."""
    from dataclasses import fields

    target_fields = {f.name for f in fields(target_cls)}
    source_fields = getattr(source, "__dataclass_fields__", {})
    return {name: getattr(source, name) for name in source_fields if name in target_fields}


def _build_phase2_runtime_kwargs(cfg) -> dict[str, Any]:
    """Build runtime Phase 2 configs from the hierarchical benchmark config."""
    from factorminer.agent.debate import DebateConfig as RuntimeDebateConfig
    from factorminer.agent.specialists import DEFAULT_SPECIALISTS
    from factorminer.evaluation.capacity import CapacityConfig as RuntimeCapacityConfig
    from factorminer.evaluation.causal import CausalConfig as RuntimeCausalConfig
    from factorminer.evaluation.regime import RegimeConfig as RuntimeRegimeConfig
    from factorminer.evaluation.significance import (
        SignificanceConfig as RuntimeSignificanceConfig,
    )

    debate_config = None
    if cfg.phase2.debate.enabled:
        requested = cfg.phase2.debate.num_specialists
        selected = list(DEFAULT_SPECIALISTS[:requested])
        if requested > len(DEFAULT_SPECIALISTS):
            selected = list(DEFAULT_SPECIALISTS)
        debate_config = RuntimeDebateConfig(
            specialists=selected,
            enable_critic=cfg.phase2.debate.enable_critic,
            candidates_per_specialist=cfg.phase2.debate.candidates_per_specialist,
            top_k_after_critic=cfg.phase2.debate.top_k_after_critic,
            critic_temperature=cfg.phase2.debate.critic_temperature,
        )

    causal_config = None
    if cfg.phase2.causal.enabled:
        causal_config = RuntimeCausalConfig(
            **_filter_dataclass_kwargs(cfg.phase2.causal, RuntimeCausalConfig)
        )

    regime_config = None
    if cfg.phase2.regime.enabled:
        regime_config = RuntimeRegimeConfig(
            **_filter_dataclass_kwargs(cfg.phase2.regime, RuntimeRegimeConfig)
        )

    capacity_config = None
    if cfg.phase2.capacity.enabled:
        capacity_config = RuntimeCapacityConfig(
            **_filter_dataclass_kwargs(cfg.phase2.capacity, RuntimeCapacityConfig)
        )

    significance_config = None
    if cfg.phase2.significance.enabled:
        significance_config = RuntimeSignificanceConfig(
            **_filter_dataclass_kwargs(cfg.phase2.significance, RuntimeSignificanceConfig)
        )

    return {
        "debate_config": debate_config,
        "causal_config": causal_config,
        "regime_config": regime_config,
        "capacity_config": capacity_config,
        "significance_config": significance_config,
        "enable_knowledge_graph": bool(cfg.phase2.helix.enable_knowledge_graph),
        "enable_embeddings": bool(cfg.phase2.helix.enable_embeddings),
        "enable_auto_inventor": bool(cfg.phase2.auto_inventor.enabled),
        "auto_invention_interval": int(cfg.phase2.auto_inventor.invention_interval),
        "canonicalize": bool(cfg.phase2.helix.enable_canonicalization),
        "forgetting_lambda": float(cfg.phase2.helix.forgetting_lambda),
    }


def _prepare_runtime_loop(
    cfg,
    *,
    output_dir: Path,
    dataset: EvaluationDataset,
    mock: bool,
    runtime_manifest: dict[str, Any],
):
    """Apply benchmark overrides to a canonical config and build run state."""
    from factorminer.application.runtime_context import build_run_context

    mining_fields = (
        "target_library_size",
        "batch_size",
        "max_iterations",
        "ic_threshold",
        "icir_threshold",
        "correlation_threshold",
        "replacement_ic_min",
        "replacement_ic_ratio",
    )
    evaluation_fields = (
        "fast_screen_assets",
        "num_workers",
        "backend",
        "gpu_device",
        "redundancy_metric",
        "signal_failure_policy",
    )
    for name in mining_fields:
        if name in runtime_manifest:
            setattr(cfg.mining, name, runtime_manifest[name])
    for name in evaluation_fields:
        if name in runtime_manifest:
            setattr(cfg.evaluation, name, runtime_manifest[name])
    if "memory_policy" in runtime_manifest:
        cfg.memory.policy = str(runtime_manifest["memory_policy"])
    if "memory_regime_lookback_window" in runtime_manifest:
        cfg.memory.regime_lookback_window = int(
            runtime_manifest["memory_regime_lookback_window"]
        )

    if runtime_manifest.get("relax_thresholds", mock):
        cfg.mining.ic_threshold = min(float(cfg.mining.ic_threshold), 0.0)
        cfg.mining.icir_threshold = min(float(cfg.mining.icir_threshold), -1.0)
        cfg.mining.correlation_threshold = max(
            float(cfg.mining.correlation_threshold), 1.1
        )

    signal_policy = str(
        runtime_manifest.get(
            "signal_failure_policy",
            "synthetic" if mock else cfg.evaluation.signal_failure_policy,
        )
    )
    return build_run_context(
        cfg,
        output_dir=output_dir,
        dataset=dataset,
        signal_failure_policy=signal_policy,
        benchmark_mode=str(cfg.benchmark.mode),
    )


def _cfg_for_runtime_baseline(cfg, baseline: str):
    """Project the hierarchical config into one runtime benchmark variant."""
    runtime_cfg = _clone_cfg(cfg)

    # Start from a clean phase-2 surface so variants are explicit.
    runtime_cfg.phase2.causal.enabled = False
    runtime_cfg.phase2.regime.enabled = False
    runtime_cfg.phase2.capacity.enabled = False
    runtime_cfg.phase2.significance.enabled = False
    runtime_cfg.phase2.debate.enabled = False
    runtime_cfg.phase2.auto_inventor.enabled = False
    runtime_cfg.phase2.helix.enabled = False
    runtime_cfg.phase2.helix.enable_knowledge_graph = False
    runtime_cfg.phase2.helix.enable_embeddings = False
    runtime_cfg.phase2.helix.enable_canonicalization = False

    if baseline in {"ralph_loop", "factor_miner", "factor_miner_no_memory"}:
        runtime_cfg.benchmark.mode = "paper"
        return runtime_cfg

    runtime_cfg.benchmark.mode = "research"
    runtime_cfg.phase2.helix.enabled = True
    runtime_cfg.phase2.helix.enable_canonicalization = True
    runtime_cfg.phase2.helix.enable_knowledge_graph = True
    runtime_cfg.phase2.helix.enable_embeddings = True
    runtime_cfg.phase2.debate.enabled = True
    runtime_cfg.phase2.causal.enabled = True
    runtime_cfg.phase2.regime.enabled = True
    runtime_cfg.phase2.capacity.enabled = True
    runtime_cfg.phase2.significance.enabled = True

    if baseline == "helix_no_memory":
        runtime_cfg.phase2.helix.enable_knowledge_graph = False
        runtime_cfg.phase2.helix.enable_embeddings = False
    elif baseline == "helix_no_debate":
        runtime_cfg.phase2.debate.enabled = False
    elif baseline == "helix_no_significance":
        runtime_cfg.phase2.significance.enabled = False
    elif baseline == "helix_no_capacity":
        runtime_cfg.phase2.capacity.enabled = False
    elif baseline == "helix_no_regime":
        runtime_cfg.phase2.regime.enabled = False
    elif baseline == "helix_no_causal":
        runtime_cfg.phase2.causal.enabled = False
    elif baseline == "helix_no_canonicalize":
        runtime_cfg.phase2.helix.enable_canonicalization = False

    return runtime_cfg


def _real_mining_loop_type(baseline: str, runtime_manifest: dict[str, Any]) -> str:
    """Resolve the loop type for a runtime mining request."""
    loop_type = str(runtime_manifest.get("loop_type", "")).strip().lower()
    if loop_type in {"ralph", "helix"}:
        return loop_type
    if baseline in {
        "helix_phase2",
        "helix_no_memory",
        "helix_no_debate",
        "helix_no_significance",
        "helix_no_capacity",
        "helix_no_regime",
        "helix_no_causal",
        "helix_no_canonicalize",
    }:
        return "helix"
    if baseline in {"factor_miner", "factor_miner_no_memory", "ralph_loop"}:
        return "ralph"
    return "ralph"


def _runtime_loop_provenance(
    *,
    baseline: str,
    loop_type: str,
    runtime_manifest: dict[str, Any],
    runtime_output_dir: Path,
) -> dict[str, Any]:
    """Summarize the real mining run used to source benchmark factors."""
    library_json = runtime_output_dir / "factor_library.json"
    run_manifest = runtime_output_dir / "run_manifest.json"
    session_json = runtime_output_dir / "session.json"
    session_log_json = runtime_output_dir / "session_log.json"
    checkpoint_dir = runtime_output_dir / "checkpoint"

    source_files: dict[str, dict[str, str]] = {}
    for label, path in {
        "library_json": library_json,
        "run_manifest_json": run_manifest,
        "session_json": session_json,
        "session_log_json": session_log_json,
        "checkpoint_library_json": checkpoint_dir / "library.json",
        "checkpoint_run_manifest_json": checkpoint_dir / "run_manifest.json",
        "checkpoint_session_json": checkpoint_dir / "session.json",
        "checkpoint_loop_state_json": checkpoint_dir / "loop_state.json",
    }.items():
        if path.exists():
            source_files[label] = {
                "path": str(path),
                "sha256": _file_sha256(path),
            }

    provenance: dict[str, Any] = {
        "kind": "runtime_loop",
        "source": baseline,
        "loop_type": loop_type,
        "baseline_kind": "runtime_loop",
        "requested_runtime_manifest": _json_safe(runtime_manifest),
        "runtime_output_dir": str(runtime_output_dir),
        "source_files": source_files,
        "run_manifest_summary": _json_summary(run_manifest),
        "session_summary": _session_summary(session_json),
        "session_log_summary": _json_summary(session_log_json),
        "library_summary": {},
    }

    if library_json.exists():
        try:
            library = load_library(runtime_output_dir / "factor_library")
        except Exception as exc:  # pragma: no cover - defensive provenance capture
            provenance["library_summary"] = {
                "path": str(library_json),
                "load_error": str(exc),
            }
        else:
            provenance["library_summary"] = {
                "path": str(library_json),
                "factor_count": library.size,
                "diagnostics": library.get_diagnostics(),
            }

    return provenance


def _run_runtime_mining_loop(
    cfg,
    *,
    baseline: str,
    dataset: EvaluationDataset,
    output_dir: Path,
    runtime_manifest: dict[str, Any] | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    """Run a real RalphLoop/HelixLoop and return its factor library."""
    runtime_manifest = dict(runtime_manifest or {})
    loop_type = _real_mining_loop_type(baseline, runtime_manifest)
    runtime_output_dir = _ensure_dir(output_dir / "benchmark" / "table1" / baseline / "runtime")
    runtime_cfg = _cfg_for_runtime_baseline(cfg, baseline)
    run_context = _prepare_runtime_loop(
        runtime_cfg,
        output_dir=runtime_output_dir,
        dataset=dataset,
        mock=mock or bool(runtime_manifest.get("mock", False)),
        runtime_manifest=runtime_manifest,
    )
    provider = _build_runtime_provider(
        runtime_cfg, mock=mock or bool(runtime_manifest.get("mock", False))
    )

    if loop_type == "helix":
        from factorminer.core.helix_loop import HelixLoop

        phase2_kwargs = _build_phase2_runtime_kwargs(runtime_cfg)
        loop = HelixLoop(
            config=runtime_cfg,
            data_tensor=dataset.data_tensor,
            returns=dataset.returns,
            llm_provider=provider,
            volume=_extract_volume_panel(dataset),
            run_context=run_context,
            **phase2_kwargs,
        )
    else:
        from factorminer.core.ralph_loop import RalphLoop

        loop = RalphLoop(
            config=runtime_cfg,
            data_tensor=dataset.data_tensor,
            returns=dataset.returns,
            llm_provider=provider,
            run_context=run_context,
        )

    checkpoint_interval = int(runtime_manifest.get("checkpoint_interval", 0 if mock else 1))
    loop.checkpoint_interval = checkpoint_interval

    if runtime_manifest.get("checkpoint_path"):
        loop.load_session(str(runtime_manifest["checkpoint_path"]))

    target_size = int(runtime_cfg.mining.target_library_size)
    max_iterations = int(runtime_cfg.mining.max_iterations)
    library = loop.run(target_size=target_size, max_iterations=max_iterations)
    provenance = _runtime_loop_provenance(
        baseline=baseline,
        loop_type=loop_type,
        runtime_manifest={
            **runtime_manifest,
            "target_library_size": target_size,
            "max_iterations": max_iterations,
        },
        runtime_output_dir=runtime_output_dir,
    )
    return {
        "baseline": baseline,
        "loop_type": loop_type,
        "library": library,
        "provenance": provenance,
        "runtime_output_dir": str(runtime_output_dir),
        "target_library_size": target_size,
        "max_iterations": max_iterations,
    }


def _strategy_ablation_raw_config(cfg) -> dict[str, Any]:
    raw = getattr(cfg, "_raw", {})
    if not isinstance(raw, dict):
        return {}
    benchmark_raw = raw.get("benchmark", {})
    if not isinstance(benchmark_raw, dict):
        return {}
    strategy_raw = benchmark_raw.get("strategy_ablation", {})
    return dict(strategy_raw) if isinstance(strategy_raw, dict) else {}


def _runtime_strategy_backends(
    requested: list[str] | tuple[str, ...] | None,
    *,
    default_backend: str,
) -> list[str]:
    try:
        from factorminer.operators import torch_available
    except Exception:  # pragma: no cover - optional dependency
        def torch_available() -> bool:
            return False

    available = {
        "numpy": True,
        "c": c_backend_available(),
        "gpu": torch_available(),
    }
    candidates = list(requested or [default_backend, "c", "gpu"])
    ordered: list[str] = []
    for backend in candidates:
        name = str(backend).strip().lower()
        if not name or name in ordered:
            continue
        if available.get(name, False):
            ordered.append(name)
    return ordered or ["numpy"]


def _mean_universe_metric(
    payload: dict[str, Any],
    metric_group: str,
    metric_name: str,
) -> float | None:
    values: list[float] = []
    for universe_payload in payload.get("universes", {}).values():
        group = universe_payload.get(metric_group, {})
        value = group.get(metric_name)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def run_table1_benchmark(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    baseline_names: list[str] | None = None,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
    use_runtime_loops: bool = False,
) -> dict:
    """Run the strict Top-K freeze benchmark across all configured universes."""
    if runtime_manifests is None:
        runtime_manifests = getattr(cfg.benchmark, "runtime_manifests", None)
    use_runtime_loops = bool(
        use_runtime_loops or getattr(cfg.benchmark, "runtime_loops", False) or runtime_manifests
    )
    benchmark_dir = _ensure_dir(output_dir / "benchmark" / "table1")
    baseline_names = baseline_names or list(cfg.benchmark.baselines)
    freeze_cfg = _cfg_with_overrides(cfg, cfg.benchmark.freeze_universe)
    freeze_dataset, freeze_hash = load_benchmark_dataset(
        freeze_cfg,
        data_path=data_path,
        raw_df=raw_df,
        universe=cfg.benchmark.freeze_universe,
        mock=mock,
    )
    freeze_dataset_contract = _benchmark_dataset_contract(freeze_cfg, freeze_dataset)

    summary: dict[str, dict] = {}
    for baseline in baseline_names:
        runtime_manifest = _runtime_manifest_value(runtime_manifests, baseline)
        runtime_contract = build_benchmark_runtime_contract(
            cfg,
            freeze_dataset_contract,
            baseline=baseline,
            runtime_manifest=runtime_manifest,
        )
        runtime_baseline = bool(runtime_manifest) or (
            use_runtime_loops
            and baseline in (RUNTIME_LOOP_BASELINES | {"factor_miner", "factor_miner_no_memory"})
        )

        if runtime_baseline:
            runtime_result = _run_runtime_mining_loop(
                cfg,
                baseline=baseline,
                dataset=freeze_dataset,
                output_dir=output_dir,
                runtime_manifest=runtime_manifest,
                mock=mock,
            )
            factors = list(runtime_result["library"].list_factors())
            provenance = runtime_result["provenance"]
            candidate_count = len(factors)
        else:
            entries = _get_baseline_entries(
                baseline,
                cfg.benchmark.seed,
                factor_miner_library_path=factor_miner_library_path,
                factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
            )
            factors = _factors_from_entries(entries)
            provenance = _baseline_provenance(
                baseline,
                factor_miner_library_path=factor_miner_library_path,
                factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
                candidate_count=len(entries),
                seed=cfg.benchmark.seed,
            )
            candidate_count = len(entries)

        artifacts = evaluate_factors(
            factors,
            freeze_dataset,
            signal_failure_policy="reject",
        )

        library_cfg = _cfg_with_overrides(cfg, cfg.benchmark.freeze_universe)
        if baseline == "factor_miner_no_memory":
            library_cfg.mining.ic_threshold = 0.02
            library_cfg.mining.correlation_threshold = 0.85
        library, library_stats = build_benchmark_library(
            artifacts,
            library_cfg,
            split_name="train",
            ic_threshold=library_cfg.mining.ic_threshold,
            correlation_threshold=library_cfg.mining.correlation_threshold,
        )
        frozen = select_frozen_top_k(
            artifacts,
            library,
            top_k=cfg.benchmark.freeze_top_k,
            split_name="train",
        )

        baseline_result = {
            "baseline": baseline,
            "mode": cfg.benchmark.mode,
            "metric_version": METRIC_VERSION,
            "freeze_universe": cfg.benchmark.freeze_universe,
            "candidate_count": candidate_count,
            "runtime_contract": runtime_contract.to_dict(),
            "paper_protocol": runtime_contract.paper_protocol,
            "walk_forward_contract": runtime_contract.walk_forward.to_dict(),
            "stress_contract": runtime_contract.stress.to_dict(),
            "strategy_grid_contract": runtime_contract.strategy_grid.to_dict(),
            "freeze_dataset_contract": dict(freeze_dataset_contract),
            "freeze_library_size": library.size,
            "freeze_stats": library_stats,
            "frozen_top_k": [
                {
                    "name": artifact.name,
                    "formula": artifact.formula,
                    "category": artifact.category,
                    "train_ic": artifact.split_stats["train"]["ic_paper_mean"],
                    "train_ic_mean": artifact.split_stats["train"]["ic_mean"],
                    "train_ic_abs_mean": artifact.split_stats["train"]["ic_abs_mean"],
                    "train_icir": artifact.split_stats["train"]["ic_paper_icir"],
                }
                for artifact in frozen
            ],
            "universes": {},
        }

        dataset_hashes = {cfg.benchmark.freeze_universe: freeze_hash}
        for universe in cfg.benchmark.report_universes:
            universe_cfg = _cfg_with_overrides(cfg, universe)
            dataset, dataset_hash = load_benchmark_dataset(
                universe_cfg,
                data_path=data_path,
                raw_df=raw_df,
                universe=universe,
                mock=mock,
            )
            dataset_hashes[universe] = dataset_hash
            baseline_result["universes"][universe] = evaluate_frozen_set(
                frozen,
                dataset,
                split_name="test",
                fit_split="train",
                cost_bps=list(runtime_contract.stress.cost_bps),
                capacity_levels=list(runtime_contract.stress.capacity_levels),
            )

        result_path = benchmark_dir / f"{baseline}.json"
        manifest_path = benchmark_dir / f"{baseline}_manifest.json"
        baseline_result["provenance"] = provenance
        _write_json(result_path, baseline_result)
        manifest = BenchmarkManifest(
            benchmark_name="table1",
            mode=cfg.benchmark.mode,
            seed=cfg.benchmark.seed,
            metric_version=METRIC_VERSION,
            baseline=baseline,
            freeze_universe=cfg.benchmark.freeze_universe,
            report_universes=list(cfg.benchmark.report_universes),
            train_period=list(cfg.data.train_period),
            test_period=list(cfg.data.test_period),
            freeze_top_k=cfg.benchmark.freeze_top_k,
            signal_failure_policy="reject",
            default_target=cfg.data.default_target,
            target_stack=[target.get("name", "") for target in cfg.data.targets],
            primary_objective=cfg.research.primary_objective,
            dataset_hashes=dataset_hashes,
            artifact_paths={
                "result": str(result_path),
                "manifest": str(manifest_path),
            },
            runtime_contract=runtime_contract.to_dict(),
            baseline_provenance={baseline: provenance},
            warnings=[],
        )
        _save_manifest(manifest_path, manifest)
        summary[baseline] = baseline_result

    return summary


def run_ablation_memory_benchmark(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Compare the default FactorMiner lane to the relaxed no-memory lane."""
    use_runtime_loops = bool(runtime_manifests or getattr(cfg.benchmark, "runtime_loops", False))
    comparison = run_table1_benchmark(
        cfg,
        output_dir,
        data_path=data_path,
        raw_df=raw_df,
        mock=mock,
        baseline_names=["factor_miner", "factor_miner_no_memory"],
        factor_miner_library_path=factor_miner_library_path,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
        runtime_manifests=runtime_manifests,
        use_runtime_loops=use_runtime_loops,
    )
    result = {}
    for baseline, payload in comparison.items():
        freeze_stats = payload["freeze_stats"]
        succeeded = max(freeze_stats.get("succeeded", 0), 1)
        result[baseline] = {
            "library_size": payload["freeze_library_size"],
            "high_quality_yield": freeze_stats.get("admitted", 0) / succeeded,
            "redundancy_rejection_rate": freeze_stats.get("correlation_rejections", 0) / succeeded,
            "replacements": freeze_stats.get("replaced", 0),
        }
    out_path = _ensure_dir(output_dir / "benchmark" / "ablation") / "memory_ablation.json"
    _write_json(out_path, result)
    return result


def run_ablation_strategy_benchmark(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    baseline: str | None = None,
    memory_policies: list[str] | None = None,
    dependence_metrics: list[str] | None = None,
    backends: list[str] | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Compare runtime loop variants across memory policy × dependence × backend."""
    strategy_cfg = _strategy_ablation_raw_config(cfg)
    baseline_name = str(baseline or strategy_cfg.get("baseline", "factor_miner"))
    memory_policies = list(
        memory_policies
        or strategy_cfg.get(
            "memory_policies",
            ["paper", "none", "kg", "family_aware", "regime_aware"],
        )
    )
    dependence_metrics = list(
        dependence_metrics
        or strategy_cfg.get(
            "dependence_metrics",
            ["spearman", "pearson", "distance_correlation"],
        )
    )
    backends = _runtime_strategy_backends(
        backends or strategy_cfg.get("backends"),
        default_backend=getattr(cfg.evaluation, "backend", "numpy"),
    )
    base_runtime_manifest = _runtime_manifest_value(runtime_manifests, baseline_name)
    freeze_cfg = _cfg_with_overrides(cfg, cfg.benchmark.freeze_universe)
    freeze_dataset, _ = load_benchmark_dataset(
        freeze_cfg,
        data_path=data_path,
        raw_df=raw_df,
        universe=cfg.benchmark.freeze_universe,
        mock=mock,
    )
    freeze_dataset_contract = _benchmark_dataset_contract(freeze_cfg, freeze_dataset)
    strategy_contract = _build_strategy_grid_contract(
        cfg,
        baseline=baseline_name,
        runtime_manifest={
            "memory_policies": memory_policies,
            "dependence_metrics": dependence_metrics,
            "backends": backends,
            **base_runtime_manifest,
        },
    )

    comparisons: list[dict[str, Any]] = []
    for memory_policy in memory_policies:
        for dependence_metric in dependence_metrics:
            for backend in backends:
                combo_name = f"{memory_policy}__{dependence_metric}__{backend}"
                combo_output_dir = output_dir / "benchmark" / "ablation" / "strategy_grid" / combo_name
                combo_manifest = {
                    **base_runtime_manifest,
                    "memory_policy": memory_policy,
                    "redundancy_metric": dependence_metric,
                    "backend": backend,
                }
                combo_payload = run_table1_benchmark(
                    cfg,
                    combo_output_dir,
                    data_path=data_path,
                    raw_df=raw_df,
                    mock=mock,
                    baseline_names=[baseline_name],
                    runtime_manifests={baseline_name: combo_manifest},
                    use_runtime_loops=True,
                )[baseline_name]
                freeze_stats = combo_payload.get("freeze_stats", {})
                succeeded = max(int(freeze_stats.get("succeeded", 0)), 1)
                comparisons.append(
                    {
                        "name": combo_name,
                        "baseline": baseline_name,
                        "memory_policy": memory_policy,
                        "dependence_metric": dependence_metric,
                        "backend": backend,
                        "runtime_contract": build_benchmark_runtime_contract(
                            cfg,
                            freeze_dataset_contract,
                            baseline=baseline_name,
                            runtime_manifest=combo_manifest,
                        ).to_dict(),
                        "artifacts_root": str(combo_output_dir),
                        "freeze_library_size": int(combo_payload.get("freeze_library_size", 0)),
                        "freeze_stats": freeze_stats,
                        "high_quality_yield": float(freeze_stats.get("admitted", 0)) / succeeded,
                        "redundancy_rejection_rate": (
                            float(freeze_stats.get("correlation_rejections", 0)) / succeeded
                        ),
                        "mean_test_library_ic": _mean_universe_metric(
                            combo_payload,
                            "library",
                            "ic",
                        ),
                        "mean_test_library_icir": _mean_universe_metric(
                            combo_payload,
                            "library",
                            "icir",
                        ),
                        "mean_test_library_rho": _mean_universe_metric(
                            combo_payload,
                            "library",
                            "avg_abs_rho",
                        ),
                        "universes": combo_payload.get("universes", {}),
                        "provenance": combo_payload.get("provenance", {}),
                    }
                )

    leaderboard = sorted(
        comparisons,
        key=lambda item: (
            item["mean_test_library_ic"] is not None,
            item["mean_test_library_ic"] or float("-inf"),
            item["mean_test_library_icir"] or float("-inf"),
            -float(item["mean_test_library_rho"] or 1.0),
        ),
        reverse=True,
    )
    result = {
        "baseline": baseline_name,
        "memory_policies": memory_policies,
        "dependence_metrics": dependence_metrics,
        "backends": backends,
        "strategy_grid_contract": strategy_contract.to_dict(),
        "comparisons": comparisons,
        "leaderboard": leaderboard,
        "best": leaderboard[0] if leaderboard else None,
    }
    out_path = _ensure_dir(output_dir / "benchmark" / "ablation") / "strategy_grid.json"
    _write_json(out_path, result)
    return result


def run_cost_pressure_benchmark(
    cfg,
    output_dir: Path,
    *,
    baseline: str = "factor_miner",
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    factor_miner_library_path: str | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Run cost-pressure analysis for one baseline on the configured universes."""
    use_runtime_loops = bool(runtime_manifests or getattr(cfg.benchmark, "runtime_loops", False))
    payload = run_table1_benchmark(
        cfg,
        output_dir,
        data_path=data_path,
        raw_df=raw_df,
        mock=mock,
        baseline_names=[baseline],
        factor_miner_library_path=factor_miner_library_path,
        runtime_manifests=runtime_manifests,
        use_runtime_loops=use_runtime_loops,
    )[baseline]
    result = {
        universe: {
            "combinations": {
                name: {
                    "cost_pressure": metrics.get("cost_pressure", {}),
                    "capacity_pressure": metrics.get("capacity_pressure"),
                }
                for name, metrics in universe_payload["combinations"].items()
            }
        }
        for universe, universe_payload in payload["universes"].items()
    }
    out_path = _ensure_dir(output_dir / "benchmark" / "cost_pressure") / f"{baseline}.json"
    _write_json(out_path, result)
    return result


def run_cpcv_benchmark(
    cfg,
    *,
    data_path: str | None = None,
    mock: bool = False,
    baseline: str = "factor_miner",
) -> dict:
    """Run Combinatorial Purged CV + Probability of Backtest Overfitting diagnostics.

    Companion to `run_table1_benchmark`'s Top-K freeze evaluation: rather
    than one frozen train/test split, every candidate in `baseline`'s
    catalog is scored on every `CombinatorialPurgedCV` path built from the
    freeze universe, producing an ``(n_trials, n_paths)`` out-of-sample
    long-short performance matrix. That matrix drives
    `ProbabilityOfBacktestOverfitting` (Bailey, Borwein, Lopez de Prado &
    Zhu, 2017) and, for the best-performing trial's per-path OOS series,
    `DeflatedSharpeCalculator` -- the same paper's companion statistic,
    already implemented in `evaluation.significance`. Together they answer
    "is the *best* candidate's edge real, or an artifact of trying many
    candidates against a validation scheme that leaks information".

    Parameters
    ----------
    cfg
        Hierarchical benchmark config (see `factorminer.utils.config.Config`).
    data_path : str, optional
        Market data path; ignored when `mock=True`.
    mock : bool
        Use deterministic mock market data instead of `data_path`.
    baseline : str
        Candidate catalog id understood by `_get_baseline_entries`
        (e.g. ``"factor_miner"``, ``"alpha101_classic"``, ``"gplearn"``).

    Returns
    -------
    dict
        JSON-safe manifest with `cpcv_config`, `cpcv`, `pbo`, and
        `deflated_sharpe` sections plus best-trial provenance.
    """
    from factorminer.evaluation.cross_validation import (
        CombinatorialPurgedCV,
        CrossValidationConfig,
        ProbabilityOfBacktestOverfitting,
    )
    from factorminer.evaluation.portfolio import PortfolioBacktester
    from factorminer.evaluation.significance import (
        DeflatedSharpeCalculator,
        SignificanceConfig,
    )

    freeze_cfg = _cfg_with_overrides(cfg, cfg.benchmark.freeze_universe)
    dataset, dataset_hash = load_benchmark_dataset(
        freeze_cfg,
        data_path=data_path,
        universe=cfg.benchmark.freeze_universe,
        mock=mock,
    )

    entries = _get_baseline_entries(baseline, cfg.benchmark.seed)
    factors = _factors_from_entries(entries)
    artifacts = evaluate_factors(factors, dataset, signal_failure_policy="reject")
    succeeded = [artifact for artifact in artifacts if artifact.succeeded]

    cv_config = CrossValidationConfig()
    manifest: dict[str, Any] = {
        "benchmark_name": "cpcv",
        "baseline": baseline,
        "freeze_universe": cfg.benchmark.freeze_universe,
        "dataset_hash": dataset_hash,
        "candidate_count": len(factors),
        "succeeded_count": len(succeeded),
        "cpcv_config": asdict(cv_config),
        "cpcv": None,
        "pbo": None,
        "deflated_sharpe": None,
        "best_trial": None,
        "warnings": [],
    }

    if len(succeeded) < 2:
        manifest["warnings"].append(
            "Fewer than 2 factors recomputed successfully; PBO/DSR require multiple trials."
        )
        return _json_safe(manifest)

    full_split = dataset.get_split("full")
    n_samples = int(full_split.returns.shape[1])
    cv = CombinatorialPurgedCV(cv_config)
    try:
        splits = cv.split(n_samples, label_horizon=1)
    except ValueError as exc:
        manifest["warnings"].append(f"CPCV split failed: {exc}")
        return _json_safe(manifest)

    backtester = PortfolioBacktester()
    returns_panel = full_split.returns.T  # (T, N assets)
    trial_names = [artifact.name for artifact in succeeded]
    n_trials = len(succeeded)
    n_paths = len(splits)
    is_oos_matrix = np.full((n_trials, n_paths), np.nan, dtype=np.float64)

    for trial_idx, artifact in enumerate(succeeded):
        stats = backtester.quintile_backtest(artifact.signals_full.T, returns_panel)
        ls_series = np.asarray(stats["ls_net_series"], dtype=np.float64)
        for split in splits:
            test_values = ls_series[split.test_indices]
            finite = test_values[np.isfinite(test_values)]
            if finite.size:
                is_oos_matrix[trial_idx, split.path_id] = float(np.mean(finite))

    valid_path_mask = np.all(np.isfinite(is_oos_matrix), axis=0)
    clean_matrix = is_oos_matrix[:, valid_path_mask]

    manifest["cpcv"] = {
        "n_samples": n_samples,
        "n_paths": n_paths,
        "n_valid_paths": int(clean_matrix.shape[1]),
        "label_horizon": 1,
    }

    if clean_matrix.shape[1] < cv_config.min_paths_for_pbo:
        manifest["warnings"].append(
            f"Only {clean_matrix.shape[1]} usable CPCV paths (< "
            f"min_paths_for_pbo={cv_config.min_paths_for_pbo}); skipping PBO."
        )
    else:
        pbo_result = ProbabilityOfBacktestOverfitting(cv_config).compute(clean_matrix)
        manifest["pbo"] = {
            "pbo": pbo_result.pbo,
            "n_combinations": pbo_result.n_combinations,
            "passes": pbo_result.passes,
            "logit_values": [float(value) for value in pbo_result.logit_values],
        }

    mean_oos = np.nanmean(is_oos_matrix, axis=1)
    best_idx = int(np.nanargmax(mean_oos))
    best_name = trial_names[best_idx]
    best_series = is_oos_matrix[best_idx][np.isfinite(is_oos_matrix[best_idx])]

    # `best_series` is one performance value per CPCV path (a subset of test
    # groups), not one value per calendar period -- annualization_factor=1.0
    # keeps the Deflated Sharpe on the same path-level scale as `pbo`
    # instead of implying a false calendar annualization.
    dsr_result = DeflatedSharpeCalculator(SignificanceConfig()).compute(
        best_name, best_series, n_trials=n_trials, annualization_factor=1.0
    )
    manifest["deflated_sharpe"] = {
        "factor_name": dsr_result.factor_name,
        "raw_sharpe": dsr_result.raw_sharpe,
        "deflated_sharpe": dsr_result.deflated_sharpe,
        "haircut": dsr_result.haircut,
        "p_value": dsr_result.p_value,
        "n_trials": dsr_result.n_trials,
        "passes": dsr_result.passes,
    }
    manifest["best_trial"] = {
        "name": best_name,
        "mean_oos_performance": float(mean_oos[best_idx]),
    }

    return _json_safe(manifest)


def _time_callable(fn, repeats: int = 3) -> float:
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return min(timings) * 1000.0


def run_efficiency_benchmark(cfg, output_dir: Path) -> dict:
    """Benchmark operator-level and factor-level compute time."""
    periods, assets = cfg.benchmark.efficiency_panel_shape
    matrix = np.random.RandomState(cfg.benchmark.seed).randn(assets, periods).astype(np.float64)
    other = np.random.RandomState(cfg.benchmark.seed + 1).randn(assets, periods).astype(np.float64)

    from factorminer.operators import torch_available
    from factorminer.operators.gpu_backend import to_tensor
    from factorminer.operators.registry import execute_operator
    from factorminer.utils.visualization import plot_efficiency_benchmark

    operator_bench: dict[str, dict[str, float | None]] = {"numpy": {}, "c": {}, "gpu": {}}

    def _backend_inputs(backend: str):
        if backend == "gpu":
            return to_tensor(matrix), to_tensor(other)
        return matrix, other

    operators = {
        "Add": lambda backend: execute_operator("Add", *_backend_inputs(backend), backend=backend),
        "Mean": lambda backend: execute_operator(
            "Mean", _backend_inputs(backend)[0], params={"window": 20}, backend=backend
        ),
        "Delta": lambda backend: execute_operator(
            "Delta", _backend_inputs(backend)[0], params={"window": 5}, backend=backend
        ),
        "TsRank": lambda backend: execute_operator(
            "TsRank", _backend_inputs(backend)[0], params={"window": 20}, backend=backend
        ),
        "Corr": lambda backend: execute_operator(
            "Corr", *_backend_inputs(backend), params={"window": 20}, backend=backend
        ),
        "CsRank": lambda backend: execute_operator(
            "CsRank", _backend_inputs(backend)[0], backend=backend
        ),
    }
    for op_name, runner in operators.items():
        operator_bench["numpy"][op_name] = _time_callable(lambda r=runner: r("numpy"))
        operator_bench["c"][op_name] = (
            _time_callable(lambda r=runner: r("c")) if c_backend_available() else None
        )
        if torch_available():
            operator_bench["gpu"][op_name] = _time_callable(lambda r=runner: r("gpu"))
        else:
            operator_bench["gpu"][op_name] = None

    factor_bench: dict[str, dict[str, float | None]] = {"numpy": {}, "c": {}, "gpu": {}}
    factor_specs = {
        "momentum_volume": lambda backend: execute_operator(
            "CsRank",
            execute_operator(
                "Mul",
                execute_operator(
                    "Return", _backend_inputs(backend)[0], params={"window": 5}, backend=backend
                ),
                execute_operator(
                    "Div",
                    _backend_inputs(backend)[1],
                    execute_operator(
                        "Mean", _backend_inputs(backend)[1], params={"window": 20}, backend=backend
                    ),
                    backend=backend,
                ),
                backend=backend,
            ),
            backend=backend,
        ),
        "vwap_gap": lambda backend: execute_operator(
            "Neg",
            execute_operator(
                "CsRank",
                execute_operator(
                    "Div",
                    execute_operator("Sub", *_backend_inputs(backend), backend=backend),
                    execute_operator(
                        "Add",
                        _backend_inputs(backend)[1],
                        to_tensor(np.full_like(other, 1e-8))
                        if backend == "gpu"
                        else np.full_like(other, 1e-8),
                        backend=backend,
                    ),
                    backend=backend,
                ),
                backend=backend,
            ),
            backend=backend,
        ),
    }
    for formula_name, runner in factor_specs.items():
        factor_bench["numpy"][formula_name] = _time_callable(lambda r=runner: r("numpy"))
        factor_bench["c"][formula_name] = (
            _time_callable(lambda r=runner: r("c")) if c_backend_available() else None
        )
        if torch_available():
            factor_bench["gpu"][formula_name] = _time_callable(lambda r=runner: r("gpu"))
        else:
            factor_bench["gpu"][formula_name] = None

    bench_dir = _ensure_dir(output_dir / "benchmark" / "efficiency")
    plot_efficiency_benchmark(
        {
            backend: {k: v for k, v in values.items() if v is not None}
            for backend, values in operator_bench.items()
        },
        save_path=str(bench_dir / "operator_efficiency.png"),
    )
    plot_efficiency_benchmark(
        {
            backend: {k: v for k, v in values.items() if v is not None}
            for backend, values in factor_bench.items()
        },
        save_path=str(bench_dir / "factor_efficiency.png"),
    )
    result = {
        "panel_shape": {"periods": periods, "assets": assets},
        "operator_level_ms": operator_bench,
        "factor_level_ms": factor_bench,
        "available_backends": {
            "numpy": True,
            "c": c_backend_available(),
            "gpu": torch_available(),
        },
    }
    _write_json(bench_dir / "efficiency.json", result)
    return result


def run_benchmark_suite(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Run the benchmark suite and return the artifact index."""
    if runtime_manifests is None:
        runtime_manifests = getattr(cfg.benchmark, "runtime_manifests", None)
    use_runtime_loops = bool(runtime_manifests or getattr(cfg.benchmark, "runtime_loops", False))
    strategy_cfg = _strategy_ablation_raw_config(cfg)
    run_strategy_ablation = bool(use_runtime_loops or strategy_cfg.get("enabled", False))
    results = {
        "table1": run_table1_benchmark(
            cfg,
            output_dir,
            data_path=data_path,
            raw_df=raw_df,
            mock=mock,
            factor_miner_library_path=factor_miner_library_path,
            factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
            runtime_manifests=runtime_manifests,
            use_runtime_loops=use_runtime_loops,
        ),
        "ablation_memory": run_ablation_memory_benchmark(
            cfg,
            output_dir,
            data_path=data_path,
            raw_df=raw_df,
            mock=mock,
            factor_miner_library_path=factor_miner_library_path,
            factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
            runtime_manifests=runtime_manifests,
        ),
        "ablation_strategy": (
            run_ablation_strategy_benchmark(
                cfg,
                output_dir,
                data_path=data_path,
                raw_df=raw_df,
                mock=mock,
                runtime_manifests=runtime_manifests,
            )
            if run_strategy_ablation
            else {
                "skipped": True,
                "reason": (
                    "runtime loops disabled; enable benchmark.strategy_ablation.enabled "
                    "or supply runtime manifests"
                ),
            }
        ),
        "cost_pressure": run_cost_pressure_benchmark(
            cfg,
            output_dir,
            data_path=data_path,
            raw_df=raw_df,
            mock=mock,
            factor_miner_library_path=factor_miner_library_path,
            runtime_manifests=runtime_manifests,
        ),
        "efficiency": run_efficiency_benchmark(cfg, output_dir),
    }
    _write_json(_ensure_dir(output_dir / "benchmark") / "suite.json", results)
    return results


def run_runtime_mining_benchmark(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Run the benchmark suite with explicit real-loop manifests when provided."""
    return run_benchmark_suite(
        cfg,
        output_dir,
        data_path=data_path,
        raw_df=raw_df,
        mock=mock,
        factor_miner_library_path=factor_miner_library_path,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
        runtime_manifests=runtime_manifests,
    )


# ---------------------------------------------------------------------------
# Consolidated Phase-2 reporting
# ---------------------------------------------------------------------------


def _reported_universe(payload: dict[str, Any], cfg) -> dict[str, Any]:
    universes = payload.get("universes", {})
    for name in getattr(cfg.benchmark, "report_universes", []):
        if name in universes:
            return universes[name]
    return next(iter(universes.values()), {})


def _method_result_from_runtime_payload(
    method: str,
    payload: dict[str, Any],
    cfg,
) -> MethodResult:
    evaluation = _reported_universe(payload, cfg)
    library = evaluation.get("library", {})
    combinations = evaluation.get("combinations", {})
    selections = evaluation.get("selections", {})
    equal_weight = combinations.get("equal_weight", {})
    ic_weighted = combinations.get("ic_weighted", {})
    lasso = selections.get("lasso", {})
    xgboost = selections.get("xgboost", {})
    freeze_stats = payload.get("freeze_stats", {})
    succeeded = max(int(freeze_stats.get("succeeded", 0)), 1)
    ic_series = np.asarray(equal_weight.get("ic_series", []), dtype=np.float64)
    return MethodResult(
        method=method,
        library_ic=float(library.get("ic", 0.0) or 0.0),
        library_icir=float(library.get("icir", 0.0) or 0.0),
        avg_abs_rho=float(library.get("avg_abs_rho", 0.0) or 0.0),
        ew_ic=float(equal_weight.get("ic", 0.0) or 0.0),
        ew_icir=float(equal_weight.get("icir", 0.0) or 0.0),
        icw_ic=float(ic_weighted.get("ic", 0.0) or 0.0),
        icw_icir=float(ic_weighted.get("icir", 0.0) or 0.0),
        lasso_ic=float(lasso.get("ic", 0.0) or 0.0),
        lasso_icir=float(lasso.get("icir", 0.0) or 0.0),
        xgb_ic=float(xgboost.get("ic", 0.0) or 0.0),
        xgb_icir=float(xgboost.get("icir", 0.0) or 0.0),
        n_factors=int(evaluation.get("factor_count", 0) or 0),
        admission_rate=float(freeze_stats.get("admitted", 0)) / succeeded,
        avg_turnover=float(equal_weight.get("turnover", 0.0) or 0.0),
        ic_series=ic_series if ic_series.size else None,
    )


def _comparison_frames(
    methods: list[str],
    results: dict[str, MethodResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    library = pd.DataFrame(
        [
            {
                "method": method,
                "ic_pct": results[method].library_ic * 100.0,
                "icir": results[method].library_icir,
                "avg_abs_rho": results[method].avg_abs_rho,
                "n_factors": results[method].n_factors,
                "avg_turnover": results[method].avg_turnover,
            }
            for method in methods
        ]
    )
    combinations = pd.DataFrame(
        [
            {
                "method": method,
                "ew_ic_pct": results[method].ew_ic * 100.0,
                "ew_icir": results[method].ew_icir,
                "icw_ic_pct": results[method].icw_ic * 100.0,
                "icw_icir": results[method].icw_icir,
            }
            for method in methods
        ]
    )
    selections = pd.DataFrame(
        [
            {
                "method": method,
                "lasso_ic_pct": results[method].lasso_ic * 100.0,
                "lasso_icir": results[method].lasso_icir,
                "xgb_ic_pct": results[method].xgb_ic * 100.0,
                "xgb_icir": results[method].xgb_icir,
                "best_ic_pct": max(results[method].lasso_ic, results[method].xgb_ic)
                * 100.0,
            }
            for method in methods
        ]
    )
    return library, combinations, selections


def run_phase2_comparison(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    baseline_methods: list[str] | None = None,
    n_target_factors: int = 40,
    n_runs: int = 1,
) -> tuple[BenchmarkResult, dict[str, Any]]:
    """Build the Phase-2 report entirely from canonical Table-1 artifacts."""
    if n_runs != 1:
        warnings.warn(
            "run_phase2_comparison now executes one provenance-complete runtime run per method",
            RuntimeWarning,
            stacklevel=2,
        )
    runtime_cfg = _clone_cfg(cfg)
    runtime_cfg.mining.target_library_size = n_target_factors
    methods = baseline_methods or [
        "random_exploration",
        "alpha101_classic",
        "alpha101_adapted",
        "ralph_loop",
        "helix_phase2",
    ]
    table1 = run_table1_benchmark(
        runtime_cfg,
        output_dir,
        data_path=data_path,
        raw_df=raw_df,
        mock=mock,
        baseline_names=methods,
        use_runtime_loops=True,
    )
    method_results = {
        method: _method_result_from_runtime_payload(method, table1[method], runtime_cfg)
        for method in methods
    }
    library_frame, combination_frame, selection_frame = _comparison_frames(
        methods,
        method_results,
    )
    efficiency = run_efficiency_benchmark(runtime_cfg, output_dir)
    speed_rows: list[dict[str, Any]] = []
    for backend, timings in efficiency.get("operator_level_ms", {}).items():
        for name, milliseconds in timings.items():
            if milliseconds is not None:
                speed_rows.append(
                    {"name": name, "time_ms": milliseconds, "type": f"operator/{backend}"}
                )
    speed_frame = pd.DataFrame(speed_rows)

    runtime_payloads: dict[str, list[dict[str, Any]]] = {}
    turnover_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for method in methods:
        payload = table1[method]
        evaluation = _reported_universe(payload, runtime_cfg)
        combinations = evaluation.get("combinations", {})
        projected = {
            "method": method,
            "run_id": 0,
            "frozen_top_k": payload.get("frozen_top_k", []),
            "library": evaluation.get("library", {}),
            "combinations": combinations,
            "selections": evaluation.get("selections", {}),
            "provenance": payload.get("provenance", {}),
        }
        runtime_payloads[method] = [projected]
        turnover_rows.append(
            {
                "method": method,
                "run_id": 0,
                **{
                    f"{name}_turnover": float(metrics.get("turnover", 0.0) or 0.0)
                    for name, metrics in combinations.items()
                },
            }
        )
        for combination, metrics in combinations.items():
            for cost_bps, cost_metrics in metrics.get("cost_pressure", {}).items():
                cost_rows.append(
                    {
                        "method": method,
                        "run_id": 0,
                        "combination": combination,
                        "cost_bps": float(cost_bps),
                        **{
                            key: cost_metrics.get(key, 0.0)
                            for key in ("ic", "icir", "turnover", "long_short", "monotonicity")
                        },
                    }
                )

    statistical_tests: dict[str, Any] = {}
    helix = method_results.get("helix_phase2")
    ralph = method_results.get("ralph_loop")
    if helix is not None and ralph is not None and helix.ic_series is not None and ralph.ic_series is not None:
        statistical_tests = StatisticalComparisonTests(runtime_cfg.benchmark.seed).run_all_tests(
            helix.ic_series,
            ralph.ic_series,
        )

    artifacts = {
        "runtime_root": str((output_dir / "benchmark").resolve()),
        "runtime_payloads": runtime_payloads,
        "table1": table1,
        "efficiency": efficiency,
    }
    result = BenchmarkResult(
        methods=methods,
        factor_library_metrics=library_frame,
        combination_metrics=combination_frame,
        selection_metrics=selection_frame,
        speed_metrics=speed_frame,
        statistical_tests=statistical_tests,
        raw_method_results={method: [result] for method, result in method_results.items()},
        turnover_metrics=pd.DataFrame(turnover_rows),
        cost_pressure_metrics=pd.DataFrame(cost_rows),
        runtime_artifacts=artifacts,
    )
    return result, artifacts


def run_phase2_ablation_study(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    configs_to_run: list[str] | None = None,
    n_target_factors: int = 40,
    n_runs: int = 1,
) -> AblationResult:
    """Run Phase-2 variants through the canonical runtime-loop benchmark."""
    if n_runs != 1:
        warnings.warn("Phase-2 ablations now run once per variant", RuntimeWarning, stacklevel=2)
    runtime_cfg = _clone_cfg(cfg)
    runtime_cfg.mining.target_library_size = n_target_factors
    configs = configs_to_run or [
        "full",
        "no_debate",
        "no_causal",
        "no_canonicalize",
        "no_regime",
        "no_capacity",
        "no_significance",
        "no_memory",
    ]
    baselines = {
        "full": "helix_phase2",
        "no_debate": "helix_no_debate",
        "no_causal": "helix_no_causal",
        "no_canonicalize": "helix_no_canonicalize",
        "no_regime": "helix_no_regime",
        "no_capacity": "helix_no_capacity",
        "no_significance": "helix_no_significance",
        "no_memory": "helix_no_memory",
    }
    results: dict[str, MethodResult] = {}
    for config_name in configs:
        if config_name not in baselines:
            logger.warning("Unknown runtime ablation config: %s", config_name)
            continue
        baseline = baselines[config_name]
        payload = run_table1_benchmark(
            runtime_cfg,
            output_dir / "runtime_ablation" / config_name,
            data_path=data_path,
            raw_df=raw_df,
            mock=mock,
            baseline_names=[baseline],
            use_runtime_loops=True,
        )[baseline]
        result = _method_result_from_runtime_payload(config_name, payload, runtime_cfg)
        results[config_name] = result

    rows: list[dict[str, Any]] = []
    full = results.get("full")
    if full is not None:
        for config_name, result in results.items():
            if config_name == "full":
                continue
            rows.append(
                {
                    "config": config_name,
                    "method": result.method,
                    "delta_library_ic": result.library_ic - full.library_ic,
                    "delta_library_icir": result.library_icir - full.library_icir,
                    "delta_ew_ic": result.ew_ic - full.ew_ic,
                    "delta_icw_ic": result.icw_ic - full.icw_ic,
                    "delta_lasso_ic": result.lasso_ic - full.lasso_ic,
                    "delta_xgb_ic": result.xgb_ic - full.xgb_ic,
                    "delta_turnover": result.avg_turnover - full.avg_turnover,
                }
            )
    return AblationResult(configs=configs, results=results, contributions=pd.DataFrame(rows))
