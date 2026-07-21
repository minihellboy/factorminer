"""Real-loop construction and execution for runtime benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from factorminer.benchmark.datasets import _clone_cfg
from factorminer.benchmark.frozen_evaluation import _extract_volume_panel
from factorminer.benchmark.provenance import (
    _file_sha256,
    _json_safe,
    _json_summary,
    _session_summary,
)
from factorminer.benchmark.reporting import _ensure_dir
from factorminer.core.library_io import load_library
from factorminer.evaluation.runtime import EvaluationDataset
from factorminer.operators.c_backend import backend_available as c_backend_available

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
        cfg.memory.regime_lookback_window = int(runtime_manifest["memory_regime_lookback_window"])

    if runtime_manifest.get("relax_thresholds", mock):
        cfg.mining.ic_threshold = min(float(cfg.mining.ic_threshold), 0.0)
        cfg.mining.icir_threshold = min(float(cfg.mining.icir_threshold), -1.0)
        cfg.mining.correlation_threshold = max(float(cfg.mining.correlation_threshold), 1.1)

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
