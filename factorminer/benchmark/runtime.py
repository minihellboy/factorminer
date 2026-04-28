"""Strict paper/research benchmark runners built on runtime recomputation."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factorminer.architecture import PaperProtocol
from factorminer.benchmark.catalogs import (
    ALPHA101_CLASSIC,
    CandidateEntry,
    build_alpha101_adapted,
    build_alphaagent_style,
    build_alphaforge_style,
    build_factor_miner_catalog,
    build_gplearn_style,
    build_random_exploration,
    dedupe_entries,
    entries_from_library,
)
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import load_library
from factorminer.core.session import MiningSession
from factorminer.evaluation.metrics import METRIC_VERSION
from factorminer.evaluation.runtime import (
    EvaluationDataset,
    FactorEvaluationArtifact,
    compute_correlation_matrix,
    evaluate_factors,
    load_runtime_dataset,
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
}


@dataclass
class BenchmarkManifest:
    """Serializable description of one benchmark run."""

    benchmark_name: str
    mode: str
    seed: int
    metric_version: str
    baseline: str
    freeze_universe: str
    report_universes: list[str]
    train_period: list[str]
    test_period: list[str]
    freeze_top_k: int
    signal_failure_policy: str
    default_target: str
    target_stack: list[str]
    primary_objective: str
    dataset_hashes: dict[str, str]
    artifact_paths: dict[str, str]
    runtime_contract: dict[str, Any] = field(default_factory=dict)
    baseline_provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WalkForwardBenchmarkContract:
    """Canonical train/test freeze contract used by the benchmark runtime."""

    freeze_universe: str
    report_universes: list[str]
    train_period: list[str]
    test_period: list[str]
    freeze_top_k: int
    fit_split: str = "train"
    eval_split: str = "test"
    default_target: str = "paper"
    signal_failure_policy: str = "reject"
    dataset_contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class StressBenchmarkContract:
    """Canonical transaction-cost and capacity stress contract."""

    cost_bps: list[float]
    capacity_levels: list[float]
    base_capacity_usd: float
    net_icir_threshold: float
    ic_degradation_limit: float

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class StrategyGridBenchmarkContract:
    """Canonical strategy grid for benchmark ablations."""

    baseline: str
    enabled: bool
    memory_policies: list[str]
    dependence_metrics: list[str]
    backends: list[str]
    selected_memory_policy: str | None = None
    selected_dependence_metric: str | None = None
    selected_backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class BenchmarkRuntimeContract:
    """Complete runtime benchmark contract emitted to manifests/results."""

    paper_protocol: dict[str, Any]
    walk_forward: WalkForwardBenchmarkContract
    stress: StressBenchmarkContract
    strategy_grid: StrategyGridBenchmarkContract
    runtime_manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "paper_protocol": self.paper_protocol,
                "walk_forward": self.walk_forward.to_dict(),
                "stress": self.stress.to_dict(),
                "strategy_grid": self.strategy_grid.to_dict(),
                "runtime_manifest": self.runtime_manifest,
            }
        )


def _clone_cfg(cfg):
    cloned = copy.deepcopy(cfg)
    cloned._raw = copy.deepcopy(getattr(cfg, "_raw", {}))
    return cloned


def _cfg_with_overrides(cfg, universe: str, mode: str | None = None):
    cloned = _clone_cfg(cfg)
    cloned.data.universe = universe
    if mode is not None:
        cloned.benchmark.mode = mode
    if cloned.benchmark.mode == "paper":
        cloned.evaluation.signal_failure_policy = "reject"
        cloned.research.enabled = False
        cloned.phase2.causal.enabled = False
        cloned.phase2.regime.enabled = False
        cloned.phase2.capacity.enabled = False
        cloned.phase2.significance.enabled = False
        cloned.phase2.debate.enabled = False
        cloned.phase2.auto_inventor.enabled = False
        cloned.phase2.helix.enabled = False
    else:
        cloned.research.enabled = True
    return cloned


def _data_hash(df: pd.DataFrame) -> str:
    sample = df.sort_values(["datetime", "asset_id"]).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
    return digest.hexdigest()


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


def _default_capacity_levels() -> list[float]:
    from factorminer.evaluation.capacity import CapacityConfig as RuntimeCapacityConfig

    return list(RuntimeCapacityConfig().capacity_levels)


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


def _extract_volume_panel(dataset: EvaluationDataset) -> np.ndarray | None:
    """Best-effort extraction of a dollar-volume panel for Helix capacity checks."""
    for key in ("$amt", "$volume"):
        panel = dataset.data_dict.get(key)
        if panel is not None and np.any(np.isfinite(panel)):
            return np.asarray(panel, dtype=np.float64)
    return None


def _split_volume_panel(
    dataset: EvaluationDataset,
    split_name: str,
) -> np.ndarray | None:
    """Align the available volume panel to one dataset split."""
    panel = _extract_volume_panel(dataset)
    if panel is None:
        return None
    split = dataset.get_split(split_name)
    if panel.ndim != 2 or panel.shape[1] < len(split.indices):
        return None
    return np.asarray(panel[:, split.indices], dtype=np.float64)


def _capacity_pressure_summary(
    *,
    factor_name: str,
    signals: np.ndarray,
    returns: np.ndarray,
    volume: np.ndarray,
    capacity_levels: list[float],
) -> dict[str, Any]:
    """Compute a compact capacity-stress summary for one factor/composite."""
    from factorminer.evaluation.capacity import CapacityConfig as RuntimeCapacityConfig
    from factorminer.evaluation.capacity import CapacityEstimator

    cap_cfg = RuntimeCapacityConfig(capacity_levels=list(capacity_levels))
    estimate = CapacityEstimator(
        np.asarray(returns, dtype=np.float64).T,
        np.asarray(volume, dtype=np.float64),
        cap_cfg,
    ).estimate(
        factor_name,
        np.asarray(signals, dtype=np.float64).T,
    )
    return {
        "factor_name": factor_name,
        "max_capacity_usd": float(estimate.max_capacity_usd),
        "break_even_cost_bps": float(estimate.break_even_cost_bps),
        "capacity_curve": {
            str(capital): float(degradation)
            for capital, degradation in estimate.capacity_curve.items()
        },
    }


def _build_runtime_loop_config(
    cfg,
    *,
    output_dir: Path,
    dataset: EvaluationDataset,
    mock: bool,
    runtime_manifest: dict[str, Any],
):
    """Build the flat loop config consumed by RalphLoop/HelixLoop."""
    from factorminer.core.config import MiningConfig as LoopMiningConfig

    target_library_size = int(
        runtime_manifest.get(
            "target_library_size",
            getattr(cfg.mining, "target_library_size", 110),
        )
    )
    max_iterations = int(
        runtime_manifest.get(
            "max_iterations",
            getattr(cfg.mining, "max_iterations", 200),
        )
    )
    ic_threshold = float(
        runtime_manifest.get(
            "ic_threshold",
            getattr(cfg.mining, "ic_threshold", 0.04),
        )
    )
    icir_threshold = float(
        runtime_manifest.get(
            "icir_threshold",
            getattr(cfg.mining, "icir_threshold", 0.5),
        )
    )
    correlation_threshold = float(
        runtime_manifest.get(
            "correlation_threshold",
            getattr(cfg.mining, "correlation_threshold", 0.5),
        )
    )
    replacement_ic_min = float(
        runtime_manifest.get(
            "replacement_ic_min",
            getattr(cfg.mining, "replacement_ic_min", 0.10),
        )
    )
    replacement_ic_ratio = float(
        runtime_manifest.get(
            "replacement_ic_ratio",
            getattr(cfg.mining, "replacement_ic_ratio", 1.3),
        )
    )

    if runtime_manifest.get("relax_thresholds", mock):
        ic_threshold = min(ic_threshold, 0.0)
        icir_threshold = min(icir_threshold, -1.0)
        correlation_threshold = max(correlation_threshold, 1.1)

    loop_cfg = LoopMiningConfig(
        target_library_size=target_library_size,
        batch_size=int(runtime_manifest.get("batch_size", getattr(cfg.mining, "batch_size", 40))),
        max_iterations=max_iterations,
        ic_threshold=ic_threshold,
        icir_threshold=icir_threshold,
        correlation_threshold=correlation_threshold,
        replacement_ic_min=replacement_ic_min,
        replacement_ic_ratio=replacement_ic_ratio,
        fast_screen_assets=int(
            runtime_manifest.get(
                "fast_screen_assets",
                getattr(cfg.evaluation, "fast_screen_assets", 100),
            )
        ),
        num_workers=int(
            runtime_manifest.get("num_workers", getattr(cfg.evaluation, "num_workers", 1))
        ),
        output_dir=str(output_dir),
        backend=str(runtime_manifest.get("backend", getattr(cfg.evaluation, "backend", "numpy"))),
        gpu_device=str(
            runtime_manifest.get("gpu_device", getattr(cfg.evaluation, "gpu_device", "cuda:0"))
        ),
        redundancy_metric=str(
            runtime_manifest.get(
                "redundancy_metric",
                getattr(cfg.evaluation, "redundancy_metric", "spearman"),
            )
        ),
        signal_failure_policy=str(
            runtime_manifest.get(
                "signal_failure_policy",
                "synthetic" if mock else getattr(cfg.evaluation, "signal_failure_policy", "reject"),
            )
        ),
        memory_policy=str(
            runtime_manifest.get(
                "memory_policy",
                getattr(getattr(cfg, "memory", None), "policy", "paper"),
            )
        ),
        memory_regime_lookback_window=int(
            runtime_manifest.get(
                "memory_regime_lookback_window",
                getattr(getattr(cfg, "memory", None), "regime_lookback_window", 60),
            )
        ),
    )

    loop_cfg.research = cfg.research
    loop_cfg.benchmark_mode = str(getattr(cfg.benchmark, "mode", "paper"))
    loop_cfg.target_panels = dataset.target_panels
    loop_cfg.target_horizons = {
        name: max(int(spec.holding_bars), 1) for name, spec in dataset.target_specs.items()
    }
    return loop_cfg


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
    loop_cfg = _build_runtime_loop_config(
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
            config=loop_cfg,
            data_tensor=dataset.data_tensor,
            returns=dataset.returns,
            llm_provider=provider,
            volume=_extract_volume_panel(dataset),
            **phase2_kwargs,
        )
    else:
        from factorminer.core.ralph_loop import RalphLoop

        loop = RalphLoop(
            config=loop_cfg,
            data_tensor=dataset.data_tensor,
            returns=dataset.returns,
            llm_provider=provider,
        )

    checkpoint_interval = int(runtime_manifest.get("checkpoint_interval", 0 if mock else 1))
    loop.checkpoint_interval = checkpoint_interval

    if runtime_manifest.get("checkpoint_path"):
        loop.load_session(str(runtime_manifest["checkpoint_path"]))

    target_size = int(runtime_manifest.get("target_library_size", loop_cfg.target_library_size))
    max_iterations = int(runtime_manifest.get("max_iterations", loop_cfg.max_iterations))
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


def load_benchmark_dataset(
    cfg,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    universe: str | None = None,
    mock: bool = False,
) -> tuple[EvaluationDataset, str]:
    """Load one universe into the canonical runtime dataset."""
    if universe is None:
        universe = cfg.data.universe

    if raw_df is None:
        if mock:
            from factorminer.data.mock_data import MockConfig, generate_mock_data

            mock_shape = getattr(cfg.benchmark, "mock_panel_shape", None)
            if mock_shape is None:
                num_periods = 12_200
                num_assets = 64 if universe.lower() == "binance" else 80
            else:
                num_periods = int(mock_shape[0])
                num_assets = int(mock_shape[1])
            mock_cfg = MockConfig(
                num_assets=num_assets,
                num_periods=num_periods,
                frequency="10min",
                start_date="2024-01-02 09:30:00",
                universe=universe,
                plant_alpha=True,
                seed=cfg.benchmark.seed,
            )
            raw_df = generate_mock_data(mock_cfg)
        else:
            path = data_path
            if path is None:
                path = getattr(cfg, "_raw", {}).get("data_path")
            if path is None:
                raise ValueError("No data path specified for benchmark run")
            from factorminer.data.loader import load_market_data

            raw_df = load_market_data(path, universe=universe)

    dataset_cfg = _cfg_with_overrides(cfg, universe)
    return load_runtime_dataset(raw_df, dataset_cfg), _data_hash(raw_df)


def _factors_from_entries(entries: Iterable[CandidateEntry]) -> list[Factor]:
    return [
        Factor(
            id=idx + 1,
            name=entry.name,
            formula=entry.formula,
            category=entry.category,
            ic_mean=0.0,
            icir=0.0,
            ic_win_rate=0.0,
            max_correlation=0.0,
            batch_number=0,
        )
        for idx, entry in enumerate(entries)
    ]


def _get_baseline_entries(
    baseline: str,
    seed: int,
    *,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
) -> list[CandidateEntry]:
    if baseline == "alpha101_classic":
        return dedupe_entries(ALPHA101_CLASSIC)
    if baseline == "alpha101_adapted":
        return dedupe_entries(build_alpha101_adapted())
    if baseline == "random_exploration":
        return dedupe_entries(build_random_exploration(seed))
    if baseline == "gplearn":
        return dedupe_entries(build_gplearn_style(seed))
    if baseline == "alphaforge_style":
        return dedupe_entries(build_alphaforge_style())
    if baseline == "alphaagent_style":
        return dedupe_entries(build_alphaagent_style())
    if baseline == "factor_miner":
        if factor_miner_library_path:
            return dedupe_entries(
                entries_from_library(load_library(_base_path(factor_miner_library_path)))
            )
        return dedupe_entries(build_factor_miner_catalog())
    if baseline == "factor_miner_no_memory":
        if factor_miner_no_memory_library_path:
            return dedupe_entries(
                entries_from_library(load_library(_base_path(factor_miner_no_memory_library_path)))
            )
        return dedupe_entries(build_random_exploration(seed + 101, count=200))
    raise KeyError(f"Unknown benchmark baseline: {baseline}")


def _base_path(path: str) -> str:
    p = Path(path)
    return str(p.with_suffix("")) if p.suffix == ".json" else str(p)


def build_benchmark_library(
    artifacts: Iterable[FactorEvaluationArtifact],
    cfg,
    *,
    split_name: str = "train",
    ic_threshold: float | None = None,
    correlation_threshold: float | None = None,
) -> tuple[FactorLibrary, dict[str, int]]:
    """Build a library from candidate artifacts under the paper admission rules."""
    ic_threshold = cfg.mining.ic_threshold if ic_threshold is None else ic_threshold
    correlation_threshold = (
        cfg.mining.correlation_threshold if correlation_threshold is None else correlation_threshold
    )
    library = FactorLibrary(
        correlation_threshold=correlation_threshold,
        ic_threshold=ic_threshold,
        dependence_metric=getattr(cfg.evaluation, "redundancy_metric", "spearman"),
    )

    stats = {
        "succeeded": 0,
        "admitted": 0,
        "replaced": 0,
        "threshold_rejections": 0,
        "correlation_rejections": 0,
    }

    ordered = [artifact for artifact in artifacts if artifact.succeeded]
    ordered.sort(
        key=lambda artifact: artifact.split_stats[split_name]["ic_paper_mean"],
        reverse=True,
    )
    stats["succeeded"] = len(ordered)

    for artifact in ordered:
        split_stats = artifact.split_stats[split_name]
        candidate_ic = float(split_stats["ic_paper_mean"])
        candidate_signals = artifact.split_signals[split_name]
        if candidate_ic < ic_threshold:
            stats["threshold_rejections"] += 1
            continue

        max_corr = (
            library._max_correlation_with_library(candidate_signals)  # noqa: SLF001
            if library.size
            else 0.0
        )
        factor = Factor(
            id=0,
            name=artifact.name,
            formula=artifact.formula,
            category=artifact.category,
            ic_mean=float(split_stats["ic_mean"]),
            ic_paper_mean=candidate_ic,
            ic_abs_mean=float(split_stats["ic_abs_mean"]),
            icir=float(split_stats["icir"]),
            ic_paper_icir=float(split_stats["ic_paper_icir"]),
            ic_win_rate=float(split_stats["ic_win_rate"]),
            max_correlation=max_corr,
            batch_number=0,
            signals=candidate_signals,
        )
        admitted, _ = library.check_admission(candidate_ic, candidate_signals)
        if admitted:
            library.admit_factor(factor)
            stats["admitted"] += 1
            continue

        replace, replace_id, _ = library.check_replacement(
            candidate_ic,
            candidate_signals,
            ic_min=cfg.mining.replacement_ic_min,
            ic_ratio=cfg.mining.replacement_ic_ratio,
        )
        if replace and replace_id is not None:
            library.replace_factor(replace_id, factor)
            stats["replaced"] += 1
            continue

        stats["correlation_rejections"] += 1

    return library, stats


def select_frozen_top_k(
    artifacts: Iterable[FactorEvaluationArtifact],
    library: FactorLibrary,
    *,
    top_k: int,
    split_name: str = "train",
    min_ic: float = 0.05,
    min_icir: float = 0.5,
) -> list[FactorEvaluationArtifact]:
    """Freeze the paper Top-K set from train-split recomputed metrics."""
    admitted_formulas = {factor.formula for factor in library.list_factors()}
    succeeded = [artifact for artifact in artifacts if artifact.succeeded]
    admitted = [
        artifact
        for artifact in succeeded
        if artifact.formula in admitted_formulas
        and artifact.split_stats[split_name]["ic_paper_mean"] >= min_ic
        and artifact.split_stats[split_name]["ic_paper_icir"] >= min_icir
    ]
    admitted.sort(
        key=lambda artifact: artifact.split_stats[split_name]["ic_paper_mean"],
        reverse=True,
    )
    selected: list[FactorEvaluationArtifact] = admitted[:top_k]
    selected_formulas = {artifact.formula for artifact in selected}

    if len(selected) < top_k:
        remainder = [
            artifact for artifact in succeeded if artifact.formula not in selected_formulas
        ]
        remainder.sort(
            key=lambda artifact: artifact.split_stats[split_name]["ic_paper_mean"],
            reverse=True,
        )
        selected.extend(remainder[: top_k - len(selected)])

    return selected


def _abs_icir_from_series(ic_series: np.ndarray) -> float:
    valid = ic_series[np.isfinite(ic_series)]
    if len(valid) < 3:
        return 0.0
    std = float(np.std(valid, ddof=1))
    if std < 1e-12:
        return 0.0
    return abs(float(np.mean(valid))) / std


def _normalize_backtest_stats(stats: dict) -> dict[str, float]:
    ic_series = np.asarray(stats.get("ic_series", []), dtype=np.float64)
    valid_ic = ic_series[np.isfinite(ic_series)]
    signed_ic = float(stats.get("ic_mean", 0.0))
    paper_ic = abs(signed_ic)
    return {
        "metric_version": METRIC_VERSION,
        "ic": paper_ic,
        "ic_mean": signed_ic,
        "ic_paper_mean": paper_ic,
        "ic_abs_mean": float(np.mean(np.abs(valid_ic))) if valid_ic.size else 0.0,
        "icir": _abs_icir_from_series(ic_series),
        "ic_win_rate": float(stats.get("ic_win_rate", 0.0)),
        "long_short": float(stats.get("ls_return", 0.0)),
        "monotonicity": float(stats.get("monotonicity", 0.0)),
        "turnover": float(stats.get("avg_turnover", 0.0)),
    }


def _avg_abs_rho(artifacts: list[FactorEvaluationArtifact], split_name: str) -> float:
    if len(artifacts) < 2:
        return 0.0
    corr = np.abs(compute_correlation_matrix(artifacts, split_name))
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _weighted_composite(
    factor_signals: dict[int, np.ndarray],
    weights: dict[int, float],
) -> np.ndarray:
    ordered = [(fid, factor_signals[fid], weights.get(fid, 0.0)) for fid in factor_signals]
    if not ordered:
        raise ValueError("Cannot build weighted composite from zero factors")
    total = sum(abs(weight) for _, _, weight in ordered)
    if total < 1e-12:
        total = float(len(ordered))
        ordered = [(fid, signal, 1.0) for fid, signal, _ in ordered]
    composite = np.zeros_like(ordered[0][1], dtype=np.float64)
    for _, signal, weight in ordered:
        composite += signal * (weight / total)
    return composite


def evaluate_frozen_set(
    frozen: list[FactorEvaluationArtifact],
    dataset: EvaluationDataset,
    *,
    split_name: str = "test",
    fit_split: str = "train",
    cost_bps: list[float] | None = None,
    capacity_levels: list[float] | None = None,
) -> dict:
    """Evaluate one frozen factor set on one universe."""
    if cost_bps is None:
        cost_bps = [1.0, 4.0, 7.0, 10.0, 11.0]
    if capacity_levels is None:
        capacity_levels = _default_capacity_levels()

    factors = _factors_from_entries(
        CandidateEntry(
            name=artifact.name,
            formula=artifact.formula,
            category=artifact.category,
        )
        for artifact in frozen
    )
    artifacts = evaluate_factors(factors, dataset, signal_failure_policy="reject")
    succeeded = [artifact for artifact in artifacts if artifact.succeeded]

    result = {
        "factor_count": len(succeeded),
        "library": {
            "ic": 0.0,
            "icir": 0.0,
            "avg_abs_rho": 0.0,
        },
        "combinations": {},
        "selections": {},
        "stress": {
            "cost_bps": [float(value) for value in cost_bps],
            "capacity_levels": [float(value) for value in capacity_levels],
        },
        "warnings": [],
    }
    if not succeeded:
        result["warnings"].append("No frozen factors recomputed successfully on this universe")
        return result

    result["library"] = {
        "ic": float(
            np.mean([artifact.split_stats[split_name]["ic_paper_mean"] for artifact in succeeded])
        ),
        "icir": float(
            np.mean([artifact.split_stats[split_name]["ic_paper_icir"] for artifact in succeeded])
        ),
        "metric_version": METRIC_VERSION,
        "avg_abs_rho": _avg_abs_rho(succeeded, split_name),
    }

    artifact_map = {artifact.factor_id: artifact for artifact in succeeded}
    fit_signals = {
        artifact.factor_id: artifact.split_signals[fit_split].T for artifact in succeeded
    }
    eval_signals = {
        artifact.factor_id: artifact.split_signals[split_name].T for artifact in succeeded
    }
    fit_returns = dataset.get_split(fit_split).returns.T
    eval_returns = dataset.get_split(split_name).returns.T
    eval_volume = _split_volume_panel(dataset, split_name)

    from factorminer.evaluation.combination import FactorCombiner
    from factorminer.evaluation.portfolio import PortfolioBacktester
    from factorminer.evaluation.selection import FactorSelector

    combiner = FactorCombiner()
    backtester = PortfolioBacktester()
    selector = FactorSelector()

    fit_ic_values = {
        artifact.factor_id: artifact.split_stats[fit_split]["ic_mean"] for artifact in succeeded
    }

    combos = {
        "equal_weight": combiner.equal_weight(eval_signals),
        "ic_weighted": combiner.ic_weighted(eval_signals, fit_ic_values),
        "orthogonal": combiner.orthogonal(eval_signals),
    }
    for name, composite in combos.items():
        stats = backtester.quintile_backtest(composite, eval_returns)
        result["combinations"][name] = _normalize_backtest_stats(stats)
        result["combinations"][name]["ic_series"] = _json_safe(
            np.asarray(stats.get("ic_series", []), dtype=np.float64).tolist()
        )
        result["combinations"][name]["turnover_series"] = _json_safe(
            np.asarray(stats.get("turnover_series", []), dtype=np.float64).tolist()
        )
        result["combinations"][name]["cost_pressure"] = {
            str(cost): _normalize_backtest_stats(
                backtester.quintile_backtest(
                    composite, eval_returns, transaction_cost_bps=float(cost)
                )
            )
            for cost in cost_bps
        }
        if eval_volume is not None:
            result["combinations"][name]["capacity_pressure"] = _capacity_pressure_summary(
                factor_name=name,
                signals=composite,
                returns=eval_returns,
                volume=eval_volume,
                capacity_levels=capacity_levels,
            )

    selection_specs = {}
    try:
        selection_specs["lasso"] = selector.lasso_selection(fit_signals, fit_returns)
    except Exception as exc:
        result["warnings"].append(f"lasso unavailable: {exc}")
    try:
        selection_specs["forward_stepwise"] = selector.forward_stepwise(fit_signals, fit_returns)
    except Exception as exc:
        result["warnings"].append(f"forward_stepwise unavailable: {exc}")
    try:
        selection_specs["xgboost"] = selector.xgboost_selection(fit_signals, fit_returns)
    except Exception as exc:
        result["warnings"].append(f"xgboost unavailable: {exc}")

    for name, ranking in selection_specs.items():
        if not ranking:
            result["selections"][name] = {"factor_count": 0}
            continue
        selected_ids = [factor_id for factor_id, _ in ranking]
        selected_eval = {factor_id: eval_signals[factor_id] for factor_id in selected_ids}
        if name == "lasso":
            weights = {factor_id: score for factor_id, score in ranking}
            composite = _weighted_composite(selected_eval, weights)
        elif name == "xgboost":
            weights = {
                factor_id: score
                * np.sign(artifact_map[factor_id].split_stats[fit_split]["ic_mean"] or 1.0)
                for factor_id, score in ranking
            }
            composite = _weighted_composite(selected_eval, weights)
        else:
            signs = {
                factor_id: np.sign(artifact_map[factor_id].split_stats[fit_split]["ic_mean"] or 1.0)
                for factor_id in selected_ids
            }
            composite = _weighted_composite(selected_eval, signs)
        stats = backtester.quintile_backtest(composite, eval_returns)
        result["selections"][name] = {
            "factor_count": len(selected_ids),
            **_normalize_backtest_stats(stats),
            "ic_series": _json_safe(
                np.asarray(stats.get("ic_series", []), dtype=np.float64).tolist()
            ),
            "turnover_series": _json_safe(
                np.asarray(stats.get("turnover_series", []), dtype=np.float64).tolist()
            ),
        }

    return result


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(_json_safe(payload), fp, indent=2, sort_keys=False, allow_nan=False)


def _save_manifest(path: Path, manifest: BenchmarkManifest) -> None:
    _write_json(path, asdict(manifest))


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
