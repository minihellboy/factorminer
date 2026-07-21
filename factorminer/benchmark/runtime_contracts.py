"""Construction of immutable benchmark runtime contracts."""

from __future__ import annotations

from typing import Any

import numpy as np

from factorminer.architecture import PaperProtocol
from factorminer.benchmark.contracts import (
    BenchmarkRuntimeContract,
    StrategyGridBenchmarkContract,
    StressBenchmarkContract,
    WalkForwardBenchmarkContract,
)
from factorminer.benchmark.frozen_evaluation import _default_capacity_levels


def _strategy_ablation_raw_config(cfg) -> dict[str, Any]:
    raw = getattr(cfg, "_raw", {})
    if not isinstance(raw, dict):
        return {}
    benchmark_raw = raw.get("benchmark", {})
    if not isinstance(benchmark_raw, dict):
        return {}
    strategy_raw = benchmark_raw.get("strategy_ablation", {})
    return dict(strategy_raw) if isinstance(strategy_raw, dict) else {}


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


def _build_stress_contract(
    cfg, runtime_manifest: dict[str, Any] | None = None
) -> StressBenchmarkContract:
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
            str(runtime_manifest["backend"])
            if runtime_manifest.get("backend") is not None
            else None
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
