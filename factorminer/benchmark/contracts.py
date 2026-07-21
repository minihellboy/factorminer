"""Versioned contracts and manifests for benchmark execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert benchmark payload values to strict JSON-compatible objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


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
    validation_period: list[str] = field(default_factory=list)
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
    validation_period: list[str] = field(default_factory=list)
    selection_split: str = "train"
    purge_bars: int = 0
    embargo_bars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass(frozen=True)
class StressBenchmarkContract:
    """Canonical transaction-cost and capacity stress contract."""

    cost_bps: list[float]
    capacity_levels: list[float]
    base_capacity_usd: float
    net_icir_threshold: float
    ic_degradation_limit: float

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


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
        return json_safe(asdict(self))


@dataclass(frozen=True)
class BenchmarkRuntimeContract:
    """Complete runtime benchmark contract emitted to manifests/results."""

    paper_protocol: dict[str, Any]
    walk_forward: WalkForwardBenchmarkContract
    stress: StressBenchmarkContract
    strategy_grid: StrategyGridBenchmarkContract
    runtime_manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "paper_protocol": self.paper_protocol,
                "walk_forward": self.walk_forward.to_dict(),
                "stress": self.stress.to_dict(),
                "strategy_grid": self.strategy_grid.to_dict(),
                "runtime_manifest": self.runtime_manifest,
            }
        )
