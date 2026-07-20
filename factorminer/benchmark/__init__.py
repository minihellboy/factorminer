"""Canonical benchmark runtime surface.

`factorminer.benchmark.runtime` is the primary benchmark API.
Legacy Helix benchmarking helpers are preserved behind lazy, deprecated
exports so older scripts keep working while the runtime path stays canonical.
"""

from __future__ import annotations

import warnings
from importlib import import_module

from factorminer.benchmark.runtime import (
    BenchmarkManifest,
    BenchmarkRuntimeContract,
    StrategyGridBenchmarkContract,
    StressBenchmarkContract,
    WalkForwardBenchmarkContract,
    build_benchmark_library,
    build_benchmark_runtime_contract,
    evaluate_frozen_set,
    load_benchmark_dataset,
    run_ablation_memory_benchmark,
    run_ablation_strategy_benchmark,
    run_benchmark_suite,
    run_cost_pressure_benchmark,
    run_cpcv_benchmark,
    run_efficiency_benchmark,
    run_runtime_mining_benchmark,
    run_table1_benchmark,
    select_frozen_top_k,
)

_LEGACY_EXPORTS = {
    "HelixBenchmark": ("factorminer.benchmark.runtime", "HelixBenchmark"),
    "BenchmarkResult": ("factorminer.benchmark.runtime", "BenchmarkResult"),
    "MethodResult": ("factorminer.benchmark.runtime", "MethodResult"),
    "DMTestResult": ("factorminer.benchmark.runtime", "DMTestResult"),
    "StatisticalComparisonTests": (
        "factorminer.benchmark.runtime",
        "StatisticalComparisonTests",
    ),
    "SpeedBenchmark": ("factorminer.benchmark.runtime", "SpeedBenchmark"),
    "OperatorSpeedResult": ("factorminer.benchmark.runtime", "OperatorSpeedResult"),
    "PipelineSpeedResult": ("factorminer.benchmark.runtime", "PipelineSpeedResult"),
}

_OPTIONAL_EXPORTS = {
    "AblationStudy": ("factorminer.benchmark.ablation", "AblationStudy"),
    "AblationResult": ("factorminer.benchmark.ablation", "AblationResult"),
    "AblatedMethodRunner": ("factorminer.benchmark.ablation", "AblatedMethodRunner"),
    "ABLATION_CONFIGS": ("factorminer.benchmark.ablation", "ABLATION_CONFIGS"),
    "ABLATION_LABELS": ("factorminer.benchmark.ablation", "ABLATION_LABELS"),
    "run_full_ablation_study": ("factorminer.benchmark.ablation", "run_full_ablation_study"),
}


def __getattr__(name: str):
    if name in _LEGACY_EXPORTS:
        warnings.warn(
            f"factorminer.benchmark.{name} is legacy; use factorminer.benchmark.runtime instead",
            DeprecationWarning,
            stacklevel=2,
        )
        module_name, attr_name = _LEGACY_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    if name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BenchmarkManifest",
    "BenchmarkRuntimeContract",
    "StressBenchmarkContract",
    "StrategyGridBenchmarkContract",
    "WalkForwardBenchmarkContract",
    "build_benchmark_library",
    "build_benchmark_runtime_contract",
    "evaluate_frozen_set",
    "load_benchmark_dataset",
    "run_ablation_memory_benchmark",
    "run_ablation_strategy_benchmark",
    "run_benchmark_suite",
    "run_cost_pressure_benchmark",
    "run_cpcv_benchmark",
    "run_efficiency_benchmark",
    "run_runtime_mining_benchmark",
    "run_table1_benchmark",
    "select_frozen_top_k",
    *list(_LEGACY_EXPORTS),
    *list(_OPTIONAL_EXPORTS),
]
