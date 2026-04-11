"""Canonical benchmark runtime surface.

`factorminer.benchmark.runtime` is the primary benchmark API.
Legacy Helix benchmarking helpers are preserved behind lazy, deprecated
exports so older scripts keep working while the runtime path stays canonical.
"""

from __future__ import annotations

from importlib import import_module
import warnings

from factorminer.benchmark.runtime import (
    BenchmarkManifest,
    BenchmarkRuntimeContract,
    StressBenchmarkContract,
    StrategyGridBenchmarkContract,
    WalkForwardBenchmarkContract,
    build_benchmark_library,
    build_benchmark_runtime_contract,
    evaluate_frozen_set,
    load_benchmark_dataset,
    run_ablation_memory_benchmark,
    run_ablation_strategy_benchmark,
    run_benchmark_suite,
    run_cost_pressure_benchmark,
    run_efficiency_benchmark,
    run_runtime_mining_benchmark,
    run_table1_benchmark,
    select_frozen_top_k,
)

_LEGACY_EXPORTS = {
    "HelixBenchmark": ("factorminer.benchmark.helix_benchmark", "HelixBenchmark"),
    "BenchmarkResult": ("factorminer.benchmark.helix_benchmark", "BenchmarkResult"),
    "MethodResult": ("factorminer.benchmark.helix_benchmark", "MethodResult"),
    "DMTestResult": ("factorminer.benchmark.helix_benchmark", "DMTestResult"),
    "StatisticalComparisonTests": (
        "factorminer.benchmark.helix_benchmark",
        "StatisticalComparisonTests",
    ),
    "SpeedBenchmark": ("factorminer.benchmark.helix_benchmark", "SpeedBenchmark"),
    "OperatorSpeedResult": ("factorminer.benchmark.helix_benchmark", "OperatorSpeedResult"),
    "PipelineSpeedResult": ("factorminer.benchmark.helix_benchmark", "PipelineSpeedResult"),
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
            "factorminer.benchmark.%s is legacy; use factorminer.benchmark.runtime instead"
            % name,
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
    "run_efficiency_benchmark",
    "run_runtime_mining_benchmark",
    "run_table1_benchmark",
    "select_frozen_top_k",
    *list(_LEGACY_EXPORTS),
    *list(_OPTIONAL_EXPORTS),
]
