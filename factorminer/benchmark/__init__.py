"""Benchmark runners for paper-faithful and Helix research evaluation.

`factorminer.benchmark.runtime` is the canonical benchmark surface.
`helix_benchmark` remains available as a legacy report/analysis module.
"""

from factorminer.benchmark.runtime import (
    BenchmarkManifest,
    build_benchmark_library,
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
from factorminer.benchmark.helix_benchmark import (
    HelixBenchmark,
    BenchmarkResult,
    MethodResult,
    DMTestResult,
    StatisticalComparisonTests,
    SpeedBenchmark,
    OperatorSpeedResult,
    PipelineSpeedResult,
)

try:  # pragma: no cover - optional in trimmed checkouts
    from factorminer.benchmark.ablation import (
        AblationStudy,
        AblationResult,
        AblatedMethodRunner,
        ABLATION_CONFIGS,
        ABLATION_LABELS,
        run_full_ablation_study,
    )
except Exception:  # pragma: no cover - optional in trimmed checkouts
    AblationStudy = None
    AblationResult = None
    AblatedMethodRunner = None
    ABLATION_CONFIGS = None
    ABLATION_LABELS = None
    run_full_ablation_study = None

__all__ = [
    # legacy runtime benchmark
    "BenchmarkManifest",
    "build_benchmark_library",
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
    # helix benchmark
    "HelixBenchmark",
    "BenchmarkResult",
    "MethodResult",
    "DMTestResult",
    "StatisticalComparisonTests",
    "SpeedBenchmark",
    "OperatorSpeedResult",
    "PipelineSpeedResult",
    # ablation
    "AblationStudy",
    "AblationResult",
    "AblatedMethodRunner",
    "ABLATION_CONFIGS",
    "ABLATION_LABELS",
    "run_full_ablation_study",
]
