"""Canonical benchmark API without eager runtime initialization."""

from __future__ import annotations

from factorminer._lazy_exports import public_dir, resolve_export

_RUNTIME_MODULE = "factorminer.benchmark.runtime"
_CANONICAL_EXPORTS = {
    name: _RUNTIME_MODULE
    for name in (
        "BenchmarkManifest",
        "BenchmarkRuntimeContract",
        "StrategyGridBenchmarkContract",
        "StressBenchmarkContract",
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
        "run_phase2_ablation_study",
        "run_phase2_comparison",
        "run_table1_benchmark",
        "select_frozen_top_k",
    )
}
_STATISTICS_EXPORTS = {
    name: "factorminer.benchmark.statistics"
    for name in (
        "BenchmarkResult",
        "MethodResult",
        "DMTestResult",
        "StatisticalComparisonTests",
        "OperatorSpeedResult",
        "PipelineSpeedResult",
    )
}
_SPEED_EXPORTS = {
    "SpeedBenchmark": "factorminer.benchmark.speed",
}
_RUNNER_EXPORTS = {
    name: "factorminer.benchmark.runners"
    for name in (
        "AblationStudy",
        "AblationResult",
        "AblatedMethodRunner",
        "ABLATION_CONFIGS",
        "ABLATION_LABELS",
        "run_full_ablation_study",
    )
}
_RECEIPT_EXPORTS = {
    name: "factorminer.benchmark.receipt"
    for name in (
        "build_research_receipt",
        "generate_commitment_nonce",
        "seal_private_commitment",
        "write_receipt",
        "verify_research_receipt",
        "ReceiptVerificationResult",
    )
}
_EXPORTS = {
    **_CANONICAL_EXPORTS,
    **_STATISTICS_EXPORTS,
    **_SPEED_EXPORTS,
    **_RUNNER_EXPORTS,
    **_RECEIPT_EXPORTS,
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), name, _EXPORTS)


def __dir__() -> list[str]:
    return public_dir(globals(), _EXPORTS)
