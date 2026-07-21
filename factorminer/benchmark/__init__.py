"""Canonical benchmark API without eager runtime initialization."""

from __future__ import annotations

from factorminer._lazy_exports import public_dir, resolve_export

_RUNTIME_MODULE = "factorminer.benchmark.runtime"
_RUNTIME_EXPORTS = {
    name: _RUNTIME_MODULE
    for name in (
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
    )
}
_CONTRACT_EXPORTS = {
    name: "factorminer.benchmark.contracts"
    for name in (
        "BenchmarkManifest",
        "BenchmarkRuntimeContract",
        "StrategyGridBenchmarkContract",
        "StressBenchmarkContract",
        "WalkForwardBenchmarkContract",
    )
}
_RUNTIME_CONTRACT_EXPORTS = {
    "build_benchmark_runtime_contract": "factorminer.benchmark.runtime_contracts",
}
_DATASET_EXPORTS = {
    name: "factorminer.benchmark.datasets"
    for name in (
        "build_benchmark_library",
        "load_benchmark_dataset",
    )
}
_FROZEN_EVALUATION_EXPORTS = {
    name: "factorminer.benchmark.frozen_evaluation"
    for name in (
        "evaluate_frozen_set",
        "select_frozen_top_k",
    )
}
_STATISTICS_EXPORTS = {
    name: "factorminer.benchmark.statistics"
    for name in (
        "AblationResult",
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
        "publish_portable_bundle",
        "seal_private_commitment",
        "write_receipt",
        "verify_research_receipt",
        "ReceiptVerificationResult",
    )
}
_EXPORTS = {
    **_RUNTIME_EXPORTS,
    **_CONTRACT_EXPORTS,
    **_RUNTIME_CONTRACT_EXPORTS,
    **_DATASET_EXPORTS,
    **_FROZEN_EVALUATION_EXPORTS,
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
