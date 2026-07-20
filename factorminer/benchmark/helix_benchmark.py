"""Deprecated import shim for the canonical benchmark runtime.

All benchmark execution, statistical comparisons, and report contracts now
live in :mod:`factorminer.benchmark.runtime`. This module intentionally holds
no benchmark implementation; it exists only so older imports continue to
resolve during the deprecation window.
"""

from __future__ import annotations

import warnings

from factorminer.benchmark.runtime import (
    _LEGACY_RUNTIME_MESSAGE,
    AblationResult,
    BenchmarkResult,
    DMTestResult,
    HelixBenchmark,
    MethodResult,
    OperatorSpeedResult,
    PipelineSpeedResult,
    SpeedBenchmark,
    StatisticalComparisonTests,
    _build_mock_data_dict,
    _json_safe,
    run_phase2_ablation_study,
    run_phase2_comparison,
)

warnings.warn(_LEGACY_RUNTIME_MESSAGE, DeprecationWarning, stacklevel=2)

__all__ = [
    "_build_mock_data_dict",
    "_json_safe",
    "AblationResult",
    "BenchmarkResult",
    "DMTestResult",
    "HelixBenchmark",
    "MethodResult",
    "OperatorSpeedResult",
    "PipelineSpeedResult",
    "SpeedBenchmark",
    "StatisticalComparisonTests",
    "run_phase2_ablation_study",
    "run_phase2_comparison",
]
