"""Focused tests for shared loop orchestration helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from factorminer.architecture import IterationPayload
from factorminer.core.loop_services import LoopExecutionService


class _DummyStage:
    def __init__(self, fn):
        self._fn = fn

    def run(self, loop, payload):
        self._fn(loop, payload)


class _DummyReporter:
    def __init__(self) -> None:
        self.logged_batches = []

    def log_batch(self, **stats):
        self.logged_batches.append(stats)


class _DummySessionLogger:
    def __init__(self) -> None:
        self.iterations = []
        self.factors = []

    def log_iteration(self, record):
        self.iterations.append(record)

    def log_factor(self, record):
        self.factors.append(record)


class _DummyLoop:
    def __init__(self) -> None:
        self.iteration = 7
        self.reporter = _DummyReporter()
        self._session_logger = _DummySessionLogger()
        self.stages = {}
        self.library = SimpleNamespace(size=0)
        self.config = SimpleNamespace(
            ic_threshold=0.04,
            icir_threshold=0.5,
            correlation_threshold=0.5,
        )
        self.data_tensor = np.zeros((3, 8, 2))
        self._session = None

    def _compute_stats(self, results, admitted, elapsed):
        return {
            "iteration": self.iteration,
            "ic_passed": 1,
            "corr_passed": 1,
            "admitted": len(admitted),
            "replaced": 0,
            "library_size": 3,
            "elapsed_seconds": elapsed,
        }

    def _empty_stats(self):
        return {"iteration": self.iteration, "library_size": 3}


def test_stage_chain_executes_in_order() -> None:
    loop = _DummyLoop()
    service = LoopExecutionService(loop)
    payload = service.new_payload(batch_size=5)
    trace = []

    def retrieve(_loop, current):
        trace.append("retrieve")
        current.memory_signal = {"signal": 1}

    def generate(_loop, current):
        trace.append("generate")
        current.candidates = [("f1", "Neg($close)")]

    def evaluate(_loop, current):
        trace.append("evaluate")
        current.results = [SimpleNamespace(parse_ok=True)]

    def update(_loop, current):
        trace.append("library_update")
        current.admitted_results = [SimpleNamespace(admitted=True)]

    def distill(_loop, current):
        trace.append("distill")
        current.stage_metrics["distilled"] = True

    loop.stages = {
        "retrieve": _DummyStage(retrieve),
        "generate": _DummyStage(generate),
        "evaluate": _DummyStage(evaluate),
        "library_update": _DummyStage(update),
        "distill": _DummyStage(distill),
    }

    service.run_stage_chain(payload, ("retrieve", "generate", "evaluate", "library_update", "distill"))

    assert trace == ["retrieve", "generate", "evaluate", "library_update", "distill"]
    assert payload.memory_signal == {"signal": 1}
    assert payload.candidates == [("f1", "Neg($close)")]
    assert len(payload.admitted_results) == 1
    assert payload.admitted_results[0].admitted is True
    assert payload.stage_metrics["distilled"] is True


def test_empty_generation_reason_uses_canonicalization_hint() -> None:
    loop = _DummyLoop()
    service = LoopExecutionService(loop)
    payload = IterationPayload(iteration=1, batch_size=4)
    payload.stage_metrics["candidates_before_canon"] = 3

    assert service.candidate_count(payload) == 3
    assert service.describe_empty_generation(payload) == "all candidates removed by canonicalization"


def test_log_telemetry_emits_iteration_and_factor_records() -> None:
    loop = _DummyLoop()
    service = LoopExecutionService(loop)
    payload = IterationPayload(iteration=7, batch_size=4)
    payload.results = [
        SimpleNamespace(
            formula="Neg($close)",
            ic_mean=0.12,
            icir=1.2,
            max_correlation=0.3,
            parse_ok=True,
            admitted=True,
            rejection_reason="",
            replaced=None,
        ),
        SimpleNamespace(
            formula="BadFormula",
            ic_mean=0.0,
            icir=0.0,
            max_correlation=0.0,
            parse_ok=False,
            admitted=False,
            rejection_reason="Parse failure",
            replaced=None,
        ),
    ]

    stats = {
        "iteration": 7,
        "ic_passed": 1,
        "corr_passed": 1,
        "admitted": 1,
        "replaced": 0,
        "library_size": 3,
    }
    telemetry = service.build_telemetry(payload, stats, elapsed=1.5, candidates_generated=2)
    service.log_telemetry(telemetry)

    assert loop.reporter.logged_batches == [stats]
    assert len(loop._session_logger.iterations) == 1
    assert loop._session_logger.iterations[0].candidates_generated == 2
    assert loop._session_logger.iterations[0].best_ic == 0.12
    assert len(loop._session_logger.factors) == 2


def test_zero_admission_guidance_explains_tiny_strict_runs() -> None:
    loop = _DummyLoop()
    loop._session = SimpleNamespace(
        total_iterations=2,
        get_summary=lambda: {"total_candidates": 12, "total_admitted": 0},
    )
    service = LoopExecutionService(loop)

    warning = service.zero_admission_guidance(target_size=3, max_iterations=2)

    assert warning is not None
    assert "No factors were admitted" in warning
    assert "3 assets x 8 periods" in warning
    assert "ic=0.04" in warning
