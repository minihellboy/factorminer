"""Pluggable loop stages for Ralph and Helix iteration execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IterationPayload:
    """Mutable state passed through each loop stage."""

    iteration: int
    batch_size: int
    library_state: dict[str, Any] = field(default_factory=dict)
    memory_signal: dict[str, Any] = field(default_factory=dict)
    prompt_context: dict[str, Any] = field(default_factory=dict)
    candidates: list[tuple[str, str]] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)
    admitted_results: list[Any] = field(default_factory=list)
    stage_metrics: dict[str, Any] = field(default_factory=dict)


class LoopStage(ABC):
    """Abstract loop stage."""

    name: str

    @abstractmethod
    def run(self, loop: Any, payload: IterationPayload) -> None:
        raise NotImplementedError


class RetrieveStage(LoopStage):
    """Retrieve memory priors for the next generation step."""

    name = "retrieve"

    def __init__(self, retrieve_fn: Callable[[Any, IterationPayload], dict[str, Any]]) -> None:
        self._retrieve_fn = retrieve_fn

    def run(self, loop: Any, payload: IterationPayload) -> None:
        payload.library_state = loop.library.get_state_summary()
        payload.memory_signal = self._retrieve_fn(loop, payload)


class GenerateStage(LoopStage):
    """Generate candidate formulas from prompt context."""

    name = "generate"

    def __init__(
        self, generate_fn: Callable[[Any, IterationPayload], list[tuple[str, str]]]
    ) -> None:
        self._generate_fn = generate_fn

    def run(self, loop: Any, payload: IterationPayload) -> None:
        payload.candidates = self._generate_fn(loop, payload)


class EvaluateStage(LoopStage):
    """Evaluate candidates under the active admission protocol."""

    name = "evaluate"

    def __init__(self, evaluate_fn: Callable[[Any, IterationPayload], list[Any]]) -> None:
        self._evaluate_fn = evaluate_fn

    def run(self, loop: Any, payload: IterationPayload) -> None:
        payload.results = self._evaluate_fn(loop, payload)


class LibraryUpdateStage(LoopStage):
    """Apply admissions and replacements to the library."""

    name = "library_update"

    def __init__(self, update_fn: Callable[[Any, IterationPayload], list[Any]]) -> None:
        self._update_fn = update_fn

    def run(self, loop: Any, payload: IterationPayload) -> None:
        payload.admitted_results = self._update_fn(loop, payload)


class DistillStage(LoopStage):
    """Distill evaluated trajectories back into memory."""

    name = "distill"

    def __init__(self, distill_fn: Callable[[Any, IterationPayload], None]) -> None:
        self._distill_fn = distill_fn

    def run(self, loop: Any, payload: IterationPayload) -> None:
        self._distill_fn(loop, payload)
