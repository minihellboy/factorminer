"""Mining budget and candidate-evaluation contracts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Budget Tracker
# ---------------------------------------------------------------------------


@dataclass
class BudgetTracker:
    """Tracks resource consumption across the mining session.

    Monitors LLM token usage, GPU compute time, and wall-clock time
    so the loop can stop early when a budget is exhausted.
    """

    max_llm_calls: int = 0  # 0 = unlimited
    max_wall_seconds: float = 0  # 0 = unlimited

    # Running totals
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    compute_seconds: float = 0.0
    wall_start: float = field(default_factory=time.time)

    def record_llm_call(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.llm_calls += 1
        self.llm_prompt_tokens += prompt_tokens
        self.llm_completion_tokens += completion_tokens

    def record_compute(self, seconds: float) -> None:
        self.compute_seconds += seconds

    @property
    def wall_elapsed(self) -> float:
        return time.time() - self.wall_start

    @property
    def total_tokens(self) -> int:
        return self.llm_prompt_tokens + self.llm_completion_tokens

    def is_exhausted(self) -> bool:
        """True if any budget limit has been reached."""
        if self.max_llm_calls > 0 and self.llm_calls >= self.max_llm_calls:
            return True
        if self.max_wall_seconds > 0 and self.wall_elapsed >= self.max_wall_seconds:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "total_tokens": self.total_tokens,
            "compute_seconds": round(self.compute_seconds, 2),
            "wall_elapsed_seconds": round(self.wall_elapsed, 2),
        }


# ---------------------------------------------------------------------------
# Candidate evaluation result
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Result of evaluating a single candidate factor."""

    factor_name: str
    formula: str
    parse_ok: bool = False
    ic_mean: float = 0.0
    ic_paper_mean: float = 0.0
    ic_abs_mean: float = 0.0
    icir: float = 0.0
    ic_paper_icir: float = 0.0
    ic_win_rate: float = 0.0
    max_correlation: float = 0.0
    correlated_with: str = ""
    admitted: bool = False
    replaced: int | None = None  # ID of replaced factor, if any
    rejection_reason: str = ""
    stage_passed: int = 0  # 0=parse/IC fail, 1=IC pass, 2=corr pass, 3=dedup pass, 4=admitted
    signals: np.ndarray | None = None
    target_stats: dict[str, dict] = field(default_factory=dict)
    research_score: float = 0.0
    research_lcb: float = 0.0
    residual_ic: float = 0.0
    projection_loss: float = 0.0
    effective_rank_gain: float = 0.0
    score_vector: dict[str, Any] | None = None
    # Parent-formula lineage for EditAwareMemoryPolicy + MRM developmental history.
    parent_formula: str = ""
    parent_ic_paper_mean: float | None = None
    edit_type: str = ""
    edit_motif: str = ""
    secondary_parent_formula: str = ""


# ---------------------------------------------------------------------------
