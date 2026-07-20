"""Agora-style sealed multi-evaluator joint search (research mode).

Source: arXiv:2606.29194 (Agora / Sealed Joint Search, Panda AI, Jun 2026).

Why this exists
---------------
A *single fixed* evaluation objective is gameable. An LLM-guided (or any
search) process can overfit / Goodhart whatever one scorer rewards — a real
reward-hacking and adversarial-robustness concern, not just a research
nicety. Agora's fix is to:

1. run **multiple differently-biased evaluators** simultaneously,
2. keep their internals **sealed** from the generation process, and
3. **track disagreement** rather than collapsing early to a single vote.

How this complements ``EvaluationKernel``
-----------------------------------------
``EvaluationKernel`` remains the canonical single-objective evaluation engine
used by mining loops (IC stats, library geometry admission, dedup). This
module is an **opt-in research mode alongside it**, never a replacement and
never on the default admission path. Wire it only when you explicitly want a
multi-evaluator agreement gate on top of (or after) kernel evaluation.

Paper caveat (do not oversell)
------------------------------
The source paper reports a strong single-seed sealed holdout Sharpe (1.87 vs
1.334 best baseline) but flags a **−0.755 cross-seed mean** as a real caveat.
Treat this as a research mode with genuine single-seed variance risk, not a
proven default-strength upgrade.

Sealed provenance boundary
--------------------------
Generation / ``PromptContextBuilder`` must only ever see
:class:`SealedFeedback` — coarse agreement signals (passed N of M, rank,
disagreement rate). Raw per-evaluator scores, weights, component breakdowns,
and formulas live in the private sibling module
``factorminer.architecture._sealed_evaluator_panel`` and are **not**
re-exported here. Do not import that private module from generation code.

Security considerations
-----------------------
- Anti-gaming: sealing evaluator internals is the structural control against
  the generator directly optimizing one known score formula.
- LLM-as-judge (optional): uses the existing provider abstraction; ``--mock``
  and missing providers fall back to a deterministic blend. LLM text is
  treated as generated output only — never as new system instructions if
  later surfaced in a prompt.
- No network surface is introduced by this module itself.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

# Metric snapshot type + array helper are safe to re-export (no weights).
# Concrete evaluators, score objects, and weight vectors stay private —
# imported under underscored aliases so ``from sealed_joint_search import *``
# and casual ``hasattr`` inspection cannot reach them.
from factorminer.architecture._sealed_evaluator_panel import (
    CandidateObservation,
    observation_from_arrays,
)
from factorminer.architecture._sealed_evaluator_panel import (
    Evaluator as _Evaluator,
)
from factorminer.architecture._sealed_evaluator_panel import (
    EvaluatorScore as _EvaluatorScore,
)
from factorminer.architecture._sealed_evaluator_panel import (
    build_default_evaluators as _build_default_evaluators,
)
from factorminer.architecture._sealed_evaluator_panel import (
    run_panel as _run_panel,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = [
    "AgreementRule",
    "CandidateObservation",
    "SealedFeedback",
    "SealedJointSearchConfig",
    "SealedPromotionDecision",
    "SealedJointSearchEngine",
    "SealedSearchReport",
    "build_sealed_engine",
    "observation_from_arrays",
    "RESEARCH_MODE_CAVEAT",
]


RESEARCH_MODE_CAVEAT = (
    "Sealed joint search is an opt-in research mode (Agora, arXiv:2606.29194). "
    "It does not replace EvaluationKernel default admission. The source paper "
    "flags material single-seed variance (cross-seed mean Sharpe −0.755) — "
    "do not treat sealed promotion as a proven default-strength upgrade."
)


class AgreementRule(str, Enum):
    """Multi-evaluator agreement rule for promotion eligibility."""

    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    ALL_BUT_ONE = "all_but_one"
    THRESHOLD = "threshold"  # need >= min_agree absolute passes


@dataclass(frozen=True)
class SealedJointSearchConfig:
    """Feature-local config for sealed multi-evaluator search.

    Frozen dataclass pattern matches ``SignificanceConfig`` / ``CapacityConfig``.
    Defaults keep the mode conservative and opt-in.
    """

    enabled: bool = False  # opt-in; never on by default
    agreement_rule: AgreementRule = AgreementRule.MAJORITY
    min_agree: int = 2  # used when rule is THRESHOLD; also floor for majority
    include_llm_judge: bool = True
    # When True, research logs may retain internal scores; prompt feedback never does.
    retain_internal_scores: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.min_agree < 1:
            raise ValueError("min_agree must be >= 1")


@dataclass(frozen=True)
class SealedFeedback:
    """Coarse, generation-safe feedback for one candidate.

    This is the **only** structure eligible to flow into
    ``PromptContextBuilder`` / future prompts. It deliberately omits:
    - per-evaluator numeric scores
    - weight vectors / component breakdowns
    - pass thresholds
    - raw rationale strings that might leak formula weights

    On ``agreement_fraction`` naming
    --------------------------------
    ``agreement_fraction`` is ``n_passed / n_evaluators`` -- a **pass
    rate**, not a chance-corrected inter-rater reliability statistic
    (e.g. Fleiss' kappa) or a rank-concordance statistic (e.g. Kendall's
    W). This is a deliberate choice, not an unexamined shortcut:

    1. The source paper (Agora, arXiv:2606.29194) does not use either.
       Its actual disagreement mechanism is report-based -- evaluators
       "carry forward disagreement instead of voting" into the aggregated
       brief -- a qualitative signal, not a scalar reduction. Pass-rate
       plus the separate ``disagreement`` flag (0 < n_passed < n) is this
       module's simplified, numeric analogue of that same idea: it keeps
       "was there a split panel" visible instead of erasing it into a
       single vote, without claiming to reproduce Agora's full
       report-aggregation architecture.
    2. Pass-rate is the right tool for what this field is actually FOR: a
       promotion gate ("how many of N differently-biased judges say GO").
       Chance-corrected or rank-based statistics answer a different
       question ("do raters generally agree with each other") and would
       not improve this specific gating decision. Fleiss' kappa is also
       known to be unstable at small N (this panel is typically N=3):
       it can swing paradoxically even under high observed agreement,
       and gets distorted by class-imbalanced pass/fail prevalence.
    """

    candidate_name: str
    n_evaluators: int
    n_passed: int
    agreement_fraction: float
    promoted: bool
    agreement_rule: str
    # Rank among the batch by coarse agreement (1 = most agreed-pass). 0 if unknown.
    batch_rank: int = 0
    # Coarse persona labels that passed, e.g. ("ic", "novelty") — ids only, no scores.
    passed_personas: tuple[str, ...] = ()
    failed_personas: tuple[str, ...] = ()
    # Free-text coarse summary safe for prompts (no numbers from internal scores).
    summary: str = ""

    def to_prompt_dict(self) -> dict[str, Any]:
        """Dict safe to merge into prompt context extras.

        Structural guarantee: keys are a fixed allow-list with no score/weight
        fields. Tests assert the absence of internal keys.
        """
        return {
            "candidate_name": self.candidate_name,
            "n_evaluators": int(self.n_evaluators),
            "n_passed": int(self.n_passed),
            "agreement_fraction": float(self.agreement_fraction),
            "promoted": bool(self.promoted),
            "agreement_rule": str(self.agreement_rule),
            "batch_rank": int(self.batch_rank),
            "passed_personas": list(self.passed_personas),
            "failed_personas": list(self.failed_personas),
            "summary": self.summary,
            "sealed": True,
            "research_mode_caveat": RESEARCH_MODE_CAVEAT,
        }


# Keys that must NEVER appear in prompt-facing sealed feedback.
_FORBIDDEN_PROMPT_KEYS = frozenset(
    {
        "score",
        "scores",
        "components",
        "threshold",
        "thresholds",
        "weights",
        "weight",
        "ic_weight",
        "novelty_weight",
        "intervention_weight",
        "raw_scores",
        "evaluator_scores",
        "internal",
        "rationale",
        "rationales",
        "pass_threshold",
    }
)


def assert_feedback_is_sealed(payload: MappingLike) -> None:
    """Raise ``AssertionError`` if a prompt payload leaks evaluator internals."""
    if hasattr(payload, "to_prompt_dict"):
        data = payload.to_prompt_dict()
    else:
        data = dict(payload)
    bad = sorted(k for k in data if k.lower() in _FORBIDDEN_PROMPT_KEYS or _looks_internal(k))
    if bad:
        raise AssertionError(
            f"Sealed feedback leaked internal evaluator keys: {bad}. "
            "Only coarse agreement signals may reach PromptContextBuilder."
        )
    # Nested dicts / lists of dicts
    for value in data.values():
        if isinstance(value, dict):
            assert_feedback_is_sealed(value)


def _looks_internal(key: str) -> bool:
    kl = key.lower()
    needles = (
        "weight",
        "component",
        "threshold",
        "raw_score",
        "evaluator_score",
        "score_vector",
        "ic_term",
        "pass_threshold",
    )
    return any(n in kl for n in needles)


# Minimal protocol for mapping-like without importing Mapping everywhere in sigs
MappingLike = Any


@dataclass
class SealedPromotionDecision:
    """Full research-side decision for one candidate (may retain internals).

    ``agreement_fraction`` here is the same pass-rate defined on
    :class:`SealedFeedback` -- see that class's docstring for why it is a
    pass rate rather than a chance-corrected inter-rater statistic.
    """

    observation: CandidateObservation
    promoted: bool
    n_passed: int
    n_evaluators: int
    agreement_fraction: float
    agreement_rule: str
    disagreement: bool
    # Internal scores retained only when config.retain_internal_scores is True.
    internal_scores: tuple[_EvaluatorScore, ...] = ()
    feedback: SealedFeedback = field(default=None)  # type: ignore[assignment]
    batch_rank: int = 0

    def to_public_dict(self) -> dict[str, Any]:
        """Public diagnostic dict — still strips raw component weights."""
        base = self.feedback.to_prompt_dict() if self.feedback is not None else {}
        base.update(
            {
                "disagreement": bool(self.disagreement),
                "formula": self.observation.formula,
            }
        )
        return base

    def to_research_dict(self) -> dict[str, Any]:
        """Research log including per-evaluator pass bits (still no weights)."""
        public = self.to_public_dict()
        if self.internal_scores:
            public["evaluator_pass_bits"] = {
                s.evaluator_id: bool(s.passed) for s in self.internal_scores
            }
            public["evaluator_score_summaries"] = {
                s.evaluator_id: {
                    "passed": bool(s.passed),
                    "score": float(s.score),
                    "persona": s.persona,
                }
                for s in self.internal_scores
            }
        return public


@dataclass
class SealedSearchReport:
    """Batch-level report with disagreement diagnostics."""

    decisions: list[SealedPromotionDecision]
    n_candidates: int
    n_promoted: int
    n_rejected: int
    disagreement_rate: float
    mean_agreement_fraction: float
    agreement_rule: str
    evaluator_ids: tuple[str, ...]
    caveat: str = RESEARCH_MODE_CAVEAT

    def promoted_names(self) -> list[str]:
        return [d.observation.name for d in self.decisions if d.promoted]

    def rejected_names(self) -> list[str]:
        return [d.observation.name for d in self.decisions if not d.promoted]

    def sealed_feedback_batch(self) -> list[dict[str, Any]]:
        """Prompt-safe batch payload for PromptContextBuilder extras."""
        out = []
        for d in self.decisions:
            if d.feedback is None:
                continue
            payload = d.feedback.to_prompt_dict()
            assert_feedback_is_sealed(payload)
            out.append(payload)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_candidates": self.n_candidates,
            "n_promoted": self.n_promoted,
            "n_rejected": self.n_rejected,
            "disagreement_rate": self.disagreement_rate,
            "mean_agreement_fraction": self.mean_agreement_fraction,
            "agreement_rule": self.agreement_rule,
            "evaluator_ids": list(self.evaluator_ids),
            "promoted": self.promoted_names(),
            "rejected": self.rejected_names(),
            "decisions": [d.to_public_dict() for d in self.decisions],
            "caveat": self.caveat,
            "complements": (
                "Opt-in research mode alongside EvaluationKernel; "
                "does not replace default single-objective admission."
            ),
        }


def _required_passes(n_evaluators: int, config: SealedJointSearchConfig) -> int:
    rule = config.agreement_rule
    if rule is AgreementRule.UNANIMOUS:
        return n_evaluators
    if rule is AgreementRule.ALL_BUT_ONE:
        return max(1, n_evaluators - 1)
    if rule is AgreementRule.THRESHOLD:
        return min(n_evaluators, max(1, config.min_agree))
    # majority
    majority = (n_evaluators // 2) + 1
    return max(majority, min(config.min_agree, n_evaluators))


def _coarse_summary(
    name: str,
    n_passed: int,
    n_evaluators: int,
    promoted: bool,
    passed_personas: Sequence[str],
    failed_personas: Sequence[str],
) -> str:
    status = "promoted" if promoted else "held"
    passed = ", ".join(passed_personas) if passed_personas else "none"
    failed = ", ".join(failed_personas) if failed_personas else "none"
    return (
        f"{name}: passed {n_passed} of {n_evaluators} sealed evaluators "
        f"({status}). personas_passed=[{passed}] personas_failed=[{failed}]"
    )


class SealedJointSearchEngine:
    """Run sealed multi-evaluator promotion over a candidate batch.

    Opt-in only. Does not mutate any library or touch ``EvaluationKernel``
    default admission paths.
    """

    def __init__(
        self,
        config: SealedJointSearchConfig | None = None,
        *,
        llm_provider: Any | None = None,
        evaluators: Sequence[_Evaluator] | None = None,
    ) -> None:
        self.config = config or SealedJointSearchConfig(enabled=True)
        if evaluators is not None:
            self._evaluators = list(evaluators)
        else:
            self._evaluators = _build_default_evaluators(
                llm_provider=llm_provider,
                include_llm_judge=self.config.include_llm_judge,
            )
        if not self._evaluators:
            raise ValueError("SealedJointSearchEngine requires at least one evaluator")

    @property
    def evaluator_ids(self) -> tuple[str, ...]:
        return tuple(getattr(e, "evaluator_id", e.__class__.__name__) for e in self._evaluators)

    @property
    def evaluator_personas(self) -> tuple[str, ...]:
        return tuple(getattr(e, "persona", "unknown") for e in self._evaluators)

    def evaluate_one(self, observation: CandidateObservation) -> SealedPromotionDecision:
        """Score one candidate under the sealed panel and apply agreement rule."""
        scores = _run_panel(observation, self._evaluators)
        n_eval = len(scores)
        n_passed = sum(1 for s in scores if s.passed)
        need = _required_passes(n_eval, self.config)
        promoted = n_passed >= need
        agreement_fraction = float(n_passed) / float(n_eval) if n_eval else 0.0
        # Disagreement = not unanimous (paper: collapsing to a vote too early
        # was the failure mode; we surface split panels as a diagnostic).
        disagreement = not (n_passed == 0 or n_passed == n_eval)

        passed_personas = tuple(s.persona for s in scores if s.passed)
        failed_personas = tuple(s.persona for s in scores if not s.passed)
        feedback = SealedFeedback(
            candidate_name=observation.name,
            n_evaluators=n_eval,
            n_passed=n_passed,
            agreement_fraction=agreement_fraction,
            promoted=promoted,
            agreement_rule=self.config.agreement_rule.value,
            batch_rank=0,
            passed_personas=passed_personas,
            failed_personas=failed_personas,
            summary=_coarse_summary(
                observation.name,
                n_passed,
                n_eval,
                promoted,
                passed_personas,
                failed_personas,
            ),
        )
        assert_feedback_is_sealed(feedback)

        internal: tuple[_EvaluatorScore, ...] = (
            tuple(scores) if self.config.retain_internal_scores else ()
        )
        return SealedPromotionDecision(
            observation=observation,
            promoted=promoted,
            n_passed=n_passed,
            n_evaluators=n_eval,
            agreement_fraction=agreement_fraction,
            agreement_rule=self.config.agreement_rule.value,
            disagreement=disagreement,
            internal_scores=internal,
            feedback=feedback,
        )

    def evaluate_batch(
        self,
        observations: Sequence[CandidateObservation],
    ) -> SealedSearchReport:
        """Evaluate a batch, assign coarse ranks, and compute disagreement rate."""
        decisions = [self.evaluate_one(obs) for obs in observations]
        # Rank by (promoted desc, agreement_fraction desc, n_passed desc, name)
        order = sorted(
            range(len(decisions)),
            key=lambda i: (
                not decisions[i].promoted,
                -decisions[i].agreement_fraction,
                -decisions[i].n_passed,
                decisions[i].observation.name,
            ),
        )
        for rank, idx in enumerate(order, start=1):
            d = decisions[idx]
            d.batch_rank = rank
            if d.feedback is not None:
                d.feedback = SealedFeedback(
                    candidate_name=d.feedback.candidate_name,
                    n_evaluators=d.feedback.n_evaluators,
                    n_passed=d.feedback.n_passed,
                    agreement_fraction=d.feedback.agreement_fraction,
                    promoted=d.feedback.promoted,
                    agreement_rule=d.feedback.agreement_rule,
                    batch_rank=rank,
                    passed_personas=d.feedback.passed_personas,
                    failed_personas=d.feedback.failed_personas,
                    summary=d.feedback.summary,
                )

        n = len(decisions)
        n_promoted = sum(1 for d in decisions if d.promoted)
        n_disagree = sum(1 for d in decisions if d.disagreement)
        mean_agree = (
            float(np.mean([d.agreement_fraction for d in decisions])) if decisions else 0.0
        )
        report = SealedSearchReport(
            decisions=decisions,
            n_candidates=n,
            n_promoted=n_promoted,
            n_rejected=n - n_promoted,
            disagreement_rate=float(n_disagree) / float(n) if n else 0.0,
            mean_agreement_fraction=mean_agree,
            agreement_rule=self.config.agreement_rule.value,
            evaluator_ids=self.evaluator_ids,
        )
        logger.info(
            "Sealed joint search: %d/%d promoted, disagreement_rate=%.2f (%s)",
            n_promoted,
            n,
            report.disagreement_rate,
            self.config.agreement_rule.value,
        )
        return report

    def prompt_context_extras(self, report: SealedSearchReport) -> dict[str, Any]:
        """Build extras dict safe to pass into ``PromptContextBuilder.build``.

        Contains only coarse sealed feedback + the research-mode caveat.
        """
        payload = {
            "sealed_joint_search": {
                "enabled": True,
                "agreement_rule": report.agreement_rule,
                "disagreement_rate": report.disagreement_rate,
                "n_promoted": report.n_promoted,
                "n_candidates": report.n_candidates,
                "candidates": report.sealed_feedback_batch(),
                "caveat": RESEARCH_MODE_CAVEAT,
            }
        }
        # Defense in depth: walk leaves for forbidden keys.
        _assert_tree_sealed(payload)
        return payload


def _assert_tree_sealed(obj: Any, *, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in _FORBIDDEN_PROMPT_KEYS or _looks_internal(str(k)):
                raise AssertionError(f"Sealed prompt tree leaked key at {path}.{k}")
            _assert_tree_sealed(v, path=f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_tree_sealed(v, path=f"{path}[{i}]")


def build_sealed_engine(
    config: SealedJointSearchConfig | None = None,
    *,
    llm_provider: Any | None = None,
) -> SealedJointSearchEngine:
    """Factory matching other architecture builders."""
    return SealedJointSearchEngine(config=config, llm_provider=llm_provider)


# Re-export observation helper under the public name for callers who must not
# touch the private panel module. CandidateObservation is the metric snapshot
# (no weights) so exposing the type is safe; concrete Evaluator classes are not.
