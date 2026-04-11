"""Multi-agent debate orchestrator for factor generation (FactorMAD).

``DebateGenerator`` is a **drop-in replacement** for ``FactorGenerator``.
It runs multiple domain-specialist generators, collects their proposals,
passes them through a multi-dimensional ``CriticAgent`` for pre-filtering,
and returns a single ``List[CandidateFactor]`` with the same interface as
``FactorGenerator.generate_batch()``.

The full pipeline (``DebateOrchestrator``) also supports:
- SymPy-based algebraic deduplication via ``FormulaCanonicalizer``.
- ``DebateMemory`` tracking: specialist leaderboards, blind spot detection.
- Parallel specialist generation (thread-pool).
- Structured ``DebateResult`` dataclass capturing the full debate state.
"""

from __future__ import annotations

import concurrent.futures
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from factorminer.agent.critic import CriticAgent, CriticScore
from factorminer.agent.factor_generator import FactorGenerator
from factorminer.agent.llm_interface import LLMProvider
from factorminer.agent.output_parser import CandidateFactor
from factorminer.agent.prompt_builder import (
    PromptBuilder,
    normalize_factor_references,
)
from factorminer.agent.specialists import (
    DEFAULT_SPECIALISTS,
    SpecialistAgent,
    SpecialistConfig,
    SpecialistPromptBuilder,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DebateConfig
# ---------------------------------------------------------------------------

@dataclass
class DebateConfig:
    """Configuration for the multi-agent FactorMAD pipeline.

    Attributes
    ----------
    specialists : list[SpecialistConfig]
        Specialist configurations to run.  Defaults to the four
        pre-defined specialists (momentum, volatility, liquidity, regime).
    enable_critic : bool
        Whether to run the CriticAgent for multi-dimensional scoring.
    candidates_per_specialist : int
        Number of candidates each specialist generates per round.
    top_k_after_critic : int
        How many candidates the critic retains after ranking.
    critic_temperature : float
        Sampling temperature for the critic LLM call.
    enable_deduplication : bool
        Whether to use SymPy canonicalization to remove algebraic duplicates.
    enable_debate_memory : bool
        Whether to track debate history across rounds for specialist feedback.
    parallel_specialists : bool
        Whether to run specialists in parallel (thread pool).
    max_parallel_workers : int
        Maximum number of parallel threads for specialist generation.
    """

    specialists: list[SpecialistConfig] = field(
        default_factory=lambda: list(DEFAULT_SPECIALISTS)
    )
    enable_critic: bool = True
    candidates_per_specialist: int = 15
    top_k_after_critic: int = 40
    critic_temperature: float = 0.3
    enable_deduplication: bool = True
    enable_debate_memory: bool = True
    parallel_specialists: bool = True
    max_parallel_workers: int = 4


# ---------------------------------------------------------------------------
# DebateResult
# ---------------------------------------------------------------------------

@dataclass
class DebateResult:
    """Full structured result from one debate round.

    Attributes
    ----------
    all_proposals : list[str]
        Raw formula strings from all specialists.
    after_dedup : list[str]
        Formulas after SymPy algebraic deduplication.
    after_critic : list[str]
        Formulas that passed the critic pre-filter (``keep=True``).
    critic_scores : list[CriticScore]
        Full multi-dimensional scores for all candidates.
    specialist_proposals : dict[str, list[str]]
        Per-specialist formula strings before any filtering.
    specialist_success_rates : dict[str, float]
        Historical admission success rates per specialist.
    debate_stats : dict
        Summary statistics: n_proposals, n_after_dedup, n_after_critic,
        n_duplicates_removed, specialist_counts.
    """

    all_proposals: list[str] = field(default_factory=list)
    after_dedup: list[str] = field(default_factory=list)
    after_critic: list[str] = field(default_factory=list)
    critic_scores: list[CriticScore] = field(default_factory=list)
    specialist_proposals: dict[str, list[str]] = field(default_factory=dict)
    specialist_success_rates: dict[str, float] = field(default_factory=dict)
    debate_stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DebateMemory -- cross-round debate history tracking
# ---------------------------------------------------------------------------

class DebateMemory:
    """Tracks debate history across rounds: who proposed what, what got admitted.

    Used by ``DebateOrchestrator`` to maintain specialist leaderboards,
    identify blind spots (operator families nobody proposes), and surface
    patterns the critic consistently rewards.

    Parameters
    ----------
    specialist_names : list[str]
        Names of all participating specialists.
    """

    _ALL_OP_FAMILIES: list[str] = [
        "arithmetic", "statistical", "timeseries", "smoothing",
        "cross_sectional", "regression", "logical",
    ]

    def __init__(self, specialist_names: list[str]) -> None:
        self._specialist_names = list(specialist_names)
        self._proposal_history: dict[str, list[tuple]] = {
            name: [] for name in specialist_names
        }
        self._rounds: list[dict[str, Any]] = []
        self._best_critic_patterns: list[str] = []

    def record_round(
        self,
        debate_result: DebateResult,
        admissions: list[str] | None = None,
    ) -> None:
        """Record outcome of one debate round.

        Parameters
        ----------
        debate_result : DebateResult
            The result of the debate round.
        admissions : list[str] or None
            Formulas ultimately admitted to the library after IC evaluation.
        """
        admissions = admissions or []
        admission_set = set(admissions)

        for spec_name, formulas in debate_result.specialist_proposals.items():
            for formula in formulas:
                was_admitted = formula in admission_set
                self._proposal_history.setdefault(spec_name, []).append(
                    (formula, was_admitted)
                )

        for score in debate_result.critic_scores:
            if score.composite_score >= 0.7 and score.formula in admission_set:
                self._best_critic_patterns.append(score.formula)

        self._rounds.append({
            "n_proposals": len(debate_result.all_proposals),
            "n_after_dedup": len(debate_result.after_dedup),
            "n_after_critic": len(debate_result.after_critic),
            "n_admissions": len(admissions),
            "specialist_counts": {
                name: len(formulas)
                for name, formulas in debate_result.specialist_proposals.items()
            },
        })

    def get_specialist_leaderboard(self) -> list[dict[str, Any]]:
        """Return specialist performance sorted by admission rate.

        Returns
        -------
        list[dict]
            Each dict has keys: ``name``, ``proposed``, ``admitted``,
            ``admission_rate``.  Sorted by ``admission_rate`` descending.
        """
        rows: list[dict[str, Any]] = []
        for name in self._specialist_names:
            history = self._proposal_history.get(name, [])
            proposed = len(history)
            admitted = sum(1 for _, was_admitted in history if was_admitted)
            rate = admitted / max(proposed, 1)
            rows.append({
                "name": name,
                "proposed": proposed,
                "admitted": admitted,
                "admission_rate": rate,
            })
        rows.sort(key=lambda r: r["admission_rate"], reverse=True)
        return rows

    def get_best_critic_patterns(self) -> list[str]:
        """Return formula patterns the critic loved that were also admitted."""
        return list(self._best_critic_patterns[-20:])

    def get_blind_spots(self) -> dict[str, list[str]]:
        """Detect operator families that no specialist is proposing.

        Returns
        -------
        dict[str, list[str]]
            ``"underused_families"``: operator families with low proposal count.
            ``"overused_families"``: operator families with disproportionate use.
        """
        from factorminer.agent.critic import _OP_CATEGORIES, _extract_operators

        family_counts: Counter = Counter()
        total_proposals = 0

        for history in self._proposal_history.values():
            for formula, _ in history:
                ops = _extract_operators(formula)
                for op in ops:
                    family = _OP_CATEGORIES.get(op, "other")
                    family_counts[family] += 1
                total_proposals += 1

        if total_proposals == 0:
            return {
                "underused_families": self._ALL_OP_FAMILIES,
                "overused_families": [],
            }

        avg_count = total_proposals / len(self._ALL_OP_FAMILIES)
        underused = [
            f for f in self._ALL_OP_FAMILIES
            if family_counts.get(f, 0) < avg_count * 0.4
        ]
        overused = [
            f for f in self._ALL_OP_FAMILIES
            if family_counts.get(f, 0) > avg_count * 2.5
        ]
        return {"underused_families": underused, "overused_families": overused}

    def get_memory_summary_for_specialist(self, specialist_name: str) -> str:
        """Return a brief performance summary for a specific specialist."""
        history = self._proposal_history.get(specialist_name, [])
        if not history:
            return f"{specialist_name}: no history yet."
        proposed = len(history)
        admitted = sum(1 for _, a in history if a)
        rate = admitted / proposed
        return (
            f"{specialist_name}: {proposed} proposed, {admitted} admitted "
            f"({rate:.1%} rate)."
        )

    @property
    def total_rounds(self) -> int:
        return len(self._rounds)


# ---------------------------------------------------------------------------
# DebateOrchestrator -- full pipeline
# ---------------------------------------------------------------------------

class DebateOrchestrator:
    """Orchestrates the full multi-agent FactorMAD debate cycle.

    Flow per round:
    1. All specialists generate proposals (optionally in parallel).
    2. Merge all proposals into a single pool.
    3. SymPy algebraic deduplication (optional).
    4. Critic multi-dimensional pre-scoring.
    5. Top-fraction selection for expensive IC evaluation.
    6. Return structured ``DebateResult``.

    Parameters
    ----------
    specialists : list[SpecialistAgent]
        Specialist agent instances.
    critic : CriticAgent
        Critic agent for pre-filtering.
    canonicalizer : FormulaCanonicalizer or None
        Optional SymPy canonicalizer for algebraic deduplication.
    parallel_specialists : bool
        Whether to run specialists concurrently.
    max_workers : int
        Max thread pool workers when parallel is enabled.
    """

    def __init__(
        self,
        specialists: list[SpecialistAgent],
        critic: CriticAgent,
        canonicalizer: Any | None = None,
        parallel_specialists: bool = True,
        max_workers: int = 4,
    ) -> None:
        self.specialists = specialists
        self.critic = critic
        self.canonicalizer = canonicalizer
        self.parallel_specialists = parallel_specialists
        self.max_workers = max_workers

    def run_debate_round(
        self,
        n_per_specialist: int = 15,
        memory_signal: dict[str, Any] | None = None,
        library_diagnostics: dict[str, Any] | None = None,
        regime_context: str = "",
        forbidden_patterns: list[str] | None = None,
        existing_factors: list[str] | None = None,
    ) -> DebateResult:
        """Run one full debate round and return structured results.

        Parameters
        ----------
        n_per_specialist : int
            Number of proposals to request from each specialist.
        memory_signal : dict or None
            Experience memory priors.
        library_diagnostics : dict or None
            Current library state.
        regime_context : str
            Current market regime description.
        forbidden_patterns : list[str] or None
            Global forbidden structural patterns.
        existing_factors : list[str] or None
            Formulas already in the library.

        Returns
        -------
        DebateResult
            Full structured result including all proposals, dedup, and
            critic scores.
        """
        memory_signal = memory_signal or {}
        library_diagnostics = library_diagnostics or {}
        forbidden_patterns = forbidden_patterns or []
        existing_factors = normalize_factor_references(existing_factors)

        # Step 1: Specialist generation
        if self.parallel_specialists and len(self.specialists) > 1:
            specialist_proposals = self._generate_parallel(
                n_per_specialist=n_per_specialist,
                memory_signal=memory_signal,
                library_diagnostics=library_diagnostics,
                regime_context=regime_context,
                forbidden_patterns=forbidden_patterns,
                existing_factors=existing_factors,
            )
        else:
            specialist_proposals: dict[str, list[str]] = {}
            for spec in self.specialists:
                formulas = spec.generate_proposals(
                    n_proposals=n_per_specialist,
                    memory_signal=memory_signal,
                    library_diagnostics=library_diagnostics,
                    regime_context=regime_context,
                    forbidden_patterns=forbidden_patterns,
                    existing_factors=existing_factors,
                )
                specialist_proposals[spec.name] = formulas
                logger.info(
                    "Specialist %s: %d proposals", spec.name, len(formulas)
                )

        # Step 2: Merge all proposals
        all_proposals: list[str] = []
        formula_to_specialist: dict[str, str] = {}
        for spec_name, formulas in specialist_proposals.items():
            for f in formulas:
                if f not in formula_to_specialist:
                    all_proposals.append(f)
                    formula_to_specialist[f] = spec_name

        logger.info(
            "Debate round: %d total proposals from %d specialists",
            len(all_proposals),
            len(self.specialists),
        )

        # Step 3: SymPy deduplication
        after_dedup = self._deduplicate(all_proposals)
        n_removed = len(all_proposals) - len(after_dedup)
        logger.info(
            "Deduplication: removed %d algebraic duplicates (%d remain)",
            n_removed,
            len(after_dedup),
        )

        # Step 4: Build CandidateFactor proposals for critic
        proposals_cf: dict[str, list[CandidateFactor]] = {}
        for formula in after_dedup:
            spec_name = formula_to_specialist.get(formula, "unknown")
            from factorminer.agent.output_parser import _try_build_candidate
            existing_count = len(proposals_cf.get(spec_name, []))
            cf = _try_build_candidate(
                f"{spec_name.lower()}_factor_{existing_count + 1}",
                formula,
            )
            proposals_cf.setdefault(spec_name, []).append(cf)

        # Step 5: Critic scoring
        mem_str = _flatten_memory_signal(memory_signal)
        critic_scores = self.critic._score_proposals(
            proposals=proposals_cf,
            existing_factors=existing_factors,
            memory_signal=mem_str,
            regime_context=regime_context,
        )

        # Step 6: Collect kept formulas
        after_critic = [cs.formula for cs in critic_scores if cs.keep]
        logger.info(
            "Critic pre-filter: %d/%d candidates kept (keep=True)",
            len(after_critic),
            len(after_dedup),
        )

        success_rates = {spec.name: spec.success_rate for spec in self.specialists}
        debate_stats = {
            "n_proposals": len(all_proposals),
            "n_after_dedup": len(after_dedup),
            "n_after_critic": len(after_critic),
            "n_duplicates_removed": n_removed,
            "specialist_counts": {
                name: len(formulas)
                for name, formulas in specialist_proposals.items()
            },
        }

        return DebateResult(
            all_proposals=all_proposals,
            after_dedup=after_dedup,
            after_critic=after_critic,
            critic_scores=critic_scores,
            specialist_proposals=specialist_proposals,
            specialist_success_rates=success_rates,
            debate_stats=debate_stats,
        )

    def _generate_parallel(
        self,
        n_per_specialist: int,
        memory_signal: dict[str, Any],
        library_diagnostics: dict[str, Any],
        regime_context: str,
        forbidden_patterns: list[str],
        existing_factors: list[str],
    ) -> dict[str, list[str]]:
        """Generate from all specialists concurrently using a thread pool."""
        results: dict[str, list[str]] = {}

        def _run_specialist(spec: SpecialistAgent) -> tuple:
            formulas = spec.generate_proposals(
                n_proposals=n_per_specialist,
                memory_signal=memory_signal,
                library_diagnostics=library_diagnostics,
                regime_context=regime_context,
                forbidden_patterns=forbidden_patterns,
                existing_factors=existing_factors,
            )
            return spec.name, formulas

        n_workers = min(self.max_workers, len(self.specialists))
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_run_specialist, spec): spec.name
                for spec in self.specialists
            }
            for future in concurrent.futures.as_completed(futures):
                spec_name = futures[future]
                try:
                    name, formulas = future.result()
                    results[name] = formulas
                    logger.info(
                        "Specialist %s (parallel): %d proposals",
                        name,
                        len(formulas),
                    )
                except Exception as exc:
                    logger.warning(
                        "Specialist %s parallel generation failed: %s",
                        spec_name,
                        exc,
                    )
                    results[spec_name] = []

        return results

    def _deduplicate(self, formulas: list[str]) -> list[str]:
        """Remove algebraic duplicates using SymPy canonicalizer if available."""
        if self.canonicalizer is None:
            seen: set = set()
            unique: list[str] = []
            for f in formulas:
                if f not in seen:
                    unique.append(f)
                    seen.add(f)
            return unique

        from factorminer.core.parser import try_parse
        seen_hashes: set = set()
        unique: list[str] = []
        for formula in formulas:
            tree = try_parse(formula)
            if tree is None:
                if formula not in {u for u in unique}:
                    unique.append(formula)
                continue
            try:
                canon_hash = self.canonicalizer.canonicalize(tree)
            except Exception:
                canon_hash = formula
            if canon_hash not in seen_hashes:
                unique.append(formula)
                seen_hashes.add(canon_hash)

        return unique


# ---------------------------------------------------------------------------
# DebateGenerator -- drop-in replacement for FactorGenerator
# ---------------------------------------------------------------------------

class DebateGenerator:
    """Multi-agent debate-based factor generator (drop-in for FactorGenerator).

    Uses the full FactorMAD pipeline: multiple specialist proposers,
    algebraic deduplication, and multi-dimensional critic pre-filtering.

    Parameters
    ----------
    llm_provider : LLMProvider
        LLM backend shared across all specialists and the critic.
    debate_config : DebateConfig or None
        Pipeline configuration.  Uses defaults if ``None``.
    prompt_builder : PromptBuilder or None
        Optional base prompt builder (its system prompt is used as the
        base for specialist prompt builders).
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        debate_config: DebateConfig | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.config = debate_config or DebateConfig()

        base_system_prompt = (
            prompt_builder.system_prompt if prompt_builder else None
        )

        # Build SpecialistAgent instances
        self._specialist_agents: list[SpecialistAgent] = []
        self._specialist_generators: dict[str, FactorGenerator] = {}

        for spec in self.config.specialists:
            agent = SpecialistAgent(
                config=spec,
                llm=self.llm_provider,
                base_system_prompt=base_system_prompt,
            )
            self._specialist_agents.append(agent)

            specialist_pb = SpecialistPromptBuilder(
                specialist_config=spec,
                base_system_prompt=base_system_prompt,
            )
            gen = FactorGenerator(
                llm_provider=self.llm_provider,
                prompt_builder=specialist_pb,
                temperature=spec.temperature,
            )
            self._specialist_generators[spec.name] = gen

        # Build critic
        self._critic: CriticAgent | None = None
        if self.config.enable_critic:
            self._critic = CriticAgent(
                llm_provider=self.llm_provider,
                temperature=self.config.critic_temperature,
            )

        # Canonicalizer for deduplication
        self._canonicalizer = None
        if self.config.enable_deduplication:
            try:
                from factorminer.core.canonicalizer import FormulaCanonicalizer
                self._canonicalizer = FormulaCanonicalizer()
            except Exception as exc:
                logger.warning(
                    "Could not initialise FormulaCanonicalizer: %s. "
                    "Falling back to string dedup.",
                    exc,
                )

        # Debate orchestrator
        if self._critic is not None:
            self._orchestrator: DebateOrchestrator | None = DebateOrchestrator(
                specialists=self._specialist_agents,
                critic=self._critic,
                canonicalizer=self._canonicalizer,
                parallel_specialists=self.config.parallel_specialists,
                max_workers=self.config.max_parallel_workers,
            )
        else:
            self._orchestrator = None

        # Debate memory
        self._debate_memory: DebateMemory | None = None
        if self.config.enable_debate_memory:
            specialist_names = [s.name for s in self.config.specialists]
            self._debate_memory = DebateMemory(specialist_names=specialist_names)

        self._last_debate_result: DebateResult | None = None
        self._generation_count = 0

    def generate_batch(
        self,
        memory_signal: dict[str, Any] | None = None,
        library_state: dict[str, Any] | None = None,
        batch_size: int = 40,
    ) -> list[CandidateFactor]:
        """Generate a batch of candidate factors via multi-agent debate.

        Signature is identical to ``FactorGenerator.generate_batch``
        so this class is a true drop-in replacement.

        Parameters
        ----------
        memory_signal : dict or None
            Memory priors for prompt injection.
        library_state : dict or None
            Current factor library state.
        batch_size : int
            Target number of candidates to return.

        Returns
        -------
        list[CandidateFactor]
            Ranked candidate factors.
        """
        memory_signal = memory_signal or {}
        library_state = library_state or {}

        self._generation_count += 1
        batch_id = self._generation_count

        logger.info(
            "Debate batch #%d: %d specialists, critic=%s, per_specialist=%d",
            batch_id,
            len(self._specialist_agents),
            self.config.enable_critic,
            self.config.candidates_per_specialist,
        )

        existing_factors = normalize_factor_references(
            library_state.get("recent_admissions", [])
        )
        regime_context = str(memory_signal.get("regime_context", ""))

        if self._orchestrator is not None:
            debate_result = self._orchestrator.run_debate_round(
                n_per_specialist=self.config.candidates_per_specialist,
                memory_signal=memory_signal,
                library_diagnostics=library_state,
                regime_context=regime_context,
                existing_factors=existing_factors,
            )
            self._last_debate_result = debate_result

            if self._debate_memory is not None:
                self._debate_memory.record_round(debate_result)

            result = self._debate_result_to_candidates(
                debate_result=debate_result,
                top_k=min(batch_size, self.config.top_k_after_critic),
            )

        else:
            # No critic: run specialist generators and merge
            proposals: dict[str, list[CandidateFactor]] = {}
            for spec_name, generator in self._specialist_generators.items():
                candidates = generator.generate_batch(
                    memory_signal=memory_signal,
                    library_state=library_state,
                    batch_size=self.config.candidates_per_specialist,
                )
                proposals[spec_name] = candidates
                logger.info(
                    "Specialist %s produced %d candidates", spec_name, len(candidates)
                )

            result = []
            seen_formulas: set = set()
            for spec_name, candidates in proposals.items():
                for c in candidates:
                    if c.formula not in seen_formulas:
                        result.append(c)
                        seen_formulas.add(c.formula)

            result = result[:batch_size]
            # Store a minimal DebateResult for consistency
            specialist_proposals = {
                name: [c.formula for c in cands]
                for name, cands in proposals.items()
            }
            self._last_debate_result = DebateResult(
                all_proposals=[f for fl in specialist_proposals.values() for f in fl],
                after_dedup=[c.formula for c in result],
                after_critic=[c.formula for c in result],
                critic_scores=[],
                specialist_proposals=specialist_proposals,
                specialist_success_rates={},
                debate_stats={"n_proposals": len(result)},
            )

        result = self._tag_specialist_source_from_agents(result)

        logger.info(
            "Debate batch #%d complete: returning %d candidates",
            batch_id,
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Public inspection helpers
    # ------------------------------------------------------------------

    @property
    def last_debate_result(self) -> DebateResult | None:
        """The ``DebateResult`` from the most recent ``generate_batch`` call."""
        return self._last_debate_result

    @property
    def debate_memory(self) -> DebateMemory | None:
        """The ``DebateMemory`` tracking history across rounds."""
        return self._debate_memory

    def get_specialist_leaderboard(self) -> list[dict[str, Any]] | None:
        """Return specialist admission leaderboard if memory is enabled."""
        if self._debate_memory is not None:
            return self._debate_memory.get_specialist_leaderboard()
        return None

    def get_blind_spots(self) -> dict[str, list[str]] | None:
        """Return operator family blind spots if memory is enabled."""
        if self._debate_memory is not None:
            return self._debate_memory.get_blind_spots()
        return None

    def update_specialist_admissions(
        self,
        admitted_formulas: list[str],
        rejected_formulas: list[str] | None = None,
        rejection_reasons: list[str] | None = None,
    ) -> None:
        """Feed evaluation results back to specialist agents and debate memory.

        Should be called after IC evaluation to close the feedback loop.

        Parameters
        ----------
        admitted_formulas : list[str]
            Formula strings admitted to the library.
        rejected_formulas : list[str] or None
            Formula strings that failed IC evaluation.
        rejection_reasons : list[str] or None
            Reasons for rejection (parallel to ``rejected_formulas``).
        """
        if self._last_debate_result is None:
            return

        rejected_formulas = rejected_formulas or []
        rejection_reasons = rejection_reasons or []

        for spec_agent in self._specialist_agents:
            spec_admitted = [
                f for f in admitted_formulas
                if f in self._last_debate_result.specialist_proposals.get(
                    spec_agent.name, []
                )
            ]
            spec_rejected = [
                f for f in rejected_formulas
                if f in self._last_debate_result.specialist_proposals.get(
                    spec_agent.name, []
                )
            ]
            spec_reasons: list[str] = []
            for f in spec_rejected:
                try:
                    idx = rejected_formulas.index(f)
                    spec_reasons.append(
                        rejection_reasons[idx] if idx < len(rejection_reasons)
                        else "unknown"
                    )
                except ValueError:
                    spec_reasons.append("unknown")

            spec_agent.update_domain_memory(
                admitted=spec_admitted,
                rejected=spec_rejected,
                reasons=spec_reasons,
            )

        if self._debate_memory is not None and self._last_debate_result is not None:
            self._debate_memory.record_round(
                debate_result=self._last_debate_result,
                admissions=admitted_formulas,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _debate_result_to_candidates(
        self,
        debate_result: DebateResult,
        top_k: int,
    ) -> list[CandidateFactor]:
        """Convert DebateResult critic scores into CandidateFactor objects."""
        from factorminer.agent.output_parser import _try_build_candidate

        kept_scores = [cs for cs in debate_result.critic_scores if cs.keep]
        kept_scores.sort(key=lambda cs: cs.composite_score, reverse=True)
        kept_scores = kept_scores[:top_k]

        result: list[CandidateFactor] = []
        seen_formulas: set = set()

        for cs in kept_scores:
            if cs.formula in seen_formulas:
                continue
            cf = _try_build_candidate(cs.factor_name, cs.formula)
            if cf.is_valid:
                cf.category = f"specialist:{cs.source_specialist}/{cf.category}"
                result.append(cf)
                seen_formulas.add(cs.formula)

        if not result:
            for formula in debate_result.after_critic[:top_k]:
                if formula in seen_formulas:
                    continue
                cf = _try_build_candidate(
                    f"debate_factor_{len(result)+1}", formula
                )
                if cf.is_valid:
                    result.append(cf)
                    seen_formulas.add(formula)

        return result

    def _tag_specialist_source_from_agents(
        self,
        candidates: list[CandidateFactor],
    ) -> list[CandidateFactor]:
        """Tag candidate source if not already embedded in category."""
        for c in candidates:
            if not c.category.startswith("specialist:"):
                if self._last_debate_result:
                    for spec_name, formulas in (
                        self._last_debate_result.specialist_proposals.items()
                    ):
                        if c.formula in formulas:
                            c.category = f"specialist:{spec_name}/{c.category}"
                            break
        return candidates

    # ------------------------------------------------------------------
    # Legacy static helpers (backward compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _scores_to_candidates(
        scores: list[CriticScore],
        proposals: dict[str, list[CandidateFactor]],
    ) -> list[CandidateFactor]:
        """Map CriticScore objects back to CandidateFactor instances."""
        lookup: dict[str, CandidateFactor] = {}
        for candidates in proposals.values():
            for c in candidates:
                lookup[c.name] = c

        result: list[CandidateFactor] = []
        seen: set = set()
        for score in scores:
            candidate = lookup.get(score.factor_name)
            if candidate is not None and score.factor_name not in seen:
                result.append(candidate)
                seen.add(score.factor_name)

        return result

    @staticmethod
    def _tag_specialist_source(
        candidates: list[CandidateFactor],
        proposals: dict[str, list[CandidateFactor]],
    ) -> list[CandidateFactor]:
        """Add specialist source information to each candidate's category."""
        source_map: dict[str, str] = {}
        for spec_name, spec_candidates in proposals.items():
            for c in spec_candidates:
                source_map[c.name] = spec_name

        for c in candidates:
            spec_name = source_map.get(c.name, "unknown")
            if not c.category.startswith("specialist:"):
                c.category = f"specialist:{spec_name}/{c.category}"

        return candidates


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _flatten_memory_signal(memory_signal: dict[str, Any]) -> str:
    """Flatten a memory signal dict to a compact string."""
    parts: list[str] = []
    for key in (
        "recommended_directions", "strategic_insights",
        "complementary_patterns", "prompt_text",
    ):
        val = memory_signal.get(key)
        if isinstance(val, list):
            parts.extend(str(v) for v in val)
        elif isinstance(val, str) and val:
            parts.append(val)
    return " ".join(parts)
