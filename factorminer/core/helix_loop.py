"""The Helix Loop: 5-stage self-evolving factor discovery with Phase 2 extensions.

Extends the base Ralph Loop with:
  1. RETRIEVE  -- KG + embeddings + flat memory hybrid retrieval
  2. PROPOSE   -- Multi-agent debate (specialists + critic) or standard generation
  3. SYNTHESIZE -- SymPy canonicalization to eliminate mathematical duplicates
  4. VALIDATE  -- Standard pipeline + causal + regime + capacity + significance
  5. DISTILL   -- Standard memory evolution + KG update + online forgetting

All Phase 2 components are optional: when none are enabled the Helix Loop
behaves identically to the Ralph Loop and is a full drop-in replacement.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from factorminer.agent.llm_interface import LLMProvider
from factorminer.architecture import (
    DistillStage,
    EvaluateStage,
    GenerateStage,
    IterationPayload,
    KnowledgeGraphService,
    LibraryUpdateStage,
    OnlineForgettingService,
    RetrieveStage,
)
from factorminer.core.factor_library import FactorLibrary
from factorminer.core.parser import try_parse
from factorminer.core.ralph_loop import (
    EvaluationResult,
    RalphLoop,
)
from factorminer.evaluation.metrics import compute_ic
from factorminer.memory.memory_store import ExperienceMemory
from factorminer.memory.retrieval import retrieve_memory
from factorminer.utils.logging import FactorRecord, IterationRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional imports -- resolved at call time with graceful fallback
# ---------------------------------------------------------------------------


def _try_import_debate():
    try:
        from factorminer.agent.debate import DebateConfig, DebateGenerator

        return DebateGenerator, DebateConfig
    except ImportError:
        return None, None


def _try_import_canonicalizer():
    try:
        from factorminer.core.canonicalizer import FormulaCanonicalizer

        return FormulaCanonicalizer
    except ImportError:
        return None


def _try_import_causal():
    try:
        from factorminer.evaluation.causal import CausalConfig, CausalValidator

        return CausalValidator, CausalConfig
    except ImportError:
        return None, None


def _try_import_regime():
    try:
        from factorminer.evaluation.regime import (
            RegimeAwareEvaluator,
            RegimeConfig,
            RegimeDetector,
        )

        return RegimeDetector, RegimeAwareEvaluator, RegimeConfig
    except ImportError:
        return None, None, None


def _try_import_capacity():
    try:
        from factorminer.evaluation.capacity import CapacityConfig, CapacityEstimator

        return CapacityEstimator, CapacityConfig
    except ImportError:
        return None, None


def _try_import_significance():
    try:
        from factorminer.evaluation.significance import (
            BootstrapICTester,
            DeflatedSharpeCalculator,
            FDRController,
            SignificanceConfig,
        )

        return BootstrapICTester, FDRController, DeflatedSharpeCalculator, SignificanceConfig
    except ImportError:
        return None, None, None, None


def _try_import_kg():
    try:
        from factorminer.memory.knowledge_graph import FactorKnowledgeGraph, FactorNode

        return FactorKnowledgeGraph, FactorNode
    except ImportError:
        return None, None


def _try_import_kg_retrieval():
    try:
        from factorminer.memory.kg_retrieval import retrieve_memory_enhanced

        return retrieve_memory_enhanced
    except ImportError:
        return None


def _try_import_embedder():
    try:
        from factorminer.memory.embeddings import FormulaEmbedder

        return FormulaEmbedder
    except ImportError:
        return None


def _try_import_auto_inventor():
    try:
        from factorminer.operators.auto_inventor import OperatorInventor

        return OperatorInventor
    except ImportError:
        return None


def _try_import_custom_store():
    try:
        from factorminer.operators.custom import CustomOperatorStore

        return CustomOperatorStore
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# HelixLoop
# ---------------------------------------------------------------------------


class HelixLoop(RalphLoop):
    """Enhanced 5-stage Helix Loop for self-evolving factor discovery.

    Extends the Ralph Loop with:
    1. RETRIEVE: KG + embeddings + flat memory hybrid retrieval
    2. PROPOSE: Multi-agent debate (specialists + critic) or standard generation
    3. SYNTHESIZE: SymPy canonicalization to eliminate mathematical duplicates
    4. VALIDATE: Standard pipeline + causal + regime + capacity + significance
    5. DISTILL: Standard memory evolution + KG update + online forgetting

    All Phase 2 components are optional and default to off. When none are
    enabled, the Helix Loop behaves identically to the Ralph Loop.

    Parameters
    ----------
    config : Any
        Mining configuration object.
    data_tensor : np.ndarray
        Market data tensor D in R^(M x T x F).
    returns : np.ndarray
        Forward returns array R in R^(M x T).
    llm_provider : LLMProvider, optional
        LLM provider for factor generation.
    memory : ExperienceMemory, optional
        Pre-populated experience memory.
    library : FactorLibrary, optional
        Pre-populated factor library.
    debate_config : DebateConfig, optional
        Configuration for multi-agent debate generation.
        When provided, replaces standard FactorGenerator.
    enable_knowledge_graph : bool
        Enable factor knowledge graph for lineage and structural analysis.
    enable_embeddings : bool
        Enable semantic formula embeddings for similarity search.
    enable_auto_inventor : bool
        Enable periodic auto-invention of new operators.
    auto_invention_interval : int
        Run auto-invention every N iterations.
    canonicalize : bool
        Enable SymPy-based formula canonicalization for deduplication.
    forgetting_lambda : float
        Exponential decay factor for online forgetting (0-1).
    causal_config : CausalConfig, optional
        Configuration for causal validation (Granger + intervention).
    regime_config : RegimeConfig, optional
        Configuration for regime-aware IC evaluation.
    capacity_config : CapacityConfig, optional
        Configuration for capacity-aware cost evaluation.
    significance_config : SignificanceConfig, optional
        Configuration for bootstrap CI + FDR + deflated Sharpe.
    volume : np.ndarray, optional
        Dollar volume array (M, T) required for capacity estimation.
    """

    def __init__(
        self,
        config: Any,
        data_tensor: np.ndarray,
        returns: np.ndarray,
        llm_provider: LLMProvider | None = None,
        memory: ExperienceMemory | None = None,
        library: FactorLibrary | None = None,
        # Phase 2 extensions
        debate_config: Any | None = None,
        enable_knowledge_graph: bool = False,
        enable_embeddings: bool = False,
        enable_auto_inventor: bool = False,
        auto_invention_interval: int = 10,
        canonicalize: bool = True,
        forgetting_lambda: float = 0.95,
        causal_config: Any | None = None,
        regime_config: Any | None = None,
        capacity_config: Any | None = None,
        significance_config: Any | None = None,
        volume: np.ndarray | None = None,
    ) -> None:
        # Initialize base RalphLoop
        super().__init__(
            config=config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=llm_provider,
            memory=memory,
            library=library,
        )

        # Store Phase 2 configuration
        self._debate_config = debate_config
        self._enable_kg = enable_knowledge_graph
        self._enable_embeddings = enable_embeddings
        self._enable_auto_inventor = enable_auto_inventor
        self._auto_invention_interval = auto_invention_interval
        self._canonicalize = canonicalize
        self._forgetting_lambda = forgetting_lambda
        self._causal_config = causal_config
        self._regime_config = regime_config
        self._capacity_config = capacity_config
        self._significance_config = significance_config
        self._volume = volume
        self._kg_service = KnowledgeGraphService()
        self._forgetting_service = OnlineForgettingService(forgetting_lambda=forgetting_lambda)

        # Track iterations without admissions for forgetting
        self._no_admission_streak: int = 0

        # Initialize Phase 2 components
        self._debate_generator: Any | None = None
        self._canonicalizer: Any | None = None
        self._causal_validator: Any | None = None
        self._regime_detector: Any | None = None
        self._regime_evaluator: Any | None = None
        self._regime_classification: Any | None = None
        self._capacity_estimator: Any | None = None
        self._bootstrap_tester: Any | None = None
        self._fdr_controller: Any | None = None
        self._kg: Any | None = None
        self._embedder: Any | None = None
        self._auto_inventor: Any | None = None
        self._custom_op_store: Any | None = None

        self._init_phase2_components(llm_provider)
        self.stages.update(
            {
                "retrieve": RetrieveStage(self._stage_retrieve_helix),
                "generate": GenerateStage(self._stage_generate_helix),
                "evaluate": EvaluateStage(self._stage_evaluate_helix),
                "library_update": LibraryUpdateStage(self._stage_library_update_helix),
                "distill": DistillStage(self._stage_distill_helix),
            }
        )

    # ------------------------------------------------------------------
    # Phase 2 component initialization
    # ------------------------------------------------------------------

    def _init_phase2_components(self, llm_provider: LLMProvider | None) -> None:
        """Initialize all Phase 2 components based on configuration."""

        # -- Debate generator --
        if self._debate_config is not None:
            DebateGeneratorCls, _ = _try_import_debate()
            if DebateGeneratorCls is not None:
                try:
                    self._debate_generator = DebateGeneratorCls(
                        llm_provider=llm_provider or self.generator.llm,
                        debate_config=self._debate_config,
                    )
                    logger.info("Helix: multi-agent debate generator enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init debate generator: %s", exc)
            else:
                logger.warning("Helix: debate_config provided but debate module unavailable")

        # -- Canonicalizer --
        if self._canonicalize:
            FormulaCanonCls = _try_import_canonicalizer()
            if FormulaCanonCls is not None:
                try:
                    self._canonicalizer = FormulaCanonCls()
                    logger.info("Helix: SymPy canonicalization enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init canonicalizer: %s", exc)
            else:
                logger.warning("Helix: canonicalize=True but sympy/canonicalizer unavailable")

        # -- Causal validator --
        if self._causal_config is not None:
            CausalValidatorCls, _ = _try_import_causal()
            if CausalValidatorCls is not None:
                logger.info("Helix: causal validation enabled")
            else:
                logger.warning("Helix: causal_config provided but causal module unavailable")

        # -- Regime evaluator --
        if self._regime_config is not None:
            RegimeDetectorCls, RegimeEvalCls, _ = _try_import_regime()
            if RegimeDetectorCls is not None and RegimeEvalCls is not None:
                try:
                    self._regime_detector = RegimeDetectorCls(self._regime_config)
                    self._regime_classification = self._regime_detector.classify(self.returns)
                    self._regime_evaluator = RegimeEvalCls(
                        returns=self.returns,
                        regime=self._regime_classification,
                        config=self._regime_config,
                    )
                    logger.info("Helix: regime-aware evaluation enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init regime evaluator: %s", exc)
            else:
                logger.warning("Helix: regime_config provided but regime module unavailable")

        # -- Capacity estimator --
        if self._capacity_config is not None:
            CapacityEstCls, _ = _try_import_capacity()
            if CapacityEstCls is not None:
                if self._volume is not None:
                    try:
                        self._capacity_estimator = CapacityEstCls(
                            returns=self.returns,
                            volume=self._volume,
                            config=self._capacity_config,
                        )
                        logger.info("Helix: capacity-aware evaluation enabled")
                    except Exception as exc:
                        logger.warning("Helix: failed to init capacity estimator: %s", exc)
                else:
                    logger.warning("Helix: capacity_config provided but no volume data supplied")
            else:
                logger.warning("Helix: capacity_config provided but capacity module unavailable")

        # -- Significance testing --
        if self._significance_config is not None:
            BootstrapCls, FDRCls, _, _ = _try_import_significance()
            if BootstrapCls is not None and FDRCls is not None:
                try:
                    self._bootstrap_tester = BootstrapCls(self._significance_config)
                    self._fdr_controller = FDRCls(self._significance_config)
                    logger.info("Helix: significance testing enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init significance testing: %s", exc)
            else:
                logger.warning(
                    "Helix: significance_config provided but significance module unavailable"
                )

        # -- Knowledge graph --
        if self._enable_kg:
            KGCls, _ = _try_import_kg()
            if KGCls is not None:
                try:
                    self._kg = KGCls()
                    logger.info("Helix: knowledge graph enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init knowledge graph: %s", exc)
            else:
                logger.warning(
                    "Helix: enable_knowledge_graph=True but knowledge_graph module unavailable"
                )

        # -- Embedder --
        if self._enable_embeddings:
            EmbedderCls = _try_import_embedder()
            if EmbedderCls is not None:
                try:
                    self._embedder = EmbedderCls()
                    self._prime_embedder_from_library()
                    logger.info("Helix: formula embeddings enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init embedder: %s", exc)
            else:
                logger.warning("Helix: enable_embeddings=True but embeddings module unavailable")

        # -- Auto inventor --
        if self._enable_auto_inventor:
            InventorCls = _try_import_auto_inventor()
            CustomStoreCls = _try_import_custom_store()
            if InventorCls is not None:
                try:
                    self._auto_inventor = InventorCls(
                        llm_provider=llm_provider or self.generator.llm,
                        data_tensor=self.data_tensor,
                        returns=self.returns,
                    )
                    logger.info("Helix: auto operator invention enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init auto inventor: %s", exc)

            if CustomStoreCls is not None:
                output_dir = getattr(self.config, "output_dir", "./output")
                try:
                    self._custom_op_store = CustomStoreCls(
                        store_dir=str(Path(output_dir) / "custom_operators")
                    )
                    logger.info("Helix: custom operator store enabled")
                except Exception as exc:
                    logger.warning("Helix: failed to init custom operator store: %s", exc)
            else:
                logger.warning(
                    "Helix: enable_auto_inventor=True but custom operator store unavailable"
                )

    # ------------------------------------------------------------------
    # Override: _run_iteration with 5-stage Helix flow
    # ------------------------------------------------------------------

    def _run_iteration(self, batch_size: int) -> dict[str, Any]:
        """Execute one iteration of the 5-stage Helix Loop.

        Stages:
          1. RETRIEVE  -- enhanced memory retrieval (KG + embeddings + flat)
          2. PROPOSE   -- debate or standard factor generation
          3. SYNTHESIZE -- canonicalize and deduplicate candidates
          4. VALIDATE  -- standard pipeline + causal + regime + capacity + significance
          5. DISTILL   -- memory evolution + KG update + forgetting

        Returns
        -------
        dict
            Iteration statistics.
        """
        t0 = time.time()
        payload = IterationPayload(iteration=self.iteration, batch_size=batch_size)

        self.stages["retrieve"].run(self, payload)
        self.stages["generate"].run(self, payload)
        self.budget.record_llm_call()

        if not payload.candidates:
            logger.warning(
                "Helix iteration %d: generator produced 0 candidates",
                self.iteration,
            )
            return self._empty_stats()

        if not payload.candidates:
            logger.warning(
                "Helix iteration %d: all candidates removed by canonicalization",
                self.iteration,
            )
            return self._empty_stats()

        self.stages["evaluate"].run(self, payload)
        self.stages["library_update"].run(self, payload)

        provenance_library_state = {
            **payload.library_state,
            "diagnostics": self.library.get_diagnostics(),
        }

        self._attach_factor_provenance(
            payload.admitted_results,
            library_state=provenance_library_state,
            memory_signal=payload.memory_signal,
            phase2_summary={
                "enabled_features": self._phase2_features(),
                "phase2_rejections": payload.stage_metrics.get("phase2_rejections", 0),
            },
            generator_family=self._generator_family(),
        )

        self.stages["distill"].run(self, payload)

        # Build stats
        elapsed = time.time() - t0
        self.budget.record_compute(elapsed)
        stats = self._compute_stats(payload.results, payload.admitted_results, elapsed)
        stats.update(payload.stage_metrics)
        stats["iteration"] = self.iteration

        # Log to reporter and session logger
        self.reporter.log_batch(**stats)
        if self._session_logger:
            ic_values = [r.ic_mean for r in payload.results if r.parse_ok]
            record = IterationRecord(
                iteration=self.iteration,
                candidates_generated=int(
                    payload.stage_metrics.get("candidates_before_canon", len(payload.candidates))
                ),
                ic_passed=stats["ic_passed"],
                correlation_passed=stats["corr_passed"],
                admitted=stats["admitted"],
                rejected=int(
                    payload.stage_metrics.get("candidates_before_canon", len(payload.candidates))
                )
                - stats["admitted"],
                replaced=stats["replaced"],
                library_size=self.library.size,
                best_ic=max(ic_values) if ic_values else 0.0,
                mean_ic=float(np.mean(ic_values)) if ic_values else 0.0,
                elapsed_seconds=elapsed,
            )
            self._session_logger.log_iteration(record)

            for r in payload.results:
                factor_rec = FactorRecord(
                    expression=r.formula,
                    ic=r.ic_mean if r.parse_ok else None,
                    icir=r.icir if r.parse_ok else None,
                    max_correlation=r.max_correlation if r.parse_ok else None,
                    admitted=r.admitted,
                    rejection_reason=r.rejection_reason or None,
                    replaced_factor=str(r.replaced) if r.replaced else None,
                )
                self._session_logger.log_factor(factor_rec)

        return stats

    def _stage_retrieve_helix(
        self,
        _loop: HelixLoop,
        payload: IterationPayload,
    ) -> dict[str, Any]:
        return self._helix_retrieve(payload.library_state)

    def _stage_generate_helix(
        self,
        _loop: HelixLoop,
        payload: IterationPayload,
    ) -> list[tuple[str, str]]:
        payload.prompt_context = self.prompt_context_builder.build(
            payload.memory_signal,
            payload.library_state,
            batch_size=payload.batch_size,
            extras={"dataset_contract": self.dataset_contract.to_dict()},
        )
        proposed = self._helix_propose(
            payload.prompt_context, payload.library_state, payload.batch_size
        )
        payload.stage_metrics["candidates_before_canon"] = len(proposed)
        deduped, n_canon_dupes, n_semantic_dupes = self._canonicalize_and_dedup(proposed)
        payload.stage_metrics["canonical_duplicates_removed"] = n_canon_dupes
        payload.stage_metrics["semantic_duplicates_removed"] = n_semantic_dupes
        return deduped

    def _stage_evaluate_helix(
        self,
        _loop: HelixLoop,
        payload: IterationPayload,
    ) -> list[EvaluationResult]:
        results = self.pipeline.evaluate_batch(payload.candidates)
        self.lifecycle_store.record_batch_results(self.iteration, results)
        return results

    def _stage_library_update_helix(
        self,
        _loop: HelixLoop,
        payload: IterationPayload,
    ) -> list[EvaluationResult]:
        admitted_results = self._update_library(payload.results)
        rejected_by_phase2 = self._helix_validate(payload.results, admitted_results)
        payload.stage_metrics["phase2_rejections"] = rejected_by_phase2
        return [result for result in admitted_results if result.admitted]

    def _stage_distill_helix(
        self,
        _loop: HelixLoop,
        payload: IterationPayload,
    ) -> None:
        super()._stage_distill(self, payload)
        self._helix_distill(payload.results, payload.admitted_results)
        if self._auto_inventor is not None and self.iteration % self._auto_invention_interval == 0:
            self._run_auto_invention()

    # ------------------------------------------------------------------
    # Stage 1: Enhanced retrieval
    # ------------------------------------------------------------------

    def _helix_retrieve(self, library_state: dict[str, Any]) -> dict[str, Any]:
        """Stage 1 RETRIEVE: KG + embeddings + flat memory hybrid retrieval.

        Falls back to standard retrieve_memory if no KG/embedder is available.
        """
        retrieve_enhanced_fn = _try_import_kg_retrieval()

        if retrieve_enhanced_fn is not None and (
            self._kg is not None or self._embedder is not None
        ):
            try:
                return retrieve_enhanced_fn(
                    memory=self.memory,
                    library_state=library_state,
                    kg=self._kg,
                    embedder=self._embedder,
                )
            except Exception as exc:
                logger.warning("Helix: enhanced retrieval failed, falling back: %s", exc)

        return retrieve_memory(self.memory, library_state=library_state)

    # ------------------------------------------------------------------
    # Stage 2: Debate or standard proposal
    # ------------------------------------------------------------------

    def _helix_propose(
        self,
        memory_signal: dict[str, Any],
        library_state: dict[str, Any],
        batch_size: int,
    ) -> list[tuple[str, str]]:
        """Stage 2 PROPOSE: Use debate generator or standard generator.

        Returns list of (name, formula) tuples compatible with the
        validation pipeline.
        """
        if self._debate_generator is not None:
            try:
                debate_candidates = self._debate_generator.generate_batch(
                    memory_signal=memory_signal,
                    library_state=library_state,
                    batch_size=batch_size,
                )
                # Convert CandidateFactor objects to (name, formula) tuples
                tuples: list[tuple[str, str]] = []
                for c in debate_candidates:
                    tuples.append((c.name, c.formula))
                if tuples:
                    logger.info(
                        "Helix: debate generator produced %d candidates",
                        len(tuples),
                    )
                    return tuples
                logger.warning(
                    "Helix: debate generator returned 0 candidates, "
                    "falling back to standard generator"
                )
            except Exception as exc:
                logger.warning("Helix: debate generation failed, falling back: %s", exc)

        # Standard generation
        return self.generator.generate_batch(
            memory_signal=memory_signal,
            library_state=library_state,
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # Stage 3: Canonicalization + deduplication
    # ------------------------------------------------------------------

    def _canonicalize_and_dedup(
        self, candidates: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], int, int]:
        """Stage 3 SYNTHESIZE: Remove mathematically equivalent candidates.

        Uses SymPy-based canonicalization to detect algebraic duplicates
        before evaluation, saving compute.

        Returns
        -------
        tuple of (deduplicated_candidates, n_canonical_duplicates_removed,
        n_semantic_duplicates_removed)
        """
        if self._canonicalizer is None and self._embedder is None:
            return candidates, 0, 0

        seen_hashes: dict[str, str] = {}  # hash -> first factor name
        unique: list[tuple[str, str]] = []
        n_canon_dupes = 0
        n_semantic_dupes = 0

        for name, formula in candidates:
            tree = try_parse(formula)
            if tree is not None and self._canonicalizer is not None:
                try:
                    canon_hash = self._canonicalizer.canonicalize(tree)
                except Exception as exc:
                    logger.debug("Helix: canonicalization failed for '%s': %s", name, exc)
                else:
                    if canon_hash in seen_hashes:
                        n_canon_dupes += 1
                        logger.debug(
                            "Helix: canonical duplicate '%s' matches '%s'",
                            name,
                            seen_hashes[canon_hash],
                        )
                        continue
                    seen_hashes[canon_hash] = name

            semantic_match = self._semantic_duplicate_target(formula)
            if semantic_match is not None:
                n_semantic_dupes += 1
                logger.debug(
                    "Helix: semantic duplicate '%s' matches library factor '%s'",
                    name,
                    semantic_match,
                )
                continue

            unique.append((name, formula))

        if n_canon_dupes > 0:
            logger.info(
                "Helix: canonicalization removed %d/%d duplicate candidates",
                n_canon_dupes,
                len(candidates),
            )

        if n_semantic_dupes > 0:
            logger.info(
                "Helix: embedding screen removed %d/%d library-adjacent candidates",
                n_semantic_dupes,
                len(candidates),
            )

        return unique, n_canon_dupes, n_semantic_dupes

    # ------------------------------------------------------------------
    # Stage 4: Extended validation
    # ------------------------------------------------------------------

    def _helix_validate(
        self,
        results: list[EvaluationResult],
        admitted_results: list[EvaluationResult],
    ) -> int:
        """Stage 4 extended VALIDATE: causal + regime + capacity + significance.

        Runs Phase 2 validation on admitted candidates and revokes admission
        for those that fail. Returns the number of Phase 2 rejections.
        """
        if not admitted_results:
            self._no_admission_streak += 1
            return 0

        rejected = 0

        # Collect admitted results that still have signals for extended checks
        to_check = [r for r in admitted_results if r.signals is not None]
        if not to_check:
            self._no_admission_streak = (
                0 if any(r.admitted for r in admitted_results) else self._no_admission_streak + 1
            )
            return 0

        # -- Causal validation --
        if self._causal_config is not None:
            rejected += self._validate_causal(to_check, results)

        # -- Regime validation --
        if self._regime_evaluator is not None:
            rejected += self._validate_regime(to_check, results)

        # -- Capacity validation --
        if self._capacity_estimator is not None:
            rejected += self._validate_capacity(to_check, results)

        # -- Significance testing (batch-level FDR) --
        if self._bootstrap_tester is not None and self._fdr_controller is not None:
            rejected += self._validate_significance(to_check, results)

        if rejected > 0:
            logger.info(
                "Helix: Phase 2 validation rejected %d/%d admitted candidates",
                rejected,
                len(admitted_results),
            )

        if any(r.admitted for r in admitted_results):
            self._no_admission_streak = 0
        else:
            self._no_admission_streak += 1

        return rejected

    def _validate_causal(
        self,
        to_check: list[EvaluationResult],
        all_results: list[EvaluationResult],
    ) -> int:
        """Run causal validation (Granger + intervention) on admitted candidates."""
        CausalValidatorCls, _ = _try_import_causal()
        if CausalValidatorCls is None:
            return 0

        # Collect library signals for controls
        library_signals: dict[str, np.ndarray] = {}
        for f in self.library.list_factors():
            if f.signals is not None:
                library_signals[f.name] = f.signals

        try:
            validator = CausalValidatorCls(
                returns=self.returns,
                data_tensor=self.data_tensor,
                library_signals=library_signals,
                config=self._causal_config,
            )
        except Exception as exc:
            logger.warning("Helix: causal validator creation failed: %s", exc)
            return 0

        rejected = 0
        threshold = getattr(self._causal_config, "robustness_threshold", 0.4)

        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                result = validator.validate(r.factor_name, r.signals)
                if not result.passes:
                    self._revoke_admission(
                        r,
                        all_results,
                        f"Causal: robustness_score={result.robustness_score:.3f} < {threshold}",
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: causal rejection for '%s' (score=%.3f)",
                        r.factor_name,
                        result.robustness_score,
                    )
            except Exception as exc:
                logger.warning(
                    "Helix: causal validation error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        return rejected

    def _validate_regime(
        self,
        to_check: list[EvaluationResult],
        all_results: list[EvaluationResult],
    ) -> int:
        """Run regime-aware IC evaluation on admitted candidates."""
        if self._regime_evaluator is None:
            return 0

        rejected = 0
        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                result = self._regime_evaluator.evaluate(r.factor_name, r.signals)
                if not result.passes:
                    self._revoke_admission(
                        r,
                        all_results,
                        f"Regime: only {result.n_regimes_passing} regimes passing "
                        f"(need {getattr(self._regime_config, 'min_regimes_passing', 2)})",
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: regime rejection for '%s' (%d regimes passing)",
                        r.factor_name,
                        result.n_regimes_passing,
                    )
            except Exception as exc:
                logger.warning(
                    "Helix: regime validation error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        return rejected

    def _validate_capacity(
        self,
        to_check: list[EvaluationResult],
        all_results: list[EvaluationResult],
    ) -> int:
        """Run capacity-aware cost evaluation on admitted candidates."""
        if self._capacity_estimator is None:
            return 0

        rejected = 0
        net_icir_threshold = getattr(self._capacity_config, "net_icir_threshold", 0.3)

        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                result = self._capacity_estimator.net_cost_evaluation(
                    factor_name=r.factor_name,
                    signals=r.signals,
                )
                if not result.passes_net_threshold:
                    self._revoke_admission(
                        r,
                        all_results,
                        f"Capacity: net_icir={result.net_icir:.3f} < {net_icir_threshold}",
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: capacity rejection for '%s' (net_icir=%.3f)",
                        r.factor_name,
                        result.net_icir,
                    )
            except Exception as exc:
                logger.warning(
                    "Helix: capacity validation error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        return rejected

    def _validate_significance(
        self,
        to_check: list[EvaluationResult],
        all_results: list[EvaluationResult],
    ) -> int:
        """Run bootstrap CI + batch-level FDR correction on admitted candidates."""
        if self._bootstrap_tester is None or self._fdr_controller is None:
            return 0

        # Compute IC series for each admitted candidate and gather p-values
        ic_series_map: dict[str, np.ndarray] = {}
        result_map: dict[str, EvaluationResult] = {}

        for r in to_check:
            if not r.admitted or r.signals is None:
                continue
            try:
                ic_series = compute_ic(r.signals, self.returns)
                ic_series_map[r.factor_name] = ic_series
                result_map[r.factor_name] = r
            except Exception as exc:
                logger.warning(
                    "Helix: IC computation error for '%s': %s",
                    r.factor_name,
                    exc,
                )

        if not ic_series_map:
            return 0

        try:
            fdr_result = self._fdr_controller.batch_evaluate(ic_series_map, self._bootstrap_tester)
        except Exception as exc:
            logger.warning("Helix: FDR batch evaluation failed: %s", exc)
            return 0

        rejected = 0
        for name, is_sig in fdr_result.significant.items():
            if not is_sig:
                r = result_map.get(name)
                if r is not None and r.admitted:
                    adj_p = fdr_result.adjusted_p_values.get(name, 1.0)
                    self._revoke_admission(
                        r,
                        all_results,
                        f"Significance: FDR-adjusted p={adj_p:.4f} > "
                        f"{getattr(self._significance_config, 'fdr_level', 0.05)}",
                    )
                    rejected += 1
                    logger.debug(
                        "Helix: significance rejection for '%s' (adj_p=%.4f)",
                        name,
                        adj_p,
                    )

        return rejected

    def _revoke_admission(
        self,
        result: EvaluationResult,
        all_results: list[EvaluationResult],
        reason: str,
    ) -> None:
        """Revoke a previously admitted candidate from the library.

        Updates the EvaluationResult and removes the factor from the library.
        """
        result.admitted = False
        result.rejection_reason = reason

        # Find and remove from library by name+formula match
        try:
            for factor in self.library.list_factors():
                if factor.name == result.factor_name and factor.formula == result.formula:
                    self.library.remove_factor(factor.id)
                    self._remove_semantic_artifacts(result.factor_name)
                    logger.debug(
                        "Helix: revoked factor '%s' (id=%d): %s",
                        result.factor_name,
                        factor.id,
                        reason,
                    )
                    return
        except Exception as exc:
            logger.warning(
                "Helix: failed to revoke factor '%s': %s",
                result.factor_name,
                exc,
            )

        self._remove_semantic_artifacts(result.factor_name)

    # ------------------------------------------------------------------
    # Stage 5: Enhanced distillation
    # ------------------------------------------------------------------

    def _helix_distill(
        self,
        results: list[EvaluationResult],
        admitted_results: list[EvaluationResult],
    ) -> None:
        """Stage 5 DISTILL: KG update + embeddings + online forgetting."""

        # -- Knowledge graph updates --
        if self._kg is not None:
            self._update_knowledge_graph(results, admitted_results)

        # -- Embed newly admitted factors --
        if self._embedder is not None:
            for r in admitted_results:
                if r.admitted:
                    try:
                        self._embedder.embed(r.factor_name, r.formula)
                    except Exception as exc:
                        logger.debug(
                            "Helix: embedding failed for '%s': %s",
                            r.factor_name,
                            exc,
                        )

        # -- Online forgetting --
        self._forgetting_service.apply(self.memory, self._no_admission_streak)

    def _update_knowledge_graph(
        self,
        results: list[EvaluationResult],
        admitted_results: list[EvaluationResult],
    ) -> None:
        """Update the knowledge graph with new factor nodes and edges."""
        if self._kg is None:
            return

        for r in admitted_results:
            try:
                self._kg_service.add_result(
                    kg=self._kg,
                    result=r,
                    library=self.library,
                    iteration=self.iteration,
                    embedder=self._embedder,
                )
            except Exception as exc:
                logger.debug("Helix: failed to add factor to KG: %s", exc)

    def _remove_semantic_artifacts(self, factor_id: str) -> None:
        """Remove a factor from derived semantic stores if present."""
        if self._kg is not None:
            self._kg_service.remove_factor(kg=self._kg, embedder=self._embedder, factor_id=factor_id)

    # ------------------------------------------------------------------
    # Auto-invention
    # ------------------------------------------------------------------

    def _run_auto_invention(self) -> None:
        """Periodically propose, validate, and register new operators.

        Uses the OperatorInventor to generate novel operators from
        successful pattern context, then validates and registers them
        via CustomOperatorStore.
        """
        if self._auto_inventor is None:
            return

        logger.info("Helix: running auto-invention at iteration %d", self.iteration)

        # Gather existing operators
        try:
            from factorminer.core.types import OPERATOR_REGISTRY as SPEC_REG

            existing_ops = dict(SPEC_REG)
        except ImportError:
            existing_ops = {}

        # Gather successful pattern descriptions
        patterns = []
        for pat in self.memory.success_patterns[:10]:
            patterns.append(f"{pat.name}: {pat.description}")

        try:
            proposals = self._auto_inventor.propose_operators(
                existing_operators=existing_ops,
                successful_patterns=patterns,
            )
        except Exception as exc:
            logger.warning("Helix: auto-invention proposal failed: %s", exc)
            return

        self.budget.record_llm_call()

        validated = 0
        for proposal in proposals:
            try:
                val_result = self._auto_inventor.validate_operator(proposal)
                if val_result.valid:
                    self._register_invented_operator(proposal, val_result)
                    validated += 1
                else:
                    logger.debug(
                        "Helix: operator '%s' failed validation: %s",
                        proposal.name,
                        val_result.error,
                    )
            except Exception as exc:
                logger.warning(
                    "Helix: operator validation error for '%s': %s",
                    proposal.name,
                    exc,
                )

        logger.info(
            "Helix: auto-invention: %d/%d proposals validated and registered",
            validated,
            len(proposals),
        )

    def _register_invented_operator(
        self,
        proposal: Any,
        val_result: Any,
    ) -> None:
        """Register a validated auto-invented operator."""
        if self._custom_op_store is None:
            logger.warning(
                "Helix: no custom operator store; cannot register '%s'",
                proposal.name,
            )
            return

        try:
            from factorminer.core.types import OperatorSpec, OperatorType, SignatureType
            from factorminer.operators.custom import CustomOperator

            spec = OperatorSpec(
                name=proposal.name,
                arity=proposal.arity,
                category=OperatorType.AUTO_INVENTED,
                signature=SignatureType.TIME_SERIES_TO_TIME_SERIES,
                param_names=proposal.param_names,
                param_defaults=proposal.param_defaults,
                param_ranges={k: tuple(v) for k, v in proposal.param_ranges.items()},
                description=proposal.description,
            )

            # Compile the function
            from factorminer.operators.custom import _compile_operator_code

            fn = _compile_operator_code(proposal.numpy_code)
            if fn is None:
                logger.warning(
                    "Helix: failed to compile invented operator '%s'",
                    proposal.name,
                )
                return

            custom_op = CustomOperator(
                name=proposal.name,
                spec=spec,
                numpy_code=proposal.numpy_code,
                numpy_fn=fn,
                validation_ic=val_result.ic_contribution,
                invention_iteration=self.iteration,
                rationale=proposal.rationale,
            )

            self._custom_op_store.register(custom_op)
            logger.info(
                "Helix: registered auto-invented operator '%s' (IC=%.4f)",
                proposal.name,
                val_result.ic_contribution,
            )
        except Exception as exc:
            logger.warning(
                "Helix: failed to register operator '%s': %s",
                proposal.name,
                exc,
            )

    # ------------------------------------------------------------------
    # Enhanced checkpointing
    # ------------------------------------------------------------------

    def _checkpoint(self) -> None:
        """Save a periodic checkpoint including Phase 2 state."""
        try:
            self.save_session()
        except Exception as exc:
            logger.warning("Helix: checkpoint failed: %s", exc)

    def save_session(self, path: str | None = None) -> str:
        """Save the full mining session state including Phase 2 components.

        Extends the base RalphLoop save with:
        - Knowledge graph serialization
        - Custom operator store persistence

        Parameters
        ----------
        path : str, optional
            Directory for the checkpoint.

        Returns
        -------
        str
            Path to the saved session directory.
        """
        # Base save
        checkpoint_path = super().save_session(path)
        checkpoint_dir = Path(checkpoint_path)

        # Save knowledge graph
        if self._kg is not None:
            try:
                kg_path = checkpoint_dir / "knowledge_graph.json"
                self._kg.save(kg_path)
                logger.debug("Helix: saved knowledge graph to %s", kg_path)
            except Exception as exc:
                logger.warning("Helix: failed to save knowledge graph: %s", exc)

        # Save custom operators
        if self._custom_op_store is not None:
            try:
                self._custom_op_store.save()
                logger.debug("Helix: saved custom operators")
            except Exception as exc:
                logger.warning("Helix: failed to save custom operators: %s", exc)

        # Save helix-specific state
        helix_state = {
            "no_admission_streak": self._no_admission_streak,
            "forgetting_lambda": self._forgetting_lambda,
            "canonicalize": self._canonicalize,
            "enable_knowledge_graph": self._enable_kg,
            "enable_embeddings": self._enable_embeddings,
            "enable_auto_inventor": self._enable_auto_inventor,
        }
        try:
            with open(checkpoint_dir / "helix_state.json", "w") as f:
                json.dump(helix_state, f, indent=2)
        except Exception as exc:
            logger.warning("Helix: failed to save helix state: %s", exc)

        if self._session is not None:
            self._refresh_run_manifest(
                output_dir=str(checkpoint_dir.parent),
                artifact_paths={
                    "library": str(checkpoint_dir / "library.json"),
                    "memory": str(checkpoint_dir / "memory.json"),
                    "session": str(checkpoint_dir / "session.json"),
                    "run_manifest": str(checkpoint_dir / "run_manifest.json"),
                    "loop_state": str(checkpoint_dir / "loop_state.json"),
                    "helix_state": str(checkpoint_dir / "helix_state.json"),
                    "knowledge_graph": str(checkpoint_dir / "knowledge_graph.json"),
                },
            )
            self._persist_run_manifest(checkpoint_dir / "run_manifest.json")
            try:
                self._session.save(checkpoint_dir / "session.json")
            except Exception as exc:
                logger.warning("Helix: failed to save session metadata: %s", exc)

        return checkpoint_path

    def load_session(self, path: str) -> None:
        """Resume a mining session from a saved checkpoint.

        Extends the base RalphLoop load with Phase 2 state restoration.

        Parameters
        ----------
        path : str
            Path to the checkpoint directory.
        """
        super().load_session(path)
        checkpoint_dir = Path(path)

        # Load knowledge graph
        if self._kg is not None:
            kg_path = checkpoint_dir / "knowledge_graph.json"
            if kg_path.exists():
                KGCls, _ = _try_import_kg()
                if KGCls is not None:
                    try:
                        self._kg = KGCls.load(kg_path)
                        logger.info(
                            "Helix: loaded knowledge graph (%d factors, %d edges)",
                            self._kg.get_factor_count(),
                            self._kg.get_edge_count(),
                        )
                    except Exception as exc:
                        logger.warning("Helix: failed to load knowledge graph: %s", exc)

        # Load custom operators
        if self._custom_op_store is not None:
            try:
                self._custom_op_store.load()
            except Exception as exc:
                logger.warning("Helix: failed to load custom operators: %s", exc)

        # Load helix-specific state
        helix_state_path = checkpoint_dir / "helix_state.json"
        if helix_state_path.exists():
            try:
                with open(helix_state_path) as f:
                    helix_state = json.load(f)
                self._no_admission_streak = helix_state.get("no_admission_streak", 0)
                logger.info(
                    "Helix: restored helix state (streak=%d)",
                    self._no_admission_streak,
                )
            except Exception as exc:
                logger.warning("Helix: failed to load helix state: %s", exc)

        self._prime_embedder_from_library()
        if self._session is not None and self._session.run_manifest:
            self._run_manifest = dict(self._session.run_manifest)
        else:
            run_manifest_path = checkpoint_dir / "run_manifest.json"
            if run_manifest_path.exists():
                try:
                    with open(run_manifest_path) as f:
                        self._run_manifest = json.load(f)
                except Exception as exc:
                    logger.warning("Helix: failed to load run manifest: %s", exc)

    def _loop_type(self) -> str:
        """Label the loop for provenance and manifests."""
        return "helix"

    def _phase2_features(self) -> list[str]:
        """List the enabled Helix Phase 2 features."""
        features: list[str] = []
        if self._debate_generator is not None:
            features.append("debate")
        if self._canonicalizer is not None:
            features.append("canonicalization")
        if self._kg is not None:
            features.append("knowledge_graph")
        if self._embedder is not None:
            features.append("embeddings")
        if self._causal_validator is not None:
            features.append("causal")
        if self._regime_evaluator is not None:
            features.append("regime")
        if self._capacity_estimator is not None:
            features.append("capacity")
        if self._bootstrap_tester is not None and self._fdr_controller is not None:
            features.append("significance")
        if self._auto_inventor is not None:
            features.append("auto_inventor")
        return features

    def _generator_family(self) -> str:
        """Return the active Helix generator label for provenance."""
        if self._debate_generator is not None:
            return self._debate_generator.__class__.__name__
        return super()._generator_family()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_operators(formula: str) -> list[str]:
        """Extract operator names from a DSL formula string."""
        return re.findall(r"([A-Z][a-zA-Z]+)\(", formula)

    @staticmethod
    def _extract_features(formula: str) -> list[str]:
        """Extract feature names (e.g. $close, $volume) from a formula."""
        return re.findall(r"\$[a-zA-Z_]+", formula)

    def _prime_embedder_from_library(self) -> None:
        """Seed the embedder cache from the currently admitted library."""
        if self._embedder is None:
            return

        try:
            self._embedder.clear()
        except Exception as exc:
            logger.debug("Helix: failed to clear embedder before priming: %s", exc)
            return

        for factor in self.library.list_factors():
            if not factor.formula:
                continue
            try:
                self._embedder.embed(factor.name, factor.formula)
            except Exception as exc:
                logger.debug(
                    "Helix: failed to prime embedding for '%s': %s",
                    factor.name,
                    exc,
                )

    def _semantic_duplicate_target(self, formula: str) -> str | None:
        """Return the matched library factor if embeddings flag a near-duplicate."""
        if self._embedder is None or self.library.size == 0:
            return None

        try:
            return self._embedder.is_semantic_duplicate(formula)
        except Exception as exc:
            logger.debug("Helix: semantic duplicate check failed: %s", exc)
            return None
