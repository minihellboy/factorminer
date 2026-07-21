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
from pathlib import Path
from typing import Any

import numpy as np

from factorminer.agent.llm_interface import LLMProvider
from factorminer.application.helix_generation import HelixGenerationService
from factorminer.application.helix_validation import HelixValidationService
from factorminer.application.runtime_context import MiningRunContext
from factorminer.architecture import (
    DistillStage,
    EvaluateStage,
    GenerateStage,
    IterationPayload,
    KnowledgeGraphService,
    LibraryUpdateStage,
    OnlineForgettingService,
    Phase2ComponentFactory,
    RetrieveStage,
)
from factorminer.architecture.model_stage import ModelCoOptimizeStage
from factorminer.core.factor_library import FactorLibrary
from factorminer.core.loop_services import LoopExecutionService
from factorminer.core.ralph_loop import (
    EvaluationResult,
    RalphLoop,
)
from factorminer.memory.memory_store import ExperienceMemory

logger = logging.getLogger(__name__)


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
        checkpoint_interval: int = 1,
        run_context: MiningRunContext | None = None,
    ) -> None:
        # Initialize base RalphLoop
        super().__init__(
            config=config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=llm_provider,
            memory=memory,
            library=library,
            checkpoint_interval=checkpoint_interval,
            run_context=run_context,
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
        self._loop_services = LoopExecutionService(self)
        self._generation_capabilities = HelixGenerationService(self)
        self._phase2_validation = HelixValidationService(self)

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
        self._debate_rounds: list[dict[str, Any]] = []


        self._init_phase2_components(llm_provider)
        self.stages.update(
            {
                "retrieve": RetrieveStage(self._stage_retrieve_helix),
                "generate": GenerateStage(self._stage_generate_helix),
                "evaluate": EvaluateStage(self._stage_evaluate_helix),
                "library_update": LibraryUpdateStage(self._stage_library_update_helix),
                "distill": DistillStage(self._stage_distill_helix),
                "model_co_optimize": ModelCoOptimizeStage.from_mining_config(self.settings),
            }
        )

    # ------------------------------------------------------------------
    # Phase 2 component initialization
    # ------------------------------------------------------------------

    def _init_phase2_components(self, llm_provider: LLMProvider | None) -> None:
        """Initialize a typed Phase-2 component bundle through the shared factory."""
        self._component_factory = Phase2ComponentFactory()
        components = self._component_factory.build(
            llm_provider=llm_provider or self.generator.llm_provider,
            data_tensor=self.data_tensor,
            returns=self.returns,
            output_dir=self.settings.output_dir,
            debate_config=self._debate_config,
            canonicalize=self._canonicalize,
            causal_config=self._causal_config,
            regime_config=self._regime_config,
            capacity_config=self._capacity_config,
            significance_config=self._significance_config,
            enable_knowledge_graph=self._enable_kg,
            enable_embeddings=self._enable_embeddings,
            enable_auto_inventor=self._enable_auto_inventor,
            volume=self._volume,
        )
        self._debate_generator = components.debate_generator
        self._canonicalizer = components.canonicalizer
        self._causal_validator = components.causal_validator
        self._regime_detector = components.regime_detector
        self._regime_evaluator = components.regime_evaluator
        self._regime_classification = components.regime_classification
        self._capacity_estimator = components.capacity_estimator
        self._bootstrap_tester = components.bootstrap_tester
        self._fdr_controller = components.fdr_controller
        self._kg = components.knowledge_graph
        self._embedder = components.embedder
        self._auto_inventor = components.auto_inventor
        self._custom_op_store = components.custom_operator_store
        self._prime_embedder_from_library()

    # ------------------------------------------------------------------
    # Override: _run_iteration with 5-stage Helix flow
    # ------------------------------------------------------------------

    def _run_iteration(self, batch_size: int) -> dict[str, Any]:
        """Execute the canonical sequence with the Helix stage composition."""
        return self._loop_services.execute_iteration(
            batch_size,
            trailing_stages=("model_co_optimize",),
            phase2_summary={"enabled_features": self._phase2_features()},
        )

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
            research_archetypes=payload.memory_signal.get("research_archetypes"),
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
        self._annotate_result_lineage(results, payload.library_state)
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
        return self._generation_capabilities._helix_retrieve(library_state)

    def _helix_propose(
        self,
        memory_signal: dict[str, Any],
        library_state: dict[str, Any],
        batch_size: int,
    ) -> list[tuple[str, str]]:
        return self._generation_capabilities._helix_propose(
            memory_signal,
            library_state,
            batch_size,
        )

    def _canonicalize_and_dedup(
        self,
        candidates: list[tuple[str, str]],
    ) -> tuple[list[tuple[str, str]], int, int]:
        return self._generation_capabilities._canonicalize_and_dedup(candidates)

    # Stage 4: Extended validation
    # ------------------------------------------------------------------

    def _helix_validate(
        self,
        results: list[EvaluationResult],
        admitted_results: list[EvaluationResult],
    ) -> int:
        return self._phase2_validation.validate(results, admitted_results)

    def _revoke_admission(
        self,
        result: EvaluationResult,
        all_results: list[EvaluationResult],
        reason: str,
    ) -> None:
        self._phase2_validation.revoke(result, all_results, reason)

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
                (KGCls,) = self._component_factory.resolve(
                    "factorminer.memory.knowledge_graph",
                    "FactorKnowledgeGraph",
                )
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
