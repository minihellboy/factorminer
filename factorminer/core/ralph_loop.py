"""The Ralph Loop: self-evolving factor discovery algorithm.

Implements Algorithm 1 from the FactorMiner paper.  The loop iteratively:
  1. Retrieves memory priors from experience memory  -- R(M, L)
  2. Generates candidate factors via LLM guided by memory -- G(m, L)
  3. Evaluates candidates through a multi-stage pipeline:
     - Stage 1: Fast IC screening on M_fast assets
     - Stage 2: Correlation check against library L
     - Stage 2.5: Replacement check for correlated candidates
     - Stage 3: Intra-batch deduplication (pairwise rho < theta)
     - Stage 4: Full validation on M_full assets + trajectory collection
  4. Updates the factor library with admitted factors  -- L <- L + {alpha}
  5. Evolves the experience memory with new insights   -- E(M, F(M, tau))

The loop terminates when the library reaches the target size K or the
maximum number of iterations is exhausted.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from factorminer.agent.factor_generator import FactorGenerator
from factorminer.agent.llm_interface import LLMProvider, MockProvider
from factorminer.agent.output_parser import candidate_pairs
from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.application.mining_budget import BudgetTracker, EvaluationResult
from factorminer.application.mining_reporting import MiningReporter
from factorminer.application.research_knowledge import ResearchKnowledgeStore
from factorminer.application.run_artifacts import MiningArtifactService
from factorminer.application.runtime_context import MiningRunContext, MiningSettings
from factorminer.application.validation_pipeline import ValidationPipeline
from factorminer.architecture import (
    DatasetContract,
    DistillStage,
    EvaluateStage,
    EvaluationKernel,
    FactorAdmissionService,
    FactorFamilyDiscovery,
    FactorLifecycleStore,
    GenerateStage,
    IterationPayload,
    LibraryGeometry,
    LibraryUpdateStage,
    PaperProtocol,
    PromptContextBuilder,
    ResearchCyclePlanner,
    RetrieveStage,
    build_memory_policy,
)
from factorminer.core.factor_library import FactorLibrary
from factorminer.core.library_io import load_library, save_library
from factorminer.core.loop_services import LoopExecutionService
from factorminer.core.provenance import infer_parent_lineage
from factorminer.core.session import MiningSession
from factorminer.memory.defaults import create_default_memory
from factorminer.memory.memory_store import ExperienceMemory
from factorminer.utils.logging import MiningSessionLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The Ralph Loop
# ---------------------------------------------------------------------------


class RalphLoop:
    """Self-Evolving Factor Discovery via the Ralph Loop paradigm.

    The Ralph Loop iteratively:
      1. Retrieves memory priors from experience memory  -- R(M, L)
      2. Generates candidate factors via LLM guided by memory  -- G(m, L)
      3. Evaluates candidates through multi-stage pipeline  -- V(alpha)
      4. Updates the factor library with admitted factors  -- L <- L + {alpha}
      5. Evolves the experience memory with new insights  -- E(M, F(M, tau))

    This implements Algorithm 1 from the FactorMiner paper.
    """

    def __init__(
        self,
        config: Any,
        data_tensor: np.ndarray,
        returns: np.ndarray,
        llm_provider: LLMProvider | None = None,
        memory: ExperienceMemory | None = None,
        library: FactorLibrary | None = None,
        checkpoint_interval: int = 1,
        run_context: MiningRunContext | None = None,
    ) -> None:
        """Initialize the Ralph Loop.

        Parameters
        ----------
        config : MiningConfig
            Mining configuration (from core.config or utils.config).
        data_tensor : np.ndarray
            Market data tensor D in R^(M x T x F).
        returns : np.ndarray
            Forward returns array R in R^(M x T).
        llm_provider : LLMProvider, optional
            LLM provider for factor generation.  Defaults to MockProvider.
        memory : ExperienceMemory, optional
            Pre-populated experience memory.  Defaults to empty memory.
        library : FactorLibrary, optional
            Pre-populated factor library.  Defaults to empty library.
        checkpoint_interval : int
            Save a checkpoint every N iterations.  Set to 0 to disable
            automatic checkpointing.  Default is 1 (every iteration).
        run_context : MiningRunContext, optional
            Per-run output and materialized target state. Hierarchical config
            remains the sole source of reusable mining settings.
        """
        self.config = config
        self.settings = MiningSettings(config, run_context)
        self.data_tensor = data_tensor
        self.returns = returns
        self.checkpoint_interval = checkpoint_interval
        self.protocol = PaperProtocol.from_config(
            config,
            benchmark_mode=self.settings.benchmark_mode,
            signal_failure_policy=self.settings.signal_failure_policy,
        )

        # Core components
        self.library = library or FactorLibrary(
            correlation_threshold=self.settings.correlation_threshold,
            ic_threshold=self.settings.ic_threshold,
            dependence_metric=self.settings.redundancy_metric,
        )
        self.geometry = LibraryGeometry(self.library)
        self.admission_service = FactorAdmissionService(self.library)
        self.family_discovery = FactorFamilyDiscovery()
        self.dataset_contract = DatasetContract.from_arrays(
            config,
            data_tensor=data_tensor,
            returns=returns,
            target_panels=dict(self.settings.target_panels or {}),
            target_horizons=dict(self.settings.target_horizons or {}),
        )
        self.memory = memory if memory is not None else create_default_memory()
        self.memory_policy = build_memory_policy(config, self.protocol, returns=returns)
        self.prompt_context_builder = PromptContextBuilder(
            self.protocol,
            family_discovery=self.family_discovery,
            cycle_planner=ResearchCyclePlanner(),
        )
        self.research_knowledge = ResearchKnowledgeStore(self.settings.output_dir)
        self.lifecycle_store = FactorLifecycleStore(self.settings.output_dir)
        self.generator = FactorGenerator(
            llm_provider=llm_provider or MockProvider(),
            prompt_builder=PromptBuilder(),
        )
        self.evaluation_kernel = EvaluationKernel(
            protocol=self.protocol,
            geometry=self.geometry,
            research_config=self.settings.research,
        )
        self.pipeline = ValidationPipeline(
            data_tensor=data_tensor,
            returns=returns,
            target_panels=dict(self.settings.target_panels or {}),
            target_horizons=dict(self.settings.target_horizons or {}),
            library=self.library,
            ic_threshold=self.settings.ic_threshold,
            icir_threshold=self.settings.icir_threshold,
            replacement_ic_min=self.settings.replacement_ic_min,
            replacement_ic_ratio=self.settings.replacement_ic_ratio,
            fast_screen_assets=self.settings.fast_screen_assets,
            num_workers=self.settings.num_workers,
            research_config=self.settings.research,
            benchmark_mode=self.settings.benchmark_mode,
            redundancy_metric=self.settings.redundancy_metric,
            evaluation_kernel=self.evaluation_kernel,
        )
        self.pipeline.signal_failure_policy = self.settings.signal_failure_policy
        self.reporter = MiningReporter(self.settings.output_dir)
        self.budget = BudgetTracker()
        self.signal_failure_policy = self.settings.signal_failure_policy
        self._loop_services = LoopExecutionService(self)
        self._artifact_service = MiningArtifactService(self)

        # Session state
        self.iteration = 0
        self._session: MiningSession | None = None
        self._session_logger: MiningSessionLogger | None = None
        self._run_manifest: dict[str, Any] = {}
        self.stages = {
            "retrieve": RetrieveStage(self._stage_retrieve),
            "generate": GenerateStage(self._stage_generate),
            "evaluate": EvaluateStage(self._stage_evaluate),
            "library_update": LibraryUpdateStage(self._stage_library_update),
            "distill": DistillStage(self._stage_distill),
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        target_size: int | None = None,
        max_iterations: int | None = None,
        callback: Callable[[int, dict[str, Any]], None] | None = None,
        resume: bool = False,
    ) -> FactorLibrary:
        """Run the complete mining loop.

        Parameters
        ----------
        target_size : int, optional
            Target library size K.  Defaults to config value (110).
        max_iterations : int, optional
            Maximum iterations before stopping.  Defaults to config value.
        callback : callable, optional
            Called after each iteration with (iteration_number, stats_dict).
        resume : bool
            If True, attempt to load the latest checkpoint from the output
            directory before starting the loop.  Default is False.

        Returns
        -------
        FactorLibrary
            The constructed factor library L.
        """
        target_size = target_size or self.settings.target_library_size
        max_iterations = max_iterations or self.settings.max_iterations
        batch_size = self.settings.batch_size
        output_dir = self.settings.output_dir

        # Resume from existing checkpoint if requested
        if resume:
            checkpoint_dir = Path(output_dir) / "checkpoint"
            if checkpoint_dir.exists():
                self.load_session(str(checkpoint_dir))
                logger.info(
                    "Resuming from iteration %d with %d factors",
                    self.iteration,
                    self.library.size,
                )

        # Initialize session
        if self._session is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._session = MiningSession(
                session_id=session_id,
                config=self._serialize_config(),
                output_dir=output_dir,
            )

        self._refresh_run_manifest(
            output_dir=output_dir,
            artifact_paths={
                "output_dir": output_dir,
                "checkpoint_dir": str(Path(output_dir) / "checkpoint"),
            },
        )
        self._run_manifest["paper_protocol"] = self.protocol.runtime_contract()
        self._run_manifest["dataset_contract"] = self.dataset_contract.to_dict()
        self._run_manifest["memory_policy"] = self.memory_policy.schema()
        self._persist_run_manifest(Path(output_dir) / "run_manifest.json")

        # Initialize session logger
        self._session_logger = MiningSessionLogger(output_dir)
        self._session_logger.log_session_start(
            {
                "target_library_size": target_size,
                "batch_size": batch_size,
                "max_iterations": max_iterations,
                "resumed_from_iteration": self.iteration if resume else 0,
            }
        )
        self._session_logger.start_progress(max_iterations)

        loop_start = time.time()

        if not hasattr(self, "budget") or self.budget is None:
            self.budget = BudgetTracker()
        self.budget.wall_start = time.time()

        try:
            while self.library.size < target_size and self.iteration < max_iterations:
                # Check budget BEFORE starting a new iteration
                if self.budget.is_exhausted():
                    logger.info("Budget exhausted — stopping loop")
                    break

                self.iteration += 1
                stats = self._run_iteration(batch_size)

                # Record in session
                self._session.record_iteration(stats)

                # Callback
                if callback:
                    callback(self.iteration, stats)

                logger.info(
                    "Iteration %d: Library size=%d, Admitted=%d, Yield=%.1f%%, AvgCorr=%.3f",
                    self.iteration,
                    stats["library_size"],
                    stats["admitted"],
                    stats["yield_rate"] * 100,
                    stats.get("avg_correlation", 0),
                )

                # Periodic checkpoint
                if self.checkpoint_interval > 0 and self.iteration % self.checkpoint_interval == 0:
                    self._checkpoint()

            if self.budget.is_exhausted():
                logger.info("Budget exhausted: %s", self.budget.to_dict())

        except KeyboardInterrupt:
            logger.warning("Mining interrupted by user at iteration %d", self.iteration)
            if self._session:
                self._session.status = "interrupted"
            # Save checkpoint on interrupt so session can be resumed
            self._checkpoint()
        finally:
            elapsed = time.time() - loop_start
            zero_admission_warning = self._loop_services.zero_admission_guidance(
                target_size=target_size,
                max_iterations=max_iterations,
            )
            if zero_admission_warning:
                logger.warning(zero_admission_warning)
            if self._session_logger:
                self._session_logger.log_session_end(self.library.size, elapsed)
            self._refresh_run_manifest(
                output_dir=output_dir,
                artifact_paths={
                    "output_dir": output_dir,
                    "checkpoint_dir": str(Path(output_dir) / "checkpoint"),
                    "library": str(Path(output_dir) / "factor_library.json"),
                    "session": str(Path(output_dir) / "session.json"),
                    "run_manifest": str(Path(output_dir) / "run_manifest.json"),
                    "session_log": str(Path(output_dir) / "session_log.json"),
                },
            )
            self._persist_run_manifest(Path(output_dir) / "run_manifest.json")
            if self._session:
                self._session.finalize()
                self._session.save()

        # Final export
        lib_path = self.reporter.export_library(self.library)
        logger.info("Factor library exported to %s", lib_path)

        return self.library

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def _run_iteration(self, batch_size: int) -> dict[str, Any]:
        """Execute one canonical iteration using the Ralph stage composition."""
        return self._loop_services.execute_iteration(batch_size)

    def _stage_retrieve(
        self,
        _loop: RalphLoop,
        payload: IterationPayload,
    ) -> dict[str, Any]:
        return self.memory_policy.retrieve(self.memory, library_state=payload.library_state)

    def _stage_generate(
        self,
        _loop: RalphLoop,
        payload: IterationPayload,
    ) -> list[tuple[str, str]]:
        payload.prompt_context = self.prompt_context_builder.build(
            payload.memory_signal,
            payload.library_state,
            batch_size=payload.batch_size,
            extras={"dataset_contract": self.dataset_contract.to_dict()},
            research_archetypes=payload.memory_signal.get("research_archetypes"),
        )
        candidates = self.generator.generate_batch(
            memory_signal=payload.prompt_context,
            library_state=payload.library_state,
            batch_size=payload.batch_size,
        )
        return candidate_pairs(candidates)

    def _stage_evaluate(
        self,
        _loop: RalphLoop,
        payload: IterationPayload,
    ) -> list[EvaluationResult]:
        results = self.pipeline.evaluate_batch(payload.candidates)
        self._annotate_result_lineage(results, payload.library_state)
        self.lifecycle_store.record_batch_results(self.iteration, results)
        return results

    def _stage_library_update(
        self,
        _loop: RalphLoop,
        payload: IterationPayload,
    ) -> list[EvaluationResult]:
        return self._update_library(payload.results)

    def _stage_distill(
        self,
        _loop: RalphLoop,
        payload: IterationPayload,
    ) -> None:
        trajectory = self._build_trajectory(payload.results)
        formed = self.memory_policy.form(self.memory, trajectory, iteration=self.iteration)
        self.memory = self.memory_policy.evolve(self.memory, formed)
        self.lifecycle_store.record_memory_distillation(self.iteration, trajectory)

    # ------------------------------------------------------------------
    # Library update
    # ------------------------------------------------------------------

    def _update_library(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        """Admit passing factors into the library and handle replacements.

        Returns the list of admitted results.
        """
        return self.admission_service.admit_results(results, iteration=self.iteration)

    # ------------------------------------------------------------------
    # Trajectory builder for memory formation
    # ------------------------------------------------------------------

    def _annotate_result_lineage(
        self,
        results: list[EvaluationResult],
        library_state: Mapping[str, Any] | None,
    ) -> None:
        """Attach parent_formula lineage onto evaluation results in-place.

        Uses the admitted-library snapshot available at generation time so
        EditAwareMemoryPolicy can observe real parent→child edges on the
        subsequent distill/form step.
        """
        library_factors = [
            {
                "id": f.id,
                "name": f.name,
                "formula": f.formula,
                "ic_paper_mean": f.ic_paper_mean,
                "ic_mean": f.ic_mean,
            }
            for f in self.library.list_factors()
        ]
        for result in results:
            if result.parent_formula:
                continue
            lineage = infer_parent_lineage(
                result.formula,
                library_state,
                library_factors=library_factors,
            )
            result.parent_formula = str(lineage.get("parent_formula", "") or "")
            parent_ic = lineage.get("parent_ic_paper_mean")
            try:
                result.parent_ic_paper_mean = (
                    float(parent_ic) if parent_ic is not None else None
                )
            except (TypeError, ValueError):
                result.parent_ic_paper_mean = None
            result.edit_type = str(lineage.get("edit_type", "") or "")
            result.edit_motif = str(lineage.get("edit_motif", "") or "")
            result.secondary_parent_formula = str(
                lineage.get("secondary_parent_formula", "") or ""
            )

    def _lineage_fields(self, result: EvaluationResult) -> dict[str, Any]:
        return {
            "parent_formula": result.parent_formula or "",
            "parent_ic_paper_mean": result.parent_ic_paper_mean,
            "edit_type": result.edit_type or "",
            "edit_motif": result.edit_motif or "",
            "secondary_parent_formula": result.secondary_parent_formula or "",
            # Canonical quality aliases used by EditAwareMemoryPolicy.
            "ic_paper_mean": result.ic_paper_mean,
            "paper_ic": result.ic_paper_mean,
        }

    def _build_trajectory(self, results: list[EvaluationResult]) -> list[dict[str, Any]]:
        """Build mining trajectory tau for memory formation.

        Converts evaluation results into the dict format expected by
        ``form_memory``. Always merges parent_formula lineage so
        ``EditAwareMemoryPolicy`` can extract edit-motif edges.
        """
        by_key = {
            (r.factor_name, r.formula): r
            for r in results
        }
        lifecycle_trajectory = self.lifecycle_store.build_trajectory(self.iteration)
        if lifecycle_trajectory:
            trajectory: list[dict[str, Any]] = []
            for entry in lifecycle_trajectory:
                merged = dict(entry)
                key = (str(entry.get("factor_id", "")), str(entry.get("formula", "")))
                result = by_key.get(key)
                if result is not None:
                    merged.update(self._lineage_fields(result))
                    if "paper_ic" not in merged:
                        merged["paper_ic"] = result.ic_paper_mean
                    if "ic_paper_mean" not in merged:
                        merged["ic_paper_mean"] = result.ic_paper_mean
                trajectory.append(merged)
            return trajectory

        trajectory = []
        for r in results:
            entry: dict[str, Any] = {
                "factor_id": r.factor_name,
                "formula": r.formula,
                "ic": r.ic_mean,
                "paper_ic": r.ic_paper_mean,
                "ic_paper_mean": r.ic_paper_mean,
                "ic_abs_mean": r.ic_abs_mean,
                "icir": r.icir,
                "paper_icir": r.ic_paper_icir,
                "max_correlation": r.max_correlation,
                "correlated_with": r.correlated_with,
                "admitted": r.admitted,
                "rejection_reason": r.rejection_reason,
            }
            entry.update(self._lineage_fields(r))
            trajectory.append(entry)
        return trajectory

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def _compute_stats(
        self,
        results: list[EvaluationResult],
        admitted: list[EvaluationResult],
        elapsed: float,
    ) -> dict[str, Any]:
        """Compute per-iteration statistics."""
        n_candidates = len(results)
        diagnostics = self.library.get_diagnostics()

        # Count dedup rejections (stage_passed==2 with dedup reason)
        dedup_rejected = sum(
            1 for r in results if not r.admitted and "deduplication" in r.rejection_reason.lower()
        )

        return {
            "iteration": self.iteration,
            "candidates": n_candidates,
            "parse_ok": sum(1 for r in results if r.parse_ok),
            "ic_passed": sum(1 for r in results if r.stage_passed >= 1),
            "corr_passed": sum(1 for r in results if r.stage_passed >= 2),
            "dedup_rejected": dedup_rejected,
            "admitted": len(admitted),
            "replaced": sum(1 for r in admitted if r.replaced is not None),
            "yield_rate": len(admitted) / max(n_candidates, 1),
            "library_size": self.library.size,
            "avg_correlation": diagnostics.get("avg_correlation", 0),
            "max_correlation": diagnostics.get("max_correlation", 0),
            "elapsed_seconds": elapsed,
            "budget": self.budget.to_dict(),
        }

    def _empty_stats(self) -> dict[str, Any]:
        """Return empty stats dict for iterations with no candidates."""
        return {
            "iteration": self.iteration,
            "candidates": 0,
            "parse_ok": 0,
            "ic_passed": 0,
            "corr_passed": 0,
            "dedup_rejected": 0,
            "admitted": 0,
            "replaced": 0,
            "yield_rate": 0.0,
            "library_size": self.library.size,
            "avg_correlation": 0.0,
            "max_correlation": 0.0,
            "elapsed_seconds": 0.0,
            "budget": self.budget.to_dict(),
        }

    # ------------------------------------------------------------------
    # Category inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_category(formula: str) -> str:
        """Infer factor category from formula structure."""
        from factorminer.architecture.families import infer_family

        return infer_family(formula)

    # ------------------------------------------------------------------
    # Session persistence (save / resume)
    # ------------------------------------------------------------------

    def save_session(self, path: str | None = None) -> str:
        """Save the full mining session state for resume.

        Saves the factor library (via ``save_library``), experience memory,
        budget tracker state, session metadata, and the loop state to a
        ``checkpoint`` directory inside the output directory.

        Parameters
        ----------
        path : str, optional
            Directory for the checkpoint.  Defaults to
            ``{output_dir}/checkpoint``.

        Returns
        -------
        str
            Path to the saved checkpoint directory.
        """
        if path is not None:
            checkpoint_dir = Path(path)
            # If caller passed a dir that doesn't end with "checkpoint*",
            # nest inside it for backward compatibility
            if not checkpoint_dir.name.startswith("checkpoint"):
                checkpoint_dir = checkpoint_dir / f"checkpoint_iter{self.iteration}"
        else:
            output_dir = self.settings.output_dir
            checkpoint_dir = Path(output_dir) / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save library using library_io (JSON + optional signal cache)
        lib_base = str(checkpoint_dir / "library")
        save_library(self.library, lib_base, save_signals=True)

        # Policy serialization is the canonical persistence path for all loops.
        mem_path = str(checkpoint_dir / "memory.json")
        with open(mem_path, "w") as f:
            json.dump(self.memory_policy.serialize(self.memory), f, indent=2, default=str)

        # Save session metadata
        if self._session:
            self._session.library_path = lib_base
            self._session.memory_path = mem_path
            self._refresh_run_manifest(
                output_dir=str(checkpoint_dir.parent),
                artifact_paths={
                    "library": f"{lib_base}.json",
                    "memory": mem_path,
                    "session": str(checkpoint_dir / "session.json"),
                    "run_manifest": str(checkpoint_dir / "run_manifest.json"),
                    "loop_state": str(checkpoint_dir / "loop_state.json"),
                },
            )
            self._persist_run_manifest(checkpoint_dir / "run_manifest.json")
            self._session.save(checkpoint_dir / "session.json")

        # Save loop state (iteration counter + budget tracker)
        loop_state: dict[str, Any] = {
            "iteration": self.iteration,
            "library_size": self.library.size,
            "memory_version": self.memory.version,
            "budget": {
                "llm_calls": self.budget.llm_calls,
                "llm_prompt_tokens": self.budget.llm_prompt_tokens,
                "llm_completion_tokens": self.budget.llm_completion_tokens,
                "compute_seconds": self.budget.compute_seconds,
                "max_llm_calls": self.budget.max_llm_calls,
                "max_wall_seconds": self.budget.max_wall_seconds,
            },
        }
        with open(checkpoint_dir / "loop_state.json", "w") as f:
            json.dump(loop_state, f, indent=2)

        logger.info("Session saved to %s", checkpoint_dir)
        return str(checkpoint_dir)

    def load_session(self, path: str) -> None:
        """Resume a mining session from a saved checkpoint.

        Restores the factor library (via ``load_library``), experience
        memory, budget tracker state, iteration counter, and session
        metadata from the checkpoint directory.

        Parameters
        ----------
        path : str
            Path to the checkpoint directory.
        """
        checkpoint_dir = Path(path)

        # Load loop state (iteration counter + budget)
        loop_state_path = checkpoint_dir / "loop_state.json"
        if loop_state_path.exists():
            with open(loop_state_path) as f:
                loop_state = json.load(f)
            self.iteration = loop_state.get("iteration", 0)

            # Restore budget tracker state
            budget_data = loop_state.get("budget", {})
            if budget_data:
                self.budget.llm_calls = budget_data.get("llm_calls", self.budget.llm_calls)
                self.budget.llm_prompt_tokens = budget_data.get(
                    "llm_prompt_tokens", self.budget.llm_prompt_tokens
                )
                self.budget.llm_completion_tokens = budget_data.get(
                    "llm_completion_tokens", self.budget.llm_completion_tokens
                )
                self.budget.compute_seconds = budget_data.get(
                    "compute_seconds", self.budget.compute_seconds
                )
                self.budget.max_llm_calls = budget_data.get(
                    "max_llm_calls", self.budget.max_llm_calls
                )
                self.budget.max_wall_seconds = budget_data.get(
                    "max_wall_seconds", self.budget.max_wall_seconds
                )

            logger.info(
                "Resuming from iteration %d (library=%d)",
                self.iteration,
                loop_state.get("library_size", 0),
            )

        # Load memory
        mem_path = checkpoint_dir / "memory.json"
        if mem_path.exists():
            with open(mem_path) as f:
                mem_data = json.load(f)
            self.memory = self.memory_policy.restore(mem_data)
            logger.info(
                "Loaded memory (version=%d, %d success, %d forbidden, %d insights)",
                self.memory.version,
                len(self.memory.success_patterns),
                len(self.memory.forbidden_directions),
                len(self.memory.insights),
            )

        # Load library using library_io (supports signals + correlation matrix)
        lib_json_path = checkpoint_dir / "library.json"
        if lib_json_path.exists():
            lib_base = str(checkpoint_dir / "library")
            loaded_library = load_library(lib_base)
            # Merge into current library (preserving thresholds from config)
            self.library.factors = loaded_library.factors
            self.library._next_id = loaded_library._next_id
            self.library._id_to_index = loaded_library._id_to_index
            self.library.correlation_matrix = loaded_library.correlation_matrix
            # Update the pipeline reference so it uses the restored library
            self.pipeline.library = self.library
            logger.info("Loaded library with %d factors", self.library.size)

        # Load session metadata
        session_path = checkpoint_dir / "session.json"
        if session_path.exists():
            self._session = MiningSession.load(session_path)
            self._session.status = "running"
            self._run_manifest = dict(self._session.run_manifest or {})

        if not self._run_manifest:
            run_manifest_path = checkpoint_dir / "run_manifest.json"
            if run_manifest_path.exists():
                with open(run_manifest_path) as f:
                    self._run_manifest = json.load(f)

    @classmethod
    def resume_from(
        cls,
        checkpoint_path: str,
        config: Any,
        data_tensor: np.ndarray,
        returns: np.ndarray,
        llm_provider: LLMProvider | None = None,
        **kwargs: Any,
    ) -> RalphLoop:
        """Create a RalphLoop and restore state from a checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint directory.
        config, data_tensor, returns, llm_provider
            Same as ``__init__``.

        Returns
        -------
        RalphLoop
            A loop ready to call ``run()`` that continues from the checkpoint.
        """
        loop = cls(
            config=config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=llm_provider,
            **kwargs,
        )
        loop.load_session(checkpoint_path)
        return loop

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _checkpoint(self) -> None:
        """Save a periodic checkpoint."""
        try:
            self.save_session()
        except Exception as exc:
            logger.warning("Checkpoint failed: %s", exc)

    def _serialize_config(self) -> dict[str, Any]:
        """Serialize config to a JSON-compatible dict."""
        try:
            if hasattr(self.config, "to_dict"):
                return self.config.to_dict()
            return asdict(self.config)
        except (TypeError, AttributeError):
            # Fallback: extract known attributes
            attrs = [
                "target_library_size",
                "batch_size",
                "max_iterations",
                "ic_threshold",
                "icir_threshold",
                "correlation_threshold",
                "replacement_ic_min",
                "replacement_ic_ratio",
                "output_dir",
            ]
            return {
                attr: getattr(self.config, attr, None)
                for attr in attrs
                if getattr(self.config, attr, None) is not None
            }

    def _loop_type(self) -> str:
        """Label the loop for provenance and manifests."""
        return "ralph"

    def _phase2_features(self) -> list[str]:
        """Phase 2 feature flags used by the current loop."""
        return []

    def _refresh_run_manifest(
        self,
        *,
        output_dir: str,
        artifact_paths: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return self._artifact_service.refresh_manifest(
            output_dir=output_dir,
            artifact_paths=artifact_paths,
        )

    def _persist_run_manifest(self, path: Path) -> None:
        self._artifact_service.persist_manifest(path)

    def _attach_factor_provenance(
        self,
        admitted_results: list[EvaluationResult],
        *,
        library_state: dict[str, Any],
        memory_signal: dict[str, Any],
        phase2_summary: dict[str, Any],
        generator_family: str | None = None,
    ) -> None:
        self._artifact_service.attach_factor_provenance(
            admitted_results,
            library_state=library_state,
            memory_signal=memory_signal,
            phase2_summary=phase2_summary,
            generator_family=generator_family,
        )

    def _generator_family(self) -> str:
        """Return the active candidate generator label for provenance."""
        return self.generator.__class__.__name__
