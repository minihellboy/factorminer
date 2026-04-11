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
import re
import time
from collections.abc import Callable
from concurrent.futures import as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from factorminer.agent.llm_interface import LLMProvider, MockProvider
from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.architecture import (
    DatasetContract,
    DistillStage,
    EvaluateStage,
    EvaluationKernel,
    FactorFamilyDiscovery,
    FactorAdmissionService,
    FactorLifecycleStore,
    GenerateStage,
    IterationPayload,
    LibraryGeometry,
    LibraryUpdateStage,
    PaperProtocol,
    PromptContextBuilder,
    RetrieveStage,
    build_memory_policy,
)
from factorminer.core.factor_library import FactorLibrary
from factorminer.core.library_io import load_library, save_library
from factorminer.core.loop_services import LoopExecutionService
from factorminer.core.provenance import build_factor_provenance, build_run_manifest
from factorminer.core.session import MiningSession
from factorminer.core.types import FEATURES
from factorminer.evaluation.metrics import (
    compute_factor_stats,
)
from factorminer.evaluation.runtime import SignalComputationError, compute_tree_signals
from factorminer.memory.experience_memory import ExperienceMemoryManager
from factorminer.memory.memory_store import ExperienceMemory
from factorminer.utils.logging import MiningSessionLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
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
    icir: float = 0.0
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


# ---------------------------------------------------------------------------
# Factor Generator: wraps LLM + prompt builder + output parser
# ---------------------------------------------------------------------------


class FactorGenerator:
    """Generates candidate factors using LLM guided by memory priors."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.llm = llm_provider or MockProvider()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate_batch(
        self,
        memory_signal: dict[str, Any],
        library_state: dict[str, Any],
        batch_size: int = 40,
    ) -> list[tuple[str, str]]:
        """Generate a batch of candidate factors.

        Returns
        -------
        list of (name, formula) tuples
        """
        user_prompt = self.prompt_builder.build_user_prompt(
            memory_signal, library_state, batch_size
        )
        raw_response = self.llm.generate(
            system_prompt=self.prompt_builder.system_prompt,
            user_prompt=user_prompt,
        )
        return self._parse_response(raw_response)

    @staticmethod
    def _parse_response(raw: str) -> list[tuple[str, str]]:
        """Parse LLM output into (name, formula) pairs.

        Expected format per line:
            <number>. <name>: <formula>
        """
        candidates: list[tuple[str, str]] = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Match patterns like "1. factor_name: Formula(...)"
            m = re.match(
                r"^\d+\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$",
                line,
            )
            if m:
                name = m.group(1).strip()
                formula = m.group(2).strip()
                candidates.append((name, formula))
        return candidates


# ---------------------------------------------------------------------------
# Validation Pipeline (lightweight orchestrator)
# ---------------------------------------------------------------------------


class ValidationPipeline:
    """Multi-stage evaluation pipeline for candidate factors.

    Implements the full 4-stage evaluation from the paper:
      Stage 1: Fast IC screening on M_fast assets  -> C1
      Stage 2: Correlation check against library L  -> C2 (+ replacement for C1\\C2)
      Stage 3: Intra-batch deduplication (pairwise rho < theta)  -> C3
      Stage 4: Full validation on M_full assets + trajectory collection
    """

    def __init__(
        self,
        data_tensor: np.ndarray,
        returns: np.ndarray,
        target_panels: dict[str, np.ndarray] | None = None,
        target_horizons: dict[str, int] | None = None,
        library: FactorLibrary | None = None,
        ic_threshold: float = 0.04,
        icir_threshold: float = 0.5,
        replacement_ic_min: float = 0.10,
        replacement_ic_ratio: float = 1.3,
        fast_screen_assets: int = 100,
        num_workers: int = 1,
        research_config: Any = None,
        benchmark_mode: str = "paper",
        redundancy_metric: str = "spearman",
        evaluation_kernel: EvaluationKernel | None = None,
    ) -> None:
        self.data_tensor = data_tensor  # (M, T, F)
        self.returns = returns  # (M, T)
        self.target_panels = target_panels or {"paper": returns}
        self.target_horizons = target_horizons or {"paper": 1}
        self.library = library or FactorLibrary(
            correlation_threshold=0.5,
            ic_threshold=ic_threshold,
            dependence_metric=redundancy_metric,
        )
        self.ic_threshold = ic_threshold
        self.icir_threshold = icir_threshold
        self.replacement_ic_min = replacement_ic_min
        self.replacement_ic_ratio = replacement_ic_ratio
        self.fast_screen_assets = fast_screen_assets
        self.num_workers = num_workers
        self.signal_failure_policy = "reject"
        self.research_config = research_config
        self.benchmark_mode = benchmark_mode
        protocol_cfg = type("ProtocolCfg", (), {})()
        protocol_cfg.mining = type(
            "MiningCfg",
            (),
            {
                "ic_threshold": ic_threshold,
                "icir_threshold": icir_threshold,
                "correlation_threshold": self.library.correlation_threshold,
                "replacement_ic_min": replacement_ic_min,
                "replacement_ic_ratio": replacement_ic_ratio,
            },
        )()
        protocol_cfg.data = type("DataCfg", (), {"default_target": "paper", "targets": []})()
        protocol_cfg.benchmark = type(
            "BenchCfg",
            (),
            {
                "mode": benchmark_mode,
                "freeze_top_k": 40,
                "freeze_universe": "CSI500",
                "report_universes": [],
            },
        )()
        protocol_cfg.evaluation = type(
            "EvalCfg",
            (),
            {
                "backend": "numpy",
                "redundancy_metric": redundancy_metric,
                "signal_failure_policy": self.signal_failure_policy,
            },
        )()
        self.geometry = LibraryGeometry(self.library)
        self.kernel = evaluation_kernel or EvaluationKernel(
            protocol=PaperProtocol.from_config(protocol_cfg),
            geometry=self.geometry,
            research_config=research_config,
        )

        # Pre-compute the fast-screen asset subset indices
        M = returns.shape[0]
        if fast_screen_assets > 0 and fast_screen_assets < M:
            rng = np.random.RandomState(0)
            self._fast_indices = rng.choice(M, fast_screen_assets, replace=False)
            self._fast_indices.sort()
        else:
            self._fast_indices = np.arange(M)

    def evaluate_candidate(
        self,
        name: str,
        formula: str,
        fast_screen: bool = True,
    ) -> EvaluationResult:
        """Evaluate a single candidate through the full pipeline.

        Parameters
        ----------
        name : str
            Candidate factor name.
        formula : str
            DSL formula string.
        fast_screen : bool
            If True, Stage 1 uses M_fast assets only.  If False, uses all.
        """
        result = EvaluationResult(factor_name=name, formula=formula)

        try:
            _tree, signals = self.kernel.compute_signals(
                formula=formula,
                data_dict=self._build_data_dict(),
                returns_shape=self.returns.shape,
                signal_failure_policy=self.signal_failure_policy,
            )
        except SignalComputationError as exc:
            if "Parse failure" in str(exc):
                result.rejection_reason = "Parse failure"
            else:
                result.rejection_reason = f"Signal computation error: {exc}"
            result.stage_passed = 0
            return result
        result.parse_ok = True

        if signals is None or np.all(np.isnan(signals)):
            result.rejection_reason = "All-NaN signals"
            result.stage_passed = 0
            return result

        result.signals = signals

        # Fast IC screen on M_fast asset subset
        if fast_screen and len(self._fast_indices) < signals.shape[0]:
            fast_signals = signals[self._fast_indices, :]
            fast_returns = self.returns[self._fast_indices, :]
            fast_stats = compute_factor_stats(fast_signals, fast_returns)
            fast_ic = fast_stats["ic_abs_mean"]

            if fast_ic < self.ic_threshold:
                result.ic_mean = fast_ic
                result.rejection_reason = (
                    f"Fast-screen IC {fast_ic:.4f} < threshold {self.ic_threshold}"
                )
                result.stage_passed = 0
                return result

        # Full IC statistics on all assets
        result.target_stats = self.kernel.compute_target_stats(
            signals,
            self.returns,
            self.target_panels,
        )
        paper_stats = result.target_stats["paper"]
        result.ic_mean = paper_stats["ic_abs_mean"]
        result.icir = paper_stats["icir"]
        result.ic_win_rate = paper_stats["ic_win_rate"]

        quality = self.kernel.compute_quality_score(
            signals=signals,
            returns=self.returns,
            target_stats=result.target_stats,
            library_signals=[
                factor.signals
                for factor in self.library.list_factors()
                if factor.signals is not None
            ],
            target_horizons=self.target_horizons,
            benchmark_mode=self.benchmark_mode,
        )
        result.research_score = float(quality["research_score"])
        result.score_vector = quality["score_vector"]
        result.max_correlation = float(quality["max_correlation"])
        if result.score_vector:
            result.research_lcb = result.score_vector["lower_confidence_bound"]
            result.residual_ic = result.score_vector["geometry"]["residual_ic"]
            result.projection_loss = result.score_vector["geometry"]["projection_loss"]
            result.effective_rank_gain = result.score_vector["geometry"]["effective_rank_gain"]

        # Stage 1 gate: IC threshold (full data)
        quality_gate = result.ic_mean
        quality_label = "IC"
        if self._research_enabled():
            quality_gate = result.research_score
            quality_label = "Research score"

        if quality_gate < self.ic_threshold:
            result.rejection_reason = (
                f"{quality_label} {quality_gate:.4f} < threshold {self.ic_threshold}"
            )
            result.stage_passed = 0
            return result
        if result.icir < self.icir_threshold:
            result.rejection_reason = f"ICIR {result.icir:.4f} < threshold {self.icir_threshold}"
            result.stage_passed = 0
            return result
        result.stage_passed = 1

        if self._research_enabled():
            admitted = bool(quality["admitted"])
            reason = str(quality["admission_reason"])
            if admitted:
                result.admitted = True
                result.stage_passed = 3
                return result
            result.stage_passed = 2
            result.rejection_reason = reason
            replace_id, replace_reason = self._research_replacement(result)
            if replace_id is not None:
                result.admitted = True
                result.replaced = replace_id
                result.rejection_reason = replace_reason
                result.stage_passed = 3
            return result

        # Stage 2: Correlation check against library (admission)
        admitted, reason = self.kernel.admission_decision(result.ic_mean, signals)
        if admitted:
            result.admitted = True
            result.stage_passed = 3
            if self.library.size > 0:
                result.max_correlation = self.geometry.candidate_geometry(signals).max_dependence
            return result

        result.stage_passed = 2

        # Stage 2.5: Replacement check for candidates that failed admission
        should_replace, replace_id, replace_reason = self.kernel.replacement_decision(
            result.ic_mean,
            signals,
        )
        if should_replace and replace_id is not None:
            result.admitted = True
            result.replaced = replace_id
            result.max_correlation = self.geometry.candidate_geometry(signals).max_dependence
            result.stage_passed = 3
            return result

        # Rejected by correlation
        result.rejection_reason = reason
        if self.library.size > 0:
            result.max_correlation = self.geometry.candidate_geometry(signals).max_dependence
        return result

    def _research_enabled(self) -> bool:
        return bool(
            self.research_config is not None
            and getattr(self.research_config, "enabled", False)
            and self.benchmark_mode == "research"
        )

    def _research_replacement(self, result: EvaluationResult) -> tuple[int | None, str]:
        if result.score_vector is None or self.library.size == 0:
            return None, result.rejection_reason

        conflicting: list[tuple[int, float]] = []
        for factor in self.library.list_factors():
            if factor.signals is None:
                continue
            corr = self.library.compute_correlation(result.signals, factor.signals)
            if corr >= self.library.correlation_threshold:
                conflicting.append((factor.id, corr))
        if len(conflicting) != 1:
            return None, result.rejection_reason

        target_id, _ = conflicting[0]
        target_factor = self.library.get_factor(target_id)
        target_score = float(
            target_factor.research_metrics.get("primary_score", target_factor.ic_mean)
        )
        if result.research_score < max(
            self.replacement_ic_min, self.replacement_ic_ratio * target_score
        ):
            return None, (
                f"Research replacement score {result.research_score:.4f} "
                f"not strong enough to replace factor {target_id} ({target_score:.4f})"
            )
        return target_id, f"Research replacement over factor {target_id}"

    def evaluate_batch(self, candidates: list[tuple[str, str]]) -> list[EvaluationResult]:
        """Evaluate a batch through all stages including intra-batch dedup.

        Stage 1-2.5 are run per-candidate (optionally in parallel).
        Stage 3 (dedup) runs on all admitted candidates together.
        """
        # Stage 1 + 2 + 2.5: per-candidate evaluation
        if self.num_workers > 1:
            results = self._evaluate_parallel(candidates)
        else:
            results = []
            for name, formula in candidates:
                result = self.evaluate_candidate(name, formula)
                results.append(result)

        # Stage 3: Intra-batch deduplication
        results = self._deduplicate_batch(results)

        return results

    def _evaluate_parallel(self, candidates: list[tuple[str, str]]) -> list[EvaluationResult]:
        """Evaluate candidates using a thread pool.

        Note: uses threads rather than processes because signals arrays
        are large and sharing via processes would require serialization.
        """
        from concurrent.futures import ThreadPoolExecutor

        results: list[EvaluationResult | None] = [None] * len(candidates)

        def _eval(idx: int, name: str, formula: str) -> tuple[int, EvaluationResult]:
            return idx, self.evaluate_candidate(name, formula)

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [
                pool.submit(_eval, i, name, formula) for i, (name, formula) in enumerate(candidates)
            ]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return [r for r in results if r is not None]

    def _deduplicate_batch(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        """Stage 3: Remove intra-batch duplicates among admitted candidates.

        For candidates that passed Stages 1-2, check pairwise correlation
        within the batch.  If two admitted candidates are correlated above
        theta, keep the one with higher IC and reject the other.
        """
        before = sum(1 for result in results if result.admitted and result.signals is not None)
        quality_attr = "research_score" if self._research_enabled() else "ic_mean"
        results = self.kernel.deduplicate_results(results, quality_attr=quality_attr)
        dedup_rejected = before - sum(
            1 for result in results if result.admitted and result.signals is not None
        )
        if dedup_rejected > 0:
            logger.debug(
                "Intra-batch dedup: rejected %d/%d admitted candidates",
                dedup_rejected,
                before,
            )

        return results

    def _build_data_dict(self) -> dict[str, np.ndarray]:
        """Convert data_tensor to a dict mapping feature names to (M, T) arrays.

        Handles two formats:
          - dict: already maps ``"$close"`` etc. to ``(M, T)`` arrays.
          - np.ndarray of shape ``(M, T, F)``: sliced along the last axis
            using the canonical ``FEATURES`` ordering.
        """
        if isinstance(self.data_tensor, dict):
            return self.data_tensor

        # (M, T, F) numpy array — map each feature slice
        data_dict: dict[str, np.ndarray] = {}
        n_features = self.data_tensor.shape[2] if self.data_tensor.ndim == 3 else 0
        for i, feat_name in enumerate(FEATURES):
            if i < n_features:
                data_dict[feat_name] = self.data_tensor[:, :, i]
        return data_dict

    def _compute_signals(self, tree) -> np.ndarray | None:
        """Compute factor signals from expression tree on the data tensor.

        Evaluates the parsed expression tree against the market data using
        the tree's own ``evaluate()`` method which dispatches through the
        numpy operator implementations under the configured failure policy.
        """
        data_dict = self._build_data_dict()
        return compute_tree_signals(
            tree,
            data_dict,
            self.returns.shape,
            signal_failure_policy=self.signal_failure_policy,
        )


# ---------------------------------------------------------------------------
# Mining Reporter
# ---------------------------------------------------------------------------


class MiningReporter:
    """Lightweight reporter that logs batch results to a JSONL file."""

    def __init__(self, output_dir: str = "./output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.output_dir / "mining_batches.jsonl"

    def log_batch(self, iteration: int, **stats: Any) -> None:
        """Append a batch record to the JSONL log."""
        record = {"iteration": iteration, "timestamp": time.time()}
        record.update(stats)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def export_library(self, library: FactorLibrary, path: str | None = None) -> str:
        """Export the factor library to JSON."""
        if path is None:
            path = str(self.output_dir / "factor_library.json")
        factors = [f.to_dict() for f in library.list_factors()]
        diagnostics = library.get_diagnostics()
        payload = {
            "factors": factors,
            "diagnostics": diagnostics,
            "exported_at": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return path


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
        """
        self.config = config
        self.data_tensor = data_tensor
        self.returns = returns
        self.checkpoint_interval = checkpoint_interval
        self.protocol = PaperProtocol.from_config(config)

        # Core components
        self.library = library or FactorLibrary(
            correlation_threshold=getattr(config, "correlation_threshold", 0.5),
            ic_threshold=getattr(config, "ic_threshold", 0.04),
            dependence_metric=getattr(config, "redundancy_metric", "spearman"),
        )
        self.geometry = LibraryGeometry(self.library)
        self.admission_service = FactorAdmissionService(self.library)
        self.family_discovery = FactorFamilyDiscovery()
        self.dataset_contract = DatasetContract.from_arrays(
            config,
            data_tensor=data_tensor,
            returns=returns,
            target_panels=getattr(config, "target_panels", None),
            target_horizons=getattr(config, "target_horizons", None),
        )
        self.memory = memory or ExperienceMemory()
        self.memory_policy = build_memory_policy(config, self.protocol, returns=returns)
        self.prompt_context_builder = PromptContextBuilder(
            self.protocol,
            family_discovery=self.family_discovery,
        )
        self.lifecycle_store = FactorLifecycleStore(getattr(config, "output_dir", "./output"))
        self.memory_manager: ExperienceMemoryManager | None = None
        self.generator = FactorGenerator(
            llm_provider=llm_provider,
            prompt_builder=PromptBuilder(),
        )
        self.evaluation_kernel = EvaluationKernel(
            protocol=self.protocol,
            geometry=self.geometry,
            research_config=getattr(config, "research", None),
        )
        self.pipeline = ValidationPipeline(
            data_tensor=data_tensor,
            returns=returns,
            target_panels=getattr(config, "target_panels", None),
            target_horizons=getattr(config, "target_horizons", None),
            library=self.library,
            ic_threshold=getattr(config, "ic_threshold", 0.04),
            icir_threshold=getattr(config, "icir_threshold", 0.5),
            replacement_ic_min=getattr(config, "replacement_ic_min", 0.10),
            replacement_ic_ratio=getattr(config, "replacement_ic_ratio", 1.3),
            fast_screen_assets=getattr(config, "fast_screen_assets", 100),
            num_workers=getattr(config, "num_workers", 1),
            research_config=getattr(config, "research", None),
            benchmark_mode=getattr(config, "benchmark_mode", "paper"),
            redundancy_metric=getattr(config, "redundancy_metric", "spearman"),
            evaluation_kernel=self.evaluation_kernel,
        )
        self.pipeline.signal_failure_policy = getattr(config, "signal_failure_policy", "reject")
        self.reporter = MiningReporter(getattr(config, "output_dir", "./output"))
        self.budget = BudgetTracker()
        self.signal_failure_policy = getattr(config, "signal_failure_policy", "reject")
        self._loop_services = LoopExecutionService(self)

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
        target_size = target_size or getattr(self.config, "target_library_size", 110)
        max_iterations = max_iterations or getattr(self.config, "max_iterations", 200)
        batch_size = getattr(self.config, "batch_size", 40)
        output_dir = getattr(self.config, "output_dir", "./output")

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
        """Execute one iteration of the Ralph Loop.

        Returns
        -------
        dict
            Iteration statistics.
        """
        t0 = time.time()
        payload = self._loop_services.new_payload(batch_size)
        self._loop_services.run_stage_chain(payload, ("retrieve", "generate"))
        self.budget.record_llm_call()

        if not payload.candidates:
            logger.warning("Iteration %d: generator produced 0 candidates", self.iteration)
            return self._loop_services.empty_stats()

        self._loop_services.run_stage_chain(payload, ("evaluate", "library_update"))

        provenance_library_state = {
            **payload.library_state,
            "diagnostics": self.library.get_diagnostics(),
        }

        self._attach_factor_provenance(
            payload.admitted_results,
            library_state=provenance_library_state,
            memory_signal=payload.memory_signal,
            phase2_summary={},
            generator_family=self._generator_family(),
        )

        self._loop_services.run_stage_chain(payload, ("distill",))

        # Build stats
        elapsed = time.time() - t0
        self.budget.record_compute(elapsed)
        stats = self._loop_services.build_stats(payload, elapsed)
        telemetry = self._loop_services.build_telemetry(
            payload,
            stats,
            elapsed,
            candidates_generated=len(payload.candidates),
        )
        self._loop_services.log_telemetry(telemetry)

        return stats

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
        )
        return self.generator.generate_batch(
            memory_signal=payload.prompt_context,
            library_state=payload.library_state,
            batch_size=payload.batch_size,
        )

    def _stage_evaluate(
        self,
        _loop: RalphLoop,
        payload: IterationPayload,
    ) -> list[EvaluationResult]:
        results = self.pipeline.evaluate_batch(payload.candidates)
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

    def _build_trajectory(self, results: list[EvaluationResult]) -> list[dict[str, Any]]:
        """Build mining trajectory tau for memory formation.

        Converts evaluation results into the dict format expected by
        ``form_memory``.
        """
        lifecycle_trajectory = self.lifecycle_store.build_trajectory(self.iteration)
        if lifecycle_trajectory:
            return lifecycle_trajectory

        trajectory: list[dict[str, Any]] = []
        for r in results:
            entry: dict[str, Any] = {
                "factor_id": r.factor_name,
                "formula": r.formula,
                "ic": r.ic_mean,
                "icir": r.icir,
                "max_correlation": r.max_correlation,
                "correlated_with": r.correlated_with,
                "admitted": r.admitted,
                "rejection_reason": r.rejection_reason,
            }
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
            output_dir = getattr(self.config, "output_dir", "./output")
            checkpoint_dir = Path(output_dir) / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save library using library_io (JSON + optional signal cache)
        lib_base = str(checkpoint_dir / "library")
        save_library(self.library, lib_base, save_signals=True)

        # Save memory using ExperienceMemoryManager if available,
        # otherwise fall back to raw ExperienceMemory serialization
        mem_path = str(checkpoint_dir / "memory.json")
        if self.memory_manager is not None:
            self.memory_manager.save(mem_path)
        else:
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
            if self.memory_manager is not None:
                self.memory_manager.load(mem_path)
                self.memory = self.memory_manager.memory
            else:
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
        """Build and cache the current run manifest."""
        if self._session is None:
            return {}

        config_summary = self._serialize_config()
        dataset_summary = {
            "data_tensor_shape": list(self.data_tensor.shape),
            "returns_shape": list(self.returns.shape),
            "memory_version": self.memory.version,
            "library_size": self.library.size,
            "library_diagnostics": self.library.get_diagnostics(),
        }
        if isinstance(self.config, dict):
            benchmark_mode = str(self.config.get("benchmark_mode", "paper"))
            target_stack = list(self.config.get("target_stack", []))
        else:
            benchmark_mode = str(getattr(self.config, "benchmark_mode", "paper"))
            target_stack = list(getattr(self.config, "target_stack", []) or [])

        pipeline_targets = getattr(self.pipeline, "target_panels", None) or {}
        if pipeline_targets:
            target_stack = [
                name for name in pipeline_targets.keys() if name and name != "paper"
            ] or target_stack

        manifest = build_run_manifest(
            run_id=self._session.session_id,
            session_id=self._session.session_id,
            loop_type=self._loop_type(),
            benchmark_mode=benchmark_mode,
            created_at=self._session.start_time,
            updated_at=datetime.now().isoformat(),
            iteration=self.iteration,
            library_size=self.library.size,
            output_dir=output_dir,
            config_summary=config_summary,
            dataset_summary=dataset_summary,
            phase2_features=self._phase2_features(),
            target_stack=target_stack,
            artifact_paths=artifact_paths or {},
            notes=[],
        )
        self._run_manifest = manifest.to_dict()
        return self._run_manifest

    def _persist_run_manifest(self, path: Path) -> None:
        """Write the current run manifest to disk and mirror it into the session."""
        if self._session is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._run_manifest:
            self._refresh_run_manifest(
                output_dir=str(path.parent.parent),
                artifact_paths={"run_manifest": str(path)},
            )
        self._run_manifest.setdefault("artifact_paths", {})["run_manifest"] = str(path)
        with open(path, "w") as f:
            json.dump(self._run_manifest, f, indent=2, default=str)

        self._session.run_manifest_path = str(path)
        self._session.run_manifest = self._run_manifest

    def _attach_factor_provenance(
        self,
        admitted_results: list[EvaluationResult],
        *,
        library_state: dict[str, Any],
        memory_signal: dict[str, Any],
        phase2_summary: dict[str, Any],
        generator_family: str | None = None,
    ) -> None:
        """Stamp provenance onto library factors that survived admission."""
        if not admitted_results or self._session is None:
            return

        run_manifest = self._run_manifest or self._refresh_run_manifest(
            output_dir=getattr(self.config, "output_dir", "./output"),
            artifact_paths={},
        )

        for rank, result in enumerate(admitted_results, start=1):
            if not result.admitted:
                continue

            factor = None
            for candidate in reversed(self.library.list_factors()):
                if candidate.name == result.factor_name and candidate.formula == result.formula:
                    factor = candidate
                    break
            if factor is None:
                continue

            factor.provenance = build_factor_provenance(
                run_manifest=run_manifest,
                factor_name=factor.name,
                formula=factor.formula,
                factor_category=factor.category,
                factor_id=factor.id,
                iteration=self.iteration,
                batch_number=factor.batch_number,
                candidate_rank=rank,
                generator_family=generator_family or self._generator_family(),
                memory_signal=memory_signal,
                library_state=library_state,
                evaluation={
                    "ic_mean": factor.ic_mean,
                    "icir": factor.icir,
                    "ic_win_rate": factor.ic_win_rate,
                    "max_correlation": factor.max_correlation,
                    "research_metrics": factor.research_metrics,
                },
                admission={
                    "admitted": True,
                    "stage_passed": result.stage_passed,
                    "replaced": result.replaced,
                    "correlated_with": result.correlated_with,
                    "rejection_reason": result.rejection_reason,
                },
                phase2=phase2_summary,
                target_stack=run_manifest.get("target_stack", []),
                research_metrics=factor.research_metrics,
            ).to_dict()

    def _generator_family(self) -> str:
        """Return the active candidate generator label for provenance."""
        return self.generator.__class__.__name__
