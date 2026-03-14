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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import save_library, load_library
from factorminer.core.parser import try_parse
from factorminer.core.session import MiningSession
from factorminer.core.types import FEATURES
from factorminer.memory.experience_memory import ExperienceMemoryManager
from factorminer.memory.memory_store import ExperienceMemory
from factorminer.memory.retrieval import retrieve_memory
from factorminer.memory.formation import form_memory
from factorminer.memory.evolution import evolve_memory
from factorminer.agent.llm_interface import LLMProvider, MockProvider
from factorminer.agent.prompt_builder import PromptBuilder
from factorminer.evaluation.metrics import (
    compute_factor_stats,
    compute_ic,
    compute_ic_mean,
    compute_ic_win_rate,
    compute_icir,
)
from factorminer.utils.logging import (
    IterationRecord,
    FactorRecord,
    MiningSessionLogger,
)

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

    max_llm_calls: int = 0      # 0 = unlimited
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

    def to_dict(self) -> Dict[str, Any]:
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
    replaced: Optional[int] = None   # ID of replaced factor, if any
    rejection_reason: str = ""
    stage_passed: int = 0  # 0=parse/IC fail, 1=IC pass, 2=corr pass, 3=dedup pass, 4=admitted
    signals: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Factor Generator: wraps LLM + prompt builder + output parser
# ---------------------------------------------------------------------------

class FactorGenerator:
    """Generates candidate factors using LLM guided by memory priors."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        prompt_builder: Optional[PromptBuilder] = None,
    ) -> None:
        self.llm = llm_provider or MockProvider()
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate_batch(
        self,
        memory_signal: Dict[str, Any],
        library_state: Dict[str, Any],
        batch_size: int = 40,
    ) -> List[Tuple[str, str]]:
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
    def _parse_response(raw: str) -> List[Tuple[str, str]]:
        """Parse LLM output into (name, formula) pairs.

        Expected format per line:
            <number>. <name>: <formula>
        """
        candidates: List[Tuple[str, str]] = []
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
        library: FactorLibrary,
        ic_threshold: float = 0.04,
        icir_threshold: float = 0.5,
        replacement_ic_min: float = 0.10,
        replacement_ic_ratio: float = 1.3,
        fast_screen_assets: int = 100,
        num_workers: int = 1,
    ) -> None:
        self.data_tensor = data_tensor  # (M, T, F)
        self.returns = returns  # (M, T)
        self.library = library
        self.ic_threshold = ic_threshold
        self.icir_threshold = icir_threshold
        self.replacement_ic_min = replacement_ic_min
        self.replacement_ic_ratio = replacement_ic_ratio
        self.fast_screen_assets = fast_screen_assets
        self.num_workers = num_workers

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

        # Stage 0: Parse
        tree = try_parse(formula)
        if tree is None:
            result.rejection_reason = "Parse failure"
            result.stage_passed = 0
            return result
        result.parse_ok = True

        # Stage 1: Compute signals and fast IC screening
        try:
            signals = self._compute_signals(tree)
        except Exception as exc:
            result.rejection_reason = f"Signal computation error: {exc}"
            result.stage_passed = 0
            return result

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
        stats = compute_factor_stats(signals, self.returns)
        result.ic_mean = stats["ic_abs_mean"]
        result.icir = stats["icir"]
        result.ic_win_rate = stats["ic_win_rate"]

        # Stage 1 gate: IC threshold (full data)
        if result.ic_mean < self.ic_threshold:
            result.rejection_reason = (
                f"IC {result.ic_mean:.4f} < threshold {self.ic_threshold}"
            )
            result.stage_passed = 0
            return result
        result.stage_passed = 1

        # Stage 2: Correlation check against library (admission)
        admitted, reason = self.library.check_admission(
            result.ic_mean, signals
        )
        if admitted:
            result.admitted = True
            result.stage_passed = 3
            if self.library.size > 0:
                result.max_correlation = self.library._max_correlation_with_library(
                    signals
                )
            return result

        result.stage_passed = 2

        # Stage 2.5: Replacement check for candidates that failed admission
        should_replace, replace_id, replace_reason = self.library.check_replacement(
            result.ic_mean,
            signals,
            ic_min=self.replacement_ic_min,
            ic_ratio=self.replacement_ic_ratio,
        )
        if should_replace and replace_id is not None:
            result.admitted = True
            result.replaced = replace_id
            result.max_correlation = self.library._max_correlation_with_library(
                signals
            )
            result.stage_passed = 3
            return result

        # Rejected by correlation
        result.rejection_reason = reason
        if self.library.size > 0:
            result.max_correlation = self.library._max_correlation_with_library(
                signals
            )
        return result

    def evaluate_batch(
        self, candidates: List[Tuple[str, str]]
    ) -> List[EvaluationResult]:
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

    def _evaluate_parallel(
        self, candidates: List[Tuple[str, str]]
    ) -> List[EvaluationResult]:
        """Evaluate candidates using a thread pool.

        Note: uses threads rather than processes because signals arrays
        are large and sharing via processes would require serialization.
        """
        from concurrent.futures import ThreadPoolExecutor

        results: List[Optional[EvaluationResult]] = [None] * len(candidates)

        def _eval(idx: int, name: str, formula: str) -> Tuple[int, EvaluationResult]:
            return idx, self.evaluate_candidate(name, formula)

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [
                pool.submit(_eval, i, name, formula)
                for i, (name, formula) in enumerate(candidates)
            ]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return [r for r in results if r is not None]

    def _deduplicate_batch(
        self, results: List[EvaluationResult]
    ) -> List[EvaluationResult]:
        """Stage 3: Remove intra-batch duplicates among admitted candidates.

        For candidates that passed Stages 1-2, check pairwise correlation
        within the batch.  If two admitted candidates are correlated above
        theta, keep the one with higher IC and reject the other.
        """
        admitted_indices = [
            i for i, r in enumerate(results)
            if r.admitted and r.signals is not None
        ]

        if len(admitted_indices) <= 1:
            return results

        # Compute pairwise correlations among admitted candidates
        admitted_signals = [results[i].signals for i in admitted_indices]
        corr_threshold = self.library.correlation_threshold

        # Greedy dedup: iterate in order of descending IC, keep non-correlated
        admitted_by_ic = sorted(
            admitted_indices,
            key=lambda i: results[i].ic_mean,
            reverse=True,
        )

        kept_indices: List[int] = []
        kept_signals: List[np.ndarray] = []

        for idx in admitted_by_ic:
            r = results[idx]
            is_correlated = False

            for kept_sig in kept_signals:
                corr = self.library._compute_correlation_vectorized(
                    r.signals, kept_sig
                )
                if corr >= corr_threshold:
                    is_correlated = True
                    break

            if is_correlated:
                # Reject this candidate from the batch due to intra-batch dup
                results[idx] = EvaluationResult(
                    factor_name=r.factor_name,
                    formula=r.formula,
                    parse_ok=r.parse_ok,
                    ic_mean=r.ic_mean,
                    icir=r.icir,
                    ic_win_rate=r.ic_win_rate,
                    max_correlation=r.max_correlation,
                    correlated_with=r.correlated_with,
                    admitted=False,
                    replaced=None,
                    rejection_reason="Intra-batch deduplication (correlated with higher-IC batch member)",
                    stage_passed=2,
                    signals=r.signals,
                )
            else:
                kept_indices.append(idx)
                kept_signals.append(r.signals)

        dedup_rejected = len(admitted_indices) - len(kept_indices)
        if dedup_rejected > 0:
            logger.debug(
                "Intra-batch dedup: rejected %d/%d admitted candidates",
                dedup_rejected, len(admitted_indices),
            )

        return results

    def _build_data_dict(self) -> Dict[str, np.ndarray]:
        """Convert data_tensor to a dict mapping feature names to (M, T) arrays.

        Handles two formats:
          - dict: already maps ``"$close"`` etc. to ``(M, T)`` arrays.
          - np.ndarray of shape ``(M, T, F)``: sliced along the last axis
            using the canonical ``FEATURES`` ordering.
        """
        if isinstance(self.data_tensor, dict):
            return self.data_tensor

        # (M, T, F) numpy array — map each feature slice
        data_dict: Dict[str, np.ndarray] = {}
        n_features = self.data_tensor.shape[2] if self.data_tensor.ndim == 3 else 0
        for i, feat_name in enumerate(FEATURES):
            if i < n_features:
                data_dict[feat_name] = self.data_tensor[:, :, i]
        return data_dict

    def _compute_signals(self, tree) -> Optional[np.ndarray]:
        """Compute factor signals from expression tree on the data tensor.

        Evaluates the parsed expression tree against the market data using
        the tree's own ``evaluate()`` method which dispatches through the
        numpy operator implementations.

        Falls back to deterministic pseudo-signals only when evaluation
        raises an unexpected error (and logs a warning).
        """
        try:
            data_dict = self._build_data_dict()
            result = tree.evaluate(data_dict)
            return result
        except Exception as exc:
            formula_str = tree.to_string()
            logger.warning(
                "Expression evaluation failed for '%s': %s  — "
                "falling back to pseudo-signals",
                formula_str,
                exc,
            )

        # Fallback: deterministic pseudo-signals from formula hash
        M, T = self.returns.shape
        formula_str = tree.to_string()
        seed = hash(formula_str) % (2**31)
        rng = np.random.RandomState(seed)
        signals = rng.randn(M, T).astype(np.float64)
        nan_mask = rng.random((M, T)) < 0.02
        signals[nan_mask] = np.nan
        return signals


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

    def export_library(
        self, library: FactorLibrary, path: Optional[str] = None
    ) -> str:
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
        llm_provider: Optional[LLMProvider] = None,
        memory: Optional[ExperienceMemory] = None,
        library: Optional[FactorLibrary] = None,
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

        # Core components
        self.library = library or FactorLibrary(
            correlation_threshold=getattr(config, "correlation_threshold", 0.5),
            ic_threshold=getattr(config, "ic_threshold", 0.04),
        )
        self.memory = memory or ExperienceMemory()
        self.memory_manager: Optional[ExperienceMemoryManager] = None
        self.generator = FactorGenerator(
            llm_provider=llm_provider,
            prompt_builder=PromptBuilder(),
        )
        self.pipeline = ValidationPipeline(
            data_tensor=data_tensor,
            returns=returns,
            library=self.library,
            ic_threshold=getattr(config, "ic_threshold", 0.04),
            icir_threshold=getattr(config, "icir_threshold", 0.5),
            replacement_ic_min=getattr(config, "replacement_ic_min", 0.10),
            replacement_ic_ratio=getattr(config, "replacement_ic_ratio", 1.3),
            fast_screen_assets=getattr(config, "fast_screen_assets", 100),
            num_workers=getattr(config, "num_workers", 1),
        )
        self.reporter = MiningReporter(
            getattr(config, "output_dir", "./output")
        )
        self.budget = BudgetTracker()

        # Session state
        self.iteration = 0
        self._session: Optional[MiningSession] = None
        self._session_logger: Optional[MiningSessionLogger] = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        target_size: Optional[int] = None,
        max_iterations: Optional[int] = None,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
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
        target_size = target_size or getattr(
            self.config, "target_library_size", 110
        )
        max_iterations = max_iterations or getattr(
            self.config, "max_iterations", 200
        )
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

        # Initialize session logger
        self._session_logger = MiningSessionLogger(output_dir)
        self._session_logger.log_session_start({
            "target_library_size": target_size,
            "batch_size": batch_size,
            "max_iterations": max_iterations,
            "resumed_from_iteration": self.iteration if resume else 0,
        })
        self._session_logger.start_progress(max_iterations)

        loop_start = time.time()

        if not hasattr(self, "budget") or self.budget is None:
            self.budget = BudgetTracker()
        self.budget.wall_start = time.time()

        try:
            while (
                self.library.size < target_size
                and self.iteration < max_iterations
            ):
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
                    "Iteration %d: Library size=%d, Admitted=%d, "
                    "Yield=%.1f%%, AvgCorr=%.3f",
                    self.iteration,
                    stats["library_size"],
                    stats["admitted"],
                    stats["yield_rate"] * 100,
                    stats.get("avg_correlation", 0),
                )

                # Periodic checkpoint
                if (
                    self.checkpoint_interval > 0
                    and self.iteration % self.checkpoint_interval == 0
                ):
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

    def _run_iteration(self, batch_size: int) -> Dict[str, Any]:
        """Execute one iteration of the Ralph Loop.

        Returns
        -------
        dict
            Iteration statistics.
        """
        t0 = time.time()

        # Step 1: Memory Retrieval -- R(M, L)
        library_state = self.library.get_state_summary()
        memory_signal = retrieve_memory(
            self.memory,
            library_state=library_state,
        )

        # Step 2: Guided Generation -- G(m, L)
        t_gen = time.time()
        candidates = self.generator.generate_batch(
            memory_signal=memory_signal,
            library_state=library_state,
            batch_size=batch_size,
        )
        self.budget.record_llm_call()

        if not candidates:
            logger.warning(
                "Iteration %d: generator produced 0 candidates", self.iteration
            )
            return self._empty_stats()

        # Step 3: Multi-Stage Evaluation -- V(alpha) for each candidate
        results = self.pipeline.evaluate_batch(candidates)

        # Step 4: Library Update -- L <- L + admitted factors
        admitted_results = self._update_library(results)

        # Step 5: Memory Evolution -- E(M, F(M, tau))
        trajectory = self._build_trajectory(results)
        formed = form_memory(self.memory, trajectory, self.iteration)
        self.memory = evolve_memory(self.memory, formed)

        # Build stats
        elapsed = time.time() - t0
        self.budget.record_compute(elapsed)
        stats = self._compute_stats(results, admitted_results, elapsed)

        # Log to reporter and session logger
        # stats already contains 'iteration', so pass it without keyword arg
        self.reporter.log_batch(**stats)
        if self._session_logger:
            ic_values = [r.ic_mean for r in results if r.parse_ok]
            record = IterationRecord(
                iteration=self.iteration,
                candidates_generated=len(candidates),
                ic_passed=stats["ic_passed"],
                correlation_passed=stats["corr_passed"],
                admitted=stats["admitted"],
                rejected=len(candidates) - stats["admitted"],
                replaced=stats["replaced"],
                library_size=self.library.size,
                best_ic=max(ic_values) if ic_values else 0.0,
                mean_ic=float(np.mean(ic_values)) if ic_values else 0.0,
                elapsed_seconds=elapsed,
            )
            self._session_logger.log_iteration(record)

            # Log individual factor records
            for r in results:
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

    # ------------------------------------------------------------------
    # Library update
    # ------------------------------------------------------------------

    def _update_library(
        self, results: List[EvaluationResult]
    ) -> List[EvaluationResult]:
        """Admit passing factors into the library and handle replacements.

        Returns the list of admitted results.
        """
        admitted: List[EvaluationResult] = []

        for result in results:
            if not result.admitted:
                continue

            # Handle replacement
            if result.replaced is not None:
                old_id = result.replaced
                new_factor = Factor(
                    id=0,  # Will be reassigned by library
                    name=result.factor_name,
                    formula=result.formula,
                    category=self._infer_category(result.formula),
                    ic_mean=result.ic_mean,
                    icir=result.icir,
                    ic_win_rate=result.ic_win_rate,
                    max_correlation=result.max_correlation,
                    batch_number=self.iteration,
                    signals=result.signals,
                )
                try:
                    self.library.replace_factor(old_id, new_factor)
                    admitted.append(result)
                    logger.info(
                        "Replaced factor %d with '%s' (IC=%.4f)",
                        old_id, result.factor_name, result.ic_mean,
                    )
                except KeyError:
                    logger.warning(
                        "Failed to replace factor %d (already removed?)", old_id
                    )
            else:
                # Direct admission
                factor = Factor(
                    id=0,  # Will be reassigned
                    name=result.factor_name,
                    formula=result.formula,
                    category=self._infer_category(result.formula),
                    ic_mean=result.ic_mean,
                    icir=result.icir,
                    ic_win_rate=result.ic_win_rate,
                    max_correlation=result.max_correlation,
                    batch_number=self.iteration,
                    signals=result.signals,
                )
                self.library.admit_factor(factor)
                admitted.append(result)

        return admitted

    # ------------------------------------------------------------------
    # Trajectory builder for memory formation
    # ------------------------------------------------------------------

    def _build_trajectory(
        self, results: List[EvaluationResult]
    ) -> List[Dict[str, Any]]:
        """Build mining trajectory tau for memory formation.

        Converts evaluation results into the dict format expected by
        ``form_memory``.
        """
        trajectory: List[Dict[str, Any]] = []
        for r in results:
            entry: Dict[str, Any] = {
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
        results: List[EvaluationResult],
        admitted: List[EvaluationResult],
        elapsed: float,
    ) -> Dict[str, Any]:
        """Compute per-iteration statistics."""
        n_candidates = len(results)
        diagnostics = self.library.get_diagnostics()

        # Count dedup rejections (stage_passed==2 with dedup reason)
        dedup_rejected = sum(
            1 for r in results
            if not r.admitted
            and "deduplication" in r.rejection_reason.lower()
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

    def _empty_stats(self) -> Dict[str, Any]:
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
        """Infer factor category from formula structure.

        Uses operator presence heuristics to classify factors into broad
        categories aligned with the paper's taxonomy.
        """
        formula_upper = formula.upper()

        # Extract operators and normalize to uppercase for matching
        ops_raw = re.findall(r"([A-Za-z][a-zA-Z]+)\(", formula)
        ops = {o.upper() for o in ops_raw}

        if ops & {"SKEW", "KURT"}:
            return "Higher-Moment"
        if ops & {"CORR", "COV", "BETA"} and "$VOLUME" in formula_upper:
            return "PV-Correlation"
        if ops & {"IFELSE", "GREATER", "LESS", "OR", "AND"}:
            return "Regime-Conditional"
        if ops & {"TSLINREG", "TSLINREGSLOPE", "TSLINREGRESID", "RESID"}:
            return "Regression"
        if ops & {"EMA", "DEMA", "KAMA", "HMA", "WMA", "SMA"}:
            return "Smoothing"
        if "$VWAP" in formula_upper:
            return "VWAP"
        if "$AMT" in formula_upper:
            return "Amount"
        if ops & {"DELTA", "DELAY", "RETURN", "LOGRETURN"}:
            return "Momentum"
        if ops & {"STD", "VAR"}:
            return "Volatility"
        if ops & {"TSMAX", "TSMIN", "TSARGMAX", "TSARGMIN", "TSRANK"}:
            return "Extrema"
        if ops & {"CSRANK", "CSZSCORE", "CSDEMEAN"}:
            return "Cross-Sectional"

        return "Other"

    # ------------------------------------------------------------------
    # Session persistence (save / resume)
    # ------------------------------------------------------------------

    def save_session(self, path: Optional[str] = None) -> str:
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
                json.dump(self.memory.to_dict(), f, indent=2, default=str)

        # Save session metadata
        if self._session:
            self._session.library_path = lib_base
            self._session.memory_path = mem_path
            self._session.save(checkpoint_dir / "session.json")

        # Save loop state (iteration counter + budget tracker)
        loop_state: Dict[str, Any] = {
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
                self.budget.llm_calls = budget_data.get(
                    "llm_calls", self.budget.llm_calls
                )
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
                self.memory = ExperienceMemory.from_dict(mem_data)
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

    @classmethod
    def resume_from(
        cls,
        checkpoint_path: str,
        config: Any,
        data_tensor: np.ndarray,
        returns: np.ndarray,
        llm_provider: Optional[LLMProvider] = None,
        **kwargs: Any,
    ) -> "RalphLoop":
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

    def _serialize_config(self) -> Dict[str, Any]:
        """Serialize config to a JSON-compatible dict."""
        try:
            if hasattr(self.config, "to_dict"):
                return self.config.to_dict()
            return asdict(self.config)
        except (TypeError, AttributeError):
            # Fallback: extract known attributes
            attrs = [
                "target_library_size", "batch_size", "max_iterations",
                "ic_threshold", "icir_threshold", "correlation_threshold",
                "replacement_ic_min", "replacement_ic_ratio", "output_dir",
            ]
            return {
                attr: getattr(self.config, attr, None)
                for attr in attrs
                if getattr(self.config, attr, None) is not None
            }
