"""Multi-stage factor evaluation and validation pipeline.

Implements Algorithm 1 Step 3: the four-stage evaluation cascade that
screens, deduplicates, and validates candidate alpha factors before
admitting them to the factor library.

Stages:
    1. Fast IC screening on a subset of assets
    2. Correlation check against the existing library
    2.5. Replacement check for rejected-but-strong candidates
    3. Intra-batch deduplication
    4. Full validation on the complete asset universe

Supports parallel evaluation via a configurable multiprocessing worker pool.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from factorminer.evaluation.admission import (
    check_replacement,
)
from factorminer.evaluation.correlation import (
    batch_spearman_pairwise,
    compute_correlation_batch,
)
from factorminer.evaluation.metrics import (
    compute_factor_stats,
    compute_ic,
    compute_ic_abs_mean,
    compute_ic_mean,
    compute_ic_paper_icir,
    compute_ic_paper_mean,
    compute_icir,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CandidateFactor:
    """A candidate factor to be evaluated."""

    name: str
    formula: str
    signals: np.ndarray | None = None  # (M, T) computed signals
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a single candidate through the pipeline."""

    factor_name: str
    formula: str
    ic_series: np.ndarray | None = None
    ic_mean: float = 0.0
    ic_paper_mean: float = 0.0
    ic_abs_mean: float = 0.0
    icir: float = 0.0
    ic_paper_icir: float = 0.0
    max_correlation: float = 0.0
    correlated_with: str | None = None
    stage_passed: int = 0  # Highest stage passed (1-4), 0 if failed stage 1
    rejection_reason: str | None = None
    admitted: bool = False
    replaced: str | None = None  # ID of replaced factor if replacement occurred
    full_stats: dict | None = None  # Full stats from stage 4

    def to_trajectory_dict(self) -> dict:
        """Convert to a dict compatible with the memory formation trajectory format."""
        return {
            "factor_id": self.factor_name,
            "formula": self.formula,
            "ic": self.ic_mean,
            "paper_ic": self.ic_paper_mean,
            "ic_abs_mean": self.ic_abs_mean,
            "icir": self.icir,
            "paper_icir": self.ic_paper_icir,
            "max_correlation": self.max_correlation,
            "correlated_with": self.correlated_with or "",
            "admitted": self.admitted,
            "rejection_reason": self.rejection_reason or "",
            "replaced": self.replaced,
            "stage_passed": self.stage_passed,
        }


@dataclass
class FactorLibraryView:
    """Read-only view of the factor library for the pipeline.

    Provides the data needed for correlation checks and replacement
    decisions without exposing the full library internals.
    """

    factor_ids: list[str]
    signals: dict[str, np.ndarray]  # factor_id -> (M, T)
    ic_map: dict[str, float]  # factor_id -> absolute IC

    @property
    def size(self) -> int:
        return len(self.factor_ids)

    def get_signals_tensor(self) -> np.ndarray:
        """Return library signals as a (N, M, T) tensor.

        Returns
        -------
        np.ndarray, shape (N, M, T)
        """
        if not self.factor_ids:
            return np.array([]).reshape(0, 0, 0)
        return np.stack([self.signals[fid] for fid in self.factor_ids], axis=0)


@dataclass
class PipelineConfig:
    """Configuration for the validation pipeline."""

    # Stage 1: Fast IC screening
    ic_threshold: float = 0.04
    fast_screen_assets: int = 100

    # Stage 2: Correlation threshold
    correlation_threshold: float = 0.5

    # Stage 2.5: Replacement
    replacement_ic_min: float = 0.10
    replacement_ic_ratio: float = 1.3

    # Stage 4: Full validation
    icir_threshold: float = 0.5

    # Parallelism
    num_workers: int = 4
    backend: str = "numpy"  # "numpy" or "gpu"

    @classmethod
    def from_config(cls, mining_cfg, eval_cfg) -> PipelineConfig:
        """Build from MiningConfig and EvaluationConfig objects."""
        return cls(
            ic_threshold=mining_cfg.ic_threshold,
            correlation_threshold=mining_cfg.correlation_threshold,
            replacement_ic_min=mining_cfg.replacement_ic_min,
            replacement_ic_ratio=mining_cfg.replacement_ic_ratio,
            fast_screen_assets=eval_cfg.fast_screen_assets,
            num_workers=eval_cfg.num_workers,
            backend=eval_cfg.backend,
        )


# ---------------------------------------------------------------------------
# Worker function for multiprocessing
# ---------------------------------------------------------------------------

def _evaluate_single_candidate_ic(
    signals: np.ndarray,
    returns: np.ndarray,
) -> tuple[np.ndarray, float, float, float, float, float]:
    """Compute IC series, IC mean, and ICIR for a single candidate.

    Designed to be called in a worker process.
    """
    ic_series = compute_ic(signals, returns)
    ic_mean_val = compute_ic_mean(ic_series)
    ic_paper_mean = compute_ic_paper_mean(ic_series)
    ic_abs_mean = compute_ic_abs_mean(ic_series)
    icir_val = compute_icir(ic_series)
    ic_paper_icir = compute_ic_paper_icir(ic_series)
    return ic_series, ic_mean_val, ic_paper_mean, ic_abs_mean, icir_val, ic_paper_icir


# ---------------------------------------------------------------------------
# Validation Pipeline
# ---------------------------------------------------------------------------

class ValidationPipeline:
    """Multi-stage factor evaluation pipeline.

    Implements the cascade: Fast IC -> Correlation -> Replacement ->
    Dedup -> Full Validation.

    Parameters
    ----------
    returns : np.ndarray, shape (M, T)
        Forward returns for all assets.
    library : FactorLibraryView
        Current state of the factor library.
    config : PipelineConfig
        Pipeline configuration.
    compute_signals_fn : callable, optional
        Function(CandidateFactor, data) -> np.ndarray to compute signals
        if not pre-computed.
    data : dict, optional
        Market data dict for signal computation.
    """

    def __init__(
        self,
        returns: np.ndarray,
        library: FactorLibraryView,
        config: PipelineConfig,
        compute_signals_fn: Callable | None = None,
        data: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.returns = returns
        self.library = library
        self.config = config
        self.compute_signals_fn = compute_signals_fn
        self.data = data

        M, T = returns.shape
        # Pre-select a random subset of assets for fast screening
        if config.fast_screen_assets < M:
            rng = np.random.default_rng(42)
            self._fast_idx = rng.choice(M, size=config.fast_screen_assets, replace=False)
        else:
            self._fast_idx = np.arange(M)

        self._fast_returns = returns[self._fast_idx, :]

    def evaluate_batch(
        self,
        candidates: list[CandidateFactor],
    ) -> list[EvaluationResult]:
        """Run the full multi-stage evaluation on a batch of candidates.

        Parameters
        ----------
        candidates : list of CandidateFactor
            Each candidate should have signals pre-computed or provide
            a compute_signals_fn.

        Returns
        -------
        list of EvaluationResult
            One result per candidate, including rejected ones.
        """
        if not candidates:
            return []

        # Ensure signals are computed
        self._ensure_signals(candidates)

        results: dict[str, EvaluationResult] = {}

        logger.info(
            "Starting pipeline evaluation for %d candidates", len(candidates)
        )

        # Stage 1: Fast IC screening
        passed_s1, failed_s1 = self._stage1_ic_screen(candidates)
        for c, result in failed_s1:
            results[c.name] = result
        logger.info(
            "Stage 1 (IC screen): %d passed, %d failed",
            len(passed_s1), len(failed_s1),
        )

        if not passed_s1:
            return list(results.values())

        # Stage 2: Correlation check against library
        passed_s2, failed_s2, replacement_candidates = self._stage2_correlation_check(
            passed_s1
        )
        for c, result in failed_s2:
            results[c.name] = result
        logger.info(
            "Stage 2 (correlation): %d passed, %d failed, %d for replacement",
            len(passed_s2), len(failed_s2), len(replacement_candidates),
        )

        # Stage 2.5: Replacement check
        replaced = self._stage25_replacement_check(replacement_candidates)
        for c, result in replaced:
            results[c.name] = result
        logger.info("Stage 2.5 (replacement): %d replacements", len(replaced))

        if not passed_s2 and not replaced:
            return list(results.values())

        # Combine stage 2 passes and successful replacements
        to_dedup = list(passed_s2)
        for c, result in replaced:
            if result.admitted:
                to_dedup.append(c)

        # Stage 3: Intra-batch deduplication
        passed_s3, failed_s3 = self._stage3_batch_dedup(to_dedup)
        for c, result in failed_s3:
            results[c.name] = result
        logger.info(
            "Stage 3 (dedup): %d passed, %d failed",
            len(passed_s3), len(failed_s3),
        )

        # Stage 4: Full validation
        validated = self._stage4_full_validation(passed_s3)
        for c, result in validated:
            results[c.name] = result
        logger.info(
            "Stage 4 (full validation): %d admitted",
            sum(1 for _, r in validated if r.admitted),
        )

        return list(results.values())

    def _ensure_signals(self, candidates: list[CandidateFactor]) -> None:
        """Compute signals for candidates that don't have them yet."""
        if self.compute_signals_fn is None:
            return
        for c in candidates:
            if c.signals is None and self.data is not None:
                c.signals = self.compute_signals_fn(c, self.data)

    # ----- Stage 1: Fast IC Screening -----

    def _stage1_ic_screen(
        self,
        candidates: list[CandidateFactor],
    ) -> tuple[
        list[CandidateFactor],
        list[tuple[CandidateFactor, EvaluationResult]],
    ]:
        """Stage 1: Fast IC screening on asset subset.

        C1 = {a in C : |IC(a)| >= tau_IC}

        Returns (passed, failed) where failed includes EvaluationResults.
        """
        passed = []
        failed = []
        threshold = self.config.ic_threshold

        for c in candidates:
            if c.signals is None:
                failed.append((c, EvaluationResult(
                    factor_name=c.name,
                    formula=c.formula,
                    stage_passed=0,
                    rejection_reason="No signals computed",
                )))
                continue

            # Use fast subset
            fast_signals = c.signals[self._fast_idx, :]
            ic_series = compute_ic(fast_signals, self._fast_returns)
            valid_ic = ic_series[~np.isnan(ic_series)]

            if len(valid_ic) == 0:
                failed.append((c, EvaluationResult(
                    factor_name=c.name,
                    formula=c.formula,
                    stage_passed=0,
                    rejection_reason="No valid IC values",
                )))
                continue

            ic_mean = compute_ic_mean(ic_series)
            ic_paper_mean = compute_ic_paper_mean(ic_series)
            ic_abs_mean = compute_ic_abs_mean(ic_series)
            icir = compute_icir(ic_series)
            ic_paper_icir = compute_ic_paper_icir(ic_series)

            if ic_paper_mean < threshold:
                failed.append((c, EvaluationResult(
                    factor_name=c.name,
                    formula=c.formula,
                    ic_series=ic_series,
                    ic_mean=ic_mean,
                    ic_paper_mean=ic_paper_mean,
                    ic_abs_mean=ic_abs_mean,
                    icir=icir,
                    ic_paper_icir=ic_paper_icir,
                    stage_passed=0,
                    rejection_reason=f"Stage 1: paper IC={ic_paper_mean:.4f} < {threshold}",
                )))
            else:
                # Store fast IC for later use
                c.metadata["fast_ic_series"] = ic_series
                c.metadata["fast_ic_mean"] = ic_mean
                c.metadata["fast_ic_paper_mean"] = ic_paper_mean
                c.metadata["fast_ic_abs_mean"] = ic_abs_mean
                c.metadata["fast_icir"] = icir
                c.metadata["fast_ic_paper_icir"] = ic_paper_icir
                passed.append(c)

        return passed, failed

    # ----- Stage 2: Correlation Check -----

    def _stage2_correlation_check(
        self,
        candidates: list[CandidateFactor],
    ) -> tuple[
        list[CandidateFactor],
        list[tuple[CandidateFactor, EvaluationResult]],
        list[tuple[CandidateFactor, dict[str, float]]],
    ]:
        """Stage 2: Correlation check against the library.

        C2 = {a in C1 : max_{g in L} |rho(a,g)| < theta}

        Returns (passed, failed, replacement_candidates).
        replacement_candidates contains candidates that failed correlation
        but might qualify for replacement.
        """
        passed = []
        failed = []
        replacement_candidates = []

        if self.library.size == 0:
            # Empty library: all pass
            return candidates, failed, replacement_candidates

        theta = self.config.correlation_threshold
        lib_tensor = self.library.get_signals_tensor()

        for c in candidates:
            # Compute correlation with all library factors
            corrs = compute_correlation_batch(
                c.signals,
                lib_tensor,
                backend=self.config.backend,
            )
            abs_corrs = np.abs(corrs)
            max_idx = int(np.argmax(abs_corrs))
            max_corr = float(abs_corrs[max_idx])
            correlated_with = self.library.factor_ids[max_idx]

            if max_corr < theta:
                c.metadata["max_correlation"] = max_corr
                c.metadata["correlated_with"] = correlated_with
                passed.append(c)
            else:
                paper_ic = c.metadata.get("fast_ic_paper_mean", 0.0)

                # Check if candidate qualifies for replacement
                if paper_ic >= self.config.replacement_ic_min:
                    # Store full correlation map for replacement check
                    corr_map = {
                        fid: float(corrs[i])
                        for i, fid in enumerate(self.library.factor_ids)
                    }
                    c.metadata["max_correlation"] = max_corr
                    c.metadata["correlated_with"] = correlated_with
                    c.metadata["correlation_map"] = corr_map
                    replacement_candidates.append((c, corr_map))
                else:
                    failed.append((c, EvaluationResult(
                        factor_name=c.name,
                        formula=c.formula,
                        ic_series=c.metadata.get("fast_ic_series"),
                        ic_mean=c.metadata.get("fast_ic_mean", 0.0),
                        ic_paper_mean=paper_ic,
                        ic_abs_mean=c.metadata.get("fast_ic_abs_mean", 0.0),
                        icir=c.metadata.get("fast_icir", 0.0),
                        ic_paper_icir=c.metadata.get("fast_ic_paper_icir", 0.0),
                        max_correlation=max_corr,
                        correlated_with=correlated_with,
                        stage_passed=1,
                        rejection_reason=(
                            f"Stage 2: max|rho|={max_corr:.4f} >= {theta} "
                            f"(with {correlated_with})"
                        ),
                    )))

        return passed, failed, replacement_candidates

    # ----- Stage 2.5: Replacement Check -----

    def _stage25_replacement_check(
        self,
        replacement_candidates: list[tuple[CandidateFactor, dict[str, float]]],
    ) -> list[tuple[CandidateFactor, EvaluationResult]]:
        """Stage 2.5: Check if rejected candidates can replace library members.

        For a in C1 \\ C2, check replacement rule (Eq. 11):
            |IC(a)| >= 0.10
            |IC(a)| >= 1.3 * |IC(g)|
            |{g : |rho(a,g)| >= theta}| == 1
        """
        results = []

        for c, corr_map in replacement_candidates:
            paper_ic = c.metadata.get("fast_ic_paper_mean", 0.0)
            max_corr = c.metadata.get("max_correlation", 0.0)
            correlated_with = c.metadata.get("correlated_with")

            decision = check_replacement(
                candidate_ic_abs=paper_ic,
                max_corr=max_corr,
                correlated_with=correlated_with,
                library_ic_map=self.library.ic_map,
                correlation_map=corr_map,
                replacement_ic_min=self.config.replacement_ic_min,
                replacement_ic_ratio=self.config.replacement_ic_ratio,
                correlation_threshold=self.config.correlation_threshold,
            )

            result = EvaluationResult(
                factor_name=c.name,
                formula=c.formula,
                ic_series=c.metadata.get("fast_ic_series"),
                ic_mean=c.metadata.get("fast_ic_mean", 0.0),
                ic_paper_mean=paper_ic,
                ic_abs_mean=c.metadata.get("fast_ic_abs_mean", 0.0),
                icir=c.metadata.get("fast_icir", 0.0),
                ic_paper_icir=c.metadata.get("fast_ic_paper_icir", 0.0),
                max_correlation=max_corr,
                correlated_with=correlated_with,
                admitted=decision.admitted,
                replaced=decision.replaced_factor_id,
                stage_passed=2 if decision.admitted else 1,
                rejection_reason=decision.rejection_reason,
            )
            results.append((c, result))

        return results

    # ----- Stage 3: Batch Deduplication -----

    def _stage3_batch_dedup(
        self,
        candidates: list[CandidateFactor],
    ) -> tuple[
        list[CandidateFactor],
        list[tuple[CandidateFactor, EvaluationResult]],
    ]:
        """Stage 3: Intra-batch deduplication.

        Remove candidates that are too correlated with each other
        within the same batch, keeping the one with higher IC.
        """
        if len(candidates) <= 1:
            return candidates, []

        theta = self.config.correlation_threshold
        signals_list = [c.signals for c in candidates]
        corr_matrix = batch_spearman_pairwise(signals_list)

        # Greedy dedup: sort by IC descending, keep each if not correlated
        # with any already-kept candidate
        ic_vals = [c.metadata.get("fast_ic_paper_mean", 0.0) for c in candidates]
        order = sorted(range(len(candidates)), key=lambda i: -ic_vals[i])

        kept_indices = set()
        removed = []

        for idx in order:
            is_correlated = False
            for kept_idx in kept_indices:
                if abs(corr_matrix[idx, kept_idx]) >= theta:
                    is_correlated = True
                    removed.append((candidates[idx], EvaluationResult(
                        factor_name=candidates[idx].name,
                        formula=candidates[idx].formula,
                        ic_mean=candidates[idx].metadata.get("fast_ic_mean", 0.0),
                        ic_paper_mean=ic_vals[idx],
                        ic_abs_mean=candidates[idx].metadata.get("fast_ic_abs_mean", 0.0),
                        icir=candidates[idx].metadata.get("fast_icir", 0.0),
                        ic_paper_icir=candidates[idx].metadata.get("fast_ic_paper_icir", 0.0),
                        max_correlation=float(abs(corr_matrix[idx, kept_idx])),
                        correlated_with=candidates[kept_idx].name,
                        stage_passed=2,
                        rejection_reason=(
                            f"Stage 3: intra-batch dup with {candidates[kept_idx].name} "
                            f"(rho={corr_matrix[idx, kept_idx]:.4f})"
                        ),
                    )))
                    break
            if not is_correlated:
                kept_indices.add(idx)

        passed = [candidates[i] for i in sorted(kept_indices)]
        return passed, removed

    # ----- Stage 4: Full Validation -----

    def _stage4_full_validation(
        self,
        candidates: list[CandidateFactor],
    ) -> list[tuple[CandidateFactor, EvaluationResult]]:
        """Stage 4: Full validation on complete asset universe.

        Compute comprehensive statistics using all assets and apply
        final quality checks.
        """
        results = []

        use_parallel = self.config.num_workers > 1 and len(candidates) > 1

        if use_parallel:
            results = self._stage4_parallel(candidates)
        else:
            for c in candidates:
                result = self._validate_single(c)
                results.append((c, result))

        return results

    def _validate_single(self, c: CandidateFactor) -> EvaluationResult:
        """Run full validation for a single candidate."""
        stats = compute_factor_stats(c.signals, self.returns)
        ic_series = stats["ic_series"]
        ic_mean = stats["ic_mean"]
        ic_paper_mean = stats["ic_paper_mean"]
        ic_abs_mean = stats["ic_abs_mean"]
        icir = stats["icir"]
        ic_paper_icir = stats["ic_paper_icir"]

        max_corr = c.metadata.get("max_correlation", 0.0)
        correlated_with = c.metadata.get("correlated_with")
        replaced = c.metadata.get("replaced") if "replaced" in c.metadata else None

        # Check if previously marked as replacement
        if hasattr(c, "_replacement_target"):
            replaced = c._replacement_target

        # Apply final threshold
        if ic_paper_mean < self.config.ic_threshold:
            return EvaluationResult(
                factor_name=c.name,
                formula=c.formula,
                ic_series=ic_series,
                ic_mean=ic_mean,
                ic_paper_mean=ic_paper_mean,
                ic_abs_mean=ic_abs_mean,
                icir=icir,
                ic_paper_icir=ic_paper_icir,
                max_correlation=max_corr,
                correlated_with=correlated_with,
                stage_passed=3,
                rejection_reason=(
                    f"Stage 4: full paper IC={ic_paper_mean:.4f} < {self.config.ic_threshold}"
                ),
                admitted=False,
                full_stats=stats,
            )

        return EvaluationResult(
            factor_name=c.name,
            formula=c.formula,
            ic_series=ic_series,
            ic_mean=ic_mean,
            ic_paper_mean=ic_paper_mean,
            ic_abs_mean=ic_abs_mean,
            icir=icir,
            ic_paper_icir=ic_paper_icir,
            max_correlation=max_corr,
            correlated_with=correlated_with,
            stage_passed=4,
            admitted=True,
            replaced=replaced,
            full_stats=stats,
        )

    def _stage4_parallel(
        self,
        candidates: list[CandidateFactor],
    ) -> list[tuple[CandidateFactor, EvaluationResult]]:
        """Run stage 4 in parallel using ProcessPoolExecutor.

        Each worker evaluates one candidate independently. Since signals
        and returns are numpy arrays, they can be pickled for IPC.
        """
        results = []
        futures_map = {}

        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            for c in candidates:
                future = executor.submit(
                    _evaluate_single_candidate_ic,
                    c.signals,
                    self.returns,
                )
                futures_map[future] = c

            for future in as_completed(futures_map):
                c = futures_map[future]
                try:
                    (
                        ic_series,
                        ic_mean,
                        ic_paper_mean,
                        ic_abs_mean,
                        icir,
                        ic_paper_icir,
                    ) = future.result()

                    max_corr = c.metadata.get("max_correlation", 0.0)
                    correlated_with = c.metadata.get("correlated_with")

                    if ic_paper_mean < self.config.ic_threshold:
                        result = EvaluationResult(
                            factor_name=c.name,
                            formula=c.formula,
                            ic_series=ic_series,
                            ic_mean=ic_mean,
                            ic_paper_mean=ic_paper_mean,
                            ic_abs_mean=ic_abs_mean,
                            icir=icir,
                            ic_paper_icir=ic_paper_icir,
                            max_correlation=max_corr,
                            correlated_with=correlated_with,
                            stage_passed=3,
                            rejection_reason=(
                                f"Stage 4: full paper IC={ic_paper_mean:.4f} "
                                f"< {self.config.ic_threshold}"
                            ),
                            admitted=False,
                        )
                    else:
                        result = EvaluationResult(
                            factor_name=c.name,
                            formula=c.formula,
                            ic_series=ic_series,
                            ic_mean=ic_mean,
                            ic_paper_mean=ic_paper_mean,
                            ic_abs_mean=ic_abs_mean,
                            icir=icir,
                            ic_paper_icir=ic_paper_icir,
                            max_correlation=max_corr,
                            correlated_with=correlated_with,
                            stage_passed=4,
                            admitted=True,
                        )
                    results.append((c, result))

                except Exception as e:
                    logger.error("Worker failed for %s: %s", c.name, e)
                    results.append((c, EvaluationResult(
                        factor_name=c.name,
                        formula=c.formula,
                        stage_passed=3,
                        rejection_reason=f"Stage 4 error: {e}",
                        admitted=False,
                    )))

        return results


# ---------------------------------------------------------------------------
# Convenience: Run the full pipeline
# ---------------------------------------------------------------------------

def run_evaluation_pipeline(
    candidates: list[CandidateFactor],
    returns: np.ndarray,
    library: FactorLibraryView,
    config: PipelineConfig,
    compute_signals_fn: Callable | None = None,
    data: dict[str, np.ndarray] | None = None,
) -> list[EvaluationResult]:
    """One-shot convenience function to run the full evaluation pipeline.

    Parameters
    ----------
    candidates : list of CandidateFactor
    returns : np.ndarray, shape (M, T)
    library : FactorLibraryView
    config : PipelineConfig
    compute_signals_fn : callable, optional
    data : dict, optional

    Returns
    -------
    list of EvaluationResult
    """
    pipeline = ValidationPipeline(
        returns=returns,
        library=library,
        config=config,
        compute_signals_fn=compute_signals_fn,
        data=data,
    )
    return pipeline.evaluate_batch(candidates)
