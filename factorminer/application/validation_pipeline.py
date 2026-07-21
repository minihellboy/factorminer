"""Canonical multi-stage candidate validation service."""

from __future__ import annotations

import logging
from concurrent.futures import as_completed
from typing import Any

import numpy as np

from factorminer.application.mining_budget import EvaluationResult
from factorminer.architecture import EvaluationKernel, LibraryGeometry, PaperProtocol
from factorminer.core.factor_library import FactorLibrary
from factorminer.core.types import get_features
from factorminer.evaluation.metrics import compute_factor_stats
from factorminer.evaluation.runtime import SignalComputationError, compute_tree_signals

logger = logging.getLogger(__name__)

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
            fast_ic = fast_stats["ic_paper_mean"]

            if fast_ic < self.ic_threshold:
                result.ic_mean = fast_stats["ic_mean"]
                result.ic_paper_mean = fast_stats["ic_paper_mean"]
                result.ic_abs_mean = fast_stats["ic_abs_mean"]
                result.icir = fast_stats["icir"]
                result.ic_paper_icir = fast_stats["ic_paper_icir"]
                result.rejection_reason = (
                    f"Fast-screen paper IC {fast_ic:.4f} < threshold {self.ic_threshold}"
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
        result.ic_mean = paper_stats["ic_mean"]
        result.ic_paper_mean = paper_stats["ic_paper_mean"]
        result.ic_abs_mean = paper_stats["ic_abs_mean"]
        result.icir = paper_stats["icir"]
        result.ic_paper_icir = paper_stats["ic_paper_icir"]
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
        quality_gate = result.ic_paper_mean
        quality_label = "Paper IC"
        if self._research_enabled():
            quality_gate = result.research_score
            quality_label = "Research score"

        if quality_gate < self.ic_threshold:
            result.rejection_reason = (
                f"{quality_label} {quality_gate:.4f} < threshold {self.ic_threshold}"
            )
            result.stage_passed = 0
            return result
        icir_gate = result.ic_paper_icir
        icir_label = "Paper ICIR"
        if self._research_enabled():
            icir_gate = result.icir
            icir_label = "Signed ICIR"
        if icir_gate < self.icir_threshold:
            result.rejection_reason = (
                f"{icir_label} {icir_gate:.4f} < threshold {self.icir_threshold}"
            )
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
        admitted, reason = self.kernel.admission_decision(result.ic_paper_mean, signals)
        if admitted:
            result.admitted = True
            result.stage_passed = 3
            if self.library.size > 0:
                result.max_correlation = self.geometry.candidate_geometry(signals).max_dependence
            return result

        result.stage_passed = 2

        # Stage 2.5: Replacement check for candidates that failed admission
        should_replace, replace_id, replace_reason = self.kernel.replacement_decision(
            result.ic_paper_mean,
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
            target_factor.research_metrics.get(
                "primary_score",
                target_factor.ic_paper_mean or abs(target_factor.ic_mean),
            )
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
            using the active feature registry ordering (defaults + extras).
        """
        if isinstance(self.data_tensor, dict):
            return self.data_tensor

        # (M, T, F) numpy array — map each feature slice
        data_dict: dict[str, np.ndarray] = {}
        n_features = self.data_tensor.shape[2] if self.data_tensor.ndim == 3 else 0
        for i, feat_name in enumerate(get_features()):
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
