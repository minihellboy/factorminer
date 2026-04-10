"""Shared evaluation kernel for signals, metrics, admission, and dedup."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from factorminer.architecture.geometry import LibraryGeometry
from factorminer.architecture.paper_protocol import PaperProtocol
from factorminer.core.parser import try_parse
from factorminer.evaluation.metrics import compute_factor_stats
from factorminer.evaluation.research import (
    build_score_vector,
    compute_factor_geometry,
    passes_research_admission,
)
from factorminer.evaluation.runtime import SignalComputationError, compute_tree_signals


@dataclass
class EvaluationKernel:
    """Canonical evaluation engine used by loops and runtime analysis."""

    protocol: PaperProtocol
    geometry: LibraryGeometry
    research_config: Any = None

    def build_data_dict(self, data_tensor: Any, features: Iterable[str]) -> dict[str, np.ndarray]:
        if isinstance(data_tensor, dict):
            return data_tensor

        data_dict: dict[str, np.ndarray] = {}
        n_features = data_tensor.shape[2] if getattr(data_tensor, "ndim", 0) == 3 else 0
        for idx, feature in enumerate(features):
            if idx < n_features:
                data_dict[str(feature)] = data_tensor[:, :, idx]
        return data_dict

    def compute_signals(
        self,
        *,
        formula: str,
        data_dict: dict[str, np.ndarray],
        returns_shape: tuple[int, int],
        signal_failure_policy: str | None = None,
    ) -> tuple[Any, np.ndarray]:
        tree = try_parse(formula)
        if tree is None:
            raise SignalComputationError(f"Parse failure for '{formula}'")

        signals = compute_tree_signals(
            tree,
            data_dict,
            returns_shape,
            signal_failure_policy=signal_failure_policy or self.protocol.signal_failure_policy,
        )
        return tree, np.asarray(signals, dtype=np.float64)

    def compute_target_stats(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        target_panels: dict[str, np.ndarray] | None = None,
    ) -> dict[str, dict]:
        target_panels = target_panels or {"paper": returns}
        return {
            target_name: compute_factor_stats(signals, target_returns)
            for target_name, target_returns in target_panels.items()
        }

    def compute_quality_score(
        self,
        *,
        signals: np.ndarray,
        returns: np.ndarray,
        target_stats: dict[str, dict],
        library_signals: list[np.ndarray],
        target_horizons: dict[str, int] | None = None,
        benchmark_mode: str | None = None,
    ) -> dict[str, Any]:
        benchmark_mode = benchmark_mode or self.protocol.benchmark_mode
        target_horizons = target_horizons or {}
        if not (
            self.research_config is not None
            and getattr(self.research_config, "enabled", False)
            and benchmark_mode == "research"
        ):
            paper_stats = target_stats.get("paper") or next(iter(target_stats.values()))
            return {
                "quality_gate": float(paper_stats["ic_abs_mean"]),
                "icir": float(paper_stats["icir"]),
                "research_score": 0.0,
                "score_vector": None,
                "admitted": None,
                "admission_reason": "",
                "max_correlation": self.geometry.candidate_geometry(signals).max_correlation,
            }

        geometry = compute_factor_geometry(signals, returns, library_signals)
        score_vector = build_score_vector(
            target_stats,
            target_horizons,
            self.research_config,
            geometry,
        )
        admitted, reason = passes_research_admission(
            score_vector,
            self.research_config,
            self.protocol.correlation_threshold,
        )
        return {
            "quality_gate": float(score_vector.primary_score),
            "icir": float(target_stats["paper"]["icir"]) if "paper" in target_stats else 0.0,
            "research_score": float(score_vector.primary_score),
            "score_vector": score_vector.to_dict(),
            "admitted": admitted,
            "admission_reason": reason,
            "max_correlation": float(score_vector.geometry.max_abs_correlation),
        }

    def admission_decision(
        self, candidate_ic: float, candidate_signals: np.ndarray
    ) -> tuple[bool, str]:
        return self.geometry.check_admission(candidate_ic, candidate_signals)

    def replacement_decision(
        self, candidate_ic: float, candidate_signals: np.ndarray
    ) -> tuple[bool, int | None, str]:
        return self.geometry.check_replacement(
            candidate_ic,
            candidate_signals,
            ic_min=self.protocol.replacement_ic_min,
            ic_ratio=self.protocol.replacement_ic_ratio,
        )

    def deduplicate_results(
        self,
        results: list[Any],
        *,
        quality_attr: str = "ic_mean",
    ) -> list[Any]:
        admitted_indices = [
            idx
            for idx, result in enumerate(results)
            if getattr(result, "admitted", False) and getattr(result, "signals", None) is not None
        ]
        if len(admitted_indices) <= 1:
            return results

        corr_threshold = self.protocol.correlation_threshold
        admitted_indices = sorted(
            admitted_indices,
            key=lambda idx: float(getattr(results[idx], quality_attr, 0.0)),
            reverse=True,
        )

        kept: list[np.ndarray] = []
        for idx in admitted_indices:
            result = results[idx]
            signals = result.signals
            duplicate = False
            for kept_signals in kept:
                corr = self.geometry.library.compute_correlation(signals, kept_signals)
                if corr >= corr_threshold:
                    duplicate = True
                    break
            if duplicate:
                result.admitted = False
                result.replaced = None
                result.stage_passed = min(int(getattr(result, "stage_passed", 0)), 2)
                result.rejection_reason = (
                    "Intra-batch deduplication (correlated with higher-IC batch member)"
                )
            else:
                kept.append(signals)
        return results
