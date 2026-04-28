"""Reusable library geometry diagnostics and novelty utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from factorminer.core.factor_library import FactorLibrary


@dataclass(frozen=True)
class CandidateGeometry:
    """Geometric view of one candidate relative to the current library."""

    max_correlation: float
    max_dependence: float
    correlated_factor_ids: list[int]
    correlated_factor_names: list[str]
    novelty_score: float
    library_size: int
    dependence_metric: str


class LibraryGeometry:
    """Centralizes correlation and saturation geometry around the library."""

    def __init__(self, library: FactorLibrary) -> None:
        self.library = library

    def candidate_geometry(self, signals: np.ndarray) -> CandidateGeometry:
        if self.library.size == 0:
            return CandidateGeometry(
                max_correlation=0.0,
                max_dependence=0.0,
                correlated_factor_ids=[],
                correlated_factor_names=[],
                novelty_score=1.0,
                library_size=0,
                dependence_metric=self.library.dependence_metric.name,
            )

        correlated_ids: list[int] = []
        correlated_names: list[str] = []
        max_corr = 0.0

        for factor in self.library.list_factors():
            if factor.signals is None:
                continue
            corr = float(self.library.compute_correlation(signals, factor.signals))
            if corr > max_corr:
                max_corr = corr
            if corr >= self.library.correlation_threshold:
                correlated_ids.append(int(factor.id))
                correlated_names.append(str(factor.name))

        return CandidateGeometry(
            max_correlation=max_corr,
            max_dependence=max_corr,
            correlated_factor_ids=correlated_ids,
            correlated_factor_names=correlated_names,
            novelty_score=max(0.0, 1.0 - max_corr),
            library_size=self.library.size,
            dependence_metric=self.library.dependence_metric.name,
        )

    def replacement_target(self, signals: np.ndarray) -> int | None:
        geometry = self.candidate_geometry(signals)
        if len(geometry.correlated_factor_ids) != 1:
            return None
        return geometry.correlated_factor_ids[0]

    def check_admission(
        self,
        candidate_ic: float,
        candidate_signals: np.ndarray,
    ) -> tuple[bool, str]:
        if candidate_ic < self.library.ic_threshold:
            return False, (
                f"IC {candidate_ic:.4f} below threshold {self.library.ic_threshold}"
            )

        if self.library.size == 0:
            return True, "First factor in library"

        geometry = self.candidate_geometry(candidate_signals)
        if geometry.max_dependence >= self.library.correlation_threshold:
            return False, (
                f"Max {geometry.dependence_metric} dependence {geometry.max_dependence:.4f} "
                f">= threshold {self.library.correlation_threshold}"
            )

        return True, (
            f"Admitted: IC={candidate_ic:.4f}, "
            f"max_{geometry.dependence_metric}={geometry.max_dependence:.4f}"
        )

    def check_replacement(
        self,
        candidate_ic: float,
        candidate_signals: np.ndarray,
        *,
        ic_min: float,
        ic_ratio: float,
    ) -> tuple[bool, int | None, str]:
        if candidate_ic < ic_min:
            return False, None, f"IC {candidate_ic:.4f} below replacement floor {ic_min}"

        geometry = self.candidate_geometry(candidate_signals)
        if len(geometry.correlated_factor_ids) != 1:
            return False, None, (
                "Replacement requires exactly 1 conflicting library factor, "
                f"found {len(geometry.correlated_factor_ids)}"
            )

        target_id = geometry.correlated_factor_ids[0]
        target_factor = self.library.get_factor(target_id)
        target_ic = float(target_factor.ic_paper_mean or abs(target_factor.ic_mean))
        if candidate_ic < ic_ratio * target_ic:
            return False, None, (
                f"IC {candidate_ic:.4f} insufficient to replace factor {target_id} "
                f"(needs >= {ic_ratio} * {target_ic:.4f})"
            )

        return True, target_id, f"Replacement over factor {target_id}"

    def library_snapshot(self) -> dict[str, Any]:
        diagnostics = self.library.get_diagnostics()
        return {
            "size": self.library.size,
            "avg_correlation": diagnostics.get("avg_correlation", 0.0),
            "max_correlation": diagnostics.get("max_correlation", 0.0),
            "domain_saturation": diagnostics.get("domain_saturation", {}),
            "dependence_metric": self.library.dependence_metric.name,
        }
