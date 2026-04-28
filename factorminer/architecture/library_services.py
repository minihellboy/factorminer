"""Reusable services for factor-library mutations."""

from __future__ import annotations

import logging

from factorminer.architecture.families import infer_family
from factorminer.core.factor_library import Factor, FactorLibrary

logger = logging.getLogger(__name__)


class FactorAdmissionService:
    """Owns conversion from evaluation results into library mutations."""

    def __init__(self, library: FactorLibrary) -> None:
        self.library = library

    def admit_results(self, results: list[object], *, iteration: int) -> list[object]:
        admitted: list[object] = []
        for result in results:
            if not getattr(result, "admitted", False):
                continue

            factor = self._build_factor(result, iteration=iteration)
            replaced = getattr(result, "replaced", None)
            if replaced is not None:
                try:
                    self.library.replace_factor(replaced, factor)
                    admitted.append(result)
                    logger.info(
                        "Replaced factor %d with '%s' (paper IC=%.4f)",
                        replaced,
                        result.factor_name,
                        getattr(result, "ic_paper_mean", result.ic_mean),
                    )
                except KeyError:
                    logger.warning("Failed to replace factor %d (already removed?)", replaced)
                continue

            self.library.admit_factor(factor)
            admitted.append(result)

        return admitted

    def _build_factor(self, result: object, *, iteration: int) -> Factor:
        return Factor(
            id=0,
            name=result.factor_name,
            formula=result.formula,
            category=infer_family(result.formula),
            ic_mean=result.ic_mean,
            ic_paper_mean=getattr(result, "ic_paper_mean", abs(float(result.ic_mean))),
            ic_abs_mean=getattr(result, "ic_abs_mean", abs(float(result.ic_mean))),
            icir=result.icir,
            ic_paper_icir=getattr(result, "ic_paper_icir", abs(float(result.icir))),
            ic_win_rate=result.ic_win_rate,
            max_correlation=result.max_correlation,
            batch_number=iteration,
            signals=result.signals,
            research_metrics=result.score_vector or {},
        )
