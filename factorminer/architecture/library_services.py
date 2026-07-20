"""Reusable services for factor-library mutations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from factorminer.architecture.families import infer_family
from factorminer.core.factor_library import Factor, FactorLibrary

logger = logging.getLogger(__name__)


@dataclass
class _MigrantAdmissionResult:
    """Duck-typed admission record for a factor migrated from another population.

    Structurally compatible with the attributes `FactorAdmissionService.admit_results`
    reads off a `core.ralph_loop.EvaluationResult` (``factor_name``, ``formula``,
    ``ic_mean``, ``ic_paper_mean``, ``ic_abs_mean``, ``icir``, ``ic_paper_icir``,
    ``ic_win_rate``, ``max_correlation``, ``signals``, ``score_vector``, ``admitted``,
    ``replaced``) without importing that class -- `core.ralph_loop` already imports
    this module, so the reverse import would be circular.
    """

    factor_name: str
    formula: str
    ic_mean: float
    ic_paper_mean: float
    ic_abs_mean: float
    icir: float
    ic_paper_icir: float
    ic_win_rate: float
    max_correlation: float
    signals: np.ndarray | None
    score_vector: dict[str, Any] | None = None
    admitted: bool = True
    replaced: int | None = None

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

    def admit_migrant(
        self,
        factor: Factor,
        *,
        iteration: int,
        allow_replacement: bool = False,
    ) -> tuple[bool, str]:
        """Admit a factor migrated in from another population's library.

        Judges the migrant by exactly the same admission/replacement criteria
        `FactorLibrary` applies to a factor proposed natively by this
        library -- `FactorLibrary.check_admission` (Eq. 10), and optionally
        `FactorLibrary.check_replacement` (Eq. 11) -- then reuses
        `admit_results` for the actual mutation. A migrant is never
        force-inserted: it is rejected exactly like any other candidate that
        fails the bar.

        Parameters
        ----------
        factor : Factor
            A fully-evaluated factor (with cached ``signals``) sourced from
            another population/island.
        iteration : int
            Iteration/epoch number recorded on the admitted factor.
        allow_replacement : bool
            If True, also try the replacement path when straight admission
            fails. Default False, so a migration attempt can never *remove*
            an existing factor from the destination library -- it can only
            add one that clears the admission bar on its own.

        Returns
        -------
        (admitted, reason) : tuple[bool, str]
            Whether the migrant was admitted, and the admission/rejection
            reason from the underlying library check.
        """
        if factor.signals is None:
            return False, "migrant has no cached signals to evaluate"

        candidate_ic = float(
            factor.ic_paper_mean if factor.ic_paper_mean is not None else abs(factor.ic_mean)
        )
        admitted, reason = self.library.check_admission(candidate_ic, factor.signals)
        replaced_id: int | None = None
        if not admitted and allow_replacement:
            should_replace, replaced_id, replace_reason = self.library.check_replacement(
                candidate_ic, factor.signals
            )
            if not should_replace:
                return False, replace_reason
            admitted, reason = True, replace_reason

        if not admitted:
            return False, reason

        result = _MigrantAdmissionResult(
            factor_name=factor.name,
            formula=factor.formula,
            ic_mean=factor.ic_mean,
            ic_paper_mean=candidate_ic,
            ic_abs_mean=float(factor.ic_abs_mean if factor.ic_abs_mean is not None else abs(factor.ic_mean)),
            icir=factor.icir,
            ic_paper_icir=float(
                factor.ic_paper_icir if factor.ic_paper_icir is not None else abs(factor.icir)
            ),
            ic_win_rate=factor.ic_win_rate,
            max_correlation=factor.max_correlation,
            signals=factor.signals,
            score_vector=dict(factor.research_metrics) if factor.research_metrics else None,
            admitted=True,
            replaced=replaced_id,
        )
        admitted_results = self.admit_results([result], iteration=iteration)
        return (len(admitted_results) > 0), reason
