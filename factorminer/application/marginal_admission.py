"""Build the marginal-value basis for ``admission.policy: marginal``."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from factorminer.core.reference_catalogs import reference_catalog
from factorminer.evaluation.marginal import MarginalBasis, MarginalConfig

logger = logging.getLogger(__name__)


def build_marginal_basis(
    config: Any,
    kernel: Any,
    data_dict: dict[str, np.ndarray],
    returns: np.ndarray,
) -> MarginalBasis | None:
    """Reference basis on the mining panel, or ``None`` under the paper rule.

    Reference formulas run through the loop's evaluation kernel (and so its
    backend); each panel is ranked as it arrives rather than held in memory.
    """
    admission = getattr(config, "admission", None)
    if admission is None or admission.policy != "marginal":
        return None
    formulas = [formula for _, formula in reference_catalog(admission.reference)]
    failed: list[str] = []

    def panels():
        outcomes = kernel.iter_batch_signals(
            formulas=formulas, data_dict=data_dict, returns_shape=returns.shape
        )
        for formula, (_tree, signals, error) in zip(formulas, outcomes, strict=True):
            if error is not None or signals is None:
                failed.append(formula)
                continue
            yield signals

    basis = MarginalBasis(returns, panels(), MarginalConfig(**admission.marginal_settings()))
    if len(failed) == len(formulas):
        raise RuntimeError(f"no {admission.reference} reference factor evaluated on this panel")
    logger.info(
        "Marginal admission: %s basis, %d components explain %.1f%% of %d reference "
        "features (%d failed)",
        admission.reference, basis.size, 100 * basis.reference_explained,
        len(formulas) - len(failed), len(failed),
    )
    return basis
