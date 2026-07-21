"""Compatibility imports for dependence metrics now owned by the domain layer.

Internal code must import :mod:`factorminer.domain.dependence` directly.  This
module remains temporarily available for external callers while compatibility
paths are audited and removed in the dedicated cleanup phase.
"""

from factorminer.domain.dependence import (
    DependenceMetric,
    DistanceCorrelationMetric,
    PearsonDependenceMetric,
    SpearmanDependenceMetric,
    build_dependence_metric,
    library_span_basis,
    orthogonal_escape_score,
    residual_alignment_score,
    spectral_compression_score,
)

__all__ = [
    "DependenceMetric",
    "DistanceCorrelationMetric",
    "PearsonDependenceMetric",
    "SpearmanDependenceMetric",
    "build_dependence_metric",
    "library_span_basis",
    "orthogonal_escape_score",
    "residual_alignment_score",
    "spectral_compression_score",
]
