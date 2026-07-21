"""Stable domain contracts shared across FactorMiner application layers."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DependenceMetric",
    "DistanceCorrelationMetric",
    "EvidencePack",
    "FrozenPayload",
    "HumanAttestation",
    "PearsonDependenceMetric",
    "SpearmanDependenceMetric",
    "build_dependence_metric",
    "library_span_basis",
    "orthogonal_escape_score",
    "residual_alignment_score",
    "spectral_compression_score",
]

_ATTRIBUTE_MAP = {
    name: ("factorminer.domain.dependence", name)
    for name in __all__
}
_ATTRIBUTE_MAP.update(
    {
        name: ("factorminer.domain.evidence", name)
        for name in ("EvidencePack", "FrozenPayload", "HumanAttestation")
    }
)


def __getattr__(name: str):
    """Resolve public domain contracts without eager numerical imports."""
    if name not in _ATTRIBUTE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _ATTRIBUTE_MAP[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
