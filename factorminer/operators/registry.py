"""Central operator registry mapping names to implementations and specs.

Combines the ``OperatorSpec`` definitions from ``core.types`` with the concrete
NumPy / PyTorch function implementations from each category module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from factorminer.core.types import OPERATOR_REGISTRY as SPEC_REGISTRY
from factorminer.core.types import OperatorSpec
from factorminer.operators.arithmetic import ARITHMETIC_OPS
from factorminer.operators.c_backend import C_BACKEND_IMPLS
from factorminer.operators.crosssectional import CROSSSECTIONAL_OPS
from factorminer.operators.logical import LOGICAL_OPS
from factorminer.operators.regression import REGRESSION_OPS
from factorminer.operators.smoothing import SMOOTHING_OPS
from factorminer.operators.statistical import STATISTICAL_OPS
from factorminer.operators.timeseries import TIMESERIES_OPS

try:
    import torch

    _TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH = False

# ---------------------------------------------------------------------------
# Build unified registry: name -> (OperatorSpec, np_fn, torch_fn)
# ---------------------------------------------------------------------------

_ALL_IMPL_TABLES: list[dict[str, tuple[Callable, Callable]]] = [
    ARITHMETIC_OPS,
    STATISTICAL_OPS,
    TIMESERIES_OPS,
    CROSSSECTIONAL_OPS,
    SMOOTHING_OPS,
    REGRESSION_OPS,
    LOGICAL_OPS,
]

# Merge implementation tables
_IMPL: dict[str, tuple[Callable, Callable]] = {}
for table in _ALL_IMPL_TABLES:
    _IMPL.update(table)

# The full registry: name -> (spec, numpy_fn, torch_fn)
OPERATOR_REGISTRY: dict[str, tuple[OperatorSpec, Callable, Callable | None]] = {}

for name, spec in SPEC_REGISTRY.items():
    if name in _IMPL:
        np_fn, torch_fn = _IMPL[name]
        OPERATOR_REGISTRY[name] = (spec, np_fn, torch_fn)
    else:
        # Spec exists but no implementation yet -- register with None fns
        OPERATOR_REGISTRY[name] = (spec, None, None)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_operator(name: str) -> OperatorSpec:
    """Look up an operator spec by name."""
    if name not in OPERATOR_REGISTRY:
        raise KeyError(f"Unknown operator '{name}'. Available: {sorted(OPERATOR_REGISTRY.keys())}")
    return OPERATOR_REGISTRY[name][0]


def get_impl(name: str, backend: str = "numpy") -> Callable:
    """Return the implementation function for a given operator and backend."""
    if name not in OPERATOR_REGISTRY:
        raise KeyError(f"Unknown operator '{name}'")
    spec, np_fn, torch_fn = OPERATOR_REGISTRY[name]
    if backend == "c":
        if name in C_BACKEND_IMPLS:
            return C_BACKEND_IMPLS[name]
        if np_fn is None:
            raise NotImplementedError(f"No C/NumPy implementation for '{name}'")
        return np_fn
    if backend == "torch" or backend == "gpu":
        if torch_fn is None:
            raise NotImplementedError(f"No PyTorch implementation for '{name}'")
        return torch_fn
    if np_fn is None:
        raise NotImplementedError(f"No NumPy implementation for '{name}'")
    return np_fn


def execute_operator(
    name: str,
    *inputs: Any,
    params: dict[str, Any] | None = None,
    backend: str = "numpy",
) -> np.ndarray | torch.Tensor:
    """Execute an operator by name.

    Parameters
    ----------
    name : str
        Operator name (e.g. ``"Add"``, ``"Mean"``).
    *inputs : array-like
        Positional data inputs (1, 2, or 3 depending on arity).
    params : dict, optional
        Extra keyword parameters (e.g. ``{"window": 20}``).
    backend : str
        ``"numpy"`` or ``"torch"`` / ``"gpu"``.

    Returns
    -------
    np.ndarray or torch.Tensor
    """
    fn = get_impl(name, backend)
    kw = params or {}
    return fn(*inputs, **kw)


def list_operators(grouped: bool = True) -> list[str] | dict[str, list[str]]:
    """List all registered operator names.

    Parameters
    ----------
    grouped : bool
        If True, return a dict mapping category name -> list of op names.
        If False, return a flat sorted list.
    """
    if not grouped:
        return sorted(OPERATOR_REGISTRY.keys())

    groups: dict[str, list[str]] = {}
    for name, (spec, _, _) in OPERATOR_REGISTRY.items():
        cat = spec.category.name
        groups.setdefault(cat, []).append(name)
    for cat in groups:
        groups[cat].sort()
    return groups


def implemented_operators() -> list[str]:
    """Return names of operators that have at least a NumPy implementation."""
    return sorted(name for name, (_, np_fn, _) in OPERATOR_REGISTRY.items() if np_fn is not None)
