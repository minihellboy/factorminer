"""Custom operator storage, registration, and persistence.

Manages operators invented by the auto-inventor: registers them into the
global operator registry at runtime, and persists them to disk as JSON
metadata plus Python source files for reload across sessions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from factorminer.core.types import (
    OPERATOR_REGISTRY as SPEC_REGISTRY,
)
from factorminer.core.types import (
    OperatorSpec,
    OperatorType,
    SignatureType,
)
from factorminer.operators.registry import OPERATOR_REGISTRY as RUNTIME_REGISTRY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe compilation (shared with auto_inventor.py)
# ---------------------------------------------------------------------------

_SAFE_GLOBALS: dict[str, Any] = {
    "np": np,
    "numpy": np,
    "__builtins__": {},
}


def _compile_operator_code(code: str) -> Callable | None:
    """Compile operator code in a restricted sandbox.

    Returns the ``compute`` function or None on failure.
    """
    safe_ns: dict[str, Any] = dict(_SAFE_GLOBALS)
    try:
        exec(code, safe_ns)  # noqa: S102 -- sandboxed exec
    except Exception as exc:
        logger.warning("Failed to compile custom operator code: %s", exc)
        return None
    fn = safe_ns.get("compute")
    if fn is None or not callable(fn):
        return None
    return fn


# ---------------------------------------------------------------------------
# CustomOperator
# ---------------------------------------------------------------------------

@dataclass
class CustomOperator:
    """A validated, auto-invented operator ready for registration.

    Attributes
    ----------
    name : str
        Canonical operator name.
    spec : OperatorSpec
        Immutable specification matching the type system.
    numpy_code : str
        Python source defining ``compute``.
    numpy_fn : Callable
        Compiled compute function (not persisted; recompiled on load).
    validation_ic : float
        Information coefficient measured during validation.
    invention_iteration : int
        The search iteration in which this operator was invented.
    rationale : str
        Why this operator was proposed.
    """

    name: str
    spec: OperatorSpec
    numpy_code: str
    numpy_fn: Callable
    validation_ic: float = 0.0
    invention_iteration: int = 0
    rationale: str = ""


# ---------------------------------------------------------------------------
# CustomOperatorStore
# ---------------------------------------------------------------------------

class CustomOperatorStore:
    """Manages custom operator lifecycle: register, persist, and reload.

    Parameters
    ----------
    store_dir : str
        Directory for persisting operator metadata and source files.
    """

    def __init__(self, store_dir: str = "./output/custom_operators") -> None:
        self._store_dir = Path(store_dir)
        self._operators: dict[str, CustomOperator] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, op: CustomOperator) -> None:
        """Register a custom operator into both global registries.

        Adds the operator to:
        1. ``core.types.OPERATOR_REGISTRY`` (spec-only registry)
        2. ``operators.registry.OPERATOR_REGISTRY`` (runtime registry with impl)

        Parameters
        ----------
        op : CustomOperator
        """
        # Add to spec registry
        SPEC_REGISTRY[op.name] = op.spec

        # Add to runtime registry (spec, numpy_fn, torch_fn=None)
        RUNTIME_REGISTRY[op.name] = (op.spec, op.numpy_fn, None)

        # Track internally
        self._operators[op.name] = op
        logger.info(
            "Registered custom operator '%s' (IC=%.4f, iteration=%d)",
            op.name,
            op.validation_ic,
            op.invention_iteration,
        )

    def save(self) -> None:
        """Persist all custom operators to disk.

        Creates ``store_dir/`` with:
        - ``index.json``: metadata for all operators
        - ``<name>.py``: Python source for each operator
        """
        self._store_dir.mkdir(parents=True, exist_ok=True)

        index: list[dict[str, Any]] = []
        for name, op in self._operators.items():
            # Save Python source
            src_path = self._store_dir / f"{name}.py"
            src_path.write_text(op.numpy_code, encoding="utf-8")

            # Build metadata entry
            entry = {
                "name": op.name,
                "arity": op.spec.arity,
                "category": op.spec.category.name,
                "signature": op.spec.signature.name,
                "param_names": list(op.spec.param_names),
                "param_defaults": op.spec.param_defaults,
                "param_ranges": {
                    k: list(v) for k, v in op.spec.param_ranges.items()
                },
                "description": op.spec.description,
                "validation_ic": op.validation_ic,
                "invention_iteration": op.invention_iteration,
                "rationale": op.rationale,
            }
            index.append(entry)

        index_path = self._store_dir / "index.json"
        index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Saved %d custom operators to %s", len(index), self._store_dir
        )

    def load(self) -> None:
        """Load custom operators from disk, recompile, and re-register.

        Reads ``store_dir/index.json`` and corresponding ``.py`` source files.
        Operators that fail recompilation are skipped with a warning.
        """
        index_path = self._store_dir / "index.json"
        if not index_path.exists():
            logger.debug("No custom operator index at %s", index_path)
            return

        with open(index_path, encoding="utf-8") as f:
            index: list[dict[str, Any]] = json.load(f)

        loaded = 0
        for entry in index:
            name = entry["name"]
            src_path = self._store_dir / f"{name}.py"
            if not src_path.exists():
                logger.warning("Source file missing for custom operator '%s'", name)
                continue

            numpy_code = src_path.read_text(encoding="utf-8")
            fn = _compile_operator_code(numpy_code)
            if fn is None:
                logger.warning(
                    "Failed to recompile custom operator '%s'; skipping", name
                )
                continue

            spec = OperatorSpec(
                name=name,
                arity=entry["arity"],
                category=OperatorType[entry["category"]],
                signature=SignatureType[entry["signature"]],
                param_names=tuple(entry.get("param_names", [])),
                param_defaults=entry.get("param_defaults", {}),
                param_ranges={
                    k: tuple(v)
                    for k, v in entry.get("param_ranges", {}).items()
                },
                description=entry.get("description", ""),
            )

            op = CustomOperator(
                name=name,
                spec=spec,
                numpy_code=numpy_code,
                numpy_fn=fn,
                validation_ic=entry.get("validation_ic", 0.0),
                invention_iteration=entry.get("invention_iteration", 0),
                rationale=entry.get("rationale", ""),
            )
            self.register(op)
            loaded += 1

        logger.info("Loaded %d / %d custom operators from %s", loaded, len(index), self._store_dir)

    def list_operators(self) -> list[str]:
        """Return names of all registered custom operators."""
        return sorted(self._operators.keys())

    def get_operator(self, name: str) -> CustomOperator | None:
        """Look up a custom operator by name.

        Returns
        -------
        CustomOperator or None
        """
        return self._operators.get(name)
