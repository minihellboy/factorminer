"""Shared sandbox helpers for NumPy-only custom operators."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

SAFE_GLOBALS: dict[str, Any] = {
    "np": np,
    "numpy": np,
    "__builtins__": {},
}

BLOCKED_TOKENS: tuple[str, ...] = (
    "import ",
    "__import__",
    "os.",
    "sys.",
    "subprocess",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "getattr(",
    "setattr(",
    "delattr(",
    "globals(",
    "locals(",
    "__class__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "breakpoint(",
    "exit(",
    "quit(",
)


def find_blocked_token(code: str) -> str | None:
    """Return the first blocked token found in operator source."""
    code_lower = code.lower()
    for token in BLOCKED_TOKENS:
        if token.lower() in code_lower:
            return token
    return None


def compile_numpy_operator(code: str) -> Callable | None:
    """Compile source defining ``compute`` with only NumPy available."""
    token = find_blocked_token(code)
    if token is not None:
        logger.warning("Blocked token '%s' found in operator code", token)
        return None

    safe_ns: dict[str, Any] = dict(SAFE_GLOBALS)
    try:
        exec(code, safe_ns)  # noqa: S102 -- intentional restricted operator sandbox
    except Exception as exc:
        logger.warning("Operator compilation failed: %s", exc)
        return None

    fn = safe_ns.get("compute")
    if fn is None or not callable(fn):
        logger.warning("No callable 'compute' found in operator code")
        return None
    return cast(Callable, fn)
