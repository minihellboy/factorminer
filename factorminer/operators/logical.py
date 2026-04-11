"""Conditional and comparison operators (element-wise).

All operators are element-wise on ``(M, T)`` arrays.
Boolean-like outputs use ``1.0`` / ``0.0`` (float), not Python bool.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


# ===========================================================================
# NumPy implementations
# ===========================================================================

def if_else_np(cond: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Where cond > 0 return x, else y.  NaN in cond -> NaN."""
    result = np.where(cond > 0, x, y)
    result[np.isnan(cond)] = np.nan
    return result


def greater_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.where(x > y, 1.0, 0.0)
    out[np.isnan(x) | np.isnan(y)] = np.nan
    return out


def less_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.where(x < y, 1.0, 0.0)
    out[np.isnan(x) | np.isnan(y)] = np.nan
    return out


def greater_equal_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.where(x >= y, 1.0, 0.0)
    out[np.isnan(x) | np.isnan(y)] = np.nan
    return out


def less_equal_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.where(x <= y, 1.0, 0.0)
    out[np.isnan(x) | np.isnan(y)] = np.nan
    return out


def equal_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.where(np.abs(x - y) < 1e-10, 1.0, 0.0)
    out[np.isnan(x) | np.isnan(y)] = np.nan
    return out


def and_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.where((x > 0) & (y > 0), 1.0, 0.0)
    out[np.isnan(x) | np.isnan(y)] = np.nan
    return out


def or_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.where((x > 0) | (y > 0), 1.0, 0.0)
    out[np.isnan(x) | np.isnan(y)] = np.nan
    return out


def not_np(x: np.ndarray) -> np.ndarray:
    out = np.where(x > 0, 0.0, 1.0)
    out[np.isnan(x)] = np.nan
    return out


def sign_np(x: np.ndarray) -> np.ndarray:
    return np.sign(x)


def max2_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.fmax(x, y)


def min2_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.fmin(x, y)


def ne_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.where(np.abs(x - y) >= 1e-10, 1.0, 0.0)
    out[np.isnan(x) | np.isnan(y)] = np.nan
    return out


# ===========================================================================
# PyTorch implementations
# ===========================================================================

def if_else_torch(cond: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = torch.where(cond > 0, x, y)
    result[torch.isnan(cond)] = float("nan")
    return result


def greater_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.where(x > y, 1.0, 0.0)
    out[torch.isnan(x) | torch.isnan(y)] = float("nan")
    return out


def less_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.where(x < y, 1.0, 0.0)
    out[torch.isnan(x) | torch.isnan(y)] = float("nan")
    return out


def greater_equal_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.where(x >= y, 1.0, 0.0)
    out[torch.isnan(x) | torch.isnan(y)] = float("nan")
    return out


def less_equal_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.where(x <= y, 1.0, 0.0)
    out[torch.isnan(x) | torch.isnan(y)] = float("nan")
    return out


def equal_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.where((x - y).abs() < 1e-10, 1.0, 0.0)
    out[torch.isnan(x) | torch.isnan(y)] = float("nan")
    return out


def and_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.where((x > 0) & (y > 0), 1.0, 0.0)
    out[torch.isnan(x) | torch.isnan(y)] = float("nan")
    return out


def or_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.where((x > 0) | (y > 0), 1.0, 0.0)
    out[torch.isnan(x) | torch.isnan(y)] = float("nan")
    return out


def not_torch(x: torch.Tensor) -> torch.Tensor:
    out = torch.where(x > 0, 0.0, 1.0)
    out[torch.isnan(x)] = float("nan")
    return out


def sign_torch(x: torch.Tensor) -> torch.Tensor:
    return x.sign()


def max2_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.fmax(x, y)


def min2_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.fmin(x, y)


def ne_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.where((x - y).abs() >= 1e-10, 1.0, 0.0)
    out[torch.isnan(x) | torch.isnan(y)] = float("nan")
    return out


# ===========================================================================
# Registration table
# ===========================================================================

LOGICAL_OPS = {
    "IfElse": (if_else_np, if_else_torch),
    "Greater": (greater_np, greater_torch),
    "GreaterEqual": (greater_equal_np, greater_equal_torch),
    "Less": (less_np, less_torch),
    "LessEqual": (less_equal_np, less_equal_torch),
    "Equal": (equal_np, equal_torch),
    "Ne": (ne_np, ne_torch),
    "And": (and_np, and_torch),
    "Or": (or_np, or_torch),
    "Not": (not_np, not_torch),
}
