"""GPU acceleration utilities for FactorMiner operators.

Provides device management, tensor conversion helpers, and batch execution
for parallel factor evaluation on CUDA GPUs with automatic CPU fallback.
"""

from __future__ import annotations

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

class DeviceManager:
    """Singleton-style helper that picks the best available device."""

    def __init__(self) -> None:
        self._device: torch.device | None = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = self._select_device()
        return self._device

    @device.setter
    def device(self, dev: str | torch.device) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")
        self._device = torch.device(dev)

    @staticmethod
    def _select_device() -> torch.device:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def is_gpu(self) -> bool:
        return self.device.type in ("cuda", "mps")

    def reset(self) -> None:
        self._device = None


device_manager = DeviceManager()


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def to_tensor(
    arr: np.ndarray,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor on the target device."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")
    dev = device or device_manager.device
    dt = dtype or torch.float32
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=dt, device=torch.device("cpu")).to(dev)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor back to a NumPy array."""
    return tensor.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Batch execution helper
# ---------------------------------------------------------------------------

def batch_execute(
    fn,
    inputs: list,
    params_list: list[dict],
    backend: str = "numpy",
) -> list:
    """Execute a function over multiple parameter sets.

    Useful for evaluating many factors in parallel on the GPU by batching
    the inputs into a single large tensor operation.
    """
    results = []
    for params in params_list:
        results.append(fn(*inputs, **params))
    return results


def torch_available() -> bool:
    """Return True if PyTorch is importable."""
    return _TORCH_AVAILABLE
