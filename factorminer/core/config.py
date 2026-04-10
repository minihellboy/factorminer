"""Mining-specific configuration for the Ralph Loop.

Provides a flat configuration dataclass specifically for the mining loop,
separate from the hierarchical Config system in utils/config.py.  This
allows the RalphLoop to accept a simple, focused parameter object while
the full Config handles loading, validation, and serialization.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MiningConfig:
    """Flat configuration controlling the Ralph Loop mining process.

    This is a convenience alias that mirrors the mining-relevant fields
    from the hierarchical Config.  The RalphLoop can accept either this
    or the full ``utils.config.MiningConfig``.
    """

    target_library_size: int = 110
    batch_size: int = 40
    max_iterations: int = 200
    ic_threshold: float = 0.04
    icir_threshold: float = 0.5
    correlation_threshold: float = 0.5
    replacement_ic_min: float = 0.10
    replacement_ic_ratio: float = 1.3
    fast_screen_assets: int = 100
    num_workers: int = 40
    output_dir: str = "./output"
    gpu_device: str = "cuda:0"
    backend: str = "numpy"
    redundancy_metric: str = "spearman"
    signal_failure_policy: str = "reject"
    memory_policy: str = "paper"
    memory_regime_lookback_window: int = 60

    def validate(self) -> None:
        """Basic sanity checks on parameter values."""
        if self.target_library_size < 1:
            raise ValueError("target_library_size must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not (0.0 < self.ic_threshold < 1.0):
            raise ValueError("ic_threshold must be in (0, 1)")
        if not (0.0 < self.correlation_threshold <= 1.0):
            raise ValueError("correlation_threshold must be in (0, 1]")
        if self.replacement_ic_min <= self.ic_threshold:
            raise ValueError("replacement_ic_min must be > ic_threshold")
        if self.replacement_ic_ratio < 1.0:
            raise ValueError("replacement_ic_ratio must be >= 1.0")
        if self.backend not in ("gpu", "numpy", "c"):
            raise ValueError(
                f"backend must be one of: gpu, numpy, c (got '{self.backend}')"
            )
        if self.redundancy_metric not in ("spearman", "pearson", "distance_correlation"):
            raise ValueError(
                "redundancy_metric must be one of: spearman, pearson, distance_correlation"
            )
        if self.signal_failure_policy not in ("reject", "synthetic", "raise"):
            raise ValueError(
                "signal_failure_policy must be one of: reject, synthetic, raise"
            )
        if self.memory_policy not in (
            "paper",
            "none",
            "no_memory",
            "kg",
            "knowledge_graph",
            "family_aware",
            "family-aware",
            "regime_aware",
            "regime-aware",
        ):
            raise ValueError(
                "memory_policy must be one of: paper, none, no_memory, kg, "
                "family_aware, regime_aware"
            )
        if self.memory_regime_lookback_window < 5:
            raise ValueError("memory_regime_lookback_window must be >= 5")
