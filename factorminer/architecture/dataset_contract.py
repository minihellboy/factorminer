"""Stable dataset pipeline contracts for mining and analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _safe_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


@dataclass(frozen=True)
class DatasetContract:
    """Canonical description of how raw data became the mining tensors."""

    feature_names: list[str]
    data_shape: tuple[int, ...]
    returns_shape: tuple[int, ...]
    default_target: str
    target_names: list[str]
    target_horizons: dict[str, int] = field(default_factory=dict)
    train_period: list[str] = field(default_factory=list)
    test_period: list[str] = field(default_factory=list)
    asset_count: int = 0
    period_count: int = 0
    split_sizes: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_runtime_dataset(cls, cfg: Any, dataset: Any) -> DatasetContract:
        target_specs = getattr(dataset, "target_specs", {}) or {}
        asset_ids = getattr(dataset, "asset_ids", None)
        timestamps = getattr(dataset, "timestamps", None)
        splits = getattr(dataset, "splits", None) or {}
        return cls(
            feature_names=list(getattr(dataset, "data_dict", {}).keys()),
            data_shape=tuple(np.shape(getattr(dataset, "data_tensor", ()))),
            returns_shape=tuple(np.shape(getattr(dataset, "returns", ()))),
            default_target=str(
                getattr(dataset, "default_target", getattr(cfg.data, "default_target", "paper"))
            ),
            target_names=list((getattr(dataset, "target_panels", {}) or {}).keys())
            or [str(getattr(cfg.data, "default_target", "paper"))],
            target_horizons={
                name: max(int(getattr(spec, "holding_bars", 1)), 1)
                for name, spec in target_specs.items()
            },
            train_period=list(getattr(cfg.data, "train_period", [])),
            test_period=list(getattr(cfg.data, "test_period", [])),
            asset_count=int(_safe_len(asset_ids)),
            period_count=int(_safe_len(timestamps)),
            split_sizes={
                name: int(getattr(split, "size", 0))
                for name, split in splits.items()
            },
        )

    @classmethod
    def from_arrays(
        cls,
        cfg: Any,
        *,
        data_tensor: Any,
        returns: np.ndarray,
        target_panels: dict[str, np.ndarray] | None = None,
        target_horizons: dict[str, int] | None = None,
    ) -> DatasetContract:
        feature_names = list(getattr(getattr(cfg, "data", None), "features", []))
        return cls(
            feature_names=feature_names,
            data_shape=tuple(np.shape(data_tensor)),
            returns_shape=tuple(np.shape(returns)),
            default_target=str(getattr(getattr(cfg, "data", None), "default_target", "paper")),
            target_names=list((target_panels or {}).keys()) or ["paper"],
            target_horizons=dict(target_horizons or {}),
            train_period=list(getattr(getattr(cfg, "data", None), "train_period", [])),
            test_period=list(getattr(getattr(cfg, "data", None), "test_period", [])),
            asset_count=int(np.shape(returns)[0]) if np.ndim(returns) >= 1 else 0,
            period_count=int(np.shape(returns)[1]) if np.ndim(returns) >= 2 else 0,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
