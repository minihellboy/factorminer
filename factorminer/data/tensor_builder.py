"""Build the data tensor D in R^(M x T x F) for FactorMiner.

Converts preprocessed panel data into dense 3-D arrays indexed by
(assets, time_periods, features).  Supports numpy and optional torch backends.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default feature ordering matching the paper specification
DEFAULT_FEATURES: list[str] = [
    "open", "high", "low", "close", "volume", "amount", "vwap", "returns",
]

Backend = Literal["numpy", "torch", "cupy"]


@dataclass(frozen=True)
class TargetSpec:
    """Definition of one aligned forward-return target."""

    name: str
    entry_delay_bars: int
    holding_bars: int
    price_pair: str = "open_to_close"
    return_transform: str = "simple"

    @property
    def column_name(self) -> str:
        return "target" if self.name == "paper" else f"target_{self.name}"


@dataclass
class TensorConfig:
    """Configuration for tensor construction.

    Attributes
    ----------
    features : list of str
        Ordered feature columns to include in the tensor.
    backend : str
        ``"numpy"``, ``"torch"``, or ``"cupy"``.
    dtype : str
        Numeric dtype string (e.g. ``"float32"``).
    train_end : str or None
        Inclusive upper bound for the training period (ISO datetime).
    test_start : str or None
        Inclusive lower bound for the test period (ISO datetime).
    m_fast : int or None
        Number of assets for the fast screening subset.  When *None*,
        no fast subset is produced.
    seed : int
        Random seed for reproducible asset sampling.
    target_column : str
        Name of the column holding the target variable (created by
        :func:`compute_target`).
    """

    features: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    backend: Backend = "numpy"
    dtype: str = "float32"
    train_end: str | None = None
    test_start: str | None = None
    m_fast: int | None = None
    seed: int = 42
    target_column: str = "target"
    target_columns: list[str] = field(default_factory=list)
    default_target: str = "target"


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the target: next-bar open-to-close return.

    For each asset the target at time *t* is defined as::

        target[t] = close[t+1] / open[t+1] - 1

    The last bar of each asset has a NaN target.
    """
    return compute_targets(
        df,
        [
            TargetSpec(
                name="paper",
                entry_delay_bars=1,
                holding_bars=1,
                price_pair="open_to_close",
                return_transform="simple",
            )
        ],
    )


def compute_targets(
    df: pd.DataFrame,
    target_specs: Sequence[TargetSpec],
) -> pd.DataFrame:
    """Compute one or more named forward-return targets on the same panel."""
    df = df.sort_values(["asset_id", "datetime"]).copy()

    for spec in target_specs:
        start_col, end_col, start_offset, end_offset = _resolve_target_offsets(spec)
        start_values = df.groupby("asset_id")[start_col].shift(-start_offset)
        end_values = df.groupby("asset_id")[end_col].shift(-end_offset)
        if spec.return_transform == "log":
            df[spec.column_name] = np.log(end_values / start_values)
        else:
            df[spec.column_name] = end_values / start_values - 1.0

    return df


def _resolve_target_offsets(spec: TargetSpec) -> tuple[str, str, int, int]:
    """Map a target spec to start/end price columns and offsets."""
    if spec.entry_delay_bars < 0 or spec.holding_bars < 0:
        raise ValueError("TargetSpec entry_delay_bars and holding_bars must be >= 0")

    if spec.price_pair == "open_to_close":
        if spec.holding_bars < 1:
            raise ValueError("open_to_close targets require holding_bars >= 1")
        return (
            "open",
            "close",
            spec.entry_delay_bars,
            spec.entry_delay_bars + spec.holding_bars - 1,
        )
    if spec.price_pair == "close_to_close":
        if spec.holding_bars < 1:
            raise ValueError("close_to_close targets require holding_bars >= 1")
        return (
            "close",
            "close",
            spec.entry_delay_bars,
            spec.entry_delay_bars + spec.holding_bars,
        )
    if spec.price_pair == "open_to_open":
        if spec.holding_bars < 1:
            raise ValueError("open_to_open targets require holding_bars >= 1")
        return (
            "open",
            "open",
            spec.entry_delay_bars,
            spec.entry_delay_bars + spec.holding_bars,
        )
    if spec.price_pair == "close_to_open":
        if spec.holding_bars < 1:
            raise ValueError("close_to_open targets require holding_bars >= 1")
        return (
            "close",
            "open",
            spec.entry_delay_bars,
            spec.entry_delay_bars + spec.holding_bars,
        )
    raise ValueError(f"Unknown TargetSpec price_pair: {spec.price_pair}")


# ---------------------------------------------------------------------------
# Tensor construction helpers
# ---------------------------------------------------------------------------

def _to_backend(arr: np.ndarray, backend: Backend, dtype: str):
    """Convert a numpy array to the requested backend."""
    np_dtype = getattr(np, dtype, np.float32)
    arr = arr.astype(np_dtype)

    if backend == "numpy":
        return arr

    if backend == "torch":
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for backend='torch'. "
                "Install with: pip install torch"
            ) from exc
        torch_dtype = getattr(torch, dtype, torch.float32)
        return torch.from_numpy(arr).to(torch_dtype)

    if backend == "cupy":
        try:
            import cupy  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "CuPy is required for backend='cupy'. "
                "Install with: pip install cupy"
            ) from exc
        return cupy.asarray(arr, dtype=dtype)

    raise ValueError(f"Unknown backend: {backend}")


def _build_3d(
    df: pd.DataFrame,
    asset_ids: np.ndarray,
    timestamps: np.ndarray,
    columns: Sequence[str],
) -> np.ndarray:
    """Pivot panel data into a dense (M, T, F) numpy array."""
    M = len(asset_ids)
    T = len(timestamps)
    F = len(columns)
    tensor = np.full((M, T, F), np.nan, dtype=np.float64)

    asset_map = {a: i for i, a in enumerate(asset_ids)}
    time_map = {t: j for j, t in enumerate(timestamps)}

    df_idx = df.copy()
    df_idx["_ai"] = df_idx["asset_id"].map(asset_map)
    df_idx["_ti"] = df_idx["datetime"].map(time_map)
    df_idx = df_idx.dropna(subset=["_ai", "_ti"])
    df_idx["_ai"] = df_idx["_ai"].astype(int)
    df_idx["_ti"] = df_idx["_ti"].astype(int)

    values = df_idx[list(columns)].to_numpy(dtype=np.float64)
    tensor[df_idx["_ai"].values, df_idx["_ti"].values, :] = values

    return tensor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class TensorDataset:
    """Container for the built tensor and associated metadata.

    Attributes
    ----------
    data : array-like
        Feature tensor of shape ``(M, T, F)``.
    target : array-like or None
        Target array of shape ``(M, T)``.
    asset_ids : np.ndarray
        Asset identifier for each row in the first axis.
    timestamps : np.ndarray
        Datetime for each position in the second axis.
    feature_names : list of str
        Feature name for each slice in the third axis.
    """

    data: object  # np.ndarray | torch.Tensor | cupy.ndarray
    target: object  # same type or None
    asset_ids: np.ndarray
    timestamps: np.ndarray
    feature_names: list[str]
    targets: dict[str, object] = field(default_factory=dict)
    default_target: str = "target"


def build_tensor(
    df: pd.DataFrame,
    config: TensorConfig | None = None,
) -> TensorDataset:
    """Build a dense 3-D tensor from preprocessed panel data.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed market data.  Must include ``datetime``, ``asset_id``,
        and all columns listed in ``config.features``.
    config : TensorConfig, optional
        Build configuration.  Uses defaults when *None*.

    Returns
    -------
    TensorDataset
        Dense tensor and metadata.
    """
    if config is None:
        config = TensorConfig()

    # Validate required feature columns
    missing = [f for f in config.features if f not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing feature columns: {missing}")

    # Sorted unique axes
    asset_ids = np.sort(df["asset_id"].unique())
    timestamps = np.sort(df["datetime"].unique())

    logger.info(
        "Building tensor: %d assets x %d time steps x %d features",
        len(asset_ids),
        len(timestamps),
        len(config.features),
    )

    data_np = _build_3d(df, asset_ids, timestamps, config.features)

    # Target
    resolved_target_columns = list(config.target_columns or [config.target_column])
    target_arrays_np: dict[str, np.ndarray] = {}
    for target_column in resolved_target_columns:
        if target_column not in df.columns:
            continue
        target_np = _build_3d(df, asset_ids, timestamps, [target_column])
        target_arrays_np[target_column] = target_np[:, :, 0]

    target_np: np.ndarray | None = None
    default_target_name = config.default_target
    if target_arrays_np:
        target_np = target_arrays_np.get(default_target_name)
        if target_np is None:
            first_target = next(iter(target_arrays_np))
            target_np = target_arrays_np[first_target]
            default_target_name = first_target

    data = _to_backend(data_np, config.backend, config.dtype)
    target = _to_backend(target_np, config.backend, config.dtype) if target_np is not None else None
    targets = {
        name: _to_backend(target_arr, config.backend, config.dtype)
        for name, target_arr in target_arrays_np.items()
    }

    return TensorDataset(
        data=data,
        target=target,
        asset_ids=asset_ids,
        timestamps=timestamps,
        feature_names=list(config.features),
        targets=targets,
        default_target=default_target_name,
    )


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def temporal_split(
    ds: TensorDataset,
    train_end: str | None = None,
    test_start: str | None = None,
) -> tuple[TensorDataset, TensorDataset]:
    """Split a :class:`TensorDataset` into train and test sets along time.

    Parameters
    ----------
    ds : TensorDataset
        Full dataset.
    train_end : str, optional
        Inclusive upper bound for training timestamps.
    test_start : str, optional
        Inclusive lower bound for test timestamps.  When *None* defaults to
        the bar immediately after *train_end*.

    Returns
    -------
    tuple of TensorDataset
        ``(train, test)`` datasets.
    """
    ts = pd.to_datetime(ds.timestamps)

    if train_end is not None:
        train_mask = ts <= pd.Timestamp(train_end)
    else:
        # Default: first 80%
        split_idx = int(len(ts) * 0.8)
        train_mask = np.arange(len(ts)) < split_idx

    if test_start is not None:
        test_mask = ts >= pd.Timestamp(test_start)
    else:
        test_mask = ~train_mask

    def _slice(mask):
        idx = np.where(mask)[0]
        # np arrays: index along axis 1 (time)
        d = ds.data
        t = ds.target
        targets = ds.targets

        # Handle different backends
        if hasattr(d, "numpy"):
            # torch tensor
            d_slice = d[:, idx, :]
            t_slice = t[:, idx] if t is not None else None
        elif hasattr(d, "get"):
            # cupy
            d_slice = d[:, idx, :]
            t_slice = t[:, idx] if t is not None else None
        else:
            # numpy
            d_slice = d[:, idx, :]
            t_slice = t[:, idx] if t is not None else None

        return TensorDataset(
            data=d_slice,
            target=t_slice,
            targets={
                name: target[:, idx] if target is not None else None
                for name, target in targets.items()
            },
            default_target=ds.default_target,
            asset_ids=ds.asset_ids,
            timestamps=ds.timestamps[idx],
            feature_names=ds.feature_names,
        )

    return _slice(train_mask), _slice(test_mask)


# ---------------------------------------------------------------------------
# Asset subset sampling
# ---------------------------------------------------------------------------

def sample_assets(
    ds: TensorDataset,
    m: int,
    seed: int = 42,
) -> TensorDataset:
    """Return a random subset of *m* assets from *ds*.

    Parameters
    ----------
    ds : TensorDataset
        Full dataset.
    m : int
        Number of assets to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    TensorDataset
        Subset with *m* assets.
    """
    rng = np.random.default_rng(seed)
    M = len(ds.asset_ids)
    if m >= M:
        logger.warning("Requested m=%d >= total assets %d; returning all", m, M)
        return ds

    idx = np.sort(rng.choice(M, size=m, replace=False))
    d = ds.data
    t = ds.target
    targets = ds.targets

    if hasattr(d, "numpy"):
        d_sub = d[idx, :, :]
        t_sub = t[idx, :] if t is not None else None
    elif hasattr(d, "get"):
        d_sub = d[idx, :, :]
        t_sub = t[idx, :] if t is not None else None
    else:
        d_sub = d[idx, :, :]
        t_sub = t[idx, :] if t is not None else None

    return TensorDataset(
        data=d_sub,
        target=t_sub,
        targets={
            name: target[idx, :] if target is not None else None
            for name, target in targets.items()
        },
        default_target=ds.default_target,
        asset_ids=ds.asset_ids[idx],
        timestamps=ds.timestamps,
        feature_names=ds.feature_names,
    )


def build_pipeline(
    df: pd.DataFrame,
    config: TensorConfig | None = None,
) -> TensorDataset | tuple[TensorDataset, TensorDataset]:
    """End-to-end: compute target, build tensor, optionally split.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed market data.
    config : TensorConfig, optional
        Configuration.

    Returns
    -------
    TensorDataset or tuple
        If ``config.train_end`` or ``config.test_start`` is set, returns
        ``(train, test)``; otherwise the full dataset.
    """
    if config is None:
        config = TensorConfig()

    df = compute_target(df)
    ds = build_tensor(df, config)

    if config.train_end is not None or config.test_start is not None:
        return temporal_split(ds, train_end=config.train_end, test_start=config.test_start)

    return ds
