"""Data preprocessing pipeline for FactorMiner.

Handles derived feature computation, missing data imputation, trading halt
detection, cross-sectional standardisation, winsorisation, and quality checks.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CROSS_FILL_METHODS = frozenset({"median", "mean", "none"})


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline.

    Attributes
    ----------
    winsor_lower : float
        Lower percentile for winsorisation (0-100).
    winsor_upper : float
        Upper percentile for winsorisation (0-100). Bounds of ``0`` and
        ``100`` disable winsorisation.
    min_nonnan_ratio : float
        Minimum fraction of non-NaN values required per cross-section
        for a time step to be kept.
    ffill_limit : int or None
        Maximum number of consecutive NaN values to forward-fill within
        each intraday session.
    cross_fill_method : str
        Cross-sectional fill method after forward fill: ``"median"``,
        ``"mean"``, or ``"none"`` to leave remaining gaps missing (needed for
        unstandardised prices, where a peer median is not a valid price).
    standardise : bool
        Whether to apply cross-sectional z-score standardisation.
    halt_volume_threshold : float
        Volume below this value flags a bar as a trading halt.
    features_to_standardise : list of str
        Column names subject to standardisation and winsorisation.
    """

    winsor_lower: float = 1.0
    winsor_upper: float = 99.0
    min_nonnan_ratio: float = 0.5
    ffill_limit: int | None = None
    cross_fill_method: str = "median"
    standardise: bool = True
    halt_volume_threshold: float = 0.0
    features_to_standardise: list[str] = field(default_factory=lambda: [
        "open", "high", "low", "close", "volume", "amount", "vwap", "returns",
    ])

    def validate(self) -> None:
        if not 0.0 <= self.winsor_lower < self.winsor_upper <= 100.0:
            raise ValueError("preprocess winsor bounds must satisfy 0 <= lower < upper <= 100")
        if not 0.0 <= self.min_nonnan_ratio <= 1.0:
            raise ValueError("preprocess.min_nonnan_ratio must be in [0, 1]")
        if self.ffill_limit is not None and self.ffill_limit < 1:
            raise ValueError("preprocess.ffill_limit must be positive or null")
        if self.cross_fill_method not in CROSS_FILL_METHODS:
            raise ValueError(
                f"preprocess.cross_fill_method must be one of {sorted(CROSS_FILL_METHODS)}"
            )


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a ``vwap`` column, deriving amount / volume where none is supplied.

    A supplied ``vwap`` is kept: vendors often quote amount and volume in
    different units (e.g. thousand-yuan amount against lot volume), so
    amount / volume can differ from the true VWAP by a constant factor.
    Missing values are derived; the result is NaN when volume is zero.
    """
    df = df.copy()
    amount = df["amount"].to_numpy(dtype=np.float64, copy=False)
    volume = df["volume"].to_numpy(dtype=np.float64, copy=False)
    vwap = np.full(len(df), np.nan, dtype=np.float64)
    np.divide(amount, volume, out=vwap, where=volume > 0)
    if "vwap" in df.columns:
        supplied = pd.to_numeric(df["vwap"], errors="coerce").to_numpy(dtype=np.float64)
        vwap = np.where(np.isfinite(supplied), supplied, vwap)
    df["vwap"] = vwap
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``returns`` column: close-to-close percentage change per asset.

    Returns are computed as ``close[t] / close[t-1] - 1`` within each asset.
    The first observation per asset is NaN.
    """
    df = df.sort_values(["asset_id", "datetime"])
    df["returns"] = df.groupby("asset_id")["close"].pct_change()
    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all derived features (vwap and returns)."""
    df = compute_vwap(df)
    df = compute_returns(df)
    return df


# ---------------------------------------------------------------------------
# Trading halt handling
# ---------------------------------------------------------------------------

def flag_halts(
    df: pd.DataFrame,
    volume_threshold: float = 0.0,
) -> pd.DataFrame:
    """Add boolean ``is_halt`` column.

    A bar is considered a trading halt when:
    - Volume is exactly zero (or below *volume_threshold*), **and**
    - open == high == low == close (no price movement).
    """
    df = df.copy()
    zero_volume = df["volume"] <= volume_threshold
    flat_price = (
        (df["open"] == df["high"])
        & (df["high"] == df["low"])
        & (df["low"] == df["close"])
    )
    df["is_halt"] = zero_volume & flat_price
    n_halt = df["is_halt"].sum()
    if n_halt > 0:
        logger.info("Flagged %d halt bars (%.2f%%)", n_halt, 100 * n_halt / len(df))
    return df


def mask_halts(df: pd.DataFrame) -> pd.DataFrame:
    """Set OHLCV and derived columns to NaN for halted bars."""
    if "is_halt" not in df.columns:
        return df
    df = df.copy()
    mask = df["is_halt"]
    cols_to_nan = [
        c for c in ["open", "high", "low", "close", "volume", "amount", "vwap", "returns"]
        if c in df.columns
    ]
    df.loc[mask, cols_to_nan] = np.nan
    return df


# ---------------------------------------------------------------------------
# Missing data handling
# ---------------------------------------------------------------------------

def fill_missing(
    df: pd.DataFrame,
    ffill_limit: int | None = None,
    cross_fill_method: str = "median",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fill missing values using a two-stage strategy.

    Stage 1 – Forward fill within each (asset, date) group so that NaNs
    from halts / gaps are filled from the last valid intraday observation.

    Stage 2 – Cross-sectional fill: remaining NaNs in each time step are
    replaced with the cross-sectional median (or mean).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``datetime`` and ``asset_id`` columns.
    ffill_limit : int or None
        Max consecutive NaN values to forward-fill.
    cross_fill_method : str
        ``"median"`` or ``"mean"`` for the cross-sectional stage, or
        ``"none"`` to skip it.
    columns : sequence of str, optional
        Columns to fill.  Defaults to numeric columns.
    """
    if cross_fill_method not in CROSS_FILL_METHODS:
        raise ValueError(f"Unknown cross_fill_method: {cross_fill_method}")
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns = [c for c in columns if c in df.columns]
    missing = [c for c in columns if df[c].isna().any()]
    if not missing:
        return df

    # A single grouped fill replaces one Python callback per asset/session/column.
    day = df["datetime"].dt.normalize()
    df[missing] = df.groupby([df["asset_id"], day], sort=False)[missing].ffill(
        limit=ffill_limit
    )

    if cross_fill_method == "none":
        return df
    for col in missing:
        if not df[col].isna().any():
            continue
        cross_vals = df.groupby("datetime", sort=False)[col].transform(cross_fill_method)
        df[col] = df[col].fillna(cross_vals)
    return df


# ---------------------------------------------------------------------------
# Winsorisation
# ---------------------------------------------------------------------------

def winsorise(
    df: pd.DataFrame,
    columns: Sequence[str],
    lower: float = 1.0,
    upper: float = 99.0,
) -> pd.DataFrame:
    """Clip values in *columns* to the [lower, upper] percentile range
    computed cross-sectionally at each time step.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``datetime`` column.
    columns : sequence of str
        Columns to winsorise.
    lower, upper : float
        Percentile bounds (0-100). ``0`` and ``100`` leave values unchanged.
    """
    df = df.copy()
    if lower <= 0.0 and upper >= 100.0:
        return df
    columns = [c for c in columns if c in df.columns]

    for col in columns:
        grouped = df.groupby("datetime", sort=False)[col]
        bounds = grouped.quantile([lower / 100.0, upper / 100.0]).unstack()
        lo = df["datetime"].map(bounds[lower / 100.0])
        hi = df["datetime"].map(bounds[upper / 100.0])
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ---------------------------------------------------------------------------
# Cross-sectional standardisation
# ---------------------------------------------------------------------------

def cross_sectional_standardise(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Z-score standardise *columns* cross-sectionally at each time step.

    ``x_std = (x - mean) / std`` where mean and std are computed across
    all assets at the same datetime.  Groups with std == 0 are set to 0.
    """
    df = df.copy()
    columns = [c for c in columns if c in df.columns]

    for col in columns:
        grp = df.groupby("datetime")[col]
        mu = grp.transform("mean")
        sigma = grp.transform("std")
        sigma = sigma.replace(0, np.nan)
        df[col] = (df[col] - mu) / sigma
        df[col] = df[col].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def quality_check(
    df: pd.DataFrame,
    min_nonnan_ratio: float = 0.5,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Drop time steps where the fraction of non-NaN values across assets
    is below *min_nonnan_ratio*.

    Parameters
    ----------
    df : pd.DataFrame
        Market data with ``datetime`` and ``asset_id``.
    min_nonnan_ratio : float
        Minimum fraction (0-1) of assets with valid data at each time step.
    columns : sequence of str, optional
        Columns to check.  Defaults to OHLCV columns.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with low-coverage time steps removed.
    """
    if columns is None:
        columns = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]

    n_assets = df["asset_id"].nunique()
    if n_assets == 0:
        return df

    valid_rows = df[list(columns)].notna().all(axis=1)
    counts = valid_rows.groupby(df["datetime"], sort=False).sum()
    valid_dts = counts.index[counts / n_assets >= min_nonnan_ratio]
    before = df["datetime"].nunique()
    df = df[df["datetime"].isin(valid_dts)]
    after = df["datetime"].nunique()
    if before > after:
        logger.info(
            "Quality check removed %d/%d time steps (min_nonnan_ratio=%.2f)",
            before - after,
            before,
            min_nonnan_ratio,
        )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess(
    df: pd.DataFrame,
    config: PreprocessConfig | None = None,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market data with at least the columns: datetime, asset_id,
        open, high, low, close, volume, amount.
    config : PreprocessConfig, optional
        Pipeline configuration.  Uses defaults when *None*.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with derived features, cleaned and
        standardised values.
    """
    if config is None:
        config = PreprocessConfig()
    config.validate()

    logger.info("Preprocessing %d rows ...", len(df))

    # 1. Derive features
    df = compute_derived_features(df)

    # 2. Flag and mask trading halts
    df = flag_halts(df, volume_threshold=config.halt_volume_threshold)
    df = mask_halts(df)

    # 3. Fill missing data
    df = fill_missing(
        df,
        ffill_limit=config.ffill_limit,
        cross_fill_method=config.cross_fill_method,
    )

    # 4. Quality check
    df = quality_check(df, min_nonnan_ratio=config.min_nonnan_ratio)

    # 5. Winsorise
    feat_cols = [c for c in config.features_to_standardise if c in df.columns]
    df = winsorise(df, columns=feat_cols, lower=config.winsor_lower, upper=config.winsor_upper)

    # 6. Cross-sectional standardisation
    if config.standardise:
        df = cross_sectional_standardise(df, columns=feat_cols)

    logger.info("Preprocessing complete: %d rows, %d columns", len(df), len(df.columns))
    return df
