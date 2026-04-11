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


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline.

    Attributes
    ----------
    winsor_lower : float
        Lower percentile for winsorisation (0-100).
    winsor_upper : float
        Upper percentile for winsorisation (0-100).
    min_nonnan_ratio : float
        Minimum fraction of non-NaN values required per cross-section
        for a time step to be kept.
    ffill_limit : int or None
        Maximum number of consecutive NaN values to forward-fill within
        each intraday session.
    cross_fill_method : str
        Cross-sectional fill method after forward fill.
        ``"median"`` or ``"mean"``.
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


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``vwap`` column: amount / volume.  NaN when volume is zero."""
    df = df.copy()
    df["vwap"] = np.where(
        df["volume"] > 0,
        df["amount"] / df["volume"],
        np.nan,
    )
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``returns`` column: close-to-close percentage change per asset.

    Returns are computed as ``close[t] / close[t-1] - 1`` within each asset.
    The first observation per asset is NaN.
    """
    df = df.copy()
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

def _extract_date(dt_series: pd.Series) -> pd.Series:
    """Return the date component of a datetime series."""
    return dt_series.dt.date


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
        ``"median"`` or ``"mean"`` for the cross-sectional stage.
    columns : sequence of str, optional
        Columns to fill.  Defaults to numeric columns.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns = [c for c in columns if c in df.columns]

    # Stage 1: forward fill within (asset, date)
    df["_date"] = _extract_date(df["datetime"])
    for col in columns:
        df[col] = df.groupby(["asset_id", "_date"])[col].transform(
            lambda s: s.ffill(limit=ffill_limit)
        )

    # Stage 2: cross-sectional fill per datetime
    if cross_fill_method == "median":
        agg_func = "median"
    elif cross_fill_method == "mean":
        agg_func = "mean"
    else:
        raise ValueError(f"Unknown cross_fill_method: {cross_fill_method}")

    for col in columns:
        cross_vals = df.groupby("datetime")[col].transform(agg_func)
        df[col] = df[col].fillna(cross_vals)

    df = df.drop(columns=["_date"])
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
        Percentile bounds (0-100).
    """
    df = df.copy()
    columns = [c for c in columns if c in df.columns]

    for col in columns:
        lo = df.groupby("datetime")[col].transform(
            lambda s: np.nanpercentile(s, lower) if s.notna().any() else np.nan
        )
        hi = df.groupby("datetime")[col].transform(
            lambda s: np.nanpercentile(s, upper) if s.notna().any() else np.nan
        )
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

    # Count non-NaN per datetime
    checks = df.groupby("datetime")[list(columns)].apply(
        lambda g: g.notna().all(axis=1).sum() / n_assets
    )
    valid_dts = checks[checks >= min_nonnan_ratio].index
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
