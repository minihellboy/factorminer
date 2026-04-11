"""Generate realistic synthetic market data for testing FactorMiner.

Produces multi-asset OHLCV data with:
- Volume clustering (GARCH-like)
- Volatility clustering
- Cross-sectional correlation via a common market factor
- Planted alpha signals for validating factor discovery
- OHLC consistency guarantees: low <= open,close <= high
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

Frequency = Literal["10min", "30min", "1h", "1d"]

_FREQ_MAP = {
    "10min": "10min",
    "30min": "30min",
    "1h": "1h",
    "1d": "1D",
}


@dataclass
class MockConfig:
    """Configuration for synthetic data generation.

    Attributes
    ----------
    num_assets : int
        Number of assets (M).
    num_periods : int
        Number of time bars (T) per asset.
    frequency : str
        Bar frequency: ``"10min"``, ``"30min"``, ``"1h"``, ``"1d"``.
    start_date : str
        Start datetime in ISO format.
    base_price : float
        Initial price level around which assets are generated.
    annual_vol : float
        Annualised volatility for the diffusion process.
    market_factor_weight : float
        Weight of the common market factor in returns (0-1).
        Higher values increase cross-sectional correlation.
    vol_persistence : float
        GARCH(1,1) persistence parameter for volatility clustering (0-1).
    volume_mean : float
        Mean daily volume per asset.
    volume_persistence : float
        AR(1) coefficient for volume clustering (0-1).
    plant_alpha : bool
        Whether to inject planted alpha signals.
    alpha_strength : float
        Signal-to-noise ratio of the planted alpha.
    alpha_assets_frac : float
        Fraction of assets that carry the planted signal.
    seed : int
        Random seed for reproducibility.
    universe : str or None
        Universe label to include in the output.
    """

    num_assets: int = 50
    num_periods: int = 1000
    frequency: Frequency = "10min"
    start_date: str = "2024-01-02 09:30:00"
    base_price: float = 50.0
    annual_vol: float = 0.25
    market_factor_weight: float = 0.3
    vol_persistence: float = 0.9
    volume_mean: float = 1_000_000.0
    volume_persistence: float = 0.85
    plant_alpha: bool = True
    alpha_strength: float = 0.02
    alpha_assets_frac: float = 0.2
    seed: int = 42
    universe: str | None = None


def _bars_per_year(freq: Frequency) -> float:
    """Approximate number of bars in a trading year."""
    trading_days = 252
    bars_per_day = {
        "10min": 24,   # 4h session / 10min
        "30min": 8,
        "1h": 4,
        "1d": 1,
    }
    return trading_days * bars_per_day[freq]


def _generate_timestamps(
    start: str,
    num_periods: int,
    freq: Frequency,
) -> pd.DatetimeIndex:
    """Create a business-aware timestamp index.

    For intraday frequencies the index skips weekends and only covers
    a simplified trading session (09:30 - 15:00 for 10min/30min bars).
    """
    pd_freq = _FREQ_MAP[freq]
    if freq == "1d":
        ts = pd.bdate_range(start=start, periods=num_periods, freq="B")
    else:
        # Generate enough intraday bars, then trim to num_periods
        days_needed = (num_periods // 24) + 10  # generous overestimate
        day_range = pd.bdate_range(start=start, periods=days_needed, freq="B")
        bars: list[pd.Timestamp] = []
        for day in day_range:
            session_start = day.replace(hour=9, minute=30, second=0)
            session_end = day.replace(hour=15, minute=0, second=0)
            day_bars = pd.date_range(session_start, session_end, freq=pd_freq)
            # Exclude the exact session end for cleaner bars
            day_bars = day_bars[day_bars < session_end]
            bars.extend(day_bars.tolist())
            if len(bars) >= num_periods:
                break
        ts = pd.DatetimeIndex(bars[:num_periods])
    return ts


def generate_mock_data(config: MockConfig | None = None) -> pd.DataFrame:
    """Generate synthetic multi-asset OHLCV + amount data.

    Parameters
    ----------
    config : MockConfig, optional
        Generation parameters.  Uses defaults when *None*.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: datetime, asset_id, open, high, low,
        close, volume, amount.  Optionally includes ``universe``.
    """
    if config is None:
        config = MockConfig()

    rng = np.random.default_rng(config.seed)
    M = config.num_assets
    T = config.num_periods

    logger.info("Generating mock data: %d assets x %d periods @ %s", M, T, config.frequency)

    timestamps = _generate_timestamps(config.start_date, T, config.frequency)
    T = len(timestamps)  # may be shorter if we ran out of session bars

    # Per-bar volatility (annualised -> per-bar)
    bar_vol = config.annual_vol / np.sqrt(_bars_per_year(config.frequency))

    # ---------------------------------------------------------------
    # Common market factor (drives cross-sectional correlation)
    # ---------------------------------------------------------------
    market_returns = rng.normal(0, bar_vol, size=T)

    # ---------------------------------------------------------------
    # Per-asset paths
    # ---------------------------------------------------------------
    asset_ids = [f"ASSET_{i:04d}" for i in range(M)]

    # Storage
    all_open = np.empty((M, T))
    all_high = np.empty((M, T))
    all_low = np.empty((M, T))
    all_close = np.empty((M, T))
    all_volume = np.empty((M, T))
    all_amount = np.empty((M, T))

    # Planted alpha: select a subset of assets
    n_alpha = max(1, int(M * config.alpha_assets_frac))
    alpha_assets = set(rng.choice(M, size=n_alpha, replace=False).tolist()) if config.plant_alpha else set()

    for i in range(M):
        # Initial price with some dispersion
        p0 = config.base_price * np.exp(rng.normal(0, 0.3))

        # GARCH-like stochastic volatility
        sigma = np.empty(T)
        sigma[0] = bar_vol
        for t in range(1, T):
            sigma[t] = (
                bar_vol * (1 - config.vol_persistence)
                + config.vol_persistence * sigma[t - 1]
                + rng.normal(0, bar_vol * 0.1)
            )
            sigma[t] = max(sigma[t], bar_vol * 0.2)  # floor

        # Idiosyncratic returns
        idio = rng.normal(0, 1, size=T) * sigma

        # Combine with market factor
        w = config.market_factor_weight
        returns = w * market_returns + (1 - w) * idio

        # Plant alpha signal: small positive drift in returns
        if i in alpha_assets:
            # Signal: positive drift correlated with lagged volume momentum
            alpha_drift = config.alpha_strength * bar_vol
            returns += alpha_drift

        # Cumulative price path (close prices)
        log_price = np.log(p0) + np.cumsum(returns)
        close = np.exp(log_price)

        # Generate intra-bar OHLC from close
        # Open = previous close + small gap noise
        open_ = np.empty(T)
        open_[0] = p0
        open_[1:] = close[:-1] * np.exp(rng.normal(0, bar_vol * 0.1, size=T - 1))

        # Intra-bar high/low
        intra_range = np.abs(rng.normal(0, sigma * 0.5, size=T))
        mid = (open_ + close) / 2
        high = np.maximum(open_, close) + intra_range
        low = np.minimum(open_, close) - intra_range
        low = np.maximum(low, mid * 0.9)  # prevent negative or absurd lows

        # Enforce OHLC consistency
        high = np.maximum(high, np.maximum(open_, close))
        low = np.minimum(low, np.minimum(open_, close))
        low = np.maximum(low, 0.01)  # price floor

        # Volume: AR(1) with log-normal noise
        log_vol = np.empty(T)
        log_vol_mean = np.log(config.volume_mean)
        log_vol[0] = log_vol_mean + rng.normal(0, 0.5)
        for t in range(1, T):
            log_vol[t] = (
                log_vol_mean * (1 - config.volume_persistence)
                + config.volume_persistence * log_vol[t - 1]
                + rng.normal(0, 0.3)
            )
        volume = np.exp(log_vol).astype(np.float64)

        # Amount = volume * vwap (approximate vwap as midpoint)
        vwap_est = (high + low + close) / 3
        amount = volume * vwap_est

        all_open[i] = open_
        all_high[i] = high
        all_low[i] = low
        all_close[i] = close
        all_volume[i] = np.round(volume)
        all_amount[i] = amount

    # ---------------------------------------------------------------
    # Assemble DataFrame
    # ---------------------------------------------------------------
    records = []
    for i in range(M):
        asset_df = pd.DataFrame({
            "datetime": timestamps,
            "asset_id": asset_ids[i],
            "open": all_open[i],
            "high": all_high[i],
            "low": all_low[i],
            "close": all_close[i],
            "volume": all_volume[i],
            "amount": all_amount[i],
        })
        records.append(asset_df)

    df = pd.concat(records, ignore_index=True)

    if config.universe is not None:
        df["universe"] = config.universe

    df = df.sort_values(["datetime", "asset_id"]).reset_index(drop=True)

    logger.info(
        "Generated %d rows: %d assets x %d periods, planted alpha in %d assets",
        len(df),
        M,
        T,
        len(alpha_assets),
    )
    return df


def generate_with_halts(
    config: MockConfig | None = None,
    halt_fraction: float = 0.01,
) -> pd.DataFrame:
    """Generate mock data with simulated trading halts.

    A fraction of (asset, time) pairs are converted to halt bars:
    open = high = low = close = last valid close, volume = 0, amount = 0.

    Parameters
    ----------
    config : MockConfig, optional
        Generation parameters.
    halt_fraction : float
        Fraction of bars to convert to halts.
    """
    df = generate_mock_data(config)
    if config is None:
        config = MockConfig()
    rng = np.random.default_rng(config.seed + 1)

    n = len(df)
    n_halt = int(n * halt_fraction)
    halt_idx = rng.choice(n, size=n_halt, replace=False)

    df.loc[halt_idx, "volume"] = 0
    df.loc[halt_idx, "amount"] = 0
    # Flatten OHLC to close (simulating last traded price)
    halt_price = df.loc[halt_idx, "close"]
    df.loc[halt_idx, "open"] = halt_price
    df.loc[halt_idx, "high"] = halt_price
    df.loc[halt_idx, "low"] = halt_price

    logger.info("Injected %d halt bars (%.2f%%)", n_halt, 100 * halt_fraction)
    return df
