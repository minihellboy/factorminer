"""Continuous-contract futures panel builder and leaf registration.

Builds a roll-adjusted front-month continuous futures panel and derives the
extra leaves FactorMiner's extensible feature registry can consume:

* ``$basis``   — futures price minus spot
* ``$spot``    — underlying spot aligned to the futures bar
* ``$premium`` — futures/spot - 1
* ``$roll_yield`` — per-bar roll yield implied by the adjustment factor
* ``$oi``      — open interest

The geometry is the same long ``(datetime, asset_id)`` panel FactorMiner already
uses for equities/crypto. Multi-tenor curve surfaces are intentionally out of
scope (see landscape §10 traps).

Offline / mock
--------------
:func:`generate_mock_futures_panel` produces a deterministic synthetic panel so
tests and ``--mock`` paths never need a live vendor. Live vendor fetch is
optional and config-gated; when enabled it still goes through the same
HTTPS + timeout + size-cap hygiene as other connectors.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from factorminer.core.types import register_features

logger = logging.getLogger(__name__)

FUTURES_FEATURE_LEAVES: tuple[str, ...] = (
    "$basis",
    "$spot",
    "$premium",
    "$roll_yield",
    "$oi",
)

FUTURES_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "$basis": "futures basis (continuous futures price minus spot)",
    "$spot": "underlying spot price aligned to the futures bar",
    "$premium": "futures premium over spot (futures/spot - 1)",
    "$roll_yield": "roll yield implied by the continuous-contract price adjustment",
    "$oi": "open interest (contracts outstanding)",
}


@dataclass(frozen=True)
class FuturesConfig:
    """Configuration for continuous-contract futures construction."""

    register_leaves: bool = True
    contract_multiplier: float = 1.0
    roll_adjust: str = "backward"  # backward | none
    min_oi: float = 0.0
    # Optional market-impact sizing defaults consumed by capacity.py
    default_multiplier: float = 1.0
    tick_size: float = 0.01

    def validate(self) -> None:
        if self.roll_adjust not in {"backward", "none"}:
            raise ValueError("roll_adjust must be 'backward' or 'none'")
        if self.contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be > 0")
        if self.default_multiplier <= 0:
            raise ValueError("default_multiplier must be > 0")


@dataclass(frozen=True)
class FuturesContractSpec:
    """Per-root contract metadata for capacity / sizing."""

    root: str
    multiplier: float = 1.0
    tick_size: float = 0.01
    currency: str = "USD"


def register_futures_features() -> list[str]:
    """Register futures leaves on the global feature registry."""
    return register_features(
        list(FUTURES_FEATURE_LEAVES),
        descriptions=FUTURES_FEATURE_DESCRIPTIONS,
    )


def _ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["asset_id"] = out["asset_id"].astype(str)
    return out


def compute_basis_leaves(
    df: pd.DataFrame,
    *,
    futures_price_col: str = "close",
    spot_col: str = "spot",
    oi_col: str = "oi",
) -> pd.DataFrame:
    """Derive basis/premium/roll_yield/oi columns on a futures panel.

    Requires ``spot`` (or *spot_col*) to be present. Missing spot yields NaN
    basis/premium rather than fabricating values.
    """
    out = _ensure_dt(df)
    if spot_col not in out.columns:
        out[spot_col] = np.nan
    if futures_price_col not in out.columns:
        raise ValueError(f"Missing futures price column {futures_price_col!r}")

    fut = pd.to_numeric(out[futures_price_col], errors="coerce")
    spot = pd.to_numeric(out[spot_col], errors="coerce")
    out["spot"] = spot
    out["basis"] = fut - spot
    with np.errstate(divide="ignore", invalid="ignore"):
        premium = np.where(
            (spot.to_numpy(dtype=np.float64) > 0) & np.isfinite(spot.to_numpy()),
            fut.to_numpy(dtype=np.float64) / spot.to_numpy(dtype=np.float64) - 1.0,
            np.nan,
        )
    out["premium"] = premium

    if oi_col in out.columns:
        out["oi"] = pd.to_numeric(out[oi_col], errors="coerce")
    elif "oi" not in out.columns:
        out["oi"] = np.nan

    if "roll_yield" not in out.columns:
        out["roll_yield"] = _infer_roll_yield(out, price_col=futures_price_col)
    return out


def _infer_roll_yield(df: pd.DataFrame, *, price_col: str = "close") -> pd.Series:
    """Infer a simple per-bar roll yield from gaps in a continuous series.

    When an explicit ``adjustment_factor`` column is present, roll yield is the
    fractional jump explained by the factor change. Otherwise returns zeros
    (no fabricated roll events).
    """
    if "adjustment_factor" in df.columns:
        adj = pd.to_numeric(df["adjustment_factor"], errors="coerce")
        grouped = adj.groupby(df["asset_id"], sort=False)
        prev = grouped.shift(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ry = np.where(
                (prev.to_numpy(dtype=np.float64) > 0) & np.isfinite(prev),
                adj.to_numpy(dtype=np.float64) / prev.to_numpy(dtype=np.float64) - 1.0,
                0.0,
            )
        return pd.Series(ry, index=df.index, dtype=np.float64)

    # No adjustment metadata: do not invent roll yield from price returns.
    return pd.Series(0.0, index=df.index, dtype=np.float64)


def backward_adjust_continuous(
    df: pd.DataFrame,
    *,
    price_cols: Sequence[str] = ("open", "high", "low", "close"),
    adjustment_col: str = "adjustment_factor",
) -> pd.DataFrame:
    """Apply backward roll adjustment using a per-bar adjustment factor.

    The newest bar keeps factor 1.0; older bars are multiplied by the cumulative
    product of roll ratios so that the continuous series has no artificial gap
    at roll dates. If *adjustment_col* is absent, the frame is returned with
    factor 1.0 filled in.
    """
    out = _ensure_dt(df)
    if adjustment_col not in out.columns:
        out[adjustment_col] = 1.0
        return out

    pieces: list[pd.DataFrame] = []
    for _, group in out.groupby("asset_id", sort=False):
        g = group.sort_values("datetime").copy()
        adj = pd.to_numeric(g[adjustment_col], errors="coerce").fillna(1.0).to_numpy()
        # Backward: scale historical prices so the last bar is unadjusted.
        # cumulative product from the end.
        ratios = np.ones_like(adj, dtype=np.float64)
        for i in range(len(adj) - 2, -1, -1):
            if adj[i] > 0 and adj[i + 1] > 0:
                ratios[i] = ratios[i + 1] * (adj[i + 1] / adj[i])
            else:
                ratios[i] = ratios[i + 1]
        # Normalize so the final ratio is 1
        if ratios[-1] != 0:
            ratios = ratios / ratios[-1]
        for col in price_cols:
            if col in g.columns:
                g[col] = pd.to_numeric(g[col], errors="coerce") * ratios
        if "amount" in g.columns:
            g["amount"] = pd.to_numeric(g["amount"], errors="coerce") * ratios
        g[adjustment_col] = ratios
        pieces.append(g)

    if not pieces:
        return out
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["datetime", "asset_id"])
        .reset_index(drop=True)
    )


def build_continuous_futures_panel(
    raw: pd.DataFrame,
    *,
    config: FuturesConfig | None = None,
    spot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a continuous futures panel with basis leaves.

    Parameters
    ----------
    raw :
        Long panel with at least ``datetime``, ``asset_id``, OHLCV columns.
        Optional ``oi``, ``adjustment_factor``, ``spot``.
    config :
        Futures construction config.
    spot :
        Optional separate spot panel with ``datetime``, ``asset_id``, ``spot``
        (or ``close``) to join onto the futures bars.
    """
    cfg = config or FuturesConfig()
    cfg.validate()
    if cfg.register_leaves:
        register_futures_features()

    out = _ensure_dt(raw)
    if cfg.roll_adjust == "backward":
        out = backward_adjust_continuous(out)

    if spot is not None and not spot.empty:
        sp = _ensure_dt(spot)
        spot_col = "spot" if "spot" in sp.columns else "close"
        sp = sp[["datetime", "asset_id", spot_col]].rename(columns={spot_col: "spot"})
        out = out.merge(sp, on=["datetime", "asset_id"], how="left", suffixes=("", "_spot"))
        if "spot_spot" in out.columns:
            out["spot"] = out["spot"].fillna(out["spot_spot"])
            out = out.drop(columns=["spot_spot"])

    out = compute_basis_leaves(out)

    # Standard amount if missing: close * volume * multiplier
    if "amount" not in out.columns or out["amount"].isna().all():
        out["amount"] = (
            pd.to_numeric(out.get("close"), errors="coerce")
            * pd.to_numeric(out.get("volume"), errors="coerce")
            * float(cfg.contract_multiplier)
        )

    if "vwap" not in out.columns:
        vol = pd.to_numeric(out.get("volume"), errors="coerce")
        amt = pd.to_numeric(out.get("amount"), errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            out["vwap"] = np.where(
                (vol.to_numpy(dtype=np.float64) > 1e-12) & np.isfinite(vol),
                amt.to_numpy(dtype=np.float64) / vol.to_numpy(dtype=np.float64),
                pd.to_numeric(out.get("close"), errors="coerce").to_numpy(),
            )

    if "returns" not in out.columns:
        out["returns"] = out.groupby("asset_id", sort=False)["close"].pct_change()

    return out.sort_values(["datetime", "asset_id"]).reset_index(drop=True)


def generate_mock_futures_panel(
    *,
    num_assets: int = 8,
    num_periods: int = 120,
    seed: int = 7,
    start: str = "2024-01-02",
    config: FuturesConfig | None = None,
) -> pd.DataFrame:
    """Deterministic synthetic continuous futures panel with spot/basis/oi.

    Suitable for ``--mock`` and unit tests. No network access.
    """
    cfg = config or FuturesConfig()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=num_periods, freq="C")
    roots = [f"FUT{i:02d}" for i in range(num_assets)]

    rows: list[dict] = []
    for a_idx, root in enumerate(roots):
        # Spot random walk
        spot0 = 50.0 + a_idx * 5.0
        spot_rets = rng.normal(0.0002, 0.01, size=num_periods)
        spot = spot0 * np.cumprod(1.0 + spot_rets)
        # Contango premium ~ annualized 3%
        premium = 0.03 * (np.arange(num_periods) % 60) / 252.0
        fut = spot * (1.0 + premium)
        # Synthetic roll every 21 bars: record adjustment factor jump
        adj = np.ones(num_periods, dtype=np.float64)
        for r in range(21, num_periods, 21):
            adj[r:] *= 1.0 + 0.002  # small backwardation/contango roll bump
        oi = rng.integers(10_000, 50_000, size=num_periods).astype(np.float64)
        vol = rng.integers(1_000, 8_000, size=num_periods).astype(np.float64)
        for t, dt in enumerate(dates):
            o = fut[t] * (1.0 + rng.normal(0, 0.001))
            h = max(o, fut[t]) * (1.0 + abs(rng.normal(0, 0.002)))
            lo = min(o, fut[t]) * (1.0 - abs(rng.normal(0, 0.002)))
            c = fut[t]
            rows.append(
                {
                    "datetime": dt,
                    "asset_id": root,
                    "open": float(o),
                    "high": float(h),
                    "low": float(lo),
                    "close": float(c),
                    "volume": float(vol[t]),
                    "amount": float(c * vol[t] * cfg.contract_multiplier),
                    "spot": float(spot[t]),
                    "oi": float(oi[t]),
                    "adjustment_factor": float(adj[t]),
                }
            )

    raw = pd.DataFrame(rows)
    return build_continuous_futures_panel(raw, config=cfg)


def notional_volume(
    price: np.ndarray,
    contracts: np.ndarray,
    *,
    multiplier: float = 1.0,
) -> np.ndarray:
    """Dollar notional volume from price × contracts × multiplier."""
    p = np.asarray(price, dtype=np.float64)
    c = np.asarray(contracts, dtype=np.float64)
    return p * c * float(multiplier)


__all__ = [
    "FUTURES_FEATURE_DESCRIPTIONS",
    "FUTURES_FEATURE_LEAVES",
    "FuturesConfig",
    "FuturesContractSpec",
    "backward_adjust_continuous",
    "build_continuous_futures_panel",
    "compute_basis_leaves",
    "generate_mock_futures_panel",
    "notional_volume",
    "register_futures_features",
]
