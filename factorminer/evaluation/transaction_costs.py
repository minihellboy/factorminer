"""Transaction cost models for realistic P&L computation.

Implements the Almgren-Chriss (2001) market impact framework, bid-ask
slippage, commissions, and A-share specific taxes. All costs are expressed
in basis points (bps) unless explicitly noted.

References
----------
Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions.
    Journal of Risk, 3(2), 5-39.
Kissell, R. (2013). The Science of Algorithmic Trading and Portfolio Management.
    Academic Press.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TradingCosts:
    """Aggregated transaction costs for a single rebalance event.

    All monetary values are in basis points (1 bps = 0.01%) of notional.

    Attributes
    ----------
    market_impact_bps : float
        Almgren-Chriss permanent + temporary impact, portfolio-level weighted
        average in bps.
    slippage_bps : float
        Bid-ask spread crossing cost (half-spread * urgency) in bps.
    commission_bps : float
        Broker commission in bps (round-trip, both legs included).
    stamp_duty_bps : float
        Stamp duty levied on the sell leg only (A-shares: 1 bps).
    total_bps : float
        Sum of all cost components.
    turnover : float
        Fraction of portfolio traded this rebalance, in [0, 2].
        0 = no trading, 1 = full one-way rebalance, 2 = full round-trip.
    details : dict
        Per-asset breakdown and intermediate quantities.
    """

    market_impact_bps: float
    slippage_bps: float
    commission_bps: float
    stamp_duty_bps: float
    total_bps: float
    turnover: float
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Market impact model (Almgren-Chriss)
# ---------------------------------------------------------------------------

class MarketImpactModel:
    """Almgren-Chriss (2001) market impact model.

    The model decomposes total market impact into:

    Permanent impact (price pressure lasting beyond the trade window)::

        g(v) = lambda_perm * sigma * sign(v) * |v/ADV|^alpha

    where alpha = 0.5 (square-root law, empirically supported for equities).

    Temporary impact (within-trade reversion; instantaneous cost)::

        h(v) = eta_temp * sigma * (v / ADV)

    Both components are expressed as fractions of price.  Multiply by 1e4
    to convert to basis points.

    Parameters
    ----------
    lambda_perm : float
        Permanent impact coefficient.  Default 0.1 (market-standard calibration
        for liquid A-shares; Almgren et al. 2005, "Direct Estimation of Equity
        Market Impact").
    eta_temp : float
        Temporary impact coefficient.  Default 0.01.
    alpha : float
        Power-law exponent for permanent impact.  Default 0.5 (square-root).
    """

    def __init__(
        self,
        lambda_perm: float = 0.1,
        eta_temp: float = 0.01,
        alpha: float = 0.5,
    ) -> None:
        if lambda_perm < 0:
            raise ValueError("lambda_perm must be >= 0")
        if eta_temp < 0:
            raise ValueError("eta_temp must be >= 0")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        self.lambda_perm = float(lambda_perm)
        self.eta_temp = float(eta_temp)
        self.alpha = float(alpha)

    # ------------------------------------------------------------------
    def compute_impact(
        self,
        trade_size: np.ndarray,
        adv: np.ndarray,
        volatility: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        """Compute total Almgren-Chriss impact for a batch of trades.

        Parameters
        ----------
        trade_size : ndarray of shape (M,)
            Absolute trade size for each asset (shares or notional units).
        adv : ndarray of shape (M,)
            Average daily volume (same units as trade_size).
        volatility : ndarray of shape (M,)
            Per-asset annualized volatility (e.g., 0.25 = 25%).
        direction : ndarray of shape (M,)
            +1.0 for buys, -1.0 for sells.  Absolute values used for
            magnitude; sign determines direction for permanent component.

        Returns
        -------
        ndarray of shape (M,)
            Total market impact in basis points per asset.  Zero for assets
            with zero trade size or zero ADV.
        """
        trade_size = np.asarray(trade_size, dtype=np.float64)
        adv = np.asarray(adv, dtype=np.float64)
        volatility = np.asarray(volatility, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)

        # Participation rate: what fraction of ADV are we trading
        participation = np.where(adv > 0, trade_size / adv, 0.0)

        # Permanent impact: lambda * sigma * participation^alpha
        # Ref: Almgren (2001) eq. (2.4) – permanent impact g(v) = lambda * sigma * |x|^alpha
        permanent = (
            self.lambda_perm
            * volatility
            * np.sign(direction)
            * np.power(np.abs(participation), self.alpha)
        )

        # Temporary impact: eta * sigma * participation
        # Ref: Almgren (2001) eq. (2.5) – temporary impact h(v) = eta * v/ADV
        temporary = self.eta_temp * volatility * np.abs(participation)

        # Total impact (fractional), convert to bps
        total_bps = (np.abs(permanent) + temporary) * 1e4

        return total_bps


# ---------------------------------------------------------------------------
# Slippage model (bid-ask spread)
# ---------------------------------------------------------------------------

class SlippageModel:
    """Bid-ask spread slippage model.

    Each trade crosses the bid-ask spread.  For a patient order (urgency=0)
    we assume the order rests on the book and earns half-spread; for an
    aggressive market order (urgency=1) we pay the full spread.  In practice,
    intraday algo execution sits around urgency=0.5.

    Per-trade slippage cost::

        slippage = spread_bps * urgency

    For a round-trip (buy + sell) this doubles.  The caller is responsible
    for applying round-trip scaling.

    Default spreads for 10-min A-share bars:
        - Liquid large-caps (CSI 300): 2-3 bps
        - Mid-cap (CSI 500): 3-5 bps
        - Small-cap: 5-10 bps
    """

    def __init__(self, default_spread_bps: float = 3.0) -> None:
        """
        Parameters
        ----------
        default_spread_bps : float
            Fallback spread used when per-asset spreads are not supplied.
        """
        self.default_spread_bps = float(default_spread_bps)

    # ------------------------------------------------------------------
    def compute_slippage(
        self,
        trade_size: np.ndarray,
        spread_bps: np.ndarray | None = None,
        urgency: float = 0.5,
    ) -> np.ndarray:
        """Compute one-way slippage for a set of trades.

        Parameters
        ----------
        trade_size : ndarray of shape (M,)
            Trade sizes (used to identify which assets are actively traded;
            zero-size trades incur no slippage).
        spread_bps : ndarray of shape (M,) or None
            Per-asset effective bid-ask spread in bps.  Falls back to
            ``default_spread_bps`` if None.
        urgency : float
            Urgency scalar in [0, 1].  0 = fully patient (resting orders),
            1 = aggressive (market orders).  Default 0.5.

        Returns
        -------
        ndarray of shape (M,)
            One-way slippage cost in bps per asset.
        """
        if not (0.0 <= urgency <= 1.0):
            raise ValueError("urgency must be in [0, 1]")

        trade_size = np.asarray(trade_size, dtype=np.float64)
        M = len(trade_size)

        if spread_bps is None:
            sp = np.full(M, self.default_spread_bps)
        else:
            sp = np.asarray(spread_bps, dtype=np.float64)

        # Only traded assets incur slippage
        traded = trade_size > 0
        cost = np.where(traded, sp * urgency, 0.0)

        return cost


# ---------------------------------------------------------------------------
# Aggregated transaction cost calculator
# ---------------------------------------------------------------------------

class TransactionCostCalculator:
    """Aggregate all transaction cost components for a portfolio rebalance.

    Components modelled
    -------------------
    1. **Market impact** – Almgren-Chriss permanent + temporary impact.
    2. **Bid-ask slippage** – half-spread crossing cost.
    3. **Commission** – fixed per-side brokerage fee.
    4. **Stamp duty** – sell-side stamp duty (A-shares only).
    5. **Financing cost** – overnight leverage cost (when applicable).

    Market defaults (A-shares, 10-min bars)
    ----------------------------------------
    * Commission: 2 bps per side (4 bps round-trip for institutional).
    * Stamp duty: 1 bps on the sell side only.
    * Spread: 3 bps (CSI 500 universe average).
    * All-in round-trip cost at modest size: ~8 bps (consistent with
      HelixFactor benchmark config ``benchmark.cost_bps`` sweep).

    Crypto defaults
    ---------------
    * Commission: 0.5 bps maker / 1.5 bps taker → 2 bps per side.
    * No stamp duty.
    * Spread: 1-2 bps for top-20 pairs.

    Parameters
    ----------
    impact_model : MarketImpactModel, optional
        Custom Almgren-Chriss model.  Defaults to standard parameterisation.
    slippage_model : SlippageModel, optional
        Custom slippage model.  Defaults to standard parameterisation.
    commission_bps : float
        One-way broker commission in bps.  Default 2 bps.
    stamp_duty_bps : float
        Sell-side stamp duty in bps.  Default 1 bps (A-shares).
    overnight_rate_annual : float
        Annualised financing rate for leveraged positions.  Default 0.0
        (no leverage).
    bars_per_year : float
        Number of bars per year used to convert overnight rate to per-bar.
        Default 252 * 24 = 6048 (10-min bars, 4-hour A-share session).
    """

    def __init__(
        self,
        impact_model: MarketImpactModel | None = None,
        slippage_model: SlippageModel | None = None,
        commission_bps: float = 2.0,
        stamp_duty_bps: float = 1.0,
        overnight_rate_annual: float = 0.0,
        bars_per_year: float = 252.0 * 24.0,
    ) -> None:
        self.impact_model = impact_model or MarketImpactModel()
        self.slippage_model = slippage_model or SlippageModel()
        self.commission_bps = float(commission_bps)
        self.stamp_duty_bps = float(stamp_duty_bps)
        self.overnight_rate_annual = float(overnight_rate_annual)
        self.bars_per_year = float(bars_per_year)

    # ------------------------------------------------------------------
    def compute_total_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        adv: np.ndarray,
        volatility: np.ndarray,
        portfolio_value: float,
        market: str = 'ashare',
        spread_bps: np.ndarray | None = None,
        urgency: float = 0.5,
    ) -> TradingCosts:
        """Compute all-in transaction costs for a single rebalance event.

        The portfolio transitions from ``old_weights`` to ``new_weights``.
        Weights are signed: positive = long, negative = short.  Their sum
        need not be 1 (allows cash + leverage).

        Parameters
        ----------
        old_weights : ndarray of shape (M,)
            Current (pre-trade) portfolio weights per asset.
        new_weights : ndarray of shape (M,)
            Target (post-trade) portfolio weights per asset.
        adv : ndarray of shape (M,)
            Average daily volume per asset in notional (same currency as
            ``portfolio_value``).
        volatility : ndarray of shape (M,)
            Per-asset annualized volatility (e.g. 0.30 = 30%).
        portfolio_value : float
            Total portfolio NAV in notional currency.
        market : str
            ``'ashare'`` or ``'crypto'``.  Controls stamp duty defaults.
        spread_bps : ndarray of shape (M,), optional
            Per-asset bid-ask spread in bps.  Falls back to model default.
        urgency : float
            Execution urgency in [0, 1].

        Returns
        -------
        TradingCosts
            Fully decomposed cost object.
        """
        old_weights = np.asarray(old_weights, dtype=np.float64)
        new_weights = np.asarray(new_weights, dtype=np.float64)
        adv = np.asarray(adv, dtype=np.float64)
        volatility = np.asarray(volatility, dtype=np.float64)

        # Weight deltas and trade notional
        delta_weights = new_weights - old_weights                     # signed
        trade_notional = np.abs(delta_weights) * portfolio_value      # always >= 0
        trade_direction = np.sign(delta_weights)                      # +1 buy, -1 sell

        # One-way turnover: sum of absolute weight changes, divided by 2 to
        # avoid double-counting buys and sells for a fully-funded portfolio.
        # Convention: turnover in [0, 1] for a single rebalance (0=no trade,
        # 1=100% of portfolio turned over on one side).
        turnover = float(np.sum(np.abs(delta_weights)) / 2.0)

        # ----------------------------------------------------------------
        # 1. Market impact (Almgren-Chriss)
        # ----------------------------------------------------------------
        impact_bps_per_asset = self.impact_model.compute_impact(
            trade_size=trade_notional,
            adv=adv,
            volatility=volatility,
            direction=trade_direction,
        )
        # Portfolio-level impact = notional-weighted average across traded assets
        total_trade_notional = float(np.sum(trade_notional))
        if total_trade_notional > 1e-12:
            impact_bps = float(
                np.sum(impact_bps_per_asset * trade_notional) / total_trade_notional
            )
        else:
            impact_bps = 0.0

        # ----------------------------------------------------------------
        # 2. Bid-ask slippage
        # ----------------------------------------------------------------
        slippage_bps_per_asset = self.slippage_model.compute_slippage(
            trade_size=trade_notional,
            spread_bps=spread_bps,
            urgency=urgency,
        )
        if total_trade_notional > 1e-12:
            slippage_bps = float(
                np.sum(slippage_bps_per_asset * trade_notional) / total_trade_notional
            )
        else:
            slippage_bps = 0.0

        # ----------------------------------------------------------------
        # 3. Commission (both buy and sell legs)
        # ----------------------------------------------------------------
        # Commission applies to both sides of each trade (enter + exit).
        # Here we apply it per side (once now + once on exit = round-trip).
        # For a rebalance we pay commission on the traded notional.
        commission_bps = self.commission_bps  # per-side, applied to traded notional

        # ----------------------------------------------------------------
        # 4. Stamp duty (sell side only)
        # ----------------------------------------------------------------
        effective_stamp = 0.0
        if market == 'ashare':
            # Identify sell trades: delta_weight < 0 (reducing long) or
            # delta_weight > 0 but old position was short (increasing short sell).
            # Simplified: stamp duty on any reduction of long exposure.
            sell_notional = np.sum(trade_notional[delta_weights < 0])
            if total_trade_notional > 1e-12:
                sell_fraction = sell_notional / total_trade_notional
                effective_stamp = self.stamp_duty_bps * sell_fraction
        # crypto: no stamp duty

        # ----------------------------------------------------------------
        # 5. Financing cost (one bar's worth of overnight carry)
        # ----------------------------------------------------------------
        # Per-bar financing cost on leveraged portion.  For a bar-length h:
        #   financing_cost = leverage * overnight_rate_annual / bars_per_year
        if self.overnight_rate_annual > 0:
            leverage = max(
                float(np.sum(np.abs(new_weights))) - 1.0, 0.0
            )  # excess over 1x
            financing_bps = (
                leverage
                * self.overnight_rate_annual
                / self.bars_per_year
                * 1e4
            )
        else:
            financing_bps = 0.0

        # ----------------------------------------------------------------
        # Aggregate
        # ----------------------------------------------------------------
        total_bps = (
            impact_bps
            + slippage_bps
            + commission_bps
            + effective_stamp
            + financing_bps
        )

        details = {
            "impact_bps_per_asset": impact_bps_per_asset,
            "slippage_bps_per_asset": slippage_bps_per_asset,
            "delta_weights": delta_weights,
            "trade_notional": trade_notional,
            "financing_bps": financing_bps,
            "sell_stamp_bps": effective_stamp,
        }

        return TradingCosts(
            market_impact_bps=impact_bps,
            slippage_bps=slippage_bps,
            commission_bps=commission_bps,
            stamp_duty_bps=effective_stamp,
            total_bps=total_bps,
            turnover=turnover,
            details=details,
        )

    # ------------------------------------------------------------------
    @classmethod
    def for_ashare(
        cls,
        lambda_perm: float = 0.1,
        eta_temp: float = 0.01,
        commission_bps: float = 2.0,
        stamp_duty_bps: float = 1.0,
        default_spread_bps: float = 3.0,
    ) -> TransactionCostCalculator:
        """Convenience constructor with A-share defaults.

        All-in round-trip cost at low turnover ≈ 7-9 bps, consistent with
        the HelixFactor benchmark sweep range.

        Parameters
        ----------
        lambda_perm : float
            Permanent impact coefficient (default 0.1).
        eta_temp : float
            Temporary impact coefficient (default 0.01).
        commission_bps : float
            One-way commission (default 2 bps).
        stamp_duty_bps : float
            Sell-side stamp duty (default 1 bps; CSRC mandated since 2023).
        default_spread_bps : float
            Default spread for assets without explicit spread data.
        """
        return cls(
            impact_model=MarketImpactModel(
                lambda_perm=lambda_perm,
                eta_temp=eta_temp,
            ),
            slippage_model=SlippageModel(default_spread_bps=default_spread_bps),
            commission_bps=commission_bps,
            stamp_duty_bps=stamp_duty_bps,
        )

    @classmethod
    def for_crypto(
        cls,
        lambda_perm: float = 0.05,
        eta_temp: float = 0.005,
        commission_bps: float = 1.0,
        default_spread_bps: float = 1.5,
    ) -> TransactionCostCalculator:
        """Convenience constructor with crypto exchange defaults.

        Parameters
        ----------
        lambda_perm : float
            Permanent impact coefficient.  Lower than A-shares due to
            continuous 24/7 liquidity provision.
        eta_temp : float
            Temporary impact coefficient.
        commission_bps : float
            Maker/taker blended commission (default 1 bps).
        default_spread_bps : float
            Default effective spread for top-20 pairs (default 1.5 bps).
        """
        return cls(
            impact_model=MarketImpactModel(
                lambda_perm=lambda_perm,
                eta_temp=eta_temp,
            ),
            slippage_model=SlippageModel(default_spread_bps=default_spread_bps),
            commission_bps=commission_bps,
            stamp_duty_bps=0.0,   # no stamp duty on crypto
        )
