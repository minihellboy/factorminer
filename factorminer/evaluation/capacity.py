"""Capacity-aware backtesting for alpha factors.

Estimates market impact via a square-root model, evaluates net-of-cost
IC / ICIR, and determines the maximum capital that a factor can absorb
before its alpha degrades beyond acceptable limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from factorminer.evaluation.metrics import compute_ic, compute_icir

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CapacityConfig:
    """Configuration for capacity-aware backtesting.

    Parameters
    ----------
    enabled : bool
        Whether capacity estimation is active.
    base_capital_usd : float
        Default capital level used when none is specified explicitly.
    capacity_levels : list[float]
        Dollar capital levels to sweep when building the capacity curve.
    ic_degradation_limit : float
        Maximum fractional IC degradation (1 - |net_IC|/|gross_IC|) before
        the factor is considered capacity-constrained.
    net_icir_threshold : float
        Minimum net ICIR for a factor to pass the cost-adjusted screen.
    sigma_annual : float
        Annualised volatility used by the square-root impact model.
    participation_limit : float
        Hard cap on the participation rate per asset (fraction of bar volume).
    top_fraction : float
        Fraction of the asset universe in the long (and short) leg.
    trading_days_per_year : float
        Number of trading days per calendar year.
    bars_per_day : float
        Number of bars (signal periods) per trading day. Default 24 assumes
        10-minute bars over a 4-hour trading session.
    """

    enabled: bool = True
    base_capital_usd: float = 1e8
    capacity_levels: list[float] = field(
        default_factory=lambda: [1e7, 5e7, 1e8, 5e8, 1e9]
    )
    ic_degradation_limit: float = 0.20
    net_icir_threshold: float = 0.3
    sigma_annual: float = 0.25
    participation_limit: float = 0.10
    top_fraction: float = 0.20
    trading_days_per_year: float = 252.0
    bars_per_day: float = 24.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MarketImpactEstimate:
    """Per-bar market impact estimate across the evaluation window.

    Attributes
    ----------
    impact_bps : np.ndarray, shape (T,)
        Estimated one-way market impact in basis points per bar.
    participation_rate : np.ndarray, shape (T,)
        Mean participation rate (fraction of bar volume) per bar.
    avg_impact_bps : float
        Time-averaged impact in basis points.
    max_impact_bps : float
        Maximum single-bar impact in basis points.
    """

    impact_bps: np.ndarray
    participation_rate: np.ndarray
    avg_impact_bps: float
    max_impact_bps: float


@dataclass
class CapacityEstimate:
    """Result of a capacity sweep for a single factor.

    Attributes
    ----------
    factor_name : str
        Identifier of the evaluated factor.
    max_capacity_usd : float
        Interpolated maximum capital (USD) before the IC degradation limit
        is breached.  ``np.inf`` if no level breaches the limit.
    capacity_curve : dict[float, float]
        Mapping from capital level (USD) to IC degradation fraction.
    break_even_cost_bps : float
        Approximate single-leg cost (bps) at which net IC drops to zero.
    """

    factor_name: str
    max_capacity_usd: float
    capacity_curve: dict[float, float]
    break_even_cost_bps: float


@dataclass
class NetCostResult:
    """Net-of-cost evaluation at a specific capital level.

    Attributes
    ----------
    factor_name : str
        Identifier of the evaluated factor.
    gross_icir : float
        ICIR computed on unadjusted returns.
    net_icir : float
        ICIR computed on impact-adjusted returns.
    gross_ls_return : float
        Mean gross long-short return per bar.
    net_ls_return : float
        Mean net long-short return per bar (gross minus round-trip impact).
    estimated_capacity_usd : float
        Capital level at which the evaluation was performed.
    impact_estimate : MarketImpactEstimate
        Detailed impact statistics.
    passes_net_threshold : bool
        ``True`` if ``net_icir >= config.net_icir_threshold``.
    """

    factor_name: str
    gross_icir: float
    net_icir: float
    gross_ls_return: float
    net_ls_return: float
    estimated_capacity_usd: float
    impact_estimate: MarketImpactEstimate
    passes_net_threshold: bool


# ---------------------------------------------------------------------------
# Square-root market impact model
# ---------------------------------------------------------------------------

class MarketImpactModel:
    """Square-root market impact model.

    The model estimates single-leg impact as::

        impact = sigma_bar * sqrt(participation_rate)

    where ``sigma_bar`` is the per-bar volatility derived from the annualised
    volatility, and the participation rate is the fraction of bar volume
    consumed by the strategy.
    """

    def __init__(self, config: CapacityConfig | None = None) -> None:
        self.config = config or CapacityConfig()
        self._sigma_bar: float = self.config.sigma_annual / np.sqrt(
            self.config.trading_days_per_year * self.config.bars_per_day
        )

    # ------------------------------------------------------------------
    def estimate_impact(
        self,
        signals: np.ndarray,
        volume: np.ndarray,
        capital: float,
    ) -> MarketImpactEstimate:
        """Estimate per-bar market impact for a given capital deployment.

        Parameters
        ----------
        signals : np.ndarray, shape (M, T)
            Factor signal matrix (used to identify quintile membership).
        volume : np.ndarray, shape (M, T)
            Dollar volume per asset per bar.  Entries <= 0 are treated as
            illiquid and assigned the participation limit.
        capital : float
            Total capital (USD) deployed by the strategy.

        Returns
        -------
        MarketImpactEstimate
        """
        M, T = signals.shape
        cfg = self.config

        n_leg = max(int(M * cfg.top_fraction), 1)
        per_asset_capital = capital / n_leg

        participation = np.full(T, np.nan, dtype=np.float64)

        for t in range(T):
            sig_t = signals[:, t]
            vol_t = volume[:, t]

            valid_sig = ~np.isnan(sig_t)
            if valid_sig.sum() < n_leg:
                participation[t] = cfg.participation_limit
                continue

            # Identify top and bottom quintile assets
            sig_filled = np.where(valid_sig, sig_t, -np.inf)
            top_idx = np.argpartition(sig_filled, -n_leg)[-n_leg:]

            # Participation rate for each selected asset
            rates = np.empty(n_leg, dtype=np.float64)
            for i, idx in enumerate(top_idx):
                v = vol_t[idx]
                if np.isnan(v) or v <= 0:
                    rates[i] = cfg.participation_limit
                else:
                    rates[i] = min(per_asset_capital / v, cfg.participation_limit)

            participation[t] = float(np.mean(rates))

        # Impact in natural units, then convert to bps
        impact = self._sigma_bar * np.sqrt(participation)
        impact_bps = impact * 1e4

        avg_impact = float(np.nanmean(impact_bps))
        max_impact = float(np.nanmax(impact_bps))

        return MarketImpactEstimate(
            impact_bps=impact_bps,
            participation_rate=participation,
            avg_impact_bps=avg_impact,
            max_impact_bps=max_impact,
        )


# ---------------------------------------------------------------------------
# Capacity estimator
# ---------------------------------------------------------------------------

class CapacityEstimator:
    """Evaluate factor capacity and net-of-cost performance.

    Parameters
    ----------
    returns : np.ndarray, shape (M, T)
        Forward returns for M assets over T bars.
    volume : np.ndarray, shape (M, T)
        Dollar volume for M assets over T bars.
    config : CapacityConfig, optional
        Configuration; uses defaults when omitted.
    """

    def __init__(
        self,
        returns: np.ndarray,
        volume: np.ndarray,
        config: CapacityConfig | None = None,
    ) -> None:
        self.returns = returns
        self.volume = volume
        self.config = config or CapacityConfig()
        self._impact_model = MarketImpactModel(self.config)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_ic(ic_series: np.ndarray) -> float:
        """Mean IC ignoring NaN."""
        valid = ic_series[~np.isnan(ic_series)]
        return float(np.mean(valid)) if len(valid) > 0 else 0.0

    def _net_returns(
        self,
        signals: np.ndarray,
        impact_bps: np.ndarray,
    ) -> np.ndarray:
        """Compute impact-adjusted returns.

        For a long-short strategy the round-trip cost is approximately
        ``2 * impact`` (entry + exit on each leg).  We subtract the cost
        uniformly from returns as a simple first-order approximation.

        Parameters
        ----------
        signals : np.ndarray, shape (M, T)
            Factor signals (unused beyond shape; cost applied uniformly).
        impact_bps : np.ndarray, shape (T,)
            One-way impact per bar in basis points.

        Returns
        -------
        np.ndarray, shape (M, T)
            Adjusted returns matrix.
        """
        cost = 2.0 * impact_bps / 1e4  # round-trip, fractional
        return self.returns - cost[np.newaxis, :]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def estimate(
        self,
        factor_name: str,
        signals: np.ndarray,
    ) -> CapacityEstimate:
        """Run a capacity sweep across configured capital levels.

        Parameters
        ----------
        factor_name : str
            Human-readable factor identifier.
        signals : np.ndarray, shape (M, T)
            Factor signal matrix.

        Returns
        -------
        CapacityEstimate
        """
        gross_ic = compute_ic(signals, self.returns)
        abs_gross_mean = abs(self._mean_ic(gross_ic))

        curve: dict[float, float] = {}
        degradations: list[float] = []
        capitals: list[float] = []

        for cap in self.config.capacity_levels:
            impact = self._impact_model.estimate_impact(signals, self.volume, cap)
            net_ret = self._net_returns(signals, impact.impact_bps)
            net_ic = compute_ic(signals, net_ret)
            abs_net_mean = abs(self._mean_ic(net_ic))

            if abs_gross_mean > 1e-12:
                deg = 1.0 - abs_net_mean / abs_gross_mean
            else:
                deg = 0.0

            curve[cap] = deg
            capitals.append(cap)
            degradations.append(deg)

        # Interpolate to find capacity at the degradation limit
        max_cap = self._interpolate_capacity(
            capitals, degradations, self.config.ic_degradation_limit
        )

        # Break-even cost: gross IC expressed in bps
        # If the full round-trip cost equals the gross L-S spread the alpha
        # vanishes.  Approximate as gross_mean_ic * 10000 (IC ~ return spread).
        break_even_bps = abs_gross_mean * 1e4

        return CapacityEstimate(
            factor_name=factor_name,
            max_capacity_usd=max_cap,
            capacity_curve=curve,
            break_even_cost_bps=break_even_bps,
        )

    def net_cost_evaluation(
        self,
        factor_name: str,
        signals: np.ndarray,
        capital: float | None = None,
    ) -> NetCostResult:
        """Evaluate a factor net of estimated market impact.

        Parameters
        ----------
        factor_name : str
            Factor identifier.
        signals : np.ndarray, shape (M, T)
            Factor signal matrix.
        capital : float, optional
            Capital to evaluate at; defaults to ``config.base_capital_usd``.

        Returns
        -------
        NetCostResult
        """
        cap = capital if capital is not None else self.config.base_capital_usd

        # Gross metrics
        gross_ic = compute_ic(signals, self.returns)
        gross_icir = compute_icir(gross_ic)

        # Impact
        impact = self._impact_model.estimate_impact(signals, self.volume, cap)

        # Net metrics
        net_ret = self._net_returns(signals, impact.impact_bps)
        net_ic = compute_ic(signals, net_ret)
        net_icir = compute_icir(net_ic)

        # Gross / net long-short return (mean across time of Q5-Q1 proxy)
        gross_ls = float(np.nanmean(self.returns.mean(axis=0)))
        # Simplified: subtract round-trip impact from L-S return
        net_ls = gross_ls - 2.0 * impact.avg_impact_bps / 1e4

        return NetCostResult(
            factor_name=factor_name,
            gross_icir=gross_icir,
            net_icir=net_icir,
            gross_ls_return=gross_ls,
            net_ls_return=net_ls,
            estimated_capacity_usd=cap,
            impact_estimate=impact,
            passes_net_threshold=net_icir >= self.config.net_icir_threshold,
        )

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_capacity(
        capitals: list[float],
        degradations: list[float],
        limit: float,
    ) -> float:
        """Linearly interpolate the capital at which degradation hits *limit*.

        Returns ``np.inf`` if all tested levels are below the limit, or the
        smallest tested level if even that exceeds the limit.
        """
        if not capitals:
            return 0.0

        # Find first crossing
        for i in range(len(degradations)):
            if degradations[i] >= limit:
                if i == 0:
                    return capitals[0]
                # Linear interpolation between [i-1] and [i]
                d0, d1 = degradations[i - 1], degradations[i]
                c0, c1 = capitals[i - 1], capitals[i]
                if abs(d1 - d0) < 1e-12:
                    return c0
                frac = (limit - d0) / (d1 - d0)
                return c0 + frac * (c1 - c0)

        # Never breached the limit
        return float("inf")
