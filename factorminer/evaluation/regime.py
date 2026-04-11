"""Regime-aware factor validation.

Classifies market periods into BULL / BEAR / SIDEWAYS regimes using
rolling return and volatility statistics, then evaluates factor IC
within each regime to ensure robustness across market conditions.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------

class MarketRegime(enum.Enum):
    """Market regime labels."""

    BULL = 0
    BEAR = 1
    SIDEWAYS = 2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegimeConfig:
    """Parameters controlling regime detection and per-regime IC validation.

    Attributes
    ----------
    enabled : bool
        Whether regime-aware evaluation is active.
    lookback_window : int
        Rolling window length (periods) for mean-return and volatility
        estimation.
    bull_return_threshold : float
        Minimum rolling-mean return to qualify as BULL (when volatility is
        also below the threshold).
    bear_return_threshold : float
        Maximum rolling-mean return to qualify as BEAR.
    volatility_percentile : float
        Percentile (0-1) of rolling volatility used to compute the
        vol_threshold separating low-vol (BULL) from high-vol environments.
    min_regime_ic : float
        Minimum mean |IC| required within a single regime for it to "pass".
    min_regimes_passing : int
        How many regimes must pass for the factor to be accepted.
    """

    enabled: bool = True
    lookback_window: int = 60
    bull_return_threshold: float = 0.0
    bear_return_threshold: float = 0.0
    volatility_percentile: float = 0.7
    min_regime_ic: float = 0.03
    min_regimes_passing: int = 2


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class RegimeClassification:
    """Output of :class:`RegimeDetector.classify`.

    Attributes
    ----------
    labels : np.ndarray, shape (T,)
        Integer regime codes per period (0=BULL, 1=BEAR, 2=SIDEWAYS).
    periods : Dict[MarketRegime, np.ndarray]
        Boolean masks of shape (T,) indicating which periods belong to
        each regime.
    stats : Dict[MarketRegime, Dict[str, float]]
        Descriptive statistics per regime: ``mean_return``, ``volatility``,
        ``n_periods``.
    """

    labels: np.ndarray
    periods: dict[MarketRegime, np.ndarray]
    stats: dict[MarketRegime, dict[str, float]]


# ---------------------------------------------------------------------------
# Regime detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Classify time periods into market regimes.

    Parameters
    ----------
    config : RegimeConfig
        Regime detection parameters.
    """

    def __init__(self, config: RegimeConfig | None = None) -> None:
        self.config = config or RegimeConfig()

    # ----- public API -----

    def classify(self, returns: np.ndarray) -> RegimeClassification:
        """Classify each period into a market regime.

        Parameters
        ----------
        returns : np.ndarray, shape (M, T)
            Forward returns for *M* assets over *T* periods.

        Returns
        -------
        RegimeClassification
        """
        cfg = self.config
        M, T = returns.shape

        # Cross-sectional average return per period (handles NaN)
        market_return = np.nanmean(returns, axis=0)  # (T,)

        # Rolling statistics
        rolling_mean = self._rolling_nanmean(market_return, cfg.lookback_window)
        rolling_vol = self._rolling_nanstd(market_return, cfg.lookback_window)

        # Volatility threshold from valid (non-NaN) rolling vol values
        valid_vol = rolling_vol[~np.isnan(rolling_vol)]
        if len(valid_vol) > 0:
            vol_threshold = float(
                np.percentile(valid_vol, cfg.volatility_percentile * 100)
            )
        else:
            vol_threshold = np.inf  # fallback: nothing qualifies as low-vol

        # Assign labels
        labels = np.full(T, MarketRegime.SIDEWAYS.value, dtype=np.int64)

        # BEAR: rolling_return < bear_threshold  (checked first)
        bear_mask = rolling_mean < cfg.bear_return_threshold
        labels[bear_mask] = MarketRegime.BEAR.value

        # BULL: rolling_return > bull_threshold AND rolling_vol < vol_threshold
        bull_mask = (rolling_mean > cfg.bull_return_threshold) & (
            rolling_vol < vol_threshold
        )
        labels[bull_mask] = MarketRegime.BULL.value

        # First lookback_window periods default to SIDEWAYS
        labels[: cfg.lookback_window] = MarketRegime.SIDEWAYS.value

        # Build boolean masks & stats
        periods: dict[MarketRegime, np.ndarray] = {}
        stats: dict[MarketRegime, dict[str, float]] = {}

        for regime in MarketRegime:
            mask = labels == regime.value
            periods[regime] = mask
            regime_returns = market_return[mask]
            valid = regime_returns[~np.isnan(regime_returns)]
            stats[regime] = {
                "mean_return": float(np.mean(valid)) if len(valid) > 0 else 0.0,
                "volatility": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
                "n_periods": int(mask.sum()),
            }

        return RegimeClassification(labels=labels, periods=periods, stats=stats)

    # ----- helpers -----

    @staticmethod
    def _rolling_nanmean(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling mean that ignores NaN, returning NaN for the first *window-1* entries."""
        T = len(arr)
        out = np.full(T, np.nan, dtype=np.float64)
        for t in range(window - 1, T):
            chunk = arr[t - window + 1 : t + 1]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) > 0:
                out[t] = float(np.mean(valid))
        return out

    @staticmethod
    def _rolling_nanstd(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling std (ddof=1) that ignores NaN."""
        T = len(arr)
        out = np.full(T, np.nan, dtype=np.float64)
        for t in range(window - 1, T):
            chunk = arr[t - window + 1 : t + 1]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) > 1:
                out[t] = float(np.std(valid, ddof=1))
        return out


# ---------------------------------------------------------------------------
# Per-regime IC result
# ---------------------------------------------------------------------------

@dataclass
class RegimeICResult:
    """Evaluation result for a single factor across market regimes.

    Attributes
    ----------
    factor_name : str
        Human-readable factor identifier.
    regime_ic : Dict[MarketRegime, float]
        Mean |IC| per regime.
    regime_icir : Dict[MarketRegime, float]
        ICIR per regime.
    regime_n_periods : Dict[MarketRegime, int]
        Number of valid IC periods per regime.
    n_regimes_passing : int
        How many regimes met the ``min_regime_ic`` threshold.
    passes : bool
        Whether ``n_regimes_passing >= config.min_regimes_passing``.
    overall_regime_score : float
        Weighted average |IC| across regimes (weighted by n_periods).
    """

    factor_name: str
    regime_ic: dict[MarketRegime, float]
    regime_icir: dict[MarketRegime, float]
    regime_n_periods: dict[MarketRegime, int]
    n_regimes_passing: int
    passes: bool
    overall_regime_score: float


# ---------------------------------------------------------------------------
# Regime-aware evaluator
# ---------------------------------------------------------------------------

class RegimeAwareEvaluator:
    """Evaluate factor IC within each market regime.

    Parameters
    ----------
    returns : np.ndarray, shape (M, T)
        Forward returns.
    regime : RegimeClassification
        Pre-computed regime classification.
    config : RegimeConfig
        Thresholds and evaluation parameters.
    """

    def __init__(
        self,
        returns: np.ndarray,
        regime: RegimeClassification,
        config: RegimeConfig | None = None,
    ) -> None:
        self.returns = returns
        self.regime = regime
        self.config = config or RegimeConfig()

    # ----- public API -----

    def evaluate(self, factor_name: str, signals: np.ndarray) -> RegimeICResult:
        """Evaluate a single factor across regimes.

        Parameters
        ----------
        factor_name : str
            Identifier for reporting.
        signals : np.ndarray, shape (M, T)
            Factor signal matrix.

        Returns
        -------
        RegimeICResult
        """
        cfg = self.config
        min_periods = cfg.lookback_window * 2

        regime_ic: dict[MarketRegime, float] = {}
        regime_icir: dict[MarketRegime, float] = {}
        regime_n_periods: dict[MarketRegime, int] = {}

        for regime in MarketRegime:
            mask = self.regime.periods[regime]
            n_regime = int(mask.sum())

            if n_regime < min_periods:
                regime_ic[regime] = 0.0
                regime_icir[regime] = 0.0
                regime_n_periods[regime] = n_regime
                continue

            # Extract time-sliced sub-arrays
            indices = np.where(mask)[0]
            sig_sub = signals[:, indices]
            ret_sub = self.returns[:, indices]

            ic_series = self._compute_ic(sig_sub, ret_sub)
            valid_ic = ic_series[~np.isnan(ic_series)]

            mean_abs_ic = float(np.mean(np.abs(valid_ic))) if len(valid_ic) > 0 else 0.0
            icir = self._compute_icir(valid_ic)

            regime_ic[regime] = mean_abs_ic
            regime_icir[regime] = icir
            regime_n_periods[regime] = int(len(valid_ic))

        # Count passing regimes
        n_passing = sum(
            1
            for r in MarketRegime
            if regime_n_periods[r] >= min_periods and regime_ic[r] >= cfg.min_regime_ic
        )

        # Weighted average IC
        total_weight = sum(regime_n_periods[r] for r in MarketRegime)
        if total_weight > 0:
            overall_score = sum(
                regime_ic[r] * regime_n_periods[r] for r in MarketRegime
            ) / total_weight
        else:
            overall_score = 0.0

        return RegimeICResult(
            factor_name=factor_name,
            regime_ic=regime_ic,
            regime_icir=regime_icir,
            regime_n_periods=regime_n_periods,
            n_regimes_passing=n_passing,
            passes=n_passing >= cfg.min_regimes_passing,
            overall_regime_score=overall_score,
        )

    def evaluate_batch(
        self,
        candidates: dict[str, np.ndarray],
    ) -> dict[str, RegimeICResult]:
        """Evaluate multiple factors.

        Parameters
        ----------
        candidates : Dict[str, np.ndarray]
            Mapping of factor name to signal matrix (M, T).

        Returns
        -------
        Dict[str, RegimeICResult]
        """
        return {name: self.evaluate(name, sig) for name, sig in candidates.items()}

    # ----- IC helpers (mirrors metrics.py conventions) -----

    @staticmethod
    def _compute_ic(signals: np.ndarray, returns: np.ndarray) -> np.ndarray:
        """Cross-sectional Spearman IC per period.

        Replicates the logic in ``metrics.compute_ic`` to keep this module
        self-contained while matching the project convention.
        """
        M, T = signals.shape
        ic_series = np.full(T, np.nan, dtype=np.float64)

        for t in range(T):
            s = signals[:, t]
            r = returns[:, t]
            valid = ~(np.isnan(s) | np.isnan(r))
            n = valid.sum()
            if n < 5:
                continue
            rs = rankdata(s[valid])
            rr = rankdata(r[valid])
            rs_m = rs - rs.mean()
            rr_m = rr - rr.mean()
            denom = np.sqrt((rs_m ** 2).sum() * (rr_m ** 2).sum())
            if denom < 1e-12:
                ic_series[t] = 0.0
            else:
                ic_series[t] = (rs_m * rr_m).sum() / denom

        return ic_series

    @staticmethod
    def _compute_icir(valid_ic: np.ndarray) -> float:
        """ICIR = mean(IC) / std(IC)."""
        if len(valid_ic) < 3:
            return 0.0
        std = float(np.std(valid_ic, ddof=1))
        if std < 1e-12:
            return 0.0
        return float(np.mean(valid_ic)) / std


# ---------------------------------------------------------------------------
# Phase 2: Streaming regime detection (added for HelixFactor)
# ---------------------------------------------------------------------------

class TrendRegime(enum.Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


class VolRegime(enum.Enum):
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    NORMAL_VOL = "normal_vol"


class MeanRevRegime(enum.Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    RANDOM_WALK = "random_walk"


@dataclass
class RegimeState:
    """Composite regime state: trend + vol + mean-reversion classification."""
    trend: TrendRegime = TrendRegime.NEUTRAL
    vol: VolRegime = VolRegime.NORMAL_VOL
    mean_rev: MeanRevRegime = MeanRevRegime.RANDOM_WALK

    def to_dict(self) -> dict:
        return {
            "trend": self.trend.value,
            "vol": self.vol.value,
            "mean_rev": self.mean_rev.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegimeState:
        return cls(
            trend=TrendRegime(d.get("trend", "neutral")),
            vol=VolRegime(d.get("vol", "normal_vol")),
            mean_rev=MeanRevRegime(d.get("mean_rev", "random_walk")),
        )

    def __str__(self) -> str:
        return f"{self.trend.value}/{self.vol.value}/{self.mean_rev.value}"

    def label(self) -> str:
        return str(self)


@dataclass
class StreamingRegimeConfig:
    """Configuration for StreamingRegimeDetector."""
    fast_alpha: float = 0.1          # EW decay for fast stats
    slow_alpha: float = 0.02         # EW decay for slow (baseline) stats
    trend_sigma_threshold: float = 1.0  # sigmas above/below zero for BULL/BEAR
    vol_high_quantile: float = 0.75  # quantile threshold for HIGH_VOL
    vol_low_quantile: float = 0.25   # quantile threshold for LOW_VOL
    hurst_lags: tuple = (2, 4, 8, 16)  # lags for variance-ratio Hurst estimate
    hmm_smoothing: float = 0.3       # sticky-state weight (0 = no smoothing)
    history_maxlen: int = 500        # max regime history records


class StreamingRegimeDetector:
    """Bar-by-bar O(1) regime classifier using exponentially-weighted stats.

    Detects three independent regime axes:
    - Trend: BULL / BEAR / NEUTRAL  (EW mean vs threshold)
    - Volatility: HIGH / LOW / NORMAL  (EW std vs quantile buffer)
    - Mean-reversion: TRENDING / MEAN_REVERTING / RANDOM_WALK (Hurst via variance ratio)
    """

    def __init__(self, config: StreamingRegimeConfig | None = None) -> None:
        self.config = config or StreamingRegimeConfig()
        # Exponentially-weighted moments
        self._ew_mean: float = 0.0
        self._ew_var: float = 0.0          # fast (for current vol)
        self._ew_var_slow: float = 0.0     # slow (baseline)
        self._n: int = 0
        # Rolling buffers for variance-ratio Hurst
        self._return_buffer: list = []
        self._vol_buffer: list = []        # rolling realized vol samples
        # Regime history
        from collections import deque
        self._history: deque = deque(maxlen=self.config.history_maxlen)
        self._transition_counts: dict = {}
        self._current: RegimeState = RegimeState()
        import threading
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        returns: np.ndarray,        # (M,) cross-sectional returns at this bar
        volumes: np.ndarray | None = None,  # (M,) optional — unused currently
    ) -> RegimeState:
        """Process one bar and return updated RegimeState."""
        with self._lock:
            r = float(np.nanmean(returns))
            vol = float(np.nanstd(returns)) if len(returns) > 1 else 0.0
            self._update_moments(r, vol)
            new_state = self._classify()
            self._apply_smoothing(new_state)
            self._record_transition(self._current, new_state)
            self._current = new_state
            self._history.append(new_state)
            return new_state

    def get_current_regime(self) -> RegimeState:
        with self._lock:
            return self._current

    def get_regime_history(self, lookback: int = 20) -> list:
        with self._lock:
            hist = list(self._history)
            return hist[-lookback:] if lookback else hist

    def regime_transition_probability(self) -> dict:
        """Return dict of 'from/to' → empirical probability."""
        with self._lock:
            total = sum(self._transition_counts.values())
            if total == 0:
                return {}
            return {k: v / total for k, v in self._transition_counts.items()}

    def reset(self) -> None:
        with self._lock:
            self.__init__(config=self.config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_moments(self, r: float, vol: float) -> None:
        fa, sa = self.config.fast_alpha, self.config.slow_alpha
        if self._n == 0:
            self._ew_mean = r
            self._ew_var = vol ** 2
            self._ew_var_slow = vol ** 2
        else:
            self._ew_mean = fa * r + (1 - fa) * self._ew_mean
            self._ew_var = fa * (r - self._ew_mean) ** 2 + (1 - fa) * self._ew_var
            self._ew_var_slow = sa * vol ** 2 + (1 - sa) * self._ew_var_slow
        self._n += 1
        self._return_buffer.append(r)
        self._vol_buffer.append(vol)
        if len(self._return_buffer) > 200:
            self._return_buffer.pop(0)
            self._vol_buffer.pop(0)

    def _classify(self) -> RegimeState:
        return RegimeState(
            trend=self._classify_trend(),
            vol=self._classify_vol(),
            mean_rev=self._classify_mean_rev(),
        )

    def _classify_trend(self) -> TrendRegime:
        sigma = float(np.sqrt(max(self._ew_var, 1e-16)))
        n = max(self._n, 1)
        se = sigma / (n ** 0.5)
        thresh = self.config.trend_sigma_threshold * se
        if self._ew_mean > thresh:
            return TrendRegime.BULL
        elif self._ew_mean < -thresh:
            return TrendRegime.BEAR
        return TrendRegime.NEUTRAL

    def _classify_vol(self) -> VolRegime:
        if len(self._vol_buffer) < 10:
            return VolRegime.NORMAL_VOL
        arr = np.array(self._vol_buffer)
        cur = float(np.sqrt(max(self._ew_var, 0.0)))
        hi = float(np.quantile(arr, self.config.vol_high_quantile))
        lo = float(np.quantile(arr, self.config.vol_low_quantile))
        if cur > hi:
            return VolRegime.HIGH_VOL
        elif cur < lo:
            return VolRegime.LOW_VOL
        return VolRegime.NORMAL_VOL

    def _classify_mean_rev(self) -> MeanRevRegime:
        buf = self._return_buffer
        if len(buf) < 32:
            return MeanRevRegime.RANDOM_WALK
        arr = np.array(buf)
        lags = [lag for lag in self.config.hurst_lags if lag < len(arr)]
        if not lags:
            return MeanRevRegime.RANDOM_WALK
        ratios = []
        for lag in lags:
            var_lag = float(np.var(arr[lag:] - arr[:-lag]))
            var_1 = float(np.var(np.diff(arr))) if len(arr) > 1 else 1e-16
            if var_1 > 1e-16:
                ratios.append(var_lag / (lag * var_1))
        if not ratios:
            return MeanRevRegime.RANDOM_WALK
        hurst_proxy = float(np.mean(ratios))
        if hurst_proxy > 1.1:
            return MeanRevRegime.TRENDING
        elif hurst_proxy < 0.9:
            return MeanRevRegime.MEAN_REVERTING
        return MeanRevRegime.RANDOM_WALK

    def _apply_smoothing(self, new_state: RegimeState) -> None:
        """HMM-inspired: resist single-bar flips via smoothing weight."""
        w = self.config.hmm_smoothing
        if w <= 0 or self._current is None:
            return
        # If smoothing weight is high and current state differs, revert in-place
        # (simple sticky-state: only update if change is "strong enough")
        # We achieve this by probabilistic rejection — deterministic version:
        # keep current if random draw < smoothing weight (approximate)
        import random
        if (new_state.trend != self._current.trend or
                new_state.vol != self._current.vol):
            if random.random() < w:
                new_state.trend = self._current.trend
                new_state.vol = self._current.vol

    def _record_transition(self, old: RegimeState, new: RegimeState) -> None:
        key = f"{old.label()}->{new.label()}"
        self._transition_counts[key] = self._transition_counts.get(key, 0) + 1
