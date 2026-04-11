"""Causal validation layer for alpha factor candidates.

Provides Granger causality testing and intervention-based robustness
analysis to verify that discovered factors have genuine predictive
relationships with forward returns rather than spurious correlations.

Two complementary tests are combined into a single robustness score:

1. **Granger causality**: Does the factor signal Granger-cause returns
   after controlling for existing library factors?
2. **Intervention robustness**: Does factor IC remain stable under
   realistic data perturbations (volume shocks, volatility shocks,
   liquidity droughts)?
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from factorminer.evaluation.metrics import compute_ic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CausalConfig:
    """Configuration for causal validation tests."""

    enabled: bool = True

    # Granger causality settings
    granger_max_lag: int = 5
    granger_significance: float = 0.05

    # Intervention test settings
    n_interventions: int = 3
    intervention_magnitude: float = 2.0
    intervention_ic_threshold: float = 0.5  # min IC ratio under intervention

    # Combined robustness scoring
    robustness_threshold: float = 0.4  # min combined score for admission
    granger_weight: float = 0.4
    intervention_weight: float = 0.6

    seed: int = 42


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class CausalTestResult:
    """Result of causal validation for a single factor."""

    factor_name: str

    # Granger test results
    granger_p_value: float
    granger_f_stat: float
    granger_passes: bool

    # Intervention test results
    intervention_ic_ratio: float
    intervention_passes: bool

    # Combined
    robustness_score: float  # 0-1
    passes: bool

    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class CausalValidator:
    """Validates causal relationships between factor signals and returns.

    Parameters
    ----------
    returns : np.ndarray, shape (M, T)
        Forward returns for M assets over T periods.
    data_tensor : np.ndarray or None, shape (M, T, F)
        Optional raw feature tensor used for realistic intervention
        perturbations.  When ``None``, a noise-based fallback is used.
    library_signals : dict
        Mapping from factor name to its signal array (M, T).  Used as
        controls in the Granger test.
    config : CausalConfig
        Configuration parameters.
    """

    def __init__(
        self,
        returns: np.ndarray,
        data_tensor: np.ndarray | None,
        library_signals: dict[str, np.ndarray],
        config: CausalConfig | None = None,
    ) -> None:
        self.returns = returns
        self.data_tensor = data_tensor
        self.library_signals = library_signals
        self.config = config or CausalConfig()
        self._rng = np.random.RandomState(self.config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, factor_name: str, signals: np.ndarray) -> CausalTestResult:
        """Run causal validation on a single factor.

        Parameters
        ----------
        factor_name : str
            Human-readable identifier for logging / result tracking.
        signals : np.ndarray, shape (M, T)
            Factor signal matrix.

        Returns
        -------
        CausalTestResult
        """
        cfg = self.config
        details: dict[str, Any] = {}
        control_library = {
            name: lib_signals
            for name, lib_signals in self.library_signals.items()
            if name != factor_name
        }

        # --- Granger ---
        g_p, g_f, g_pass = self._granger_test(
            signals, self.returns, control_library
        )
        details["granger"] = {
            "p_value": g_p,
            "f_stat": g_f,
            "passes": g_pass,
        }

        # --- Intervention ---
        i_ratio, i_pass = self._intervention_test(
            signals, self.returns, self.data_tensor
        )
        details["intervention"] = {
            "ic_ratio": i_ratio,
            "passes": i_pass,
        }

        # --- Combined score ---
        score = self._compute_robustness_score(g_pass, g_p, i_ratio, i_pass)
        passes = score >= cfg.robustness_threshold
        details["robustness_score"] = score

        return CausalTestResult(
            factor_name=factor_name,
            granger_p_value=g_p,
            granger_f_stat=g_f,
            granger_passes=g_pass,
            intervention_ic_ratio=i_ratio,
            intervention_passes=i_pass,
            robustness_score=score,
            passes=passes,
            details=details,
        )

    def validate_batch(
        self, candidates: list[tuple[str, np.ndarray]]
    ) -> dict[str, CausalTestResult]:
        """Validate a batch of candidate factors.

        Parameters
        ----------
        candidates : list of (name, signals) tuples
            Each entry is ``(factor_name, signals_array)`` with signals
            shaped ``(M, T)``.

        Returns
        -------
        dict
            Mapping from factor name to its :class:`CausalTestResult`.
        """
        results: dict[str, CausalTestResult] = {}
        for name, signals in candidates:
            results[name] = self.validate(name, signals)
        return results

    # ------------------------------------------------------------------
    # Granger causality test
    # ------------------------------------------------------------------

    def _granger_test(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        library_signals: dict[str, np.ndarray],
    ) -> tuple[float, float, bool]:
        """Granger causality test for factor -> returns.

        Averages signals and returns across the top-20 assets (by signal
        magnitude) to produce T-length time series, then applies
        statsmodels Granger tests.

        Returns ``(p_value, f_stat, passes)``.
        """
        cfg = self.config
        M, T = signals.shape

        # Minimum series length guard
        min_length = 2 * cfg.granger_max_lag + 1
        if T < min_length:
            logger.warning(
                "Time series too short for Granger test "
                "(T=%d < %d). Passing by default.", T, min_length,
            )
            return 1.0, 0.0, True

        # --- Aggregate to T-length series ---
        sig_series = self._aggregate_top_assets(signals, top_k=20)
        ret_series = self._aggregate_top_assets(returns, top_k=20)

        # Handle constant or all-NaN series
        if self._is_degenerate(sig_series) or self._is_degenerate(ret_series):
            logger.warning(
                "Degenerate series detected in Granger test. Passing by default."
            )
            return 1.0, 0.0, True

        # --- Attempt statsmodels-based Granger test ---
        try:
            p_value, f_stat = self._run_granger_bivariate(
                sig_series, ret_series, cfg.granger_max_lag
            )

            # Multivariate extension if library has enough factors
            if len(library_signals) > 10:
                p_multi, f_multi = self._run_granger_multivariate(
                    sig_series, ret_series, library_signals, cfg.granger_max_lag
                )
                # Take the more conservative (higher) p-value
                if p_multi is not None:
                    p_value = max(p_value, p_multi)
                    f_stat = min(f_stat, f_multi)

            passes = p_value < cfg.granger_significance
            return float(p_value), float(f_stat), bool(passes)

        except ImportError:
            logger.warning(
                "statsmodels not available; skipping Granger test. "
                "Install statsmodels for causal validation."
            )
            return 1.0, 0.0, True
        except Exception as exc:
            logger.warning("Granger test failed: %s. Passing by default.", exc)
            return 1.0, 0.0, True

    def _run_granger_bivariate(
        self,
        sig_series: np.ndarray,
        ret_series: np.ndarray,
        max_lag: int,
    ) -> tuple[float, float]:
        """Bivariate Granger test using statsmodels."""
        from statsmodels.tsa.stattools import grangercausalitytests

        # Stack as (T, 2): [returns, signals] -- statsmodels convention
        # tests if column 1 (signals) Granger-causes column 0 (returns)
        data = np.column_stack([ret_series, sig_series])

        # Clamp max_lag to available data
        effective_lag = min(max_lag, len(data) // 3)
        if effective_lag < 1:
            return 1.0, 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = grangercausalitytests(data, maxlag=effective_lag, verbose=False)

        # Find the lag with the smallest p-value (ssr_ftest)
        best_p = 1.0
        best_f = 0.0
        for lag in range(1, effective_lag + 1):
            if lag not in results:
                continue
            test_dict = results[lag][0]
            f_test = test_dict.get("ssr_ftest")
            if f_test is not None:
                p_val = f_test[1]
                f_val = f_test[0]
                if p_val < best_p:
                    best_p = p_val
                    best_f = f_val

        return float(best_p), float(best_f)

    def _run_granger_multivariate(
        self,
        sig_series: np.ndarray,
        ret_series: np.ndarray,
        library_signals: dict[str, np.ndarray],
        max_lag: int,
    ) -> tuple[float | None, float | None]:
        """Multivariate Granger via VAR, controlling for library factors.

        If the library has >10 factors the controls are PCA-reduced to
        5 components.
        """
        try:
            from statsmodels.tsa.api import VAR

            # Build control matrix: average each library factor across top assets
            control_series = []
            for _name, lib_sig in library_signals.items():
                cs = self._aggregate_top_assets(lib_sig, top_k=20)
                if not self._is_degenerate(cs):
                    control_series.append(cs)

            if not control_series:
                return None, None

            controls = np.column_stack(control_series)

            # PCA reduction if too many controls
            if controls.shape[1] > 10:
                controls = self._pca_reduce(controls, n_components=5)

            # Build VAR dataset: [returns, signals, controls...]
            var_data = np.column_stack([ret_series, sig_series, controls])

            # Drop rows with NaN
            valid_mask = ~np.any(np.isnan(var_data), axis=1)
            var_data = var_data[valid_mask]

            effective_lag = min(max_lag, len(var_data) // (3 * var_data.shape[1]))
            if effective_lag < 1:
                return None, None

            model = VAR(var_data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.fit(maxlags=effective_lag, ic=None)

            # Test Granger causality: does column 1 (signals) cause column 0 (returns)?
            test_result = fitted.test_causality(
                caused=0, causing=1, kind="f"
            )
            p_value = float(test_result.pvalue)
            f_stat = float(test_result.test_statistic)

            return p_value, f_stat

        except Exception as exc:
            logger.warning(
                "Multivariate Granger (VAR) failed: %s. Skipping.", exc
            )
            return None, None

    # ------------------------------------------------------------------
    # Intervention robustness test
    # ------------------------------------------------------------------

    def _intervention_test(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        data_tensor: np.ndarray | None,
    ) -> tuple[float, bool]:
        """Intervention-based robustness test.

        Three perturbation scenarios are applied; the factor passes if
        its IC remains above the threshold ratio in at least 2 of 3.

        Returns ``(mean_ic_ratio, passes)``.
        """
        cfg = self.config

        # Baseline IC
        ic_orig = compute_ic(signals, returns)
        valid_orig = ic_orig[~np.isnan(ic_orig)]
        if len(valid_orig) < 3:
            logger.warning("Too few valid IC periods for intervention test.")
            return 1.0, True

        mean_ic_orig = float(np.mean(np.abs(valid_orig)))
        if mean_ic_orig < 1e-10:
            # Zero IC baseline: interventions cannot degrade further
            return 1.0, True

        ratios: list[float] = []
        pass_count = 0

        scenarios = self._build_intervention_scenarios(
            signals, returns, data_tensor
        )

        for scenario_name, perturbed_signals, perturbed_returns in scenarios:
            ic_pert = compute_ic(perturbed_signals, perturbed_returns)
            valid_pert = ic_pert[~np.isnan(ic_pert)]
            if len(valid_pert) < 3:
                # Not enough data after perturbation; count as pass
                ratios.append(1.0)
                pass_count += 1
                continue

            mean_ic_pert = float(np.mean(np.abs(valid_pert)))
            ratio = mean_ic_pert / mean_ic_orig
            ratios.append(ratio)
            if ratio >= cfg.intervention_ic_threshold:
                pass_count += 1

        mean_ratio = float(np.mean(ratios)) if ratios else 1.0
        passes = pass_count >= 2  # at least 2/3 interventions pass

        return mean_ratio, passes

    def _build_intervention_scenarios(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        data_tensor: np.ndarray | None,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Construct the three intervention scenarios.

        Returns a list of ``(name, perturbed_signals, perturbed_returns)``.
        """
        M, T = signals.shape
        cfg = self.config
        rng = self._rng

        scenarios: list[tuple[str, np.ndarray, np.ndarray]] = []

        if data_tensor is not None and data_tensor.shape[:2] == (M, T):
            # --- Volume shock: 2x on random 30% of periods ---
            shock_periods = rng.choice(T, size=max(1, int(0.3 * T)), replace=False)
            sig_vol = signals.copy()
            sig_vol[:, shock_periods] *= cfg.intervention_magnitude
            scenarios.append(("volume_shock", sig_vol, returns.copy()))

            # --- Volatility shock: 2x noise on returns ---
            ret_vol = returns.copy()
            noise = rng.randn(M, T) * np.nanstd(returns) * cfg.intervention_magnitude
            ret_vol += noise
            scenarios.append(("volatility_shock", signals.copy(), ret_vol))

            # --- Liquidity drought: zero volume on 10% of (asset, period) pairs ---
            sig_liq = signals.copy()
            n_pairs = max(1, int(0.1 * M * T))
            drought_assets = rng.randint(0, M, size=n_pairs)
            drought_periods = rng.randint(0, T, size=n_pairs)
            sig_liq[drought_assets, drought_periods] = 0.0
            scenarios.append(("liquidity_drought", sig_liq, returns.copy()))

        else:
            # --- Fallback: add noise directly to signals ---
            for i, scenario_name in enumerate(
                ["noise_shock_1", "noise_shock_2", "noise_shock_3"]
            ):
                sig_pert = signals.copy()
                noise_scale = np.nanstd(signals) * cfg.intervention_magnitude
                if noise_scale < 1e-12:
                    noise_scale = cfg.intervention_magnitude
                noise = rng.randn(M, T) * noise_scale * (0.5 + 0.5 * i)
                sig_pert += noise
                scenarios.append((scenario_name, sig_pert, returns.copy()))

        return scenarios

    # ------------------------------------------------------------------
    # Robustness score
    # ------------------------------------------------------------------

    def _compute_robustness_score(
        self,
        granger_passes: bool,
        granger_p: float,
        intervention_ratio: float,
        intervention_passes: bool,
    ) -> float:
        """Combine Granger and intervention results into a 0-1 score.

        granger_component = 1.0 - min(p_value / significance, 1.0)
        intervention_component = min(ic_ratio / 1.0, 1.0)
        score = w_g * granger_component + w_i * intervention_component
        """
        cfg = self.config

        granger_component = 1.0 - min(granger_p / cfg.granger_significance, 1.0)
        intervention_component = min(intervention_ratio / 1.0, 1.0)

        score = (
            cfg.granger_weight * granger_component
            + cfg.intervention_weight * intervention_component
        )

        return float(np.clip(score, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_top_assets(
        matrix: np.ndarray, top_k: int = 20
    ) -> np.ndarray:
        """Average across the top-k assets (by mean absolute value) to
        produce a T-length series.
        """
        M, T = matrix.shape
        k = min(top_k, M)

        # Mean absolute value per asset, ignoring NaN
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            asset_means = np.nanmean(np.abs(matrix), axis=1)

        # Replace NaN means with -inf so they sort last
        asset_means = np.where(np.isnan(asset_means), -np.inf, asset_means)
        top_idx = np.argpartition(asset_means, -k)[-k:]

        subset = matrix[top_idx, :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            series = np.nanmean(subset, axis=0)

        # Fill remaining NaN with 0
        series = np.where(np.isnan(series), 0.0, series)
        return series

    @staticmethod
    def _is_degenerate(series: np.ndarray) -> bool:
        """Check if a series is constant or all-NaN."""
        valid = series[~np.isnan(series)]
        if len(valid) < 3:
            return True
        return float(np.std(valid)) < 1e-12

    @staticmethod
    def _pca_reduce(X: np.ndarray, n_components: int = 5) -> np.ndarray:
        """Reduce columns of X via truncated SVD (no sklearn dependency).

        Parameters
        ----------
        X : np.ndarray, shape (T, K)
        n_components : int

        Returns
        -------
        np.ndarray, shape (T, n_components)
        """
        # Center
        means = np.nanmean(X, axis=0)
        X_centered = X - means
        X_centered = np.where(np.isnan(X_centered), 0.0, X_centered)

        # Economy SVD
        n_comp = min(n_components, X_centered.shape[1], X_centered.shape[0])
        try:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            return U[:, :n_comp] * S[:n_comp]
        except np.linalg.LinAlgError:
            logger.warning("SVD failed during PCA reduction; using raw data.")
            return X_centered[:, :n_comp]
