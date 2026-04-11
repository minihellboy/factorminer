"""Statistical significance testing for alpha factors.

Provides block bootstrap confidence intervals, Benjamini-Hochberg FDR
control, and Deflated Sharpe Ratio (Bailey & López de Prado, 2014) to
guard against data-snooping and multiple-testing bias in factor research.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import kurtosis, norm, skew

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SignificanceConfig:
    """Configuration for all significance tests."""

    enabled: bool = True
    bootstrap_n_samples: int = 1000
    bootstrap_block_size: int = 20
    bootstrap_confidence: float = 0.95
    fdr_level: float = 0.05
    deflated_sharpe_enabled: bool = True
    min_deflated_sharpe: float = 0.0
    seed: int = 42


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCIResult:
    """Result of a block bootstrap confidence interval for mean |IC|."""

    factor_name: str
    ic_mean: float
    ci_lower: float
    ci_upper: float
    ic_std_boot: float
    ci_excludes_zero: bool


class BootstrapICTester:
    """Block bootstrap tester for IC series significance.

    Uses circular block bootstrap to preserve time-series autocorrelation
    when constructing confidence intervals for mean |IC|.

    Parameters
    ----------
    config : SignificanceConfig
        Bootstrap parameters (n_samples, block_size, confidence, seed).
    """

    def __init__(self, config: SignificanceConfig) -> None:
        self._config = config
        self._rng = np.random.RandomState(config.seed)

    # ----- public API -----

    def compute_ci(
        self, factor_name: str, ic_series: np.ndarray
    ) -> BootstrapCIResult:
        """Compute block-bootstrap CI for mean |IC|.

        Parameters
        ----------
        factor_name : str
            Human-readable factor identifier.
        ic_series : np.ndarray, shape (T,)
            IC time series (NaN entries are dropped before resampling).

        Returns
        -------
        BootstrapCIResult
        """
        valid = ic_series[~np.isnan(ic_series)]
        T = len(valid)
        if T == 0:
            return BootstrapCIResult(
                factor_name=factor_name,
                ic_mean=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                ic_std_boot=0.0,
                ci_excludes_zero=False,
            )

        abs_valid = np.abs(valid)
        ic_mean = float(np.mean(abs_valid))

        boot_means = self._block_bootstrap_means(abs_valid)

        alpha = 1.0 - self._config.bootstrap_confidence
        ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        ic_std_boot = float(np.std(boot_means, ddof=1))

        return BootstrapCIResult(
            factor_name=factor_name,
            ic_mean=ic_mean,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ic_std_boot=ic_std_boot,
            ci_excludes_zero=ci_lower > 0,
        )

    def compute_p_value(self, ic_series: np.ndarray) -> float:
        """Estimate a two-sided p-value for non-zero mean IC.

        Uses a sign-flip randomization test on the observed IC series.
        Under the null of no predictive signal, flipping the sign of each
        period's IC leaves the distribution unchanged while preserving the
        magnitude structure of the observed sample.
        """
        valid = ic_series[~np.isnan(ic_series)]
        T = len(valid)
        if T == 0:
            return 1.0

        observed = float(abs(np.mean(valid)))
        if observed < 1e-15:
            return 1.0

        null_means = np.empty(self._config.bootstrap_n_samples, dtype=np.float64)
        for i in range(self._config.bootstrap_n_samples):
            signs = self._rng.choice((-1.0, 1.0), size=T)
            null_means[i] = abs(float(np.mean(valid * signs)))

        exceedances = int(np.sum(null_means >= observed))
        return float((exceedances + 1) / (len(null_means) + 1))

    # ----- internals -----

    def _effective_block_size(self, T: int) -> int:
        """Adaptive block size: min(configured, T // 10), at least 1."""
        bs = self._config.bootstrap_block_size
        adaptive = max(T // 10, 1)
        return min(bs, adaptive)

    def _block_bootstrap_means(self, series: np.ndarray) -> np.ndarray:
        """Generate bootstrap distribution of the sample mean.

        Parameters
        ----------
        series : np.ndarray, shape (T,)
            Already cleaned (no NaN) series values.

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Bootstrap sample means.
        """
        T = len(series)
        block_size = self._effective_block_size(T)
        n_blocks = int(math.ceil(T / block_size))
        n_samples = self._config.bootstrap_n_samples

        boot_means = np.empty(n_samples, dtype=np.float64)
        max_start = T - block_size  # last valid block start

        for i in range(n_samples):
            # Sample block start indices with replacement
            starts = self._rng.randint(0, max_start + 1, size=n_blocks)
            # Concatenate blocks and truncate to length T
            indices = np.concatenate(
                [np.arange(s, s + block_size) for s in starts]
            )[:T]
            boot_means[i] = series[indices].mean()

        return boot_means


# ---------------------------------------------------------------------------
# FDR Control (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

@dataclass
class FDRResult:
    """Result of Benjamini-Hochberg FDR correction."""

    raw_p_values: dict[str, float]
    adjusted_p_values: dict[str, float]
    significant: dict[str, bool]
    n_discoveries: int
    fdr_level: float


class FDRController:
    """Benjamini-Hochberg FDR correction for multiple factor testing.

    Parameters
    ----------
    config : SignificanceConfig
    """

    def __init__(self, config: SignificanceConfig) -> None:
        self._config = config

    def apply_fdr(self, p_values: dict[str, float]) -> FDRResult:
        """Apply Benjamini-Hochberg procedure.

        Parameters
        ----------
        p_values : Dict[str, float]
            Mapping of factor_name -> raw p-value.

        Returns
        -------
        FDRResult
        """
        if not p_values:
            return FDRResult(
                raw_p_values={},
                adjusted_p_values={},
                significant={},
                n_discoveries=0,
                fdr_level=self._config.fdr_level,
            )

        names = list(p_values.keys())
        raw = np.array([p_values[n] for n in names], dtype=np.float64)
        m = len(raw)

        # Sort ascending
        order = np.argsort(raw)
        sorted_raw = raw[order]

        # BH adjusted p-values: p_adj[i] = min(p[i] * m / (i+1), 1.0)
        adjusted = np.empty(m, dtype=np.float64)
        for idx in range(m):
            rank = idx + 1  # 1-indexed rank
            adjusted[idx] = min(sorted_raw[idx] * m / rank, 1.0)

        # Enforce monotonicity from bottom up
        for idx in range(m - 2, -1, -1):
            adjusted[idx] = min(adjusted[idx], adjusted[idx + 1])

        # Map back to original order
        inv_order = np.empty(m, dtype=int)
        inv_order[order] = np.arange(m)
        adjusted_orig = adjusted[inv_order]

        adjusted_dict: dict[str, float] = {}
        significant_dict: dict[str, bool] = {}
        for i, name in enumerate(names):
            adjusted_dict[name] = float(adjusted_orig[i])
            significant_dict[name] = adjusted_orig[i] <= self._config.fdr_level

        return FDRResult(
            raw_p_values=dict(p_values),
            adjusted_p_values=adjusted_dict,
            significant=significant_dict,
            n_discoveries=sum(significant_dict.values()),
            fdr_level=self._config.fdr_level,
        )

    def batch_evaluate(
        self,
        ic_series_map: dict[str, np.ndarray],
        bootstrap_tester: BootstrapICTester,
    ) -> FDRResult:
        """Compute bootstrap p-values for all factors, then apply BH.

        Parameters
        ----------
        ic_series_map : Dict[str, np.ndarray]
            Mapping of factor_name -> IC series (T,).
        bootstrap_tester : BootstrapICTester

        Returns
        -------
        FDRResult
        """
        p_values: dict[str, float] = {}
        for name, ic_series in ic_series_map.items():
            p_values[name] = bootstrap_tester.compute_p_value(ic_series)
        return self.apply_fdr(p_values)


# ---------------------------------------------------------------------------
# Deflated Sharpe Ratio
# ---------------------------------------------------------------------------

@dataclass
class DeflatedSharpeResult:
    """Result of Deflated Sharpe Ratio test."""

    factor_name: str
    raw_sharpe: float
    deflated_sharpe: float
    haircut: float
    p_value: float
    n_trials: int
    passes: bool


class DeflatedSharpeCalculator:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts the observed Sharpe Ratio for multiple testing by estimating
    the expected maximum Sharpe under the null hypothesis of zero skill
    across *n_trials* independent strategies.

    Parameters
    ----------
    config : SignificanceConfig
    """

    _EULER_GAMMA = 0.5772156649015329

    def __init__(self, config: SignificanceConfig) -> None:
        self._config = config

    def compute(
        self,
        factor_name: str,
        ls_returns: np.ndarray,
        n_trials: int,
        annualization_factor: float = 252.0,
    ) -> DeflatedSharpeResult:
        """Compute the Deflated Sharpe Ratio for a factor's L/S returns.

        Parameters
        ----------
        factor_name : str
        ls_returns : np.ndarray, shape (T,)
            Long-short portfolio return series (NaN-free expected).
        n_trials : int
            Total number of strategy trials (including this one).
        annualization_factor : float
            Trading periods per year (default 252).

        Returns
        -------
        DeflatedSharpeResult
        """
        valid = ls_returns[~np.isnan(ls_returns)]
        T = len(valid)

        if T < 10 or n_trials < 1:
            return DeflatedSharpeResult(
                factor_name=factor_name,
                raw_sharpe=0.0,
                deflated_sharpe=0.0,
                haircut=0.0,
                p_value=1.0,
                n_trials=n_trials,
                passes=False,
            )

        # Annualised Sharpe
        mean_r = float(np.mean(valid))
        std_r = float(np.std(valid, ddof=1))
        if std_r < 1e-15:
            return DeflatedSharpeResult(
                factor_name=factor_name,
                raw_sharpe=0.0,
                deflated_sharpe=0.0,
                haircut=0.0,
                p_value=1.0,
                n_trials=n_trials,
                passes=False,
            )

        SR = (mean_r / std_r) * math.sqrt(annualization_factor)

        # Expected maximum SR under the null (Bailey & LdP, 2014)
        e_max_sr = self._expected_max_sr(n_trials)

        # Higher moments of returns
        gamma3 = float(skew(valid, bias=False))
        gamma4 = float(kurtosis(valid, fisher=True, bias=False))  # excess kurtosis

        # Variance correction incorporating skewness and kurtosis
        var_correction = (1.0 - gamma3 * SR + (gamma4 - 1.0) / 4.0 * SR ** 2) / T

        if var_correction <= 0:
            deflated_sr = 0.0
        else:
            deflated_sr = (SR - e_max_sr) / math.sqrt(var_correction)

        p_value = 1.0 - float(norm.cdf(deflated_sr))
        haircut = SR - deflated_sr

        passes = (
            deflated_sr > self._config.min_deflated_sharpe and p_value < 0.05
        )

        return DeflatedSharpeResult(
            factor_name=factor_name,
            raw_sharpe=SR,
            deflated_sharpe=deflated_sr,
            haircut=haircut,
            p_value=p_value,
            n_trials=n_trials,
            passes=passes,
        )

    @classmethod
    def _expected_max_sr(cls, n_trials: int) -> float:
        """E[max(SR)] approximation from Bailey & López de Prado (2014).

        E[max(SR)] ~ sqrt(2*ln(N)) * (1 - gamma / (2*ln(N))) + gamma / sqrt(2*ln(N))
        """
        if n_trials <= 1:
            return 0.0
        log_n = math.log(n_trials)
        sqrt_2log = math.sqrt(2.0 * log_n)
        g = cls._EULER_GAMMA
        return sqrt_2log * (1.0 - g / (2.0 * log_n)) + g / sqrt_2log


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def check_significance(
    factor_name: str,
    ic_series: np.ndarray,
    ls_returns: np.ndarray,
    n_total_trials: int,
    config: SignificanceConfig | None = None,
) -> tuple[bool, str | None, dict]:
    """Run all significance checks on a single factor.

    Executes bootstrap CI, bootstrap p-value, and (optionally) the
    Deflated Sharpe Ratio test.  Returns an overall pass/fail verdict
    with a human-readable rejection reason.

    Parameters
    ----------
    factor_name : str
    ic_series : np.ndarray, shape (T,)
    ls_returns : np.ndarray, shape (T,)
    n_total_trials : int
        Total number of factor trials (for DSR correction).
    config : SignificanceConfig, optional
        If *None*, defaults are used.

    Returns
    -------
    Tuple[bool, Optional[str], Dict]
        (passes, rejection_reason, details)
        *passes* is True when all enabled tests succeed.
        *rejection_reason* is None when *passes* is True.
        *details* contains per-test result objects.
    """
    if config is None:
        config = SignificanceConfig()

    if not config.enabled:
        return True, None, {"skipped": True}

    details: dict = {}

    # -- Bootstrap IC CI / p-value --
    bt = BootstrapICTester(config)
    ci_result = bt.compute_ci(factor_name, ic_series)
    details["bootstrap_ci"] = ci_result
    p_value = bt.compute_p_value(ic_series)
    details["bootstrap_p_value"] = p_value

    if p_value > config.fdr_level:
        return (
            False,
            f"Bootstrap p-value {p_value:.4f} exceeds alpha {config.fdr_level:.4f}",
            details,
        )

    # -- Deflated Sharpe Ratio --
    if config.deflated_sharpe_enabled:
        dsr = DeflatedSharpeCalculator(config)
        dsr_result = dsr.compute(factor_name, ls_returns, n_total_trials)
        details["deflated_sharpe"] = dsr_result

        if not dsr_result.passes:
            return (
                False,
                f"Deflated Sharpe test failed: DSR={dsr_result.deflated_sharpe:.3f}, "
                f"p={dsr_result.p_value:.4f}, haircut={dsr_result.haircut:.3f} "
                f"(n_trials={n_total_trials})",
                details,
            )

    return True, None, details
