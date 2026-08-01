"""Statistical significance testing for alpha factors.

Provides block bootstrap confidence intervals, Benjamini-Hochberg FDR
control, and Deflated Sharpe Ratio (Bailey & López de Prado, 2014) to
guard against data-snooping and multiple-testing bias in factor research.

Also provides a thin additive bridge for Agora-style sealed multi-evaluator
agreement diagnostics (see ``factorminer.architecture.sealed_joint_search``).
That bridge never changes default single-factor significance behavior.
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
    bootstrap_method: str = "circular_block"
    fdr_level: float = 0.05
    deflated_sharpe_enabled: bool = True
    min_deflated_sharpe: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class SealedAgreementConfig:
    """Opt-in config for sealed multi-evaluator agreement diagnostics.

    Additive companion to :class:`SignificanceConfig`. Default ``enabled=False``
    keeps existing significance behavior byte-identical. When enabled, callers
    may attach a coarse multi-evaluator agreement summary alongside classical
    DSR/FDR/bootstrap results — useful as an anti-Goodhart research overlay,
    not a replacement for those tests.

    See arXiv:2606.29194 (Agora). Paper caveat: single-seed variance is real;
    do not treat sealed agreement as a proven default upgrade.
    """

    enabled: bool = False
    agreement_rule: str = "majority"
    min_agree: int = 2
    include_llm_judge: bool = True


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

@dataclass
class BootstrapCIResult:
    """Result of a block bootstrap confidence interval for mean IC.

    ``ic_mean`` is the signed time-average ``mean(IC_t)``; ``ic_paper_mean``
    is its absolute value — the ``paper_ic_v2`` admission statistic. The CI
    brackets the signed mean so it is consistent with the sign-flip p-value,
    which tests ``|mean(IC_t)|``.
    """

    factor_name: str
    ic_mean: float
    ci_lower: float
    ci_upper: float
    ic_std_boot: float
    ci_excludes_zero: bool
    ic_paper_mean: float = 0.0
    bootstrap_method: str = "circular_block"


def stationary_bootstrap_indices(
    length: int,
    mean_block_size: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Draw Politis-Romano stationary-bootstrap indices.

    Blocks have geometrically distributed lengths with expectation
    ``mean_block_size``.  The first index and every restart are uniform; a
    non-restart advances circularly, preserving local serial dependence.
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if mean_block_size < 1:
        raise ValueError("mean_block_size must be >= 1")
    restart_probability = 1.0 / min(mean_block_size, length)
    indices = np.empty(length, dtype=np.int64)
    indices[0] = int(rng.randint(0, length))
    for position in range(1, length):
        if float(rng.random_sample()) < restart_probability:
            indices[position] = int(rng.randint(0, length))
        else:
            indices[position] = (indices[position - 1] + 1) % length
    return indices


class BootstrapICTester:
    """Block bootstrap tester for IC series significance.

    Uses circular block bootstrap to preserve time-series autocorrelation
    when constructing confidence intervals for the signed mean IC.

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
        """Compute a block-bootstrap CI for the signed mean IC.

        The CI targets ``mean(IC_t)`` — the statistic whose absolute value
        is the ``paper_ic_v2`` gate — so it agrees with
        :meth:`compute_p_value`, which tests ``|mean(IC_t)|`` under a
        sign-flip null. A series with magnitude but no stable direction
        (e.g. alternating ±0.1) yields a CI covering zero, not a spuriously
        positive interval on ``mean(|IC_t|)``.

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
                ic_paper_mean=0.0,
                bootstrap_method=self._config.bootstrap_method,
            )

        ic_mean = float(np.mean(valid))

        boot_means = self._block_bootstrap_means(valid)

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
            ci_excludes_zero=bool(ci_lower > 0 or ci_upper < 0),
            ic_paper_mean=abs(ic_mean),
            bootstrap_method=self._config.bootstrap_method,
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

        for i in range(n_samples):
            if self._config.bootstrap_method == "stationary":
                indices = stationary_bootstrap_indices(T, block_size, self._rng)
            elif self._config.bootstrap_method == "circular_block":
                # Circular: any start is valid, blocks wrap modulo T so every
                # observation is sampled with equal probability.
                starts = self._rng.randint(0, T, size=n_blocks)
                indices = np.concatenate(
                    [np.arange(s, s + block_size) for s in starts]
                )[:T] % T
            else:
                raise ValueError(
                    "bootstrap_method must be 'circular_block' or 'stationary'"
                )
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
    n_trials: float
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
        n_trials: float,
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
        # Bailey & Lopez de Prado's gamma4 is *raw* (non-excess) kurtosis --
        # a normal distribution has gamma4 = 3, giving the familiar
        # (gamma4 - 1) / 4 = 0.5 asymptotic variance term. scipy's default
        # `fisher=True` returns *excess* kurtosis (normal = 0); using that
        # directly here would understate var_correction for fat-tailed
        # returns (gamma4_excess > 0 is the common case for real return
        # series) and make the deflation systematically too permissive.
        gamma4 = float(kurtosis(valid, fisher=False, bias=False))  # raw kurtosis

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
    def _expected_max_sr(cls, n_trials: float) -> float:
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
# Correlation-adjusted trials and family-wide bootstrap tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectiveTrialsEstimate:
    """Correlation-adjusted independent-trial estimate for one family."""

    raw_trials: int
    effective_trials: float
    average_pairwise_correlation: float
    clipped_average_correlation: float
    eigenvalue_effective_rank: float
    method: str = "bailey_lopez_de_prado_average_correlation"


def estimate_effective_trials(trial_series: np.ndarray) -> EffectiveTrialsEstimate:
    """Estimate implied independent trials from cross-trial correlation.

    The input orientation is ``(trials, periods)``.  Bailey and López de
    Prado's interpolation ``rho + (1-rho) M`` is used for DSR.  Negative
    average correlation is clipped at zero, conservatively preventing an
    estimate larger than the number of canonical hypotheses.  The spectral
    participation ratio is reported as a second geometry diagnostic.
    """
    matrix = np.asarray(trial_series, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("trial_series must have shape (trials >= 1, periods >= 2)")
    raw_trials = int(matrix.shape[0])
    if raw_trials == 1:
        return EffectiveTrialsEstimate(1, 1.0, 1.0, 1.0, 1.0)

    normalized = matrix.copy()
    for row_index in range(raw_trials):
        row = normalized[row_index]
        finite = np.isfinite(row)
        fill = float(np.mean(row[finite])) if np.any(finite) else 0.0
        row[~finite] = fill

    correlation = np.eye(raw_trials, dtype=np.float64)
    variable = np.std(normalized, axis=1) > 1e-12
    if np.count_nonzero(variable) >= 2:
        variable_corr = np.corrcoef(normalized[variable])
        variable_indices = np.flatnonzero(variable)
        correlation[np.ix_(variable_indices, variable_indices)] = variable_corr
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)

    off_diagonal = correlation[~np.eye(raw_trials, dtype=bool)]
    average = float(np.mean(off_diagonal))
    clipped = float(np.clip(average, 0.0, 1.0))
    effective = float(clipped + (1.0 - clipped) * raw_trials)

    eigenvalues = np.linalg.eigvalsh(correlation)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    eigen_sum = float(np.sum(eigenvalues))
    eigen_square_sum = float(np.sum(eigenvalues**2))
    effective_rank = (
        eigen_sum**2 / eigen_square_sum if eigen_square_sum > 1e-15 else 1.0
    )
    effective_rank = float(np.clip(effective_rank, 1.0, raw_trials))

    return EffectiveTrialsEstimate(
        raw_trials=raw_trials,
        effective_trials=effective,
        average_pairwise_correlation=average,
        clipped_average_correlation=clipped,
        eigenvalue_effective_rank=effective_rank,
    )


@dataclass(frozen=True)
class SuperiorPredictiveAbilityResult:
    """White Reality Check and Hansen SPA results for one tried family."""

    observed_best_mean: float
    best_trial_index: int
    reality_check_p_value: float
    spa_consistent_p_value: float
    spa_lower_p_value: float
    p_value_upper: float
    n_trials: int
    n_periods: int
    bootstrap_samples: int
    mean_block_size: int
    consistent_relevant_trials: int
    passes: bool


class SuperiorPredictiveAbilityTest:
    """Stationary-bootstrap White Reality Check and Hansen SPA test.

    ``performance_differentials`` has shape ``(trials, periods)`` and uses a
    positive-is-better convention relative to the benchmark.  One shared
    bootstrap index path is applied to all trials per replication, retaining
    both serial dependence and cross-trial dependence.
    """

    def __init__(
        self,
        *,
        bootstrap_samples: int = 1000,
        mean_block_size: int | None = None,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> None:
        if bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be >= 100")
        if mean_block_size is not None and mean_block_size < 1:
            raise ValueError("mean_block_size must be >= 1")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.bootstrap_samples = bootstrap_samples
        self.mean_block_size = mean_block_size
        self.alpha = alpha
        self.seed = seed

    @staticmethod
    def _stationary_long_run_variance(
        differentials: np.ndarray,
        mean_block_size: int,
    ) -> np.ndarray:
        periods = differentials.shape[1]
        restart_probability = 1.0 / mean_block_size
        demeaned = differentials - np.mean(differentials, axis=1, keepdims=True)
        variances = np.sum(demeaned**2, axis=1) / periods
        for lag in range(1, periods):
            weight = (
                (1.0 - lag / periods) * (1.0 - restart_probability) ** lag
                + (lag / periods)
                * (1.0 - restart_probability) ** (periods - lag)
            )
            covariance = np.sum(
                demeaned[:, : periods - lag] * demeaned[:, lag:], axis=1
            ) / periods
            variances += 2.0 * weight * covariance
        return np.asarray(
            np.clip(variances, np.finfo(np.float64).eps, None),
            dtype=np.float64,
        )

    def compute(self, performance_differentials: np.ndarray) -> SuperiorPredictiveAbilityResult:
        differentials = np.asarray(performance_differentials, dtype=np.float64)
        if differentials.ndim != 2 or min(differentials.shape) < 1:
            raise ValueError("performance_differentials must be two-dimensional")
        if differentials.shape[1] < 10:
            raise ValueError("SPA/Reality Check requires at least 10 periods")
        if not np.all(np.isfinite(differentials)):
            raise ValueError("performance_differentials must contain only finite values")

        n_trials, periods = differentials.shape
        block_size = self.mean_block_size or max(1, int(math.sqrt(periods)))
        block_size = min(block_size, periods)
        means = np.mean(differentials, axis=1)
        variances = self._stationary_long_run_variance(differentials, block_size)
        threshold = -np.sqrt(
            (variances / periods) * 2.0 * np.log(np.log(periods))
        )
        consistent_relevant = means >= threshold

        # White's least-favourable Reality Check recenters every trial. Hansen's
        # consistent SPA leaves clearly inferior trials out of the bootstrap
        # maximum; the lower bound excludes every sample-inferior trial.
        upper_centers = means
        consistent_centers = np.where(consistent_relevant, means, 0.0)
        lower_centers = np.where(means >= 0.0, means, 0.0)

        observed = float(np.max(means))
        rng = np.random.RandomState(self.seed)
        upper_exceedances = 0
        consistent_exceedances = 0
        lower_exceedances = 0
        for _ in range(self.bootstrap_samples):
            indices = stationary_bootstrap_indices(periods, block_size, rng)
            resampled_means = np.mean(differentials[:, indices], axis=1)
            upper_exceedances += int(np.max(resampled_means - upper_centers) >= observed)
            consistent_exceedances += int(
                np.max(resampled_means - consistent_centers) >= observed
            )
            lower_exceedances += int(np.max(resampled_means - lower_centers) >= observed)

        denominator = self.bootstrap_samples + 1
        upper_p = (upper_exceedances + 1) / denominator
        consistent_p = (consistent_exceedances + 1) / denominator
        lower_p = (lower_exceedances + 1) / denominator
        return SuperiorPredictiveAbilityResult(
            observed_best_mean=observed,
            best_trial_index=int(np.argmax(means)),
            reality_check_p_value=float(upper_p),
            spa_consistent_p_value=float(consistent_p),
            spa_lower_p_value=float(lower_p),
            p_value_upper=float(upper_p),
            n_trials=n_trials,
            n_periods=periods,
            bootstrap_samples=self.bootstrap_samples,
            mean_block_size=block_size,
            consistent_relevant_trials=int(np.count_nonzero(consistent_relevant)),
            passes=bool(observed > 0.0 and consistent_p < self.alpha),
        )


RealityCheck = SuperiorPredictiveAbilityTest


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def check_significance(
    factor_name: str,
    ic_series: np.ndarray,
    ls_returns: np.ndarray,
    n_total_trials: float,
    config: SignificanceConfig | None = None,
    trial_family: object | None = None,
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
    if trial_family is not None:
        from factorminer.domain.trials import TrialInferenceFamily

        if not isinstance(trial_family, TrialInferenceFamily):
            raise TypeError("trial_family must be a TrialInferenceFamily")
        n_total_trials = trial_family.dsr_trial_count
        details["trial_accounting"] = {
            "source": "canonical_global_trial_ledger",
            "raw_trial_count": trial_family.raw_trial_count,
            "effective_trial_count": trial_family.effective_trial_count,
            "dsr_trial_count": trial_family.dsr_trial_count,
        }

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


# ---------------------------------------------------------------------------
# Sealed multi-evaluator agreement bridge (Agora / research mode)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SealedAgreementSummary:
    """Coarse multi-evaluator agreement summary for one candidate.

    Intentionally free of per-evaluator weight vectors and raw score
    components so it can sit next to classical significance details without
    becoming a second gameable single objective.

    ``agreement_fraction`` is ``n_passed / n_evaluators`` (a pass rate),
    not a chance-corrected inter-rater statistic (e.g. Fleiss' kappa) --
    see ``architecture.sealed_joint_search.SealedFeedback``'s docstring
    for the full reasoning (the source paper's own mechanism is
    report-based disagreement-carrying, not a kappa/concordance
    statistic, and kappa is unstable at this panel's small N anyway).
    """

    candidate_name: str
    n_evaluators: int
    n_passed: int
    agreement_fraction: float
    promoted: bool
    disagreement: bool
    agreement_rule: str
    passed_personas: tuple[str, ...]
    failed_personas: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "candidate_name": self.candidate_name,
            "n_evaluators": self.n_evaluators,
            "n_passed": self.n_passed,
            "agreement_fraction": self.agreement_fraction,
            "promoted": self.promoted,
            "disagreement": self.disagreement,
            "agreement_rule": self.agreement_rule,
            "passed_personas": list(self.passed_personas),
            "failed_personas": list(self.failed_personas),
        }


def summarize_sealed_agreement(
    candidate_name: str,
    *,
    ic_paper_mean: float,
    icir: float = 0.0,
    ic_win_rate: float = 0.5,
    ic_std: float = 0.0,
    intervention_robustness: float = 0.5,
    cpcv_ic_std: float = 0.02,
    max_library_dependence: float = 0.0,
    novelty_score: float | None = None,
    formula: str = "",
    config: SealedAgreementConfig | None = None,
) -> SealedAgreementSummary | None:
    """Run sealed multi-evaluator agreement and return a coarse summary.

    Returns ``None`` when sealed agreement is disabled (the default), so
    existing ``check_significance`` callers are unaffected. This function does
    **not** alter bootstrap/DSR/FDR logic.

    Parameters are plain metric scalars already available from FactorMiner's
    evaluation stack — no need to import evaluator weight vectors here.
    """
    cfg = config or SealedAgreementConfig()
    if not cfg.enabled:
        return None

    # Local import keeps the classical significance path free of the sealed
    # panel dependency unless this opt-in bridge is actually used.
    from factorminer.architecture.sealed_joint_search import (
        AgreementRule,
        CandidateObservation,
        SealedJointSearchConfig,
        SealedJointSearchEngine,
    )

    try:
        rule = AgreementRule(cfg.agreement_rule)
    except ValueError:
        rule = AgreementRule.MAJORITY

    novelty = (
        float(novelty_score)
        if novelty_score is not None
        else max(0.0, 1.0 - float(max_library_dependence))
    )
    obs = CandidateObservation(
        name=candidate_name,
        formula=formula,
        ic_paper_mean=float(ic_paper_mean),
        ic_mean=float(ic_paper_mean),
        ic_std=float(ic_std),
        icir=float(icir),
        ic_win_rate=float(ic_win_rate),
        intervention_robustness=float(intervention_robustness),
        cpcv_ic_std=float(cpcv_ic_std),
        cpcv_ic_mean=float(ic_paper_mean),
        max_library_dependence=float(max_library_dependence),
        novelty_score=novelty,
    )
    engine = SealedJointSearchEngine(
        SealedJointSearchConfig(
            enabled=True,
            agreement_rule=rule,
            min_agree=cfg.min_agree,
            include_llm_judge=cfg.include_llm_judge,
            retain_internal_scores=False,
        )
    )
    decision = engine.evaluate_one(obs)
    fb = decision.feedback
    return SealedAgreementSummary(
        candidate_name=candidate_name,
        n_evaluators=decision.n_evaluators,
        n_passed=decision.n_passed,
        agreement_fraction=decision.agreement_fraction,
        promoted=decision.promoted,
        disagreement=decision.disagreement,
        agreement_rule=decision.agreement_rule,
        passed_personas=fb.passed_personas if fb is not None else (),
        failed_personas=fb.failed_personas if fb is not None else (),
    )
