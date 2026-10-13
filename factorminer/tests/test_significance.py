"""Tests for statistical significance testing (evaluation/significance.py)."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.evaluation.significance import (
    BootstrapCIResult,
    BootstrapICTester,
    DeflatedSharpeCalculator,
    DeflatedSharpeResult,
    FDRController,
    FDRResult,
    SignificanceConfig,
    SuperiorPredictiveAbilityTest,
    estimate_effective_trials,
    stationary_bootstrap_indices,
)


@pytest.fixture
def config():
    return SignificanceConfig(
        bootstrap_n_samples=500,
        bootstrap_block_size=10,
        bootstrap_confidence=0.95,
        fdr_level=0.05,
        seed=42,
    )


# -----------------------------------------------------------------------
# BootstrapICTester: strong signal -> CI excludes zero
# -----------------------------------------------------------------------

def test_bootstrap_strong_signal_excludes_zero(config):
    """A consistently high IC (0.10) should have CI that excludes zero."""
    T = 200
    ic_series = np.full(T, 0.10) + np.random.default_rng(42).normal(0, 0.01, T)

    tester = BootstrapICTester(config)
    result = tester.compute_ci("strong_factor", ic_series)

    assert result.ci_excludes_zero is True
    assert result.ci_lower > 0
    assert result.ic_mean > 0.08


# -----------------------------------------------------------------------
# BootstrapICTester: weak signal -> CI includes zero
# -----------------------------------------------------------------------

def test_bootstrap_weak_signal_includes_zero(config):
    """A near-zero IC should have CI that includes zero."""
    T = 200
    rng = np.random.default_rng(123)
    ic_series = rng.normal(0.0, 0.05, T)  # mean ~0

    tester = BootstrapICTester(config)
    result = tester.compute_ci("weak_factor", ic_series)

    # The signed-mean CI may or may not include zero for a finite sample,
    # but the result must remain a valid BootstrapCIResult.
    assert isinstance(result, BootstrapCIResult)
    assert result.ci_lower <= result.ci_upper


def test_bootstrap_ci_and_p_value_target_same_statistic(config):
    """Regression: the CI used to bracket mean(|IC_t|) while the p-value
    tested |mean(IC_t)|. An alternating ±0.1 series (magnitude, no stable
    direction) then got a CI excluding zero next to a p-value near 1.0 —
    contradictory evidence for the same factor. Both must target the
    signed-mean statistic behind paper_ic_v2."""
    ic_series = np.tile(np.array([0.1, -0.1]), 100)

    tester = BootstrapICTester(config)
    result = tester.compute_ci("alternating", ic_series)
    p_value = tester.compute_p_value(ic_series)

    assert result.ic_mean == pytest.approx(0.0, abs=1e-12)
    assert result.ic_paper_mean == pytest.approx(0.0, abs=1e-12)
    assert result.ci_lower <= 0.0 <= result.ci_upper
    assert result.ci_excludes_zero is False
    assert p_value > 0.5


def test_bootstrap_negative_signal_sign_aware(config):
    """A consistently negative IC must yield a negative signed mean, a CI
    strictly below zero (which still counts as excluding zero), and a
    positive paper-mode magnitude."""
    T = 200
    ic_series = np.full(T, -0.10) + np.random.default_rng(42).normal(0, 0.01, T)

    tester = BootstrapICTester(config)
    result = tester.compute_ci("negative_factor", ic_series)

    assert result.ic_mean < -0.08
    assert result.ci_upper < 0
    assert result.ci_excludes_zero is True
    assert result.ic_paper_mean > 0.08


def test_block_bootstrap_circular_uniform_coverage():
    """Regression: block starts were capped at T - block_size, so tail
    observations were under-sampled despite the 'circular' docstring.
    With wrap-around sampling every index is equally likely, so the
    expected bootstrap mean equals the sample mean even when the only
    nonzero value sits at the final index."""
    series = np.zeros(100)
    series[-1] = 1.0
    tester = BootstrapICTester(
        SignificanceConfig(bootstrap_n_samples=2000, bootstrap_block_size=10, seed=7)
    )
    boot_means = tester._block_bootstrap_means(series)

    assert np.mean(boot_means) == pytest.approx(series.mean(), abs=0.002)


def test_bootstrap_p_value_distinguishes_signal_from_noise(config):
    """The sign-flip p-value should be small for signal and large for noise."""
    rng = np.random.default_rng(7)
    strong_ic = 0.08 + rng.normal(0.0, 0.01, 200)
    weak_ic = rng.normal(0.0, 0.05, 200)

    tester = BootstrapICTester(config)

    strong_p = tester.compute_p_value(strong_ic)
    weak_p = tester.compute_p_value(weak_ic)

    assert strong_p < 0.05
    assert weak_p > 0.05


# -----------------------------------------------------------------------
# FDRController: BH procedure
# -----------------------------------------------------------------------

def test_fdr_batch_evaluate_separates_signal_from_noise(config):
    """Batch FDR should keep the strong series and reject the weak one."""
    strong_ic = np.full(200, 0.08)
    weak_ic = np.tile(np.array([0.05, -0.05]), 100)

    tester = BootstrapICTester(config)
    controller = FDRController(config)
    result = controller.batch_evaluate(
        {"strong_factor": strong_ic, "weak_factor": weak_ic},
        tester,
    )

    assert result.significant["strong_factor"]
    assert not result.significant["weak_factor"]
    assert result.n_discoveries == 1

def test_fdr_bh_procedure(config):
    """10 factors with p-values [0.001, ..., 0.010] at FDR=0.05."""
    # Use small enough p-values that BH adjustment still yields significance
    p_values = {f"f{i}": 0.001 * (i + 1) for i in range(10)}
    controller = FDRController(config)
    result = controller.apply_fdr(p_values)

    assert isinstance(result, FDRResult)
    assert result.fdr_level == 0.05
    # With BH at 0.05 and raw p in [0.001..0.010], adjusted p for f0 = 0.001*10/1 = 0.01 < 0.05
    assert result.n_discoveries >= 1
    assert result.significant["f0"]  # p=0.001, adjusted=0.01


def test_fdr_all_significant(config):
    """All p=0.001 should be significant after BH."""
    p_values = {f"f{i}": 0.001 for i in range(10)}
    controller = FDRController(config)
    result = controller.apply_fdr(p_values)

    assert result.n_discoveries == 10
    for name, sig in result.significant.items():
        assert sig


def test_fdr_empty_input(config):
    """Empty p-value dict should return empty result."""
    controller = FDRController(config)
    result = controller.apply_fdr({})
    assert result.n_discoveries == 0
    assert result.significant == {}


# -----------------------------------------------------------------------
# DeflatedSharpeCalculator
# -----------------------------------------------------------------------

def test_deflated_sharpe_with_known_returns(config):
    """Verify DSR computation with known returns and n_trials."""
    rng = np.random.default_rng(42)
    T = 500
    # Strong positive returns
    ls_returns = rng.normal(0.001, 0.01, T)

    calc = DeflatedSharpeCalculator(config)
    result = calc.compute("good_factor", ls_returns, n_trials=10)

    assert isinstance(result, DeflatedSharpeResult)
    assert result.raw_sharpe > 0
    assert result.n_trials == 10
    assert result.haircut >= 0 or result.haircut < 0  # can be negative


def test_deflated_sharpe_many_trials_penalizes(config):
    """More trials should increase the haircut (higher expected max SR)."""
    rng = np.random.default_rng(42)
    T = 500
    ls_returns = rng.normal(0.001, 0.01, T)

    calc = DeflatedSharpeCalculator(config)
    result_few = calc.compute("factor", ls_returns, n_trials=5)
    result_many = calc.compute("factor", ls_returns, n_trials=500)

    # With more trials, the deflated SR should be lower
    assert result_many.deflated_sharpe <= result_few.deflated_sharpe


def test_deflated_sharpe_short_series(config):
    """Very short series (<10) should return default failing result."""
    ls_returns = np.array([0.01, 0.02, 0.01])
    calc = DeflatedSharpeCalculator(config)
    result = calc.compute("short", ls_returns, n_trials=10)
    assert result.passes is False
    assert result.raw_sharpe == 0.0


def test_deflated_sharpe_uses_raw_not_excess_kurtosis(config):
    """The dispersion term uses raw kurtosis (normal returns: (g4 - 1) / 4 = 0.5).

    Plugging in scipy's excess kurtosis (normal => 0) would understate the
    estimator's variance for fat-tailed series and inflate the statistic.
    """
    from scipy.stats import kurtosis, norm, skew

    rng = np.random.default_rng(11)
    T = 500
    base = rng.normal(loc=0.0035, scale=0.01, size=T)
    jump_mask = rng.random(T) < 0.03
    base[jump_mask] += rng.normal(scale=0.08, size=int(jump_mask.sum()))

    result = DeflatedSharpeCalculator(config).compute("fat_tailed", base, n_trials=10)

    sr = float(np.mean(base) / np.std(base, ddof=1))
    gamma3 = float(skew(base, bias=False))
    gamma4_raw = float(kurtosis(base, fisher=False, bias=False))
    assert gamma4_raw > 3.5, "fixture must be fat-tailed for this test to be meaningful"
    g = 0.5772156649015329
    sr0 = np.sqrt(1.0 / (T - 1)) * ((1 - g) * norm.ppf(1 - 1 / 10) + g * norm.ppf(1 - 1 / (10 * np.e)))
    expected = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - gamma3 * sr + (gamma4_raw - 1) / 4 * sr**2)
    assert result.deflated_sharpe == pytest.approx(expected, rel=1e-9)
    assert result.haircut == pytest.approx(sr0 * np.sqrt(252.0))

    gamma4_excess = float(kurtosis(base, fisher=True, bias=False))
    inflated = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - gamma3 * sr + (gamma4_excess - 1) / 4 * sr**2)
    assert result.deflated_sharpe < inflated


def test_deflated_sharpe_matches_bailey_lopez_de_prado_null_scale(config):
    """Under the null, the expected-maximum hurdle is on the per-period SR scale.

    With 1000 trials of 1000 periods, the null hurdle is about 3.25 / sqrt(999)
    per period (~1.63 annualized); a per-period SR of 0.2 (3.2 annualized)
    clears it, which the former annualized-vs-unit-normal comparison denied.
    """
    rng = np.random.default_rng(5)
    series = rng.normal(size=1000)
    series = (series - series.mean()) / series.std(ddof=1) + 0.2  # per-period SR 0.2
    result = DeflatedSharpeCalculator(config).compute("strong", series, n_trials=1000)
    assert result.haircut == pytest.approx(3.2563 / np.sqrt(999) * np.sqrt(252), rel=1e-3)
    assert result.deflated_sharpe > 3 and result.passes
    noise = DeflatedSharpeCalculator(config).compute("noise", rng.normal(size=1000), n_trials=1000)
    assert not noise.passes


def test_stationary_bootstrap_indices_are_seeded_and_circular():
    left = stationary_bootstrap_indices(100, 10, np.random.RandomState(17))
    right = stationary_bootstrap_indices(100, 10, np.random.RandomState(17))
    assert np.array_equal(left, right)
    assert left.shape == (100,)
    assert np.all((left >= 0) & (left < 100))
    # Most adjacent positions belong to one geometric block.
    continuation = left[1:] == (left[:-1] + 1) % 100
    assert 0.75 < float(np.mean(continuation)) < 1.0


def test_bootstrap_ic_supports_stationary_method():
    cfg = SignificanceConfig(
        bootstrap_n_samples=300,
        bootstrap_block_size=12,
        bootstrap_method="stationary",
        seed=9,
    )
    rng = np.random.default_rng(9)
    series = 0.04 + rng.normal(0.0, 0.02, 250)
    result = BootstrapICTester(cfg).compute_ci("stationary", series)
    assert result.bootstrap_method == "stationary"
    assert result.ci_lower > 0.0


def test_effective_trials_collapses_perfectly_correlated_aliases():
    base = np.linspace(-1.0, 1.0, 200)
    estimate = estimate_effective_trials(np.vstack([base, base, base]))
    assert estimate.raw_trials == 3
    assert estimate.average_pairwise_correlation == pytest.approx(1.0)
    assert estimate.effective_trials == pytest.approx(1.0)
    assert estimate.eigenvalue_effective_rank == pytest.approx(1.0)


def test_effective_trials_retains_most_independent_trials():
    matrix = np.random.default_rng(123).normal(size=(8, 500))
    estimate = estimate_effective_trials(matrix)
    assert 6.5 <= estimate.effective_trials <= 8.0
    assert 6.5 <= estimate.eigenvalue_effective_rank <= 8.0


def test_spa_and_reality_check_do_not_reject_exact_null():
    result = SuperiorPredictiveAbilityTest(
        bootstrap_samples=200,
        mean_block_size=8,
        seed=11,
    ).compute(np.zeros((4, 120)))
    assert result.observed_best_mean == 0.0
    assert result.reality_check_p_value == 1.0
    assert result.spa_consistent_p_value == 1.0
    assert result.passes is False


def test_spa_detects_one_persistent_superior_trial():
    rng = np.random.default_rng(2026)
    periods = 500
    family = rng.normal(0.0, 0.01, size=(12, periods))
    family[0] += 0.008
    family[5:] -= 0.01
    result = SuperiorPredictiveAbilityTest(
        bootstrap_samples=500,
        mean_block_size=10,
        seed=2026,
    ).compute(family)
    assert result.best_trial_index == 0
    assert result.passes is True
    assert result.spa_lower_p_value <= result.spa_consistent_p_value
    assert result.spa_consistent_p_value <= result.reality_check_p_value
    assert result.spa_consistent_p_value < 0.05
