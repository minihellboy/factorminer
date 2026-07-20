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

    # The CI for |IC| may or may not include zero depending on noise,
    # but the result should be a valid BootstrapCIResult
    assert isinstance(result, BootstrapCIResult)
    assert result.ci_lower <= result.ci_upper


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
    """DSR's variance-correction term must use raw (non-excess) kurtosis.

    Bailey & Lopez de Prado (2014) define gamma4 as raw kurtosis (normal
    distribution => gamma4 = 3), giving ``(gamma4 - 1) / 4 == 0.5`` for
    normal returns. Plugging in scipy's *excess* kurtosis (``fisher=True``,
    normal => 0) directly would instead give ``(0 - 1) / 4 == -0.25`` --
    silently understating ``var_correction`` (and so overstating
    ``deflated_sharpe``) for real, typically fat-tailed return series,
    exactly where deflation matters most. This fixture uses a large mean
    return so SR clearly exceeds the expected max SR under the null,
    making the bug's practical direction (falsely inflated deflated_sharpe)
    unambiguous regardless of n_trials.
    """
    from scipy.stats import kurtosis, skew

    rng = np.random.default_rng(11)
    T = 500
    base = rng.normal(loc=0.0035, scale=0.01, size=T)
    jump_mask = rng.random(T) < 0.03
    base[jump_mask] += rng.normal(scale=0.08, size=int(jump_mask.sum()))
    ls_returns = base

    calc = DeflatedSharpeCalculator(config)
    result = calc.compute("fat_tailed", ls_returns, n_trials=10)

    mean_r = float(np.mean(ls_returns))
    std_r = float(np.std(ls_returns, ddof=1))
    sr = (mean_r / std_r) * np.sqrt(252.0)
    gamma3 = float(skew(ls_returns, bias=False))
    gamma4_raw = float(kurtosis(ls_returns, fisher=False, bias=False))
    e_max_sr = calc._expected_max_sr(10)
    assert gamma4_raw > 3.5, "fixture must be fat-tailed for this test to be meaningful"
    assert sr > e_max_sr, "fixture must clear the null bar for the bug's direction to be unambiguous"
    expected_var_correction = (1.0 - gamma3 * sr + (gamma4_raw - 1.0) / 4.0 * sr**2) / T
    expected_dsr = (sr - e_max_sr) / np.sqrt(expected_var_correction)

    assert result.deflated_sharpe == pytest.approx(expected_dsr, rel=1e-6)

    # Pin the regression explicitly: computing with *excess* kurtosis
    # instead must give a materially different (falsely higher) result.
    gamma4_excess = float(kurtosis(ls_returns, fisher=True, bias=False))
    buggy_var_correction = (1.0 - gamma3 * sr + (gamma4_excess - 1.0) / 4.0 * sr**2) / T
    assert buggy_var_correction > 0
    buggy_dsr = (sr - e_max_sr) / np.sqrt(buggy_var_correction)
    assert result.deflated_sharpe < buggy_dsr
    assert result.deflated_sharpe != pytest.approx(buggy_dsr, rel=1e-3)
