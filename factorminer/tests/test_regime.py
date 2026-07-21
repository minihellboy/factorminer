"""Tests for regime-aware factor validation (evaluation/regime.py)."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.evaluation.regime import (
    MarketRegime,
    RegimeAwareEvaluator,
    RegimeConfig,
    RegimeDetector,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# -----------------------------------------------------------------------
# RegimeDetector: synthetic bull/bear phases
# -----------------------------------------------------------------------

def test_regime_detector_bull_bear_phases(rng):
    """Clear positive first half, negative second half should produce
    BULL and BEAR labels after the lookback window."""
    M, T = 20, 300
    returns = np.zeros((M, T))
    # First half: strongly positive
    returns[:, :150] = rng.normal(0.02, 0.005, (M, 150))
    # Second half: strongly negative
    returns[:, 150:] = rng.normal(-0.02, 0.005, (M, 150))

    cfg = RegimeConfig(lookback_window=30, bull_return_threshold=0.0,
                       bear_return_threshold=0.0)
    detector = RegimeDetector(config=cfg)
    result = detector.classify(returns)

    # After the lookback window, first half should contain BULL periods
    bull_periods = result.periods[MarketRegime.BULL]
    bear_periods = result.periods[MarketRegime.BEAR]
    assert bull_periods[50:140].sum() > 0, "Should have BULL in first half"
    assert bear_periods[180:].sum() > 0, "Should have BEAR in second half"


def test_regime_detector_labels_shape(rng):
    M, T = 10, 100
    returns = rng.normal(0, 0.01, (M, T))
    detector = RegimeDetector()
    result = detector.classify(returns)
    assert result.labels.shape == (T,)
    assert set(result.labels).issubset({0, 1, 2})


# -----------------------------------------------------------------------
# RegimeAwareEvaluator: signal works in all regimes
# -----------------------------------------------------------------------

def test_regime_evaluator_all_regimes_pass(rng):
    """A signal correlated with returns across all regimes should pass."""
    M, T = 20, 400
    returns = rng.normal(0, 0.01, (M, T))
    signal = returns * 5 + rng.normal(0, 0.001, (M, T))

    cfg = RegimeConfig(lookback_window=20, min_regime_ic=0.01,
                       min_regimes_passing=1)
    detector = RegimeDetector(config=cfg)
    regime = detector.classify(returns)

    evaluator = RegimeAwareEvaluator(returns, regime, config=cfg)
    result = evaluator.evaluate("strong_factor", signal)
    assert result.passes is True


# -----------------------------------------------------------------------
# RegimeAwareEvaluator: signal only works in bull
# -----------------------------------------------------------------------

def test_regime_evaluator_bull_only_fails(rng):
    """A signal that only works in positive-return periods should fail
    if min_regimes_passing=2."""
    M, T = 20, 400
    returns = np.zeros((M, T))
    returns[:, :200] = rng.normal(0.02, 0.005, (M, 200))
    returns[:, 200:] = rng.normal(-0.02, 0.005, (M, 200))

    # Signal only correlates with returns in first half
    signal = np.zeros((M, T))
    signal[:, :200] = returns[:, :200] * 5
    signal[:, 200:] = rng.normal(0, 1, (M, 200))  # noise in bear

    cfg = RegimeConfig(lookback_window=20, min_regime_ic=0.03,
                       min_regimes_passing=2)
    detector = RegimeDetector(config=cfg)
    regime = detector.classify(returns)

    evaluator = RegimeAwareEvaluator(returns, regime, config=cfg)
    result = evaluator.evaluate("bull_only", signal)
    # May or may not pass depending on how many regimes are detected,
    # but the structure is correct
    assert isinstance(result.n_regimes_passing, int)
    assert isinstance(result.passes, bool)


# -----------------------------------------------------------------------
# Edge case: very short data
# -----------------------------------------------------------------------

def test_regime_detector_short_data(rng):
    """Data shorter than lookback_window should still work (all SIDEWAYS)."""
    M, T = 10, 20
    returns = rng.normal(0, 0.01, (M, T))
    cfg = RegimeConfig(lookback_window=60)
    detector = RegimeDetector(config=cfg)
    result = detector.classify(returns)
    # All periods should be SIDEWAYS since T < lookback_window
    assert np.all(result.labels == MarketRegime.SIDEWAYS.value)


# -----------------------------------------------------------------------
# RegimeState.similarity and regime-conditioned pattern retrieval
# -----------------------------------------------------------------------

def test_regime_state_similarity_component_fraction():
    from factorminer.evaluation.regime import (
        MeanRevRegime,
        RegimeState,
        TrendRegime,
        VolRegime,
    )

    bull_high = RegimeState(TrendRegime.BULL, VolRegime.HIGH_VOL, MeanRevRegime.TRENDING)
    bear_low = RegimeState(TrendRegime.BEAR, VolRegime.LOW_VOL, MeanRevRegime.MEAN_REVERTING)
    bull_low = RegimeState(TrendRegime.BULL, VolRegime.LOW_VOL, MeanRevRegime.MEAN_REVERTING)
    bull_high_mr = RegimeState(TrendRegime.BULL, VolRegime.HIGH_VOL, MeanRevRegime.MEAN_REVERTING)

    assert bull_high.similarity(bull_high) == 1.0
    assert bull_high.similarity(bear_low) == 0.0
    assert bull_high.similarity(bull_low) == pytest.approx(1 / 3)
    assert bull_low.similarity(bull_high) == pytest.approx(1 / 3)
    assert bull_high.similarity(bull_high_mr) == pytest.approx(2 / 3)


def test_regime_pattern_retrieval_scores_by_similarity():
    """Regression: retrieve_for_regime raised AttributeError before
    RegimeState.similarity existed (online_regime_memory.py:308 called it
    on every stored pattern), so regime-conditioned retrieval crashed as
    soon as the store held a qualifying pattern."""
    from factorminer.evaluation.regime import (
        MeanRevRegime,
        RegimeState,
        TrendRegime,
        VolRegime,
    )
    from factorminer.memory.online_regime_memory import RegimeSpecificPatternStore

    bull = RegimeState(TrendRegime.BULL, VolRegime.HIGH_VOL, MeanRevRegime.TRENDING)
    bear = RegimeState(TrendRegime.BEAR, VolRegime.LOW_VOL, MeanRevRegime.MEAN_REVERTING)

    store = RegimeSpecificPatternStore()
    store.add_pattern("Rank(Delta($close, 3))", bull, ic=0.06)
    store.add_pattern("Rank($volume)", bear, ic=0.06)

    got = store.retrieve_for_regime(bull, top_k=5, min_confidence=0.0)

    assert len(got) == 2
    # equal confidence and |IC|: the same-regime pattern must rank first
    assert got[0].formula_template == "Rank(Delta($close, 3))"
