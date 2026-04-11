"""Shared pytest fixtures for FactorMiner test suite."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.memory.experience_memory import ExperienceMemoryManager

# ---------------------------------------------------------------------------
# Mock data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def small_data(rng):
    """Small (M=10, T=50) synthetic dataset dict mapping feature names to arrays."""
    M, T = 10, 50
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, (M, T)), axis=1)
    open_ = close + rng.normal(0, 0.1, (M, T))
    high = np.maximum(close, open_) + np.abs(rng.normal(0, 0.2, (M, T)))
    low = np.minimum(close, open_) - np.abs(rng.normal(0, 0.2, (M, T)))
    low = np.maximum(low, 1.0)
    volume = np.abs(rng.normal(1e6, 1e5, (M, T)))
    vwap = (high + low + close) / 3
    amt = volume * vwap
    returns = np.zeros((M, T))
    returns[:, 1:] = np.diff(close, axis=1) / close[:, :-1]

    return {
        "$open": open_,
        "$high": high,
        "$low": low,
        "$close": close,
        "$volume": volume,
        "$amt": amt,
        "$vwap": vwap,
        "$returns": returns,
    }


@pytest.fixture
def medium_data(rng):
    """Medium (M=20, T=100) synthetic dataset for evaluation tests."""
    M, T = 20, 100
    close = 50.0 + np.cumsum(rng.normal(0, 0.3, (M, T)), axis=1)
    open_ = close + rng.normal(0, 0.05, (M, T))
    high = np.maximum(close, open_) + np.abs(rng.normal(0, 0.1, (M, T)))
    low = np.minimum(close, open_) - np.abs(rng.normal(0, 0.1, (M, T)))
    low = np.maximum(low, 1.0)
    volume = np.abs(rng.normal(1e6, 1e5, (M, T)))
    vwap = (high + low + close) / 3
    amt = volume * vwap
    returns = np.zeros((M, T))
    returns[:, 1:] = np.diff(close, axis=1) / close[:, :-1]

    return {
        "$open": open_,
        "$high": high,
        "$low": low,
        "$close": close,
        "$volume": volume,
        "$amt": amt,
        "$vwap": vwap,
        "$returns": returns,
    }


# ---------------------------------------------------------------------------
# Library fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_library(rng):
    """Small FactorLibrary pre-loaded with 3 known factors."""
    lib = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)

    M, T = 20, 60
    for i in range(3):
        signals = rng.normal(0, 1, (M, T))
        factor = Factor(
            id=0,
            name=f"test_factor_{i}",
            formula="Neg($close)" if i == 0 else f"CsRank(Mean($close, {10 + i * 5}))",
            category="test",
            ic_mean=0.05 + i * 0.01,
            icir=0.8 + i * 0.1,
            ic_win_rate=0.55 + i * 0.05,
            max_correlation=0.1 * i,
            batch_number=1,
            signals=signals,
        )
        lib.admit_factor(factor)

    return lib


# ---------------------------------------------------------------------------
# Memory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_memory():
    """ExperienceMemoryManager with default patterns initialized."""
    return ExperienceMemoryManager(
        max_success_patterns=20,
        max_failure_patterns=30,
        max_insights=15,
    )


@pytest.fixture
def sample_trajectory():
    """Sample batch trajectory for memory update tests."""
    return [
        {
            "formula": "CsRank(Corr($close, $volume, 20))",
            "factor_id": "f001",
            "ic": 0.08,
            "icir": 1.2,
            "max_correlation": 0.15,
            "correlated_with": "",
            "admitted": True,
            "rejection_reason": "",
        },
        {
            "formula": "Neg(Div(Sub($close, $vwap), $vwap))",
            "factor_id": "f002",
            "ic": 0.06,
            "icir": 0.9,
            "max_correlation": 0.65,
            "correlated_with": "existing_factor_3",
            "admitted": False,
            "rejection_reason": "Max correlation 0.65 >= threshold 0.5",
        },
        {
            "formula": "IfElse(Skew($close, 20), CsRank($returns), Neg($returns))",
            "factor_id": "f003",
            "ic": 0.10,
            "icir": 1.5,
            "max_correlation": 0.20,
            "correlated_with": "",
            "admitted": True,
            "rejection_reason": "",
        },
        {
            "formula": "CsZScore(Std($returns, 10))",
            "factor_id": "f004",
            "ic": 0.03,
            "icir": 0.4,
            "max_correlation": 0.70,
            "correlated_with": "existing_factor_1",
            "admitted": False,
            "rejection_reason": "IC 0.03 below threshold 0.04",
        },
    ]
