"""Tests for Combinatorial Purged CV + Probability of Backtest Overfitting
(evaluation/cross_validation.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from factorminer.evaluation.cross_validation import (
    CombinatorialPurgedCV,
    CPCVSplit,
    CrossValidationConfig,
    PBOResult,
    ProbabilityOfBacktestOverfitting,
)

# ---------------------------------------------------------------------------
# CombinatorialPurgedCV
# ---------------------------------------------------------------------------


def test_split_count_matches_combinatorial_choice():
    """C(n_groups, n_test_groups) splits are produced, each with a unique path_id."""
    cfg = CrossValidationConfig(n_groups=6, n_test_groups=2)
    cv = CombinatorialPurgedCV(cfg)
    splits = cv.split(n_samples=300, label_horizon=1)

    assert len(splits) == math.comb(6, 2)
    assert cv.n_paths == math.comb(6, 2)
    assert [split.path_id for split in splits] == list(range(len(splits)))
    assert all(isinstance(split, CPCVSplit) for split in splits)


def test_split_purges_every_label_window_overlap():
    """No surviving train index has a label window [i, i+label_horizon) that
    overlaps any test-group index -- the core CPCV leakage guarantee."""
    cfg = CrossValidationConfig(n_groups=6, n_test_groups=2, embargo_fraction=0.02)
    cv = CombinatorialPurgedCV(cfg)
    n_samples = 300
    label_horizon = 5
    splits = cv.split(n_samples, label_horizon=label_horizon)

    for split in splits:
        test_set = set(split.test_indices.tolist())
        for i in split.train_indices:
            window = set(range(int(i), min(int(i) + label_horizon, n_samples)))
            assert not (window & test_set), (
                f"train index {i} has a label window overlapping the test set"
            )
        # train and test indices are always disjoint
        assert not (set(split.train_indices.tolist()) & test_set)


def test_split_embargoes_samples_after_test_group():
    """Samples immediately after a test group are dropped from training even
    when their own label window does not reach into the test set."""
    cfg = CrossValidationConfig(n_groups=5, n_test_groups=1, embargo_fraction=0.1)
    cv = CombinatorialPurgedCV(cfg)
    n_samples = 200
    splits = cv.split(n_samples, label_horizon=0)  # no purge effect, isolate embargo

    embargo_len = int(round(cfg.embargo_fraction * n_samples))
    assert embargo_len > 0

    # First split's test group is group 0 -- the first contiguous block.
    split0 = splits[0]
    test_end = int(split0.test_indices.max())
    embargo_zone = set(range(test_end + 1, min(test_end + 1 + embargo_len, n_samples)))
    train_set = set(split0.train_indices.tolist())
    assert not (embargo_zone & train_set)


def test_split_train_test_cover_disjoint_ranges():
    """train_indices and test_indices never intersect for any split, and every
    test group appears in exactly C(n_groups-1, n_test_groups-1) splits."""
    cfg = CrossValidationConfig(n_groups=5, n_test_groups=2, embargo_fraction=0.0)
    cv = CombinatorialPurgedCV(cfg)
    splits = cv.split(n_samples=100, label_horizon=1)

    for split in splits:
        assert set(split.train_indices.tolist()).isdisjoint(split.test_indices.tolist())
        assert split.test_indices.size > 0
        assert np.array_equal(split.test_indices, np.sort(split.test_indices))
        assert np.array_equal(split.train_indices, np.sort(split.train_indices))


def test_split_rejects_invalid_config():
    """n_samples smaller than n_groups, or n_test_groups out of range, raises."""
    cv = CombinatorialPurgedCV(CrossValidationConfig(n_groups=10, n_test_groups=2))
    with pytest.raises(ValueError):
        cv.split(n_samples=5)

    with pytest.raises(ValueError):
        CombinatorialPurgedCV(CrossValidationConfig(n_groups=3, n_test_groups=3)).split(100)


# ---------------------------------------------------------------------------
# ProbabilityOfBacktestOverfitting
# ---------------------------------------------------------------------------


def test_pbo_high_when_in_sample_best_is_pure_noise():
    """Many independent noisy trials with no genuine OOS-persistent skill:
    picking the in-sample winner is a selection-biased procedure, so PBO
    should register comfortably above the 0.5 chance baseline."""
    rng = np.random.default_rng(42)
    n_trials, n_paths = 40, 10
    matrix = rng.normal(loc=0.0, scale=1.0, size=(n_trials, n_paths))

    calc = ProbabilityOfBacktestOverfitting(CrossValidationConfig(min_paths_for_pbo=8), seed=42)
    result = calc.compute(matrix)

    assert isinstance(result, PBOResult)
    assert result.pbo > 0.5
    assert result.passes is False
    assert result.n_combinations == math.comb(n_paths, n_paths // 2)
    assert len(result.logit_values) == result.n_combinations


def test_pbo_low_when_one_trial_dominates_is_and_oos():
    """A trial with a large, consistent performance edge on every path is
    both the in-sample winner and the out-of-sample winner every time, so
    PBO should be low (well under the 0.5 chance baseline)."""
    rng = np.random.default_rng(42)
    n_trials, n_paths = 40, 10
    matrix = rng.normal(loc=0.0, scale=0.3, size=(n_trials, n_paths))
    matrix[0, :] += 3.0  # trial 0 dominates every path

    calc = ProbabilityOfBacktestOverfitting(CrossValidationConfig(min_paths_for_pbo=8), seed=42)
    result = calc.compute(matrix)

    assert result.pbo < 0.5
    assert result.passes is True


def test_pbo_requires_minimum_paths():
    """Fewer paths than config.min_paths_for_pbo raises rather than silently
    producing an unreliable estimate."""
    calc = ProbabilityOfBacktestOverfitting(CrossValidationConfig(min_paths_for_pbo=8))
    with pytest.raises(ValueError):
        calc.compute(np.zeros((5, 4)))


def test_pbo_requires_multiple_trials():
    calc = ProbabilityOfBacktestOverfitting(CrossValidationConfig(min_paths_for_pbo=4))
    with pytest.raises(ValueError):
        calc.compute(np.zeros((1, 8)))


def test_pbo_samples_combinations_when_search_space_is_large():
    """When C(n_paths, n_paths // 2) exceeds max_combinations, a deterministic
    sample of exactly max_combinations distinct half-splits is used instead
    of blowing up combinatorially."""
    rng = np.random.default_rng(7)
    n_trials, n_paths = 12, 20  # C(20, 10) = 184,756
    matrix = rng.normal(size=(n_trials, n_paths))

    calc = ProbabilityOfBacktestOverfitting(
        CrossValidationConfig(min_paths_for_pbo=8), max_combinations=500, seed=7
    )
    result = calc.compute(matrix)

    assert result.n_combinations == 500
    assert 0.0 <= result.pbo <= 1.0

    # Determinism: same seed/matrix -> identical result.
    result_again = calc.compute(matrix)
    assert result_again.pbo == result.pbo
    assert result_again.logit_values == result.logit_values
