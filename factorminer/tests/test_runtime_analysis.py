"""Unit tests for strict runtime recomputation helpers."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from factorminer.core.factor_library import Factor
from factorminer.core.parser import try_parse
from factorminer.evaluation.metrics import compute_factor_stats
from factorminer.evaluation.runtime import (
    EvaluationDataset,
    SignalComputationError,
    build_runtime_dataset_from_arrays,
    compute_tree_signals,
    evaluate_factors,
    generate_synthetic_signals,
)


def _build_dataset(data_dict: dict[str, np.ndarray]) -> EvaluationDataset:
    timestamps = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(i, "D") for i in range(50)]
    )
    returns = data_dict["$returns"]
    feature_order = [
        "$open",
        "$high",
        "$low",
        "$close",
        "$volume",
        "$amt",
        "$vwap",
        "$returns",
    ]
    return build_runtime_dataset_from_arrays(
        data_dict,
        returns,
        feature_order=feature_order,
        timestamps=timestamps,
        asset_ids=np.array([f"A{i:02d}" for i in range(returns.shape[0])]),
        split_indices={
            "train": np.arange(25),
            "test": np.arange(25, 50),
            "full": np.arange(50),
        },
    )


def test_array_dataset_builder_rejects_misaligned_panels() -> None:
    with pytest.raises(ValueError, match="expected"):
        build_runtime_dataset_from_arrays(
            {"$close": np.ones((2, 4)), "$volume": np.ones((2, 3))},
            np.ones((2, 4)),
        )


def test_evaluate_factors_matches_direct_metric_computation(small_data):
    """Shared runtime evaluation should match direct metric recomputation."""
    dataset = _build_dataset(small_data)
    factor = Factor(
        id=1,
        name="close_neg",
        formula="Neg($close)",
        category="test",
        ic_mean=99.0,
        icir=88.0,
        ic_win_rate=0.99,
        max_correlation=0.0,
        batch_number=1,
    )

    artifact = evaluate_factors([factor], dataset, signal_failure_policy="reject")[0]
    tree = try_parse(factor.formula)
    signals = tree.evaluate(dataset.data_dict)
    expected_train = compute_factor_stats(signals[:, :25], dataset.returns[:, :25])
    expected_test = compute_factor_stats(signals[:, 25:], dataset.returns[:, 25:])

    assert artifact.succeeded
    np.testing.assert_allclose(
        artifact.split_stats["train"]["ic_series"],
        expected_train["ic_series"],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        artifact.split_stats["test"]["ic_series"],
        expected_test["ic_series"],
        equal_nan=True,
    )
    assert artifact.split_stats["train"]["ic_mean"] == pytest.approx(
        expected_train["ic_mean"]
    )
    assert artifact.split_stats["test"]["long_short"] == pytest.approx(
        expected_test["long_short"]
    )
    assert artifact.split_stats["train"]["turnover"] == pytest.approx(
        expected_train["turnover"]
    )


def test_compute_tree_signals_obeys_failure_policy():
    """Signal failures should reject, synthesize, or raise explicitly."""
    tree = try_parse("Neg($close)")
    returns_shape = (3, 7)

    with pytest.raises(SignalComputationError):
        compute_tree_signals(
            tree,
            data_dict={},
            returns_shape=returns_shape,
            signal_failure_policy="reject",
        )

    synthetic = compute_tree_signals(
        tree,
        data_dict={},
        returns_shape=returns_shape,
        signal_failure_policy="synthetic",
    )
    assert synthetic.shape == returns_shape
    assert np.isfinite(synthetic).sum() > 0

    with pytest.raises(Exception):
        compute_tree_signals(
            tree,
            data_dict={},
            returns_shape=returns_shape,
            signal_failure_policy="raise",
        )


def test_synthetic_signals_are_stable_across_processes():
    """Synthetic fallback must not depend on Python's randomized hash seed."""
    script = (
        "import json;"
        "from factorminer.evaluation.runtime import generate_synthetic_signals;"
        "arr=generate_synthetic_signals('Neg($close)', (3, 5));"
        "print(json.dumps(arr.tolist()))"
    )
    first = subprocess.check_output([sys.executable, "-c", script], text=True)
    second = subprocess.check_output([sys.executable, "-c", script], text=True)

    assert json.loads(first) == json.loads(second)

    same = generate_synthetic_signals("Neg($close)", (3, 5))
    different = generate_synthetic_signals("CsRank($close)", (3, 5))
    assert not np.array_equal(np.nan_to_num(same), np.nan_to_num(different))


def test_evaluate_factors_records_strict_recomputation_failure(small_data):
    """Strict evaluation should record failures instead of hiding them."""
    dataset = _build_dataset(dict(small_data, **{"$close": np.full((10, 50), np.nan)}))
    factor = Factor(
        id=7,
        name="broken_close",
        formula="Neg($close)",
        category="test",
        ic_mean=0.0,
        icir=0.0,
        ic_win_rate=0.0,
        max_correlation=0.0,
        batch_number=1,
    )

    artifact = evaluate_factors([factor], dataset, signal_failure_policy="reject")[0]

    assert not artifact.succeeded
    assert "Signal computation produced only NaN values" in artifact.error
