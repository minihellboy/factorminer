"""Unit tests for strict runtime recomputation helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factorminer.core.factor_library import Factor
from factorminer.core.parser import try_parse
from factorminer.evaluation.metrics import compute_factor_stats
from factorminer.evaluation.runtime import (
    DatasetSplit,
    EvaluationDataset,
    SignalComputationError,
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
    data_tensor = np.stack([data_dict[name] for name in feature_order], axis=-1)

    splits = {
        "train": DatasetSplit(
            name="train",
            indices=np.arange(25),
            timestamps=timestamps[:25],
            returns=returns[:, :25],
        ),
        "test": DatasetSplit(
            name="test",
            indices=np.arange(25, 50),
            timestamps=timestamps[25:],
            returns=returns[:, 25:],
        ),
        "full": DatasetSplit(
            name="full",
            indices=np.arange(50),
            timestamps=timestamps,
            returns=returns,
        ),
    }

    return EvaluationDataset(
        data_dict=data_dict,
        data_tensor=data_tensor,
        returns=returns,
        timestamps=timestamps,
        asset_ids=np.array([f"A{i:02d}" for i in range(returns.shape[0])]),
        splits=splits,
        processed_df=pd.DataFrame(),
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


def test_build_core_mining_config_uses_synthetic_policy_for_mock():
    """Mock mining flows should opt into synthetic fallback explicitly."""
    from factorminer.cli import _build_core_mining_config

    cfg = SimpleNamespace(
        mining=SimpleNamespace(
            target_library_size=10,
            batch_size=5,
            max_iterations=3,
            ic_threshold=0.02,
            icir_threshold=0.3,
            correlation_threshold=0.7,
            replacement_ic_min=0.10,
            replacement_ic_ratio=1.3,
        ),
        evaluation=SimpleNamespace(
            fast_screen_assets=10,
            num_workers=1,
            backend="numpy",
            gpu_device="cuda:0",
            signal_failure_policy="reject",
        ),
    )

    strict_cfg = _build_core_mining_config(cfg, output_dir=Path("/tmp"), mock=False)
    mock_cfg = _build_core_mining_config(cfg, output_dir=Path("/tmp"), mock=True)

    assert strict_cfg.signal_failure_policy == "reject"
    assert mock_cfg.signal_failure_policy == "synthetic"
