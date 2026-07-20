"""Tests for the optional-torch correlation-GraphSAGE model kind."""

from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pytest

from factorminer.evaluation import model_zoo as model_zoo_mod
from factorminer.evaluation.model_zoo import (
    _TORCH_AVAILABLE,
    ModelZooConfig,
    ModelZooEvaluator,
)


def _graphsage_panel(seed: int = 42, assets: int = 20, periods: int = 60):
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((assets, periods))
    returns = 0.12 * latent + 0.05 * rng.standard_normal((assets, periods))
    # Shared shock among a block of assets -> non-trivial corr graph.
    returns[:5] += 0.2 * rng.standard_normal((periods,))
    f_signal = latent + 0.15 * rng.standard_normal((assets, periods))
    f_noise_a = rng.standard_normal((assets, periods))
    f_noise_b = rng.standard_normal((assets, periods))
    signals = {1: f_signal, 2: f_noise_a, 3: f_noise_b}
    names = {1: "signal", 2: "noise_a", 3: "noise_b"}
    return signals, names, returns


def test_config_accepts_corr_graphsage_kind():
    cfg = ModelZooConfig(
        model_kind="corr_graphsage",
        graph_corr_threshold=0.25,
        graph_hidden_dim=4,
    )
    assert cfg.model_kind == "corr_graphsage"
    assert cfg.graph_hidden_dim == 4


def test_config_rejects_nonpositive_hidden_dim():
    with pytest.raises(ValueError, match="graph_hidden_dim"):
        ModelZooConfig(model_kind="corr_graphsage", graph_hidden_dim=0)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_graphsage_end_to_end_on_synthetic_panel():
    signals, names, returns = _graphsage_panel()
    report = ModelZooEvaluator().evaluate(
        signals,
        names,
        returns,
        config=ModelZooConfig(
            model_kind="corr_graphsage",
            train_fraction=0.7,
            permutation_repeats=4,
            graph_corr_threshold=0.2,
            graph_hidden_dim=8,
            seed=0,
        ),
        iteration=3,
    )

    assert report.model_kind == "corr_graphsage"
    assert report.n_factors == 3
    assert report.n_train_samples > 0
    assert report.n_test_samples > 0
    assert np.isfinite(report.held_out_ic)
    assert np.isfinite(report.held_out_r2)
    assert np.isfinite(report.held_out_sharpe)
    assert len(report.contributions) == 3
    # Signal factor should outrank pure noise on permutation importance.
    by_name = {c.factor_name: c for c in report.contributions}
    assert by_name["signal"].rank == 1
    assert by_name["signal"].permutation_importance_mean > by_name["noise_a"].permutation_importance_mean
    # Creative AI narrative field
    assert report.neighbor_influence_summary is not None
    assert "Correlation-GraphSAGE" in report.neighbor_influence_summary
    assert "asset_" in report.neighbor_influence_summary

    payload = report.to_dict()
    assert "neighbor_influence_summary" in payload
    json.dumps(payload)  # serializable


def test_graphsage_raises_clear_error_without_torch():
    """Without torch, corr_graphsage must fail with a clear RuntimeError."""
    signals, names, returns = _graphsage_panel(assets=8, periods=30)
    cfg = ModelZooConfig(model_kind="corr_graphsage", permutation_repeats=2, seed=1)

    with mock.patch.object(model_zoo_mod, "_TORCH_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="PyTorch"):
            ModelZooEvaluator().evaluate(signals, names, returns, config=cfg)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_graphsage_empty_library_is_graceful():
    returns = np.zeros((10, 40))
    report = ModelZooEvaluator().evaluate(
        {},
        {},
        returns,
        config=ModelZooConfig(model_kind="corr_graphsage"),
    )
    assert report.n_factors == 0
    assert report.contributions == []
    assert report.neighbor_influence_summary is None
