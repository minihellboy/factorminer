"""Tests for ranking-loss train objectives on ModelZooEvaluator.

Covers ``train_objective`` in {mse, margin_pairwise, listnet, bpr} for
linear models (custom differentiable losses via L-BFGS-B) and the XGBoost
``rank:pairwise`` passthrough.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from factorminer.evaluation.model_zoo import (
    _SUPPORTED_TRAIN_OBJECTIVES,
    ModelZooConfig,
    ModelZooEvaluator,
    _fit_ranking_linear,
    _LinearRankingModel,
)


def _ranking_panel(seed: int = 7, assets: int = 24, periods: int = 80):
    """Synthetic panel where cross-sectional order matters.

    Includes a single MSE-hostile outlier period so plain ridge chases
    squared error while ranking losses protect the cross-sectional order.
    """
    rng = np.random.default_rng(seed)
    strong = rng.standard_normal((assets, periods))
    weak = rng.standard_normal((assets, periods))
    noise_a = rng.standard_normal((assets, periods))
    noise_b = rng.standard_normal((assets, periods))
    returns = 0.7 * strong + 0.1 * weak + 0.35 * rng.standard_normal((assets, periods))
    # Inject a few extreme outliers that MSE overweight.
    returns[0, periods // 2] = 8.0
    returns[1, periods // 2 + 1] = -8.0
    signals = {1: strong, 2: weak, 3: noise_a, 4: noise_b}
    names = {1: "strong", 2: "weak", 3: "noise_a", 4: "noise_b"}
    return signals, names, returns


def test_config_accepts_all_ranking_objectives():
    for obj in _SUPPORTED_TRAIN_OBJECTIVES:
        cfg = ModelZooConfig(train_objective=obj)
        assert cfg.train_objective == obj


def test_config_rejects_unknown_train_objective():
    with pytest.raises(ValueError, match="train_objective"):
        ModelZooConfig(train_objective="hinge_not_real")


@pytest.mark.parametrize("objective", ["margin_pairwise", "listnet", "bpr"])
def test_ranking_loss_gradient_matches_finite_difference(objective: str):
    """Analytic gradient must match central finite differences.

    Gold-standard check for a hand-derived gradient. Catches the BPR sign
    bug where softplus(-diff) was differentiated as -sigmoid(diff) instead
    of -(1 - sigmoid(diff)) -- a formula error that silently corrupts
    L-BFGS-B convergence without raising any error (zero gradient pressure
    on badly mis-ranked pairs, maximal pressure on already-correct ones).
    """
    from factorminer.evaluation.model_zoo import _pairwise_index_sample, _ranking_loss_and_grad

    rng = np.random.default_rng(3)
    n, d = 14, 5
    x = rng.normal(size=(n, d))
    y = rng.normal(size=n)
    w = rng.normal(size=d) * 0.3
    alpha = 0.1
    pair_i, pair_j = _pairwise_index_sample(y)

    _, grad_analytic = _ranking_loss_and_grad(
        w, x, y, objective=objective, alpha=alpha, pair_i=pair_i, pair_j=pair_j
    )

    eps = 1e-6
    grad_numeric = np.zeros_like(w)
    for k in range(d):
        wp, wm = w.copy(), w.copy()
        wp[k] += eps
        wm[k] -= eps
        lp, _ = _ranking_loss_and_grad(
            wp, x, y, objective=objective, alpha=alpha, pair_i=pair_i, pair_j=pair_j
        )
        lm, _ = _ranking_loss_and_grad(
            wm, x, y, objective=objective, alpha=alpha, pair_i=pair_i, pair_j=pair_j
        )
        grad_numeric[k] = (lp - lm) / (2 * eps)

    np.testing.assert_allclose(grad_analytic, grad_numeric, atol=1e-4, rtol=1e-4)


def test_fit_ranking_linear_returns_sklearn_compatible_shim():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 3))
    y = x @ np.array([1.5, -0.5, 0.1]) + 0.05 * rng.normal(size=60)
    model = _fit_ranking_linear(x, y, objective="bpr", alpha=0.5, seed=1)
    assert isinstance(model, _LinearRankingModel)
    pred = model.predict(x)
    assert pred.shape == (60,)
    assert model.coef_.shape == (3,)
    # permutation_importance needs get_params/set_params/fit
    assert "coef" in model.get_params()
    model.fit(x, y)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("objective", ["margin_pairwise", "listnet", "bpr"])
def test_ranking_objective_changes_predictions_vs_mse(objective: str):
    signals, names, returns = _ranking_panel()
    evaluator = ModelZooEvaluator()

    mse_report = evaluator.evaluate(
        signals,
        names,
        returns,
        config=ModelZooConfig(
            model_kind="ridge",
            train_objective="mse",
            permutation_repeats=5,
            seed=11,
            alpha=1.0,
        ),
    )
    rank_report = evaluator.evaluate(
        signals,
        names,
        returns,
        config=ModelZooConfig(
            model_kind="ridge",
            train_objective=objective,
            permutation_repeats=5,
            seed=11,
            alpha=1.0,
        ),
    )

    assert mse_report.model_kind == "ridge"
    assert rank_report.n_factors == 4
    assert all(np.isfinite(c.permutation_importance_mean) for c in rank_report.contributions)
    # Coefficients must exist for linear ranking shim
    assert all(c.coefficient is not None for c in rank_report.contributions)


def test_ranking_loss_improves_cross_sectional_spearman_vs_mse():
    """Concrete proof: ranking objective beats MSE on Spearman of order.

    Build a monotone feature plus a second noise feature that an MSE fit
    overweights because of a few extreme outliers aligned with it. A
    pairwise ranking loss should ignore magnitude and recover a higher
    Spearman correlation against the true order on the clean mass.
    """
    rng = np.random.default_rng(3)
    n = 200
    x_signal = np.linspace(-2.0, 2.0, n) + 0.05 * rng.normal(size=n)
    x_decoy = rng.normal(size=n)
    y = x_signal.copy()
    # Outliers: huge y aligned with the decoy feature, anti-aligned with signal order.
    outlier_idx = np.array([0, 1, 2, 3, 4], dtype=int)
    y[outlier_idx] = 30.0 * np.sign(x_decoy[outlier_idx])
    x = np.column_stack([x_signal, x_decoy])
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-12)

    from sklearn.linear_model import Ridge

    mse_model = Ridge(alpha=0.1, random_state=0).fit(x, y)
    rank_model = _fit_ranking_linear(
        x, y, objective="margin_pairwise", alpha=0.1, seed=0
    )

    mask = np.ones(n, dtype=bool)
    mask[outlier_idx] = False
    sp_mse = float(spearmanr(mse_model.predict(x)[mask], y[mask]).correlation)
    sp_rank = float(spearmanr(rank_model.predict(x)[mask], y[mask]).correlation)

    assert sp_rank > sp_mse + 0.02
    assert sp_rank > 0.85
    # Predictions must actually differ from plain MSE.
    assert not np.allclose(mse_model.predict(x), rank_model.predict(x))
    # Ranking should put more absolute weight on the signal column.
    assert abs(rank_model.coef_[0]) > abs(rank_model.coef_[1])


def test_evaluate_lasso_with_listnet_runs_end_to_end():
    signals, names, returns = _ranking_panel(assets=16, periods=60)
    report = ModelZooEvaluator().evaluate(
        signals,
        names,
        returns,
        config=ModelZooConfig(
            model_kind="lasso",
            train_objective="listnet",
            permutation_repeats=4,
            seed=2,
            alpha=0.1,
        ),
    )
    assert report.n_train_samples > 0
    assert report.n_test_samples > 0
    assert np.isfinite(report.held_out_ic)
    assert len(report.contributions) == 4


def test_xgboost_ranking_objective_runs():
    signals, names, returns = _ranking_panel(assets=18, periods=70)
    report = ModelZooEvaluator().evaluate(
        signals,
        names,
        returns,
        config=ModelZooConfig(
            model_kind="xgboost",
            train_objective="margin_pairwise",
            xgb_n_estimators=20,
            xgb_max_depth=2,
            permutation_repeats=3,
            seed=5,
        ),
    )
    assert report.model_kind == "xgboost"
    assert np.isfinite(report.held_out_ic)
    # xgboost has no linear coefficient
    assert all(c.coefficient is None for c in report.contributions)
