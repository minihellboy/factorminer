"""Downstream model zoo for RD-Agent(Q)-style factor+model co-optimization.

RD-Agent(Q) (Microsoft Research + HKUST, NeurIPS 2025) reports its largest
result from *jointly* evolving the factor pool and a downstream nonlinear
prediction model, instead of mining factors in isolation and combining them
with a fixed rule. This module implements the "evaluate the downstream
model" building block of that idea: periodically fit a Ridge, Lasso, or
XGBoost model on the *current* factor library's signals against realized
returns, score it out-of-sample (held-out IC/Sharpe), and attribute each
admitted factor's marginal contribution to that model.

Marginal contribution metric
-----------------------------
Each factor's contribution is scored with **held-out permutation
importance** (``sklearn.inspection.permutation_importance``): the drop in
test-set R^2 when that factor's column is independently shuffled, averaged
over several repeats. This is chosen over raw coefficient magnitude or
gain-based feature importance because it is (a) model-agnostic -- identical
semantics across ridge, lasso, and xgboost -- and (b) computed strictly on
the held-out split, so it cannot reward a factor for overfitting the
training window. Linear models additionally report their standardized
coefficient (``coefficient``) as a supplementary, classically interpretable
field; it is ``None`` for xgboost. As a cross-check, each factor also gets
an ``ensemble_marginal_delta_ic``, produced by running the existing
``EnsembleMarginalUtilityService`` (train/test OLS ensemble-utility
machinery already used for candidate-vs-library scoring) leave-one-out
against the rest of the library.

This module never mutates factor admission decisions; it is a read-only
diagnostic surface for `architecture.model_stage.ModelCoOptimizeStage`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812 -- idiomatic PyTorch alias

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from factorminer.architecture.research_extensions import (
    EnsembleMarginalUtilityService,
    _as_panel,
    _flatten_period_samples,
    _safe_corr,
    _safe_r2,
    _standardize_train_test,
)
from factorminer.evaluation.backtest import compute_sharpe_ratio
from factorminer.evaluation.portfolio import PortfolioBacktester

logger = logging.getLogger(__name__)

_SUPPORTED_MODEL_KINDS = ("ridge", "lasso", "xgboost", "corr_graphsage")
_SUPPORTED_TRAIN_OBJECTIVES = ("mse", "margin_pairwise", "listnet", "bpr")
_MIN_SAMPLES = 5


@dataclass(frozen=True)
class ModelZooConfig:
    """Configuration for one downstream factor+model co-optimization fit.

    Parameters
    ----------
    model_kind : str
        One of ``"ridge"``, ``"lasso"``, ``"xgboost"``, ``"corr_graphsage"`` --
        the downstream model family fit on the factor library's signals
        against realized returns.
    alpha : float
        L2 (ridge) or L1 (lasso) regularization strength. Also used as the
        L2 penalty for ranking-loss linear fits. Ignored for xgboost /
        corr_graphsage.
    train_fraction : float
        Fraction of time periods (contiguous, earliest-first) used for
        training; the remainder is held out for IC/Sharpe evaluation and
        permutation importance.
    permutation_repeats : int
        Number of shuffles per factor for held-out permutation importance.
    seed : int
        Random seed for model fitting and permutation importance.
    xgb_n_estimators, xgb_max_depth, xgb_learning_rate : int, int, float
        XGBoost hyperparameters; ignored for ridge/lasso/corr_graphsage.
    train_objective : str
        Training objective. ``"mse"`` (default) is ordinary regression.
        Ranking objectives ``"margin_pairwise"``, ``"listnet"``, and
        ``"bpr"`` re-fit linear models via a differentiable pairwise /
        listwise loss, and switch XGBoost to ``rank:pairwise``.
    graph_corr_threshold : float
        Absolute return-correlation threshold used to build the per-date
        asset adjacency for ``corr_graphsage``.
    graph_hidden_dim : int
        Hidden width of the tiny GraphSAGE encoder.
    """

    model_kind: str = "ridge"
    alpha: float = 1.0
    train_fraction: float = 0.7
    permutation_repeats: int = 20
    seed: int = 42
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.1
    train_objective: str = "mse"
    graph_corr_threshold: float = 0.3
    graph_hidden_dim: int = 8

    def __post_init__(self) -> None:
        if self.model_kind not in _SUPPORTED_MODEL_KINDS:
            raise ValueError(
                f"model_kind must be one of {_SUPPORTED_MODEL_KINDS}, got {self.model_kind!r}"
            )
        if self.train_objective not in _SUPPORTED_TRAIN_OBJECTIVES:
            raise ValueError(
                f"train_objective must be one of {_SUPPORTED_TRAIN_OBJECTIVES}, "
                f"got {self.train_objective!r}"
            )
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1)")
        if self.graph_hidden_dim < 1:
            raise ValueError("graph_hidden_dim must be >= 1")


@dataclass(frozen=True)
class FactorContributionSummary:
    """Marginal contribution of one admitted factor to the fitted downstream model."""

    factor_id: int
    factor_name: str
    permutation_importance_mean: float
    permutation_importance_std: float
    coefficient: float | None
    ensemble_marginal_delta_ic: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCoOptimizationReport:
    """JSON-serializable report of one factor-library + downstream-model co-optimization pass."""

    model_kind: str
    n_factors: int
    n_train_samples: int
    n_test_samples: int
    held_out_ic: float
    held_out_r2: float
    held_out_sharpe: float
    baseline_equal_weight_ic: float
    contributions: list[FactorContributionSummary]
    generated_at_iteration: int
    neighbor_influence_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_kind": self.model_kind,
            "n_factors": self.n_factors,
            "n_train_samples": self.n_train_samples,
            "n_test_samples": self.n_test_samples,
            "held_out_ic": self.held_out_ic,
            "held_out_r2": self.held_out_r2,
            "held_out_sharpe": self.held_out_sharpe,
            "baseline_equal_weight_ic": self.baseline_equal_weight_ic,
            "contributions": [c.to_dict() for c in self.contributions],
            "generated_at_iteration": self.generated_at_iteration,
        }
        if self.neighbor_influence_summary is not None:
            payload["neighbor_influence_summary"] = self.neighbor_influence_summary
        return payload


def _make_linear_ranking_model_class() -> type:
    """Build the sklearn-compatible ranking shim with proper estimator bases."""
    from sklearn.base import BaseEstimator, RegressorMixin

    class _LinearRankingModel(BaseEstimator, RegressorMixin):  # type: ignore[misc,valid-type]
        """Minimal sklearn-compatible shim around a fitted linear weight vector.

        Exposes ``fit``/``predict``/``coef_`` so
        ``sklearn.inspection.permutation_importance`` keeps working unchanged
        for ranking-objective linear models.
        """

        def __init__(
            self, coef: np.ndarray | None = None, intercept: float = 0.0
        ) -> None:
            self.coef = (
                np.asarray(coef, dtype=np.float64).ravel()
                if coef is not None
                else np.zeros(1, dtype=np.float64)
            )
            self.intercept = float(intercept)

        @property
        def coef_(self) -> np.ndarray:
            return np.asarray(self.coef, dtype=np.float64).ravel()

        @property
        def intercept_(self) -> float:
            return float(self.intercept)

        def fit(self, x: np.ndarray, y: np.ndarray) -> Any:  # noqa: ARG002
            # Already fitted by ``_fit_ranking_linear``; no-op for sklearn APIs.
            return self

        def predict(self, x: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x, dtype=np.float64)
            return x_arr @ self.coef_ + self.intercept_

    return _LinearRankingModel


_LinearRankingModel = _make_linear_ranking_model_class()


def _pairwise_index_sample(
    y: np.ndarray, *, max_pairs: int = 2048, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ordered pairs (i, j) where y[i] > y[j]."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    # Prefer all pairs for small n; subsample otherwise.
    if n * (n - 1) // 2 <= max_pairs:
        i_idx, j_idx = np.triu_indices(n, k=1)
        higher = y[i_idx] > y[j_idx]
        lower = y[i_idx] < y[j_idx]
        hi = np.concatenate([i_idx[higher], j_idx[lower]])
        lo = np.concatenate([j_idx[higher], i_idx[lower]])
        return hi.astype(np.int64), lo.astype(np.int64)

    # Random sampling with rejection until we fill the budget or exhaust tries.
    hi_list: list[int] = []
    lo_list: list[int] = []
    tries = 0
    target = max_pairs
    max_tries = max_pairs * 20
    while len(hi_list) < target and tries < max_tries:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        tries += 1
        if i == j:
            continue
        if y[i] > y[j]:
            hi_list.append(i)
            lo_list.append(j)
        elif y[j] > y[i]:
            hi_list.append(j)
            lo_list.append(i)
    return np.asarray(hi_list, dtype=np.int64), np.asarray(lo_list, dtype=np.int64)


def _ranking_loss_and_grad(
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    objective: str,
    alpha: float,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Differentiable ranking loss + analytic gradient w.r.t. weights."""
    scores = x @ w
    reg = 0.5 * float(alpha) * float(np.dot(w, w))
    grad_reg = float(alpha) * w

    if objective == "listnet":
        # Listwise softmax cross-entropy between target and score distributions.
        t = y - np.max(y)
        s = scores - np.max(scores)
        exp_t = np.exp(t)
        exp_s = np.exp(s)
        p_t = exp_t / (np.sum(exp_t) + 1e-12)
        p_s = exp_s / (np.sum(exp_s) + 1e-12)
        loss = float(-np.sum(p_t * np.log(p_s + 1e-12)) + reg)
        grad = x.T @ (p_s - p_t) + grad_reg
        return loss, grad

    if pair_i.size == 0:
        return float(reg), grad_reg

    diff = scores[pair_i] - scores[pair_j]

    if objective == "margin_pairwise":
        # Hinge / margin pairwise: max(0, 1 - (s_i - s_j))
        margin = 1.0 - diff
        active = margin > 0.0
        loss = float(np.mean(np.maximum(margin, 0.0)) + reg)
        grad = grad_reg.copy()
        if active.any():
            scale = 1.0 / pair_i.shape[0]
            contrib = np.zeros_like(scores)
            np.add.at(contrib, pair_i[active], -scale)
            np.add.at(contrib, pair_j[active], scale)
            grad = grad + x.T @ contrib
        return loss, grad

    if objective == "bpr":
        # Bayesian Personalized Ranking: -log(sigmoid(s_i - s_j))
        # Stable softplus form: softplus(-diff) = log(1 + exp(-diff))
        neg = -diff
        # softplus
        soft = np.where(neg > 0, neg + np.log1p(np.exp(-neg)), np.log1p(np.exp(neg)))
        loss = float(np.mean(soft) + reg)
        # softplus(-diff) = -log(sigmoid(diff)), so
        # d/ddiff softplus(-diff) = -(1 - sigmoid(diff)) = sigmoid(diff) - 1.
        # (Not `-sigmoid(diff)`: that sign/formula error would apply zero
        # gradient pressure to badly mis-ranked pairs and maximal pressure
        # to already-correctly-ranked ones -- the opposite of BPR's intent.)
        sig = 1.0 / (1.0 + np.exp(-diff))
        one_minus_sig = 1.0 - sig
        scale = 1.0 / pair_i.shape[0]
        contrib = np.zeros_like(scores)
        np.add.at(contrib, pair_i, -scale * one_minus_sig)
        np.add.at(contrib, pair_j, scale * one_minus_sig)
        grad = grad_reg + x.T @ contrib
        return loss, grad

    raise ValueError(f"Unsupported ranking objective: {objective!r}")


def _fit_ranking_linear(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    objective: str,
    alpha: float,
    seed: int,
) -> _LinearRankingModel:
    """Fit a linear model under a ranking objective via L-BFGS-B."""
    from scipy.optimize import minimize

    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64).ravel()
    n_features = x.shape[1]
    pair_i, pair_j = _pairwise_index_sample(y, seed=seed)

    def fun(w: np.ndarray) -> float:
        loss, _ = _ranking_loss_and_grad(
            w, x, y, objective=objective, alpha=alpha, pair_i=pair_i, pair_j=pair_j
        )
        return loss

    def jac(w: np.ndarray) -> np.ndarray:
        _, grad = _ranking_loss_and_grad(
            w, x, y, objective=objective, alpha=alpha, pair_i=pair_i, pair_j=pair_j
        )
        return grad

    rng = np.random.default_rng(seed)
    w0 = rng.normal(scale=0.01, size=n_features)
    result = minimize(
        fun,
        w0,
        method="L-BFGS-B",
        jac=jac,
        options={"maxiter": 200, "ftol": 1e-8},
    )
    coef = np.asarray(result.x, dtype=np.float64)
    return _LinearRankingModel(coef)


class ModelZooEvaluator:
    """Fits a downstream Ridge/Lasso/XGBoost model on a factor library's signals.

    See module docstring for the marginal-contribution metric rationale.
    """

    def __init__(self, marginal_utility_service: EnsembleMarginalUtilityService | None = None) -> None:
        self._marginal_utility_service = marginal_utility_service or EnsembleMarginalUtilityService()

    def evaluate(
        self,
        factor_signals: Mapping[int, np.ndarray],
        factor_names: Mapping[int, str],
        returns: np.ndarray,
        *,
        config: ModelZooConfig | None = None,
        iteration: int = 0,
    ) -> ModelCoOptimizationReport:
        """Fit `config.model_kind` on `factor_signals` vs `returns` and score each factor.

        Parameters
        ----------
        factor_signals : Mapping[int, ndarray], each of shape (assets, periods)
            Admitted factor signals, keyed by factor id.
        factor_names : Mapping[int, str]
            Human-readable name per factor id (for the report only).
        returns : ndarray, shape (assets, periods)
            Realized forward returns aligned with the factor signals.
        config : ModelZooConfig, optional
            Downstream model configuration; defaults to a Ridge fit.
        iteration : int
            Loop iteration this pass was generated at, echoed into the report.

        Returns
        -------
        ModelCoOptimizationReport
        """
        config = config or ModelZooConfig()
        if config.model_kind == "corr_graphsage":
            return self._evaluate_graphsage(
                factor_signals,
                factor_names,
                returns,
                config=config,
                iteration=iteration,
            )
        factor_ids = sorted(factor_signals)
        if not factor_ids:
            return self._empty_report(config, iteration, n_factors=0)

        panels = [_as_panel(factor_signals[fid]) for fid in factor_ids]
        return_panel = _as_panel(returns)
        for panel in panels:
            if panel.shape != return_panel.shape:
                raise ValueError("All factor signal panels must share the returns panel shape")

        assets, periods = return_panel.shape
        if periods <= 1:
            return self._empty_report(config, iteration, n_factors=len(factor_ids))

        split = int(round(periods * config.train_fraction))
        split = min(max(split, 1), periods - 1)
        train_periods = list(range(split))
        test_periods = list(range(split, periods))

        train_x, train_y = _flatten_period_samples(panels, return_panel, train_periods)
        test_x, test_y = _flatten_period_samples(panels, return_panel, test_periods)
        if train_x.shape[0] < _MIN_SAMPLES or test_x.shape[0] < _MIN_SAMPLES:
            return self._empty_report(config, iteration, n_factors=len(factor_ids))

        train_x_std, test_x_std = _standardize_train_test(train_x, test_x)

        # Mirrors `_standardize_train_test`'s own mean/std convention, kept
        # separately so the full (period, asset) prediction grid needed for
        # the held-out Sharpe backtest below can be standardized the same
        # way without re-deriving it from a differently shaped input.
        train_mean = np.mean(train_x, axis=0, keepdims=True)
        train_std = np.std(train_x, axis=0, keepdims=True)
        train_std[train_std < 1e-12] = 1.0

        if config.model_kind in ("ridge", "lasso") and config.train_objective != "mse":
            model = _fit_ranking_linear(
                train_x_std,
                train_y,
                objective=config.train_objective,
                alpha=config.alpha,
                seed=config.seed,
            )
        else:
            model = self._build_model(config)
            model.fit(train_x_std, train_y)
        test_pred = model.predict(test_x_std)

        held_out_ic = _safe_corr(test_pred, test_y)
        held_out_r2 = _safe_r2(test_y, test_pred)
        held_out_sharpe = self._held_out_sharpe(
            model, panels, return_panel, test_periods, train_mean, train_std
        )

        equal_weight_panel = np.nanmean(np.stack(panels, axis=0), axis=0)
        eq_test_x, eq_test_y = _flatten_period_samples(
            [equal_weight_panel], return_panel, test_periods
        )
        baseline_ic = (
            _safe_corr(eq_test_x[:, 0], eq_test_y) if eq_test_x.size else 0.0
        )

        contributions = self._score_contributions(
            model=model,
            config=config,
            factor_ids=factor_ids,
            factor_names=factor_names,
            panels=panels,
            return_panel=return_panel,
            test_x_std=test_x_std,
            test_y=test_y,
        )

        return ModelCoOptimizationReport(
            model_kind=config.model_kind,
            n_factors=len(factor_ids),
            n_train_samples=int(train_x.shape[0]),
            n_test_samples=int(test_x.shape[0]),
            held_out_ic=float(held_out_ic),
            held_out_r2=float(held_out_r2),
            held_out_sharpe=float(held_out_sharpe),
            baseline_equal_weight_ic=float(baseline_ic),
            contributions=contributions,
            generated_at_iteration=int(iteration),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, config: ModelZooConfig) -> Any:
        if config.model_kind == "ridge":
            from sklearn.linear_model import Ridge

            return Ridge(alpha=config.alpha, random_state=config.seed)
        if config.model_kind == "lasso":
            from sklearn.linear_model import Lasso

            return Lasso(alpha=config.alpha, random_state=config.seed, max_iter=10000)
        if config.model_kind == "corr_graphsage":
            raise ValueError(
                "corr_graphsage uses the panel-aware _evaluate_graphsage path; "
                "do not call _build_model for it"
            )

        import xgboost as xgb

        kwargs: dict[str, Any] = dict(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            n_jobs=-1,
            verbosity=0,
            random_state=config.seed,
        )
        if config.train_objective != "mse":
            # Native XGBoost pairwise ranking objective. XGBRegressor still
            # accepts predict(); group construction is left implicit (single
            # list) which is adequate for the flattened row sample matrix.
            kwargs["objective"] = "rank:pairwise"
        return xgb.XGBRegressor(**kwargs)

    def _held_out_sharpe(
        self,
        model: Any,
        panels: Sequence[np.ndarray],
        return_panel: np.ndarray,
        test_periods: Sequence[int],
        train_mean: np.ndarray,
        train_std: np.ndarray,
    ) -> float:
        """Reshape held-out predictions into a (T, N) signal grid and Sharpe it.

        Reuses `PortfolioBacktester.quintile_backtest` (long-short quintile
        portfolio construction) and `compute_sharpe_ratio` rather than
        re-deriving a portfolio-return time series from scratch.
        """
        assets, _ = return_panel.shape
        n_test = len(test_periods)
        if n_test == 0 or assets == 0:
            return 0.0

        stacked = np.stack([panel[:, test_periods] for panel in panels], axis=-1)  # (assets, n_test, K)
        grid = np.transpose(stacked, (1, 0, 2)).reshape(n_test * assets, len(panels))
        valid = np.all(np.isfinite(grid), axis=1)
        predicted = np.full(n_test * assets, np.nan, dtype=np.float64)
        if valid.any():
            standardized = (grid[valid] - train_mean) / train_std
            predicted[valid] = model.predict(standardized)

        signal_tn = predicted.reshape(n_test, assets)
        returns_tn = return_panel[:, test_periods].T
        try:
            backtest = PortfolioBacktester().quintile_backtest(signal_tn, returns_tn)
        except Exception:  # pragma: no cover - defensive, backtest is best-effort here
            logger.warning("Held-out quintile backtest failed; reporting Sharpe=0.0", exc_info=True)
            return 0.0
        ls_series = np.asarray(backtest["ls_net_series"], dtype=np.float64)
        return compute_sharpe_ratio(ls_series)

    def _score_contributions(
        self,
        *,
        model: Any,
        config: ModelZooConfig,
        factor_ids: Sequence[int],
        factor_names: Mapping[int, str],
        panels: Sequence[np.ndarray],
        return_panel: np.ndarray,
        test_x_std: np.ndarray,
        test_y: np.ndarray,
    ) -> list[FactorContributionSummary]:
        n_factors = len(factor_ids)
        if test_x_std.shape[0] >= _MIN_SAMPLES:
            from sklearn.inspection import permutation_importance

            perm = permutation_importance(
                model,
                test_x_std,
                test_y,
                n_repeats=config.permutation_repeats,
                random_state=config.seed,
                scoring="r2",
            )
            importances_mean = perm.importances_mean
            importances_std = perm.importances_std
        else:
            importances_mean = np.zeros(n_factors)
            importances_std = np.zeros(n_factors)

        coefficients = getattr(model, "coef_", None)
        if coefficients is not None:
            coefficients = np.asarray(coefficients, dtype=np.float64).ravel()

        summaries: list[FactorContributionSummary] = []
        for idx, fid in enumerate(factor_ids):
            other_panels = [panels[j] for j in range(n_factors) if j != idx]
            other_ids = [factor_ids[j] for j in range(n_factors) if j != idx]
            try:
                utility = self._marginal_utility_service.estimate(
                    candidate=panels[idx],
                    existing_signals=other_panels,
                    returns=return_panel,
                    train_fraction=config.train_fraction,
                    candidate_name=str(factor_names.get(fid, fid)),
                    reference_names=[str(factor_names.get(oid, oid)) for oid in other_ids],
                )
                delta_ic = float(utility.delta_ic)
            except Exception:  # pragma: no cover - cross-check must never break the primary report
                logger.warning(
                    "Ensemble marginal utility cross-check failed for factor %s", fid, exc_info=True
                )
                delta_ic = 0.0

            summaries.append(
                FactorContributionSummary(
                    factor_id=int(fid),
                    factor_name=str(factor_names.get(fid, fid)),
                    permutation_importance_mean=float(importances_mean[idx]),
                    permutation_importance_std=float(importances_std[idx]),
                    coefficient=float(coefficients[idx]) if coefficients is not None else None,
                    ensemble_marginal_delta_ic=delta_ic,
                    rank=0,
                )
            )

        summaries.sort(key=lambda s: s.permutation_importance_mean, reverse=True)
        return [replace(summary, rank=rank + 1) for rank, summary in enumerate(summaries)]

    def _evaluate_graphsage(
        self,
        factor_signals: Mapping[int, np.ndarray],
        factor_names: Mapping[int, str],
        returns: np.ndarray,
        *,
        config: ModelZooConfig,
        iteration: int,
    ) -> ModelCoOptimizationReport:
        """Panel-aware correlation-GraphSAGE path (optional torch).

        Unlike ridge/lasso/xgboost, this keeps the (assets, periods) layout:
        each date is a graph whose nodes are assets, node features are the
        K factor values that day, and edges come from rolling return
        correlation above ``config.graph_corr_threshold``.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "corr_graphsage requires PyTorch. Install torch to enable "
                "the correlation-GraphSAGE model kind "
                "(pip install torch / uv add torch)."
            )

        factor_ids = sorted(factor_signals)
        if not factor_ids:
            return self._empty_report(config, iteration, n_factors=0)

        panels = [_as_panel(factor_signals[fid]) for fid in factor_ids]
        return_panel = _as_panel(returns)
        for panel in panels:
            if panel.shape != return_panel.shape:
                raise ValueError("All factor signal panels must share the returns panel shape")

        n_assets, n_periods = return_panel.shape
        n_factors = len(factor_ids)
        if n_periods <= 1 or n_assets < 2:
            return self._empty_report(config, iteration, n_factors=n_factors)

        split = int(round(n_periods * config.train_fraction))
        split = min(max(split, 1), n_periods - 1)
        train_periods = list(range(split))
        test_periods = list(range(split, n_periods))

        # Stack factors -> (assets, periods, K)
        feat = np.stack(panels, axis=-1).astype(np.float64)
        # Standardize features on train periods only (per-factor).
        train_slice = feat[:, train_periods, :]
        flat_train = train_slice.reshape(-1, n_factors)
        mu = np.nanmean(flat_train, axis=0)
        sigma = np.nanstd(flat_train, axis=0)
        sigma = np.where(sigma < 1e-12, 1.0, sigma)
        feat = (feat - mu) / sigma
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

        ret = np.nan_to_num(return_panel.astype(np.float64), nan=0.0)

        # Rolling correlation adjacency from past returns (causal: lookback ends at t-1).
        lookback = min(20, max(5, split // 2))
        threshold = float(config.graph_corr_threshold)
        adjacencies: list[np.ndarray] = []
        for t in range(n_periods):
            start = max(0, t - lookback)
            end = t  # exclusive; no leakage of date t returns
            if end - start < 3:
                adj = np.eye(n_assets, dtype=np.float64)
            else:
                window = ret[:, start:end]  # (N, W)
                # Pairwise corr across assets
                centered = window - window.mean(axis=1, keepdims=True)
                norms = np.linalg.norm(centered, axis=1, keepdims=True)
                norms = np.where(norms < 1e-12, 1.0, norms)
                normalized = centered / norms
                corr = normalized @ normalized.T
                adj = (np.abs(corr) >= threshold).astype(np.float64)
                np.fill_diagonal(adj, 1.0)
            # Row-normalize for mean-aggregator GraphSAGE
            deg = adj.sum(axis=1, keepdims=True)
            deg = np.where(deg < 1e-12, 1.0, deg)
            adjacencies.append(adj / deg)

        hidden = int(config.graph_hidden_dim)
        torch.manual_seed(int(config.seed))
        model = _CorrGraphSAGE(n_factors, hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)

        model.train()
        n_epochs = 40
        for _ in range(n_epochs):
            optimizer.zero_grad()
            losses = []
            for t in train_periods:
                x_t = torch.as_tensor(feat[:, t, :], dtype=torch.float32)
                a_t = torch.as_tensor(adjacencies[t], dtype=torch.float32)
                y_t = torch.as_tensor(ret[:, t], dtype=torch.float32)
                pred = model(x_t, a_t)
                # Prefer ranking-friendly loss when requested; else MSE.
                if config.train_objective == "mse":
                    losses.append(F.mse_loss(pred, y_t))
                else:
                    # Soft pairwise logistic on a random subset of pairs.
                    n = y_t.shape[0]
                    if n < 2:
                        continue
                    # All-pairs BPR-style for small N
                    diff_y = y_t.unsqueeze(0) - y_t.unsqueeze(1)
                    diff_p = pred.unsqueeze(0) - pred.unsqueeze(1)
                    mask = diff_y > 0
                    if mask.any():
                        losses.append(F.softplus(-diff_p[mask]).mean())
            if not losses:
                break
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_preds: list[np.ndarray] = []
            test_targets: list[np.ndarray] = []
            # Neighbor influence bookkeeping for the creative narrative.
            influence_scores = np.zeros(n_assets, dtype=np.float64)
            for t in test_periods:
                x_t = torch.as_tensor(feat[:, t, :], dtype=torch.float32)
                a_t = torch.as_tensor(adjacencies[t], dtype=torch.float32)
                pred = model(x_t, a_t).cpu().numpy()
                test_preds.append(pred)
                test_targets.append(ret[:, t])
                # Proxy: degree-weighted absolute adjacency mass as neighbor influence.
                influence_scores += np.asarray(adjacencies[t], dtype=np.float64).sum(axis=0)

        if not test_preds:
            return self._empty_report(config, iteration, n_factors=n_factors)

        pred_mat = np.stack(test_preds, axis=1)  # (assets, n_test)
        tgt_mat = np.stack(test_targets, axis=1)
        test_pred_flat = pred_mat.T.reshape(-1)
        test_y_flat = tgt_mat.T.reshape(-1)
        held_out_ic = _safe_corr(test_pred_flat, test_y_flat)
        held_out_r2 = _safe_r2(test_y_flat, test_pred_flat)

        # Sharpe via existing quintile path on (T, N) grids.
        try:
            backtest = PortfolioBacktester().quintile_backtest(pred_mat.T, tgt_mat.T)
            ls_series = np.asarray(backtest["ls_net_series"], dtype=np.float64)
            held_out_sharpe = float(compute_sharpe_ratio(ls_series))
        except Exception:  # pragma: no cover
            logger.warning("GraphSAGE held-out Sharpe backtest failed", exc_info=True)
            held_out_sharpe = 0.0

        equal_weight_panel = np.nanmean(np.stack(panels, axis=0), axis=0)
        eq_test_x, eq_test_y = _flatten_period_samples(
            [equal_weight_panel], return_panel, test_periods
        )
        baseline_ic = (
            _safe_corr(eq_test_x[:, 0], eq_test_y) if eq_test_x.size else 0.0
        )

        # Factor contributions: permutation of each factor channel on test dates.
        rng = np.random.default_rng(config.seed)
        base_ic = held_out_ic
        importances_mean = np.zeros(n_factors, dtype=np.float64)
        importances_std = np.zeros(n_factors, dtype=np.float64)
        for k in range(n_factors):
            drops: list[float] = []
            for _ in range(max(3, min(config.permutation_repeats, 10))):
                feat_perm = feat.copy()
                # Shuffle factor k across assets within each test date.
                for t in test_periods:
                    order = rng.permutation(n_assets)
                    feat_perm[:, t, k] = feat_perm[order, t, k]
                perm_preds = []
                with torch.no_grad():
                    for t in test_periods:
                        x_t = torch.as_tensor(feat_perm[:, t, :], dtype=torch.float32)
                        a_t = torch.as_tensor(adjacencies[t], dtype=torch.float32)
                        perm_preds.append(model(x_t, a_t).cpu().numpy())
                perm_flat = np.stack(perm_preds, axis=1).T.reshape(-1)
                perm_ic = _safe_corr(perm_flat, test_y_flat)
                drops.append(base_ic - perm_ic)
            importances_mean[k] = float(np.mean(drops))
            importances_std[k] = float(np.std(drops))

        # Linear readout weights as a crude "coefficient" analogue (input projection).
        with torch.no_grad():
            # Average |W| over hidden for each input factor.
            w_in = model.lin_self.weight.detach().cpu().numpy()  # (hidden, K)
            coefs = np.mean(np.abs(w_in), axis=0)

        summaries: list[FactorContributionSummary] = []
        for idx, fid in enumerate(factor_ids):
            other_panels = [panels[j] for j in range(n_factors) if j != idx]
            other_ids = [factor_ids[j] for j in range(n_factors) if j != idx]
            try:
                utility = self._marginal_utility_service.estimate(
                    candidate=panels[idx],
                    existing_signals=other_panels,
                    returns=return_panel,
                    train_fraction=config.train_fraction,
                    candidate_name=str(factor_names.get(fid, fid)),
                    reference_names=[str(factor_names.get(oid, oid)) for oid in other_ids],
                )
                delta_ic = float(utility.delta_ic)
            except Exception:  # pragma: no cover
                logger.warning(
                    "Ensemble marginal utility cross-check failed for factor %s", fid, exc_info=True
                )
                delta_ic = 0.0
            summaries.append(
                FactorContributionSummary(
                    factor_id=int(fid),
                    factor_name=str(factor_names.get(fid, fid)),
                    permutation_importance_mean=float(importances_mean[idx]),
                    permutation_importance_std=float(importances_std[idx]),
                    coefficient=float(coefs[idx]),
                    ensemble_marginal_delta_ic=delta_ic,
                    rank=0,
                )
            )
        summaries.sort(key=lambda s: s.permutation_importance_mean, reverse=True)
        contributions = [replace(s, rank=r + 1) for r, s in enumerate(summaries)]

        # Creative AI angle: name the assets whose correlation neighbors
        # most influenced predictions on the held-out window.
        top_idx = np.argsort(-influence_scores)[:3]
        neighbor_bits = []
        for i in top_idx:
            # Top neighbors of asset i on last test date by adjacency weight.
            last_t = test_periods[-1]
            raw_adj = adjacencies[last_t][i].copy()
            raw_adj[i] = 0.0
            neigh = np.argsort(-raw_adj)[:3]
            neigh_names = [f"asset_{int(j)}" for j in neigh if raw_adj[j] > 0]
            if neigh_names:
                neighbor_bits.append(
                    f"asset_{int(i)} (via {', '.join(neigh_names)})"
                )
            else:
                neighbor_bits.append(f"asset_{int(i)}")
        neighbor_summary = (
            "Correlation-GraphSAGE predictions were most influenced by neighbors of "
            + "; ".join(neighbor_bits)
            + "."
        )

        n_train_samples = n_assets * len(train_periods)
        n_test_samples = n_assets * len(test_periods)
        return ModelCoOptimizationReport(
            model_kind=config.model_kind,
            n_factors=n_factors,
            n_train_samples=int(n_train_samples),
            n_test_samples=int(n_test_samples),
            held_out_ic=float(held_out_ic),
            held_out_r2=float(held_out_r2),
            held_out_sharpe=float(held_out_sharpe),
            baseline_equal_weight_ic=float(baseline_ic),
            contributions=contributions,
            generated_at_iteration=int(iteration),
            neighbor_influence_summary=neighbor_summary,
        )

    def _empty_report(
        self, config: ModelZooConfig, iteration: int, *, n_factors: int
    ) -> ModelCoOptimizationReport:
        return ModelCoOptimizationReport(
            model_kind=config.model_kind,
            n_factors=n_factors,
            n_train_samples=0,
            n_test_samples=0,
            held_out_ic=0.0,
            held_out_r2=0.0,
            held_out_sharpe=0.0,
            baseline_equal_weight_ic=0.0,
            contributions=[],
            generated_at_iteration=int(iteration),
        )


if _TORCH_AVAILABLE:

    class _CorrGraphSAGE(nn.Module):
        """Tiny mean-aggregator GraphSAGE over a correlation asset graph.

        One message-passing step:
            h = ReLU(W_self x + W_neigh (A @ x))
            score = w_out · h
        where A is a row-normalized adjacency built from rolling return
        correlation. Kept deliberately small so CPU training on N~20, T~60
        panels finishes in well under a second.
        """

        def __init__(self, in_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.lin_self = nn.Linear(in_dim, hidden_dim, bias=True)
            self.lin_neigh = nn.Linear(in_dim, hidden_dim, bias=False)
            self.lin_out = nn.Linear(hidden_dim, 1, bias=True)

        def forward(self, x: Any, adj: Any) -> Any:
            # x: (N, K), adj: (N, N) row-normalized
            neigh = adj @ x
            h = F.relu(self.lin_self(x) + self.lin_neigh(neigh))
            return self.lin_out(h).squeeze(-1)

else:  # pragma: no cover - exercised via RuntimeError path in tests

    class _CorrGraphSAGE:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "corr_graphsage requires PyTorch. Install torch to enable "
                "the correlation-GraphSAGE model kind."
            )
