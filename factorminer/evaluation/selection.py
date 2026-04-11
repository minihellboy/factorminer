"""Factor selection methods for identifying sparse, high-value subsets.

Implements Lasso (L1), Forward Stepwise, and XGBoost-based selection
strategies for choosing an optimal subset of factors from the mined library.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class FactorSelector:
    """Select optimal factor subsets from the factor library.

    All methods accept factor signals as (T, N) arrays and forward returns
    as a (T, N) array, then return ordered lists of (factor_id, score) tuples.
    """

    # ------------------------------------------------------------------
    # Lasso (L1-regularized) selection
    # ------------------------------------------------------------------

    def lasso_selection(
        self,
        factor_signals: dict[int, np.ndarray],
        returns: np.ndarray,
        alpha: float | None = None,
    ) -> list[tuple[int, float]]:
        """Lasso: L1-regularized linear regression for factor selection.

        Paper: Only 8 factors capture 95% of IC improvement.

        Parameters
        ----------
        factor_signals : dict[int, ndarray]
            Mapping from factor ID to (T, N) signal array.
        returns : ndarray of shape (T, N)
            Forward returns aligned with factor signals.
        alpha : float or None
            L1 regularization strength.  If None, selected via cross-validation
            using LassoCV with 5 folds.

        Returns
        -------
        list of (factor_id, coefficient)
            Non-zero factors sorted by absolute coefficient (descending).
        """
        from sklearn.linear_model import Lasso, LassoCV

        ids, X, y = self._prepare_panel(factor_signals, returns)
        if len(ids) == 0:
            return []

        if alpha is None:
            model = LassoCV(cv=5, max_iter=10000, n_jobs=-1)
            model.fit(X, y)
            alpha = model.alpha_
            logger.info("LassoCV selected alpha=%.6f", alpha)

        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X, y)

        results: list[tuple[int, float]] = []
        for idx, coef in enumerate(lasso.coef_):
            if abs(coef) > 1e-10:
                results.append((ids[idx], float(coef)))

        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Forward stepwise selection
    # ------------------------------------------------------------------

    def forward_stepwise(
        self,
        factor_signals: dict[int, np.ndarray],
        returns: np.ndarray,
        max_factors: int = 20,
    ) -> list[tuple[int, float]]:
        """Forward Stepwise: greedy selection maximizing combined ICIR.

        Paper: 18 factors, ICIR=1.38.

        At each step, add the factor that yields the largest improvement in
        ICIR of the equal-weight composite of selected factors.

        Parameters
        ----------
        factor_signals : dict[int, ndarray]
            Mapping from factor ID to (T, N) signal array.
        returns : ndarray of shape (T, N)
            Forward returns.
        max_factors : int
            Maximum number of factors to select.

        Returns
        -------
        list of (factor_id, delta_ICIR)
            Factors in selection order with the ICIR improvement each contributed.
        """
        if not factor_signals:
            return []

        remaining = set(factor_signals.keys())
        selected: list[int] = []
        result: list[tuple[int, float]] = []
        current_icir = 0.0

        for _ in range(min(max_factors, len(factor_signals))):
            best_fid: int | None = None
            best_icir = current_icir
            best_delta = 0.0

            for fid in remaining:
                candidate = selected + [fid]
                icir = self._composite_icir(factor_signals, candidate, returns)
                delta = icir - current_icir
                if icir > best_icir:
                    best_fid = fid
                    best_icir = icir
                    best_delta = delta

            if best_fid is None:
                break

            selected.append(best_fid)
            remaining.discard(best_fid)
            result.append((best_fid, float(best_delta)))
            current_icir = best_icir
            logger.info(
                "Step %d: added factor %d, ICIR=%.4f (+%.4f)",
                len(selected), best_fid, current_icir, best_delta,
            )

        return result

    # ------------------------------------------------------------------
    # XGBoost importance-based selection
    # ------------------------------------------------------------------

    def xgboost_selection(
        self,
        factor_signals: dict[int, np.ndarray],
        returns: np.ndarray,
    ) -> list[tuple[int, float]]:
        """XGBoost: gradient boosting for nonlinear factor interactions.

        Paper: Best performance with ICIR=1.49, 92.6% win rate.

        Parameters
        ----------
        factor_signals : dict[int, ndarray]
            Mapping from factor ID to (T, N) signal array.
        returns : ndarray of shape (T, N)
            Forward returns.

        Returns
        -------
        list of (factor_id, importance)
            All factors sorted by gain importance (descending).
        """
        import xgboost as xgb

        ids, X, y = self._prepare_panel(factor_signals, returns)
        if len(ids) == 0:
            return []

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X, y)

        importance = model.feature_importances_  # gain-based by default
        results: list[tuple[int, float]] = [
            (ids[i], float(importance[i])) for i in range(len(ids))
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_panel(
        factor_signals: dict[int, np.ndarray],
        returns: np.ndarray,
    ) -> tuple[list[int], np.ndarray, np.ndarray]:
        """Flatten panel data to (samples, features) for sklearn-style models.

        Stacks all (T, N) arrays into (T*N, K) feature matrix and (T*N,)
        target vector, dropping rows with any NaN.

        Returns
        -------
        ids : list of int
            Factor IDs in column order.
        X : ndarray (n_samples, n_factors)
        y : ndarray (n_samples,)
        """
        if not factor_signals:
            return [], np.empty((0, 0)), np.empty(0)

        ids = sorted(factor_signals.keys())
        T, N = next(iter(factor_signals.values())).shape
        len(ids)

        # Build (T*N, K) matrix
        X = np.column_stack([
            factor_signals[fid].ravel() for fid in ids
        ])  # (T*N, K)
        y = returns.ravel()  # (T*N,)

        # Drop NaN rows
        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        return ids, X[valid], y[valid]

    @staticmethod
    def _composite_icir(
        factor_signals: dict[int, np.ndarray],
        selected_ids: list[int],
        returns: np.ndarray,
    ) -> float:
        """Compute ICIR of the equal-weight composite of selected factors.

        IC is the cross-sectional Spearman rank correlation between the
        composite signal and forward returns at each time step.  ICIR is
        mean(IC) / std(IC).

        Returns 0.0 if computation fails or std is zero.
        """
        if not selected_ids:
            return 0.0

        signals = []
        for fid in selected_ids:
            sig = factor_signals[fid].astype(np.float64)
            cs_mean = np.nanmean(sig, axis=1, keepdims=True)
            cs_std = np.nanstd(sig, axis=1, keepdims=True)
            cs_std = np.where(cs_std == 0.0, 1.0, cs_std)
            signals.append((sig - cs_mean) / cs_std)

        composite = np.nanmean(np.stack(signals, axis=0), axis=0)  # (T, N)

        T = composite.shape[0]
        ics = np.full(T, np.nan)
        for t in range(T):
            x = composite[t]
            y = returns[t]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() < 5:
                continue
            corr, _ = spearmanr(x[valid], y[valid])
            if np.isfinite(corr):
                ics[t] = corr

        finite_ics = ics[np.isfinite(ics)]
        if len(finite_ics) < 2:
            return 0.0

        ic_std = np.std(finite_ics, ddof=1)
        if ic_std < 1e-12:
            return 0.0

        return float(np.mean(finite_ics) / ic_std)
