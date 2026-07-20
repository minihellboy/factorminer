"""Risk-based multi-asset portfolio construction.

Turns a matrix of per-period asset (or per-factor strategy) return series
into a risk-aware weight vector using three well-known, dependency-free
construction methods:

- :class:`HRPOptimizer` -- Hierarchical Risk Parity (Lopez de Prado, 2016):
  hierarchical clustering of the correlation-distance matrix, quasi-
  diagonalization by dendrogram leaf order, and recursive bisection with
  inverse-variance weighting down the resulting tree.
- :class:`RiskParityOptimizer` -- naive / equal risk-contribution risk
  parity via iterative fixed-point updates that pull each asset's risk
  contribution toward an equal share of total portfolio variance.
- :class:`CVaRPortfolioOptimizer` -- Rockafellar-Uryasev CVaR-budgeted
  weights, solved as a linear program with :func:`scipy.optimize.linprog`.

All three only use numpy/scipy (already project dependencies). None of
this module produces trade or execution instructions: it computes
research-facing portfolio weight vectors from historical return series,
nothing more.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.optimize import linprog
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RiskPortfolioConfig:
    """Configuration shared by the risk-based portfolio optimizers.

    Attributes
    ----------
    linkage_method : str
        Hierarchical clustering linkage method for HRP (passed through to
        ``scipy.cluster.hierarchy.linkage``). ``"single"`` is the linkage
        used in the original HRP paper.
    risk_parity_max_iter : int
        Maximum fixed-point iterations for :class:`RiskParityOptimizer`.
    risk_parity_tol : float
        Convergence tolerance (max weight change between iterations) for
        :class:`RiskParityOptimizer`.
    cvar_alpha : float
        Default CVaR confidence level (e.g. 0.95 -> worst 5% of scenarios).
    min_weight : float
        Lower bound applied to every asset weight (default: long-only,
        ``0.0``).
    ridge : float
        Small numerical-stability term added when inverting/normalizing
        near-singular covariance matrices.
    """

    linkage_method: str = "single"
    risk_parity_max_iter: int = 500
    risk_parity_tol: float = 1e-9
    cvar_alpha: float = 0.95
    min_weight: float = 0.0
    ridge: float = 1e-10


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class RiskPortfolioResult:
    """Weights plus realized risk diagnostics for one constructed portfolio."""

    method: str
    weights: np.ndarray
    asset_ids: list

    #: Realized (in-sample) portfolio return volatility, per-period.
    realized_vol: float
    #: Realized historical CVaR of portfolio returns (positive = loss magnitude).
    realized_cvar: float
    #: Concentration diagnostic: effective number of assets, 1 / sum(w**2).
    effective_n: float
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of this result."""
        return {
            "method": self.method,
            "weights": np.asarray(self.weights, dtype=np.float64).tolist(),
            "asset_ids": list(self.asset_ids),
            "realized_vol": self.realized_vol,
            "realized_cvar": self.realized_cvar,
            "effective_n": self.effective_n,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_returns(returns: np.ndarray) -> np.ndarray:
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError(f"returns must be 2-D (T, N); got shape {returns.shape}")
    if returns.shape[0] < 2:
        raise ValueError("returns must have at least 2 periods (T >= 2)")
    if not np.all(np.isfinite(returns)):
        raise ValueError("returns must not contain NaN/inf")
    return returns


def _cov_to_corr(cov: np.ndarray, ridge: float) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix, clipped to [-1, 1]."""
    std = np.sqrt(np.clip(np.diag(cov), ridge, None))
    corr = cov / np.outer(std, std)
    return np.clip(corr, -1.0, 1.0)


def effective_n(weights: np.ndarray) -> float:
    """Effective number of assets in a portfolio: ``1 / sum(w ** 2)``.

    A concentration diagnostic. Equal-weighting N assets gives an
    effective N of exactly N; concentrating all weight in one asset gives
    an effective N of 1.
    """
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights**2))
    return 1.0 / denom if denom > 0 else 0.0


def historical_cvar(portfolio_returns: np.ndarray, alpha: float = 0.95) -> float:
    """Historical (empirical) CVaR of a portfolio return series.

    Parameters
    ----------
    portfolio_returns : np.ndarray, shape (T,)
        Realized per-period portfolio returns.
    alpha : float
        Confidence level; CVaR averages the worst ``1 - alpha`` fraction
        of losses.

    Returns
    -------
    float
        CVaR expressed as a positive loss magnitude (higher = worse).
    """
    losses = -np.asarray(portfolio_returns, dtype=np.float64)
    t = losses.shape[0]
    k = max(1, int(np.ceil((1.0 - alpha) * t)))
    worst = np.sort(losses)[::-1][:k]
    return float(np.mean(worst))


def naive_inverse_variance_weights(returns: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Naive inverse-variance weights, ignoring cross-asset correlation.

    Used as the textbook baseline that HRP is meant to improve on:
    ``w_i = (1 / var_i) / sum_j (1 / var_j)``.
    """
    returns = _validate_returns(returns)
    variances = np.clip(np.var(returns, axis=0, ddof=1), ridge, None)
    inv = 1.0 / variances
    return inv / inv.sum()


def build_result(
    method: str,
    weights: np.ndarray,
    returns: np.ndarray,
    asset_ids: list | None = None,
    config: RiskPortfolioConfig | None = None,
    details: dict | None = None,
) -> RiskPortfolioResult:
    """Build a :class:`RiskPortfolioResult` from weights and realized returns."""
    config = config or RiskPortfolioConfig()
    weights = np.asarray(weights, dtype=np.float64)
    returns = _validate_returns(returns)
    portfolio_returns = returns @ weights

    realized_vol = (
        float(np.std(portfolio_returns, ddof=1)) if portfolio_returns.shape[0] > 1 else 0.0
    )
    realized_cvar = historical_cvar(portfolio_returns, config.cvar_alpha)

    ids = list(asset_ids) if asset_ids is not None else list(range(weights.shape[0]))
    return RiskPortfolioResult(
        method=method,
        weights=weights,
        asset_ids=ids,
        realized_vol=realized_vol,
        realized_cvar=realized_cvar,
        effective_n=effective_n(weights),
        details=details or {},
    )


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity
# ---------------------------------------------------------------------------


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    """Variance of the inverse-variance portfolio restricted to `items`."""
    sub_cov = cov[np.ix_(items, items)]
    diag = np.clip(np.diag(sub_cov), 1e-16, None)
    ivp = 1.0 / diag
    ivp /= ivp.sum()
    return float(ivp @ sub_cov @ ivp)


def _hrp_recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    """Standard HRP recursive-bisection weighting given a quasi-diagonal order."""
    n = cov.shape[0]
    weights = np.ones(n, dtype=np.float64)
    clusters = [list(sort_ix)]

    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            denom = var_left + var_right
            # Split risk budget inversely to each side's cluster variance.
            alpha = 0.5 if denom <= 0 else 1.0 - var_left / denom
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters.append(left)
            next_clusters.append(right)
        clusters = next_clusters

    weights /= weights.sum()
    return weights


class HRPOptimizer:
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    Builds portfolio weights in three steps:

    1. **Tree clustering**: hierarchical clustering of assets using the
       HRP correlation distance ``d_ij = sqrt(0.5 * (1 - rho_ij))``.
    2. **Quasi-diagonalization**: reorder assets by dendrogram leaf order
       so similar (highly-correlated) assets sit next to each other.
    3. **Recursive bisection**: walk down the dendrogram, splitting the
       risk budget between the two sub-clusters at each split inversely
       to their inverse-variance-portfolio variance, until every leaf has
       a weight.

    Unlike naive inverse-variance weighting, HRP allocates risk budget
    hierarchically across the correlation structure, which typically
    avoids concentrating weight inside a single tight correlation cluster.
    """

    def __init__(self, config: RiskPortfolioConfig | None = None) -> None:
        self.config = config or RiskPortfolioConfig()

    def optimize(self, returns: np.ndarray) -> np.ndarray:
        """Compute HRP weights for a panel of asset return series.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)
            Per-period return series for N assets.

        Returns
        -------
        np.ndarray, shape (N,)
            Non-negative weights, summing to 1, in the original asset
            order (index ``i`` in the input columns).
        """
        returns = _validate_returns(returns)
        n = returns.shape[1]
        if n == 1:
            return np.array([1.0])

        cov = np.atleast_2d(np.cov(returns, rowvar=False))
        corr = _cov_to_corr(cov, self.config.ridge)

        dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
        np.fill_diagonal(dist, 0.0)
        # Symmetrize away floating-point asymmetry before condensing.
        dist = (dist + dist.T) / 2.0
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method=self.config.linkage_method)
        sort_ix = dendrogram(link, no_plot=True)["leaves"]

        return _hrp_recursive_bisection(cov, sort_ix)


# ---------------------------------------------------------------------------
# Naive Risk Parity (Equal Risk Contribution)
# ---------------------------------------------------------------------------


class RiskParityOptimizer:
    """Naive / equal risk-contribution risk parity.

    Finds weights such that every asset contributes an equal share of
    total portfolio variance: ``RC_i = w_i * (Cov @ w)_i`` is equalized
    across assets. Solved via iterative multiplicative fixed-point
    updates that shrink the objective ``sum_i (RC_i / totalvar - 1/N)**2``
    (a Newton-style step toward the equal-contribution target at each
    iteration), then re-projected onto the simplex (``w >= 0``,
    ``sum(w) = 1``).
    """

    def __init__(self, config: RiskPortfolioConfig | None = None) -> None:
        self.config = config or RiskPortfolioConfig()

    def optimize(self, returns: np.ndarray) -> np.ndarray:
        """Compute equal risk-contribution weights.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)
            Per-period return series for N assets.

        Returns
        -------
        np.ndarray, shape (N,)
            Non-negative weights, summing to 1, with approximately equal
            risk contributions per asset.
        """
        returns = _validate_returns(returns)
        n = returns.shape[1]
        if n == 1:
            return np.array([1.0])

        cov = np.atleast_2d(np.cov(returns, rowvar=False))
        cov = cov + self.config.ridge * np.eye(n)
        target = np.full(n, 1.0 / n)
        w = np.full(n, 1.0 / n)

        for _ in range(self.config.risk_parity_max_iter):
            port_var = float(w @ cov @ w)
            if port_var <= 0:
                break
            marginal = cov @ w
            rc_frac = (w * marginal) / port_var
            rc_frac = np.clip(rc_frac, 1e-12, None)
            # Multiplicative fixed-point step: shrink weights whose risk
            # contribution exceeds target, grow those below it.
            w_new = w * np.sqrt(target / rc_frac)
            w_new = np.clip(w_new, 0.0, None)
            total = w_new.sum()
            if total <= 0:
                break
            w_new = w_new / total
            if np.max(np.abs(w_new - w)) < self.config.risk_parity_tol:
                w = w_new
                break
            w = w_new

        return w

    def risk_contributions(self, returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Fractional risk contribution of each asset for the given weights."""
        returns = _validate_returns(returns)
        cov = np.atleast_2d(np.cov(returns, rowvar=False))
        weights = np.asarray(weights, dtype=np.float64)
        port_var = float(weights @ cov @ weights)
        if port_var <= 0:
            return np.zeros_like(weights)
        return (weights * (cov @ weights)) / port_var


# ---------------------------------------------------------------------------
# CVaR-optimal portfolio (Rockafellar-Uryasev linear program)
# ---------------------------------------------------------------------------


class CVaRPortfolioOptimizer:
    """CVaR-budgeted portfolio weights (Rockafellar & Uryasev, 2000).

    Minimizes the historical (scenario-based) Conditional Value-at-Risk of
    portfolio losses, subject to a fully-invested long-only budget and an
    optional minimum expected-return constraint, formulated as a linear
    program:

    .. math::

        \\min_{w, \\zeta, u} \\; \\zeta + \\frac{1}{(1-\\alpha) T} \\sum_t u_t

        \\text{s.t.} \\quad u_t \\ge -r_t^\\top w - \\zeta, \\quad u_t \\ge 0,
        \\quad \\sum_i w_i = 1, \\quad w \\ge 0

    where :math:`r_t` is the return scenario at period ``t``. Solved with
    :func:`scipy.optimize.linprog` (no external LP/QP dependency).
    """

    def __init__(self, config: RiskPortfolioConfig | None = None) -> None:
        self.config = config or RiskPortfolioConfig()

    def optimize(
        self,
        returns: np.ndarray,
        alpha: float = 0.95,
        target_return: float | None = None,
    ) -> np.ndarray:
        """Compute CVaR-optimal weights via linear programming.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)
            Per-period return scenarios for N assets.
        alpha : float
            CVaR confidence level (e.g. 0.95 minimizes average loss in the
            worst 5% of scenarios).
        target_return : float, optional
            If given, requires the portfolio's in-sample mean return to be
            at least this value.

        Returns
        -------
        np.ndarray, shape (N,)
            Non-negative weights summing to 1.
        """
        returns = _validate_returns(returns)
        t, n = returns.shape
        if n == 1:
            return np.array([1.0])
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

        # Decision vector: [w_1..w_N, zeta, u_1..u_T]
        n_vars = n + 1 + t
        c = np.zeros(n_vars)
        c[n] = 1.0  # zeta
        c[n + 1 :] = 1.0 / ((1.0 - alpha) * t)  # mean excess-loss term

        # CVaR scenario constraints: -r_t . w - zeta - u_t <= 0
        a_ub_rows = [np.zeros(n_vars) for _ in range(t)]
        b_ub_rows = [0.0] * t
        for row_t in range(t):
            row = a_ub_rows[row_t]
            row[:n] = -returns[row_t, :]
            row[n] = -1.0
            row[n + 1 + row_t] = -1.0

        if target_return is not None:
            target_row = np.zeros(n_vars)
            target_row[:n] = -np.mean(returns, axis=0)
            a_ub_rows.append(target_row)
            b_ub_rows.append(-float(target_return))

        a_ub = np.vstack(a_ub_rows)
        b_ub = np.asarray(b_ub_rows, dtype=np.float64)

        a_eq = np.zeros((1, n_vars))
        a_eq[0, :n] = 1.0
        b_eq = np.array([1.0])

        bounds = (
            [(self.config.min_weight, None)] * n
            + [(None, None)]
            + [(0.0, None)] * t
        )

        result = linprog(
            c,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if not result.success:
            logger.warning(
                "CVaR LP did not converge (%s); falling back to equal weights.",
                result.message,
            )
            return np.full(n, 1.0 / n)

        weights = np.clip(result.x[:n], 0.0, None)
        total = weights.sum()
        return weights / total if total > 0 else np.full(n, 1.0 / n)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

_METHOD_LABELS = {
    "hrp": "Hierarchical Risk Parity",
    "risk_parity": "Naive Risk Parity",
    "cvar": "CVaR-Optimal",
}


def construct_portfolio(
    returns: np.ndarray,
    method: str,
    asset_ids: list | None = None,
    config: RiskPortfolioConfig | None = None,
    alpha: float | None = None,
    target_return: float | None = None,
) -> RiskPortfolioResult:
    """Construct risk-based portfolio weights and diagnostics in one call.

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Per-period return series for N assets (or asset-level return
        proxies, e.g. per-factor long-short strategy returns).
    method : {"hrp", "risk_parity", "cvar"}
        Construction method.
    asset_ids : list, optional
        Labels for each column of `returns`, used in the returned result.
    config : RiskPortfolioConfig, optional
        Shared optimizer configuration.
    alpha : float, optional
        CVaR confidence level; only used when ``method == "cvar"``.
    target_return : float, optional
        Minimum expected return constraint; only used when
        ``method == "cvar"``.

    Returns
    -------
    RiskPortfolioResult
    """
    config = config or RiskPortfolioConfig()
    if method not in _METHOD_LABELS:
        raise ValueError(f"Unknown method '{method}'; expected one of {sorted(_METHOD_LABELS)}")

    if method == "hrp":
        weights = HRPOptimizer(config).optimize(returns)
    elif method == "risk_parity":
        weights = RiskParityOptimizer(config).optimize(returns)
    else:
        cvar_alpha = alpha if alpha is not None else config.cvar_alpha
        weights = CVaRPortfolioOptimizer(config).optimize(
            returns, alpha=cvar_alpha, target_return=target_return
        )

    return build_result(
        method=_METHOD_LABELS[method],
        weights=weights,
        returns=returns,
        asset_ids=asset_ids,
        config=config,
    )
