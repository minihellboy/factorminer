"""Expression tree data structure for alpha-factor formulas.

An expression tree is a DAG of ``Node`` objects whose leaves are raw
market-data features (``LeafNode``) or numeric constants (``ConstantNode``)
and whose internal nodes are operator applications (``OperatorNode``).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np

from factorminer.core.types import (
    FEATURE_SET,
    OperatorSpec,
)

# Epsilon for safe division / log
_EPS = 1e-10


# ---------------------------------------------------------------------------
# Node hierarchy
# ---------------------------------------------------------------------------

class Node(ABC):
    """Abstract base for every node in an expression tree."""

    @abstractmethod
    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """Compute the node's value given market data.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Maps feature names (e.g. ``"$close"``) to arrays of shape
            ``(M, T)`` where *M* is the number of stocks and *T* is the
            number of time steps.

        Returns
        -------
        np.ndarray
            Result array, typically shape ``(M, T)``.
        """

    @abstractmethod
    def to_string(self) -> str:
        """Serialize the subtree rooted at this node to a DSL formula."""

    @abstractmethod
    def depth(self) -> int:
        """Return the depth of the subtree (leaf = 1)."""

    @abstractmethod
    def size(self) -> int:
        """Return the number of nodes in the subtree."""

    @abstractmethod
    def clone(self) -> Node:
        """Return a deep copy of the subtree."""

    def __repr__(self) -> str:  # pragma: no cover
        return self.to_string()

    # Iteration helpers -----------------------------------------------------

    def iter_nodes(self) -> Iterator[Node]:
        """Yield every node in the subtree (pre-order)."""
        yield self
        if isinstance(self, OperatorNode):
            for child in self.children:
                yield from child.iter_nodes()

    def leaf_features(self) -> list[str]:
        """Return sorted unique feature names referenced by this subtree."""
        feats = {n.feature_name for n in self.iter_nodes() if isinstance(n, LeafNode)}
        return sorted(feats)


class LeafNode(Node):
    """References a raw market-data column (e.g. ``$close``)."""

    __slots__ = ("feature_name",)

    def __init__(self, feature_name: str) -> None:
        if feature_name not in FEATURE_SET:
            raise ValueError(
                f"Unknown feature '{feature_name}'. "
                f"Expected one of {sorted(FEATURE_SET)}."
            )
        self.feature_name = feature_name

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        if self.feature_name not in data:
            raise KeyError(
                f"Feature '{self.feature_name}' not found in data. "
                f"Available: {sorted(data.keys())}"
            )
        return data[self.feature_name].astype(np.float64, copy=False)

    def to_string(self) -> str:
        return self.feature_name

    def depth(self) -> int:
        return 1

    def size(self) -> int:
        return 1

    def clone(self) -> LeafNode:
        return LeafNode(self.feature_name)


class ConstantNode(Node):
    """A numeric literal embedded in the expression."""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        # Infer shape from any entry in data so the constant broadcasts.
        for arr in data.values():
            return np.full_like(arr, self.value, dtype=np.float64)
        raise ValueError("Cannot evaluate ConstantNode with empty data dict.")

    def to_string(self) -> str:
        # Produce a clean numeric literal.
        if self.value == int(self.value) and abs(self.value) < 1e12:
            return str(int(self.value))
        return f"{self.value:g}"

    def depth(self) -> int:
        return 1

    def size(self) -> int:
        return 1

    def clone(self) -> ConstantNode:
        return ConstantNode(self.value)


class OperatorNode(Node):
    """An internal node that applies an operator to child sub-trees.

    Parameters
    ----------
    operator : OperatorSpec
        The operator to apply.
    children : list[Node]
        Child expression nodes.  Length must equal ``operator.arity``.
    params : dict[str, float]
        Extra numeric parameters (e.g. ``{"window": 10}``).
    """

    __slots__ = ("operator", "children", "params")

    def __init__(
        self,
        operator: OperatorSpec,
        children: list[Node],
        params: dict[str, float] | None = None,
    ) -> None:
        self.operator = operator
        self.children = list(children)
        self.params = dict(params) if params else {}
        # Merge defaults for any missing parameter.
        for pname, pdefault in operator.param_defaults.items():
            if pname not in self.params:
                self.params[pname] = pdefault

    # ---- serialization ----------------------------------------------------

    def to_string(self) -> str:
        parts = [child.to_string() for child in self.children]
        # Append explicit numeric parameters (window etc.)
        for pname in self.operator.param_names:
            if pname in self.params:
                v = self.params[pname]
                if v == int(v) and abs(v) < 1e12:
                    parts.append(str(int(v)))
                else:
                    parts.append(f"{v:g}")
        return f"{self.operator.name}({', '.join(parts)})"

    # ---- structural queries -----------------------------------------------

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def clone(self) -> OperatorNode:
        return OperatorNode(
            operator=self.operator,
            children=[c.clone() for c in self.children],
            params=dict(self.params),
        )

    # ---- evaluation -------------------------------------------------------

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        child_vals = [c.evaluate(data) for c in self.children]
        return _dispatch_operator(self.operator, child_vals, self.params)


# ---------------------------------------------------------------------------
# Operator dispatch  (pure-numpy implementations)
# ---------------------------------------------------------------------------

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Division that returns 0 where the denominator is near zero."""
    out = np.where(np.abs(b) > _EPS, a / np.where(np.abs(b) > _EPS, b, 1.0), 0.0)
    return out


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def _safe_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.sqrt(np.abs(x))


def _rolling_apply(
    x: np.ndarray,
    window: int,
    func,
    *,
    binary_y: np.ndarray | None = None,
) -> np.ndarray:
    """Apply *func* over a rolling window along the last axis (T).

    Parameters
    ----------
    x : np.ndarray, shape (M, T)
    window : int
    func : callable  (slice_x, [slice_y]) -> scalar or 1-d
    binary_y : optional second array for bivariate rolling ops

    Returns
    -------
    np.ndarray, shape (M, T)   – leading positions filled with NaN.
    """
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(window - 1, T):
        sx = x[:, t - window + 1 : t + 1]
        if binary_y is not None:
            sy = binary_y[:, t - window + 1 : t + 1]
            out[:, t] = func(sx, sy)
        else:
            out[:, t] = func(sx)
    return out


def _ts_mean(sx: np.ndarray) -> np.ndarray:
    return np.nanmean(sx, axis=1)


def _ts_std(sx: np.ndarray) -> np.ndarray:
    return np.nanstd(sx, axis=1, ddof=1)


def _ts_var(sx: np.ndarray) -> np.ndarray:
    return np.nanvar(sx, axis=1, ddof=1)


def _ts_sum(sx: np.ndarray) -> np.ndarray:
    return np.nansum(sx, axis=1)


def _ts_prod(sx: np.ndarray) -> np.ndarray:
    return np.nanprod(sx, axis=1)


def _ts_max(sx: np.ndarray) -> np.ndarray:
    return np.nanmax(sx, axis=1)


def _ts_min(sx: np.ndarray) -> np.ndarray:
    return np.nanmin(sx, axis=1)


def _ts_argmax(sx: np.ndarray) -> np.ndarray:
    return np.nanargmax(sx, axis=1).astype(np.float64)


def _ts_argmin(sx: np.ndarray) -> np.ndarray:
    return np.nanargmin(sx, axis=1).astype(np.float64)


def _ts_median(sx: np.ndarray) -> np.ndarray:
    return np.nanmedian(sx, axis=1)


def _ts_skew(sx: np.ndarray) -> np.ndarray:
    m = np.nanmean(sx, axis=1, keepdims=True)
    s = np.nanstd(sx, axis=1, keepdims=True, ddof=1)
    s = np.where(s > _EPS, s, 1.0)
    n = sx.shape[1]
    sk = np.nanmean(((sx - m) / s) ** 3, axis=1) * n**2 / max((n - 1) * (n - 2), 1)
    return sk


def _ts_kurt(sx: np.ndarray) -> np.ndarray:
    m = np.nanmean(sx, axis=1, keepdims=True)
    s = np.nanstd(sx, axis=1, keepdims=True, ddof=1)
    s = np.where(s > _EPS, s, 1.0)
    return np.nanmean(((sx - m) / s) ** 4, axis=1) - 3.0


def _ts_rank(sx: np.ndarray) -> np.ndarray:
    """Percentile rank of the latest value within the window."""
    latest = sx[:, -1]
    rank = np.sum(sx <= latest[:, None], axis=1).astype(np.float64)
    return rank / sx.shape[1]


def _ts_corr(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    mx = np.nanmean(sx, axis=1, keepdims=True)
    my = np.nanmean(sy, axis=1, keepdims=True)
    dx, dy = sx - mx, sy - my
    cov = np.nanmean(dx * dy, axis=1)
    sx_std = np.nanstd(sx, axis=1, ddof=1)
    sy_std = np.nanstd(sy, axis=1, ddof=1)
    denom = sx_std * sy_std
    return np.where(denom > _EPS, cov / denom, 0.0)


def _ts_cov(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    mx = np.nanmean(sx, axis=1, keepdims=True)
    my = np.nanmean(sy, axis=1, keepdims=True)
    return np.nanmean((sx - mx) * (sy - my), axis=1)


def _ts_beta(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """Rolling OLS slope of x on y."""
    my = np.nanmean(sy, axis=1, keepdims=True)
    mx = np.nanmean(sx, axis=1, keepdims=True)
    dy = sy - my
    var_y = np.nansum(dy ** 2, axis=1)
    cov_xy = np.nansum((sx - mx) * dy, axis=1)
    return np.where(var_y > _EPS, cov_xy / var_y, 0.0)


def _ts_resid(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    beta = _ts_beta(sx, sy)
    my = np.nanmean(sy, axis=1, keepdims=True)
    mx = np.nanmean(sx, axis=1, keepdims=True)
    predicted = mx.squeeze(1) + beta * (sy[:, -1] - my.squeeze(1))
    return sx[:, -1] - predicted


def _ema(x: np.ndarray, window: int) -> np.ndarray:
    """Exponential moving average along the last axis."""
    alpha = 2.0 / (window + 1)
    M, T = x.shape
    out = np.empty_like(x, dtype=np.float64)
    out[:, 0] = x[:, 0]
    for t in range(1, T):
        out[:, t] = alpha * x[:, t] + (1 - alpha) * out[:, t - 1]
    return out


def _wma(x: np.ndarray, window: int) -> np.ndarray:
    """Linearly-weighted moving average."""
    weights = np.arange(1, window + 1, dtype=np.float64)
    weights /= weights.sum()
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(window - 1, T):
        out[:, t] = (x[:, t - window + 1 : t + 1] * weights[None, :]).sum(axis=1)
    return out


def _decay(x: np.ndarray, window: int) -> np.ndarray:
    """Exponentially decaying sum."""
    alpha = 2.0 / (window + 1)
    weights = np.array([alpha * (1 - alpha) ** i for i in range(window)][::-1])
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(window - 1, T):
        out[:, t] = (x[:, t - window + 1 : t + 1] * weights[None, :]).sum(axis=1)
    return out


def _cs_rank(x: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank at each time step."""
    M, T = x.shape
    out = np.empty_like(x, dtype=np.float64)
    for t in range(T):
        col = x[:, t]
        valid = ~np.isnan(col)
        ranked = np.empty(M, dtype=np.float64)
        ranked[:] = np.nan
        if valid.any():
            order = col[valid].argsort().argsort().astype(np.float64)
            ranked[valid] = (order + 1) / valid.sum()
        out[:, t] = ranked
    return out


def _cs_zscore(x: np.ndarray) -> np.ndarray:
    M, T = x.shape
    out = np.empty_like(x, dtype=np.float64)
    for t in range(T):
        col = x[:, t]
        m = np.nanmean(col)
        s = np.nanstd(col, ddof=1)
        out[:, t] = (col - m) / max(s, _EPS)
    return out


def _cs_demean(x: np.ndarray) -> np.ndarray:
    m = np.nanmean(x, axis=0, keepdims=True)
    return x - m


def _cs_scale(x: np.ndarray) -> np.ndarray:
    s = np.nansum(np.abs(x), axis=0, keepdims=True)
    s = np.where(s > _EPS, s, 1.0)
    return x / s


def _ts_linreg_slope(x: np.ndarray, window: int) -> np.ndarray:
    t_vals = np.arange(window, dtype=np.float64)
    t_mean = t_vals.mean()
    t_var = np.sum((t_vals - t_mean) ** 2)
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(window - 1, T):
        sx = x[:, t - window + 1 : t + 1]
        x_mean = np.nanmean(sx, axis=1, keepdims=True)
        cov = np.nansum((sx - x_mean) * (t_vals[None, :] - t_mean), axis=1)
        out[:, t] = cov / max(t_var, _EPS)
    return out


def _ts_linreg_intercept(x: np.ndarray, window: int) -> np.ndarray:
    t_vals = np.arange(window, dtype=np.float64)
    t_mean = t_vals.mean()
    slope = _ts_linreg_slope(x, window)
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(window - 1, T):
        sx = x[:, t - window + 1 : t + 1]
        x_mean = np.nanmean(sx, axis=1)
        out[:, t] = x_mean - slope[:, t] * t_mean
    return out


def _ts_linreg_fitted(x: np.ndarray, window: int) -> np.ndarray:
    slope = _ts_linreg_slope(x, window)
    intercept = _ts_linreg_intercept(x, window)
    t_last = float(window - 1)
    return intercept + slope * t_last


def _ts_linreg_resid(x: np.ndarray, window: int) -> np.ndarray:
    fitted = _ts_linreg_fitted(x, window)
    M, T = x.shape
    out = np.full_like(x, np.nan, dtype=np.float64)
    for t in range(window - 1, T):
        out[:, t] = x[:, t] - fitted[:, t]
    return out


# Main dispatch table -------------------------------------------------------

def _dispatch_operator(
    spec: OperatorSpec,
    children: list[np.ndarray],
    params: dict[str, float],
) -> np.ndarray:
    """Execute an operator on evaluated children, return result array."""
    name = spec.name
    w = int(params.get("window", 0))

    # -- Arithmetic ---------------------------------------------------------
    if name == "Add":
        return children[0] + children[1]
    if name == "Sub":
        return children[0] - children[1]
    if name == "Mul":
        return children[0] * children[1]
    if name == "Div":
        return _safe_div(children[0], children[1])
    if name == "Neg":
        return -children[0]
    if name == "Abs":
        return np.abs(children[0])
    if name == "Sign":
        return np.sign(children[0])
    if name == "Log":
        return _safe_log(children[0])
    if name == "Sqrt":
        return _safe_sqrt(children[0])
    if name == "Square":
        return children[0] ** 2
    if name == "Pow":
        base, exp = children
        return np.sign(base) * np.abs(base) ** exp
    if name == "Max":
        return np.maximum(children[0], children[1])
    if name == "Min":
        return np.minimum(children[0], children[1])
    if name == "Clip":
        lo = params.get("lower", -3.0)
        hi = params.get("upper", 3.0)
        return np.clip(children[0], lo, hi)
    if name == "Inv":
        return _safe_div(np.ones_like(children[0]), children[0])

    # -- Statistical (rolling) ----------------------------------------------
    if name == "Mean":
        return _rolling_apply(children[0], w, _ts_mean)
    if name == "Std":
        return _rolling_apply(children[0], w, _ts_std)
    if name == "Var":
        return _rolling_apply(children[0], w, _ts_var)
    if name == "Skew":
        return _rolling_apply(children[0], w, _ts_skew)
    if name == "Kurt":
        return _rolling_apply(children[0], w, _ts_kurt)
    if name == "Median":
        return _rolling_apply(children[0], w, _ts_median)
    if name == "Sum":
        return _rolling_apply(children[0], w, _ts_sum)
    if name == "Prod":
        return _rolling_apply(children[0], w, _ts_prod)
    if name == "TsMax":
        return _rolling_apply(children[0], w, _ts_max)
    if name == "TsMin":
        return _rolling_apply(children[0], w, _ts_min)
    if name == "TsArgMax":
        return _rolling_apply(children[0], w, _ts_argmax)
    if name == "TsArgMin":
        return _rolling_apply(children[0], w, _ts_argmin)
    if name == "TsRank":
        return _rolling_apply(children[0], w, _ts_rank)
    if name == "Quantile":
        q = params.get("q", 0.5)
        return _rolling_apply(
            children[0], w, lambda sx: np.nanquantile(sx, q, axis=1)
        )
    if name == "CountNaN":
        return _rolling_apply(
            children[0], w, lambda sx: np.sum(np.isnan(sx), axis=1).astype(np.float64)
        )
    if name == "CountNotNaN":
        return _rolling_apply(
            children[0], w, lambda sx: np.sum(~np.isnan(sx), axis=1).astype(np.float64)
        )

    # -- Time-series --------------------------------------------------------
    if name == "Delta":
        M, T = children[0].shape
        out = np.full_like(children[0], np.nan, dtype=np.float64)
        if w < T:
            out[:, w:] = children[0][:, w:] - children[0][:, :-w]
        return out
    if name == "Delay":
        M, T = children[0].shape
        out = np.full_like(children[0], np.nan, dtype=np.float64)
        if w < T:
            out[:, w:] = children[0][:, :-w]
        return out
    if name == "Return":
        M, T = children[0].shape
        out = np.full_like(children[0], np.nan, dtype=np.float64)
        if w < T:
            prev = children[0][:, :-w]
            out[:, w:] = _safe_div(children[0][:, w:] - prev, prev)
        return out
    if name == "LogReturn":
        M, T = children[0].shape
        out = np.full_like(children[0], np.nan, dtype=np.float64)
        if w < T:
            ratio = _safe_div(children[0][:, w:], np.where(np.abs(children[0][:, :-w]) > _EPS, children[0][:, :-w], 1.0))
            out[:, w:] = np.log(np.abs(ratio) + _EPS)
        return out
    if name == "Corr":
        return _rolling_apply(children[0], w, _ts_corr, binary_y=children[1])
    if name == "Cov":
        return _rolling_apply(children[0], w, _ts_cov, binary_y=children[1])
    if name == "Beta":
        return _rolling_apply(children[0], w, _ts_beta, binary_y=children[1])
    if name == "Resid":
        return _rolling_apply(children[0], w, _ts_resid, binary_y=children[1])
    if name == "WMA":
        return _wma(children[0], w)
    if name == "Decay":
        return _decay(children[0], w)
    if name == "CumSum":
        return np.nancumsum(children[0], axis=1)
    if name == "CumProd":
        return np.nancumprod(children[0], axis=1)
    if name == "CumMax":
        return np.maximum.accumulate(np.nan_to_num(children[0], nan=-np.inf), axis=1)
    if name == "CumMin":
        return np.minimum.accumulate(np.nan_to_num(children[0], nan=np.inf), axis=1)

    # -- Smoothing ----------------------------------------------------------
    if name == "EMA":
        return _ema(children[0], w)
    if name == "DEMA":
        e1 = _ema(children[0], w)
        e2 = _ema(e1, w)
        return 2 * e1 - e2
    if name == "SMA":
        return _rolling_apply(children[0], w, _ts_mean)
    if name == "KAMA":
        return _ema(children[0], w)  # simplified
    if name == "HMA":
        half_w = max(w // 2, 1)
        sqrt_w = max(int(math.sqrt(w)), 1)
        wma_half = _wma(children[0], half_w)
        wma_full = _wma(children[0], w)
        # Fill leading NaN from the shorter window with the longer
        diff = 2 * np.nan_to_num(wma_half) - np.nan_to_num(wma_full)
        return _wma(diff, sqrt_w)

    # -- Cross-sectional ----------------------------------------------------
    if name == "CsRank":
        return _cs_rank(children[0])
    if name == "CsZScore":
        return _cs_zscore(children[0])
    if name == "CsDemean":
        return _cs_demean(children[0])
    if name == "CsScale":
        return _cs_scale(children[0])
    if name == "CsNeutralize":
        return _cs_demean(children[0])  # simplified: industry-neutralize ≈ demean
    if name == "CsQuantile":
        n_bins = int(params.get("n_bins", 5))
        ranked = _cs_rank(children[0])
        return np.ceil(ranked * n_bins).clip(1, n_bins)

    # -- Regression ---------------------------------------------------------
    if name == "TsLinReg":
        return _ts_linreg_fitted(children[0], w)
    if name == "TsLinRegSlope":
        return _ts_linreg_slope(children[0], w)
    if name == "TsLinRegIntercept":
        return _ts_linreg_intercept(children[0], w)
    if name == "TsLinRegResid":
        return _ts_linreg_resid(children[0], w)

    # -- Logical / conditional ----------------------------------------------
    if name == "IfElse":
        cond, x_true, x_false = children
        return np.where(cond > 0, x_true, x_false)
    if name == "Greater":
        return (children[0] > children[1]).astype(np.float64)
    if name == "Less":
        return (children[0] < children[1]).astype(np.float64)
    if name == "Equal":
        return (np.abs(children[0] - children[1]) < _EPS).astype(np.float64)
    if name == "And":
        return ((children[0] > 0) & (children[1] > 0)).astype(np.float64)
    if name == "Or":
        return ((children[0] > 0) | (children[1] > 0)).astype(np.float64)
    if name == "Not":
        return (children[0] <= 0).astype(np.float64)

    raise NotImplementedError(f"Operator '{name}' has no evaluation implementation.")


# ---------------------------------------------------------------------------
# Expression tree wrapper
# ---------------------------------------------------------------------------

class ExpressionTree:
    """Wrapper around a root ``Node`` providing a convenient API.

    Parameters
    ----------
    root : Node
        The root node of the tree.
    """

    __slots__ = ("root",)

    def __init__(self, root: Node) -> None:
        self.root = root

    def to_string(self) -> str:
        """Serialize the full tree to a DSL formula string."""
        return self.root.to_string()

    def depth(self) -> int:
        """Return the depth of the tree."""
        return self.root.depth()

    def size(self) -> int:
        """Return the total number of nodes."""
        return self.root.size()

    def evaluate(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """Execute the formula on market data.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Maps feature names to arrays of shape ``(M, T)``.

        Returns
        -------
        np.ndarray of shape ``(M, T)``
        """
        return self.root.evaluate(data)

    def clone(self) -> ExpressionTree:
        """Return a deep copy of the tree."""
        return ExpressionTree(self.root.clone())

    def leaf_features(self) -> list[str]:
        """Return sorted unique feature names referenced by this tree."""
        return self.root.leaf_features()

    def __repr__(self) -> str:
        return f"ExpressionTree({self.to_string()})"

    def __str__(self) -> str:
        return self.to_string()
