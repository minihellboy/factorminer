"""Type system for the FactorMiner operator library.

Defines operator categories, signatures, specifications, and the canonical
set of raw market-data feature names used as leaf nodes in expression trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OperatorType(Enum):
    """High-level category for every operator."""
    ARITHMETIC = auto()
    STATISTICAL = auto()
    TIMESERIES = auto()
    CROSS_SECTIONAL = auto()
    SMOOTHING = auto()
    REGRESSION = auto()
    LOGICAL = auto()
    AUTO_INVENTED = auto()


class SignatureType(Enum):
    """Describes how an operator maps inputs to outputs.

    TIME_SERIES_TO_TIME_SERIES  – rolling / lookback along the time axis
    CROSS_SECTION_TO_CROSS_SECTION – operates across stocks at each point
    ELEMENT_WISE – pointwise on array(s), no window or cross-section logic
    REDUCE_TIME – collapses the time axis (e.g. cumulative sum)
    """
    TIME_SERIES_TO_TIME_SERIES = auto()
    CROSS_SECTION_TO_CROSS_SECTION = auto()
    ELEMENT_WISE = auto()
    REDUCE_TIME = auto()


# ---------------------------------------------------------------------------
# Operator specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorSpec:
    """Immutable descriptor for a single operator in the library.

    Parameters
    ----------
    name : str
        Canonical name used in DSL strings (e.g. ``"Add"``).
    arity : int
        Number of *expression* children (1 = unary, 2 = binary, 3 = ternary).
    category : OperatorType
        Broad category of the operator.
    signature : SignatureType
        How the operator maps inputs to outputs.
    param_names : tuple[str, ...]
        Names of extra numeric parameters (e.g. ``("window",)``).
    param_defaults : dict[str, float]
        Default value for each parameter when omitted.
    param_ranges : dict[str, tuple[float, float]]
        Valid (inclusive) range for each parameter.
    description : str
        Short human-readable description.
    """
    name: str
    arity: int
    category: OperatorType
    signature: SignatureType
    param_names: Tuple[str, ...] = ()
    param_defaults: Dict[str, float] = field(default_factory=dict)
    param_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# Canonical feature names (leaf nodes)
# ---------------------------------------------------------------------------

FEATURES: List[str] = [
    "$open",
    "$high",
    "$low",
    "$close",
    "$volume",
    "$amt",
    "$vwap",
    "$returns",
]

FEATURE_SET: frozenset = frozenset(FEATURES)


# ---------------------------------------------------------------------------
# Complete operator library  (60+ operators)
# ---------------------------------------------------------------------------

def _window_params(
    default: int = 10,
    lo: int = 2,
    hi: int = 250,
) -> Tuple[Tuple[str, ...], Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Helper returning standard (window,) parameter triple."""
    return (
        ("window",),
        {"window": float(default)},
        {"window": (float(lo), float(hi))},
    )


def _build_operator_registry() -> Dict[str, OperatorSpec]:
    """Construct the full operator registry.

    Returns a mapping from canonical operator name to its ``OperatorSpec``.
    """
    registry: Dict[str, OperatorSpec] = {}

    def _reg(
        name: str,
        arity: int,
        cat: OperatorType,
        sig: SignatureType,
        param_names: Tuple[str, ...] = (),
        param_defaults: Optional[Dict[str, float]] = None,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        desc: str = "",
    ) -> None:
        registry[name] = OperatorSpec(
            name=name,
            arity=arity,
            category=cat,
            signature=sig,
            param_names=param_names,
            param_defaults=param_defaults or {},
            param_ranges=param_ranges or {},
            description=desc,
        )

    EW = SignatureType.ELEMENT_WISE
    TS = SignatureType.TIME_SERIES_TO_TIME_SERIES
    CS = SignatureType.CROSS_SECTION_TO_CROSS_SECTION
    RT = SignatureType.REDUCE_TIME

    A = OperatorType.ARITHMETIC
    S = OperatorType.STATISTICAL
    T = OperatorType.TIMESERIES
    X = OperatorType.CROSS_SECTIONAL
    SM = OperatorType.SMOOTHING
    R = OperatorType.REGRESSION
    L = OperatorType.LOGICAL

    wp = _window_params

    # ---- Arithmetic (element-wise) ----------------------------------------
    _reg("Add", 2, A, EW, desc="x + y")
    _reg("Sub", 2, A, EW, desc="x - y")
    _reg("Mul", 2, A, EW, desc="x * y")
    _reg("Div", 2, A, EW, desc="x / y (safe division)")
    _reg("Neg", 1, A, EW, desc="-x")
    _reg("Abs", 1, A, EW, desc="|x|")
    _reg("Sign", 1, A, EW, desc="sign(x)")
    _reg("Log", 1, A, EW, desc="log(1 + |x|) * sign(x)")
    _reg("Sqrt", 1, A, EW, desc="sqrt(|x|) * sign(x)")
    _reg("Square", 1, A, EW, desc="x^2")
    _reg("Pow", 2, A, EW, desc="x^y")
    _reg("Max", 2, A, EW, desc="element-wise max(x, y)")
    _reg("Min", 2, A, EW, desc="element-wise min(x, y)")
    _reg("Clip", 1, A, EW,
         param_names=("lower", "upper"),
         param_defaults={"lower": -3.0, "upper": 3.0},
         param_ranges={"lower": (-10.0, 10.0), "upper": (-10.0, 10.0)},
         desc="clip(x, lower, upper)")
    _reg("Inv", 1, A, EW, desc="1 / x (safe)")

    # ---- Statistical (rolling window) -------------------------------------
    _reg("Mean", 1, S, TS, *wp(10), desc="rolling mean")
    _reg("Std", 1, S, TS, *wp(10), desc="rolling std dev")
    _reg("Var", 1, S, TS, *wp(10), desc="rolling variance")
    _reg("Skew", 1, S, TS, *wp(20), desc="rolling skewness")
    _reg("Kurt", 1, S, TS, *wp(20), desc="rolling kurtosis")
    _reg("Median", 1, S, TS, *wp(10), desc="rolling median")
    _reg("Sum", 1, S, TS, *wp(10), desc="rolling sum")
    _reg("Prod", 1, S, TS, *wp(10), desc="rolling product")
    _reg("TsMax", 1, S, TS, *wp(10), desc="rolling max")
    _reg("TsMin", 1, S, TS, *wp(10), desc="rolling min")
    _reg("TsArgMax", 1, S, TS, *wp(10), desc="rolling argmax")
    _reg("TsArgMin", 1, S, TS, *wp(10), desc="rolling argmin")
    _reg("TsRank", 1, S, TS, *wp(10), desc="rolling rank of latest value")
    _reg("Quantile", 1, S, TS,
         param_names=("window", "q"),
         param_defaults={"window": 10.0, "q": 0.5},
         param_ranges={"window": (2.0, 250.0), "q": (0.0, 1.0)},
         desc="rolling quantile")
    _reg("CountNaN", 1, S, TS, *wp(10), desc="rolling count of NaN")
    _reg("CountNotNaN", 1, S, TS, *wp(10), desc="rolling count of non-NaN")

    # ---- Time-series operators --------------------------------------------
    _reg("Delta", 1, T, TS, *wp(1, 1, 60), desc="x[t] - x[t-d]")
    _reg("Delay", 1, T, TS, *wp(1, 1, 60), desc="x[t-d]")
    _reg("Return", 1, T, TS, *wp(1, 1, 60), desc="x[t]/x[t-d] - 1")
    _reg("LogReturn", 1, T, TS, *wp(1, 1, 60), desc="log(x[t]/x[t-d])")
    _reg("Corr", 2, T, TS, *wp(10), desc="rolling correlation")
    _reg("Cov", 2, T, TS, *wp(10), desc="rolling covariance")
    _reg("Beta", 2, T, TS, *wp(10), desc="rolling regression beta")
    _reg("Resid", 2, T, TS, *wp(10), desc="rolling regression residual")
    _reg("WMA", 1, T, TS, *wp(10), desc="weighted moving average (linear)")
    _reg("Decay", 1, T, TS, *wp(10), desc="exponentially decaying sum")
    _reg("CumSum", 1, T, RT, desc="cumulative sum along time")
    _reg("CumProd", 1, T, RT, desc="cumulative product along time")
    _reg("CumMax", 1, T, RT, desc="cumulative max along time")
    _reg("CumMin", 1, T, RT, desc="cumulative min along time")

    # ---- Smoothing --------------------------------------------------------
    _reg("EMA", 1, SM, TS, *wp(10), desc="exponential moving average")
    _reg("DEMA", 1, SM, TS, *wp(10), desc="double EMA")
    _reg("SMA", 1, SM, TS, *wp(10), desc="simple moving average")
    _reg("KAMA", 1, SM, TS, *wp(10), desc="Kaufman adaptive moving average")
    _reg("HMA", 1, SM, TS, *wp(10), desc="Hull moving average")

    # ---- Cross-sectional --------------------------------------------------
    _reg("CsRank", 1, X, CS, desc="cross-sectional rank (percentile)")
    _reg("CsZScore", 1, X, CS, desc="cross-sectional z-score")
    _reg("CsDemean", 1, X, CS, desc="x - cross-sectional mean")
    _reg("CsScale", 1, X, CS, desc="scale to unit L1 norm cross-sectionally")
    _reg("CsNeutralize", 1, X, CS, desc="industry-neutralize")
    _reg("CsQuantile", 1, X, CS,
         param_names=("n_bins",),
         param_defaults={"n_bins": 5.0},
         param_ranges={"n_bins": (2.0, 20.0)},
         desc="cross-sectional quantile bin")

    # ---- Regression -------------------------------------------------------
    _reg("TsLinReg", 1, R, TS, *wp(20), desc="rolling linear-regression fitted value")
    _reg("TsLinRegSlope", 1, R, TS, *wp(20), desc="rolling linear-regression slope")
    _reg("TsLinRegIntercept", 1, R, TS, *wp(20), desc="rolling linear-regression intercept")
    _reg("TsLinRegResid", 1, R, TS, *wp(20), desc="rolling linear-regression residual")

    # ---- Logical / conditional --------------------------------------------
    _reg("IfElse", 3, L, EW, desc="if cond > 0 then x else y")
    _reg("Greater", 2, L, EW, desc="1.0 where x > y else 0.0")
    _reg("GreaterEqual", 2, L, EW, desc="1.0 where x >= y else 0.0")
    _reg("Less", 2, L, EW, desc="1.0 where x < y else 0.0")
    _reg("LessEqual", 2, L, EW, desc="1.0 where x <= y else 0.0")
    _reg("Equal", 2, L, EW, desc="1.0 where x == y else 0.0")
    _reg("Ne", 2, L, EW, desc="1.0 where x != y else 0.0")
    _reg("And", 2, L, EW, desc="logical and")
    _reg("Or", 2, L, EW, desc="logical or")
    _reg("Not", 1, L, EW, desc="logical not")

    return registry


OPERATOR_REGISTRY: Dict[str, OperatorSpec] = _build_operator_registry()
"""Global mapping from operator name to its ``OperatorSpec``."""


def get_operator(name: str) -> OperatorSpec:
    """Look up an operator by name, raising ``KeyError`` if unknown."""
    if name not in OPERATOR_REGISTRY:
        raise KeyError(
            f"Unknown operator '{name}'. "
            f"Available: {sorted(OPERATOR_REGISTRY.keys())}"
        )
    return OPERATOR_REGISTRY[name]
