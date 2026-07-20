"""Type system for the FactorMiner operator library.

Defines operator categories, signatures, specifications, and the canonical
set of raw market-data feature names used as leaf nodes in expression trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

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
    param_names: tuple[str, ...] = ()
    param_defaults: dict[str, float] = field(default_factory=dict)
    param_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# Canonical feature names (leaf nodes)
# ---------------------------------------------------------------------------
# The eight paper OHLCV leaves are the immutable default set. Extra numeric
# leaves (fundamentals, futures basis, ...) are registered at runtime via
# :func:`register_features` so loaders, the parser, and prompt builders share
# one composable registry instead of hard-coded parallel lists.

DEFAULT_FEATURES: list[str] = [
    "$open",
    "$high",
    "$low",
    "$close",
    "$volume",
    "$amt",
    "$vwap",
    "$returns",
]

# Live registry (defaults first). Mutated only through register_features /
# unregister_features / reset_features — never by assigning FEATURES = ...
FEATURES: list[str] = list(DEFAULT_FEATURES)

FEATURE_SET: frozenset[str] = frozenset(FEATURES)

#: Human-readable descriptions surfaced in LLM generation prompts.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "$open": "opening price",
    "$high": "highest price in the bar",
    "$low": "lowest price in the bar",
    "$close": "closing price",
    "$volume": "trading volume (shares/contracts)",
    "$amt": "trading amount (currency value / notional)",
    "$vwap": "volume-weighted average price",
    "$returns": "close-to-close returns",
    # Fundamentals (EDGAR XBRL) — registered when that lane is active
    "$eps": "earnings per share (point-in-time, as-filed)",
    "$revenue": "total revenue (point-in-time, as-filed)",
    "$book_equity": "book equity / stockholders' equity (point-in-time, as-filed)",
    "$shares_out": "shares outstanding (point-in-time, as-filed)",
    # Futures continuous-contract leaves
    "$basis": "futures basis (futures price minus spot)",
    "$spot": "underlying spot price aligned to the futures bar",
    "$premium": "futures premium over spot (futures/spot - 1)",
    "$roll_yield": "roll yield implied by the continuous-contract adjustment",
    "$oi": "open interest (contracts outstanding)",
}

# DataFrame column name special-cases (DSL leaf -> panel column).
_FEATURE_COLUMN_ALIASES: dict[str, str] = {
    "$amt": "amount",
    "amt": "amount",
    "$amount": "amount",
}
_COLUMN_FEATURE_ALIASES: dict[str, str] = {
    "amount": "$amt",
    "amt": "$amt",
}


def normalize_feature_name(name: str) -> str:
    """Return the canonical ``$``-prefixed DSL leaf name for *name*."""
    raw = str(name).strip()
    if not raw:
        raise ValueError("feature name must be non-empty")
    if raw in _COLUMN_FEATURE_ALIASES:
        return _COLUMN_FEATURE_ALIASES[raw]
    if raw.startswith("$"):
        return raw
    if raw == "amount":
        return "$amt"
    return f"${raw}"


def feature_to_column(name: str) -> str:
    """Map a DSL leaf name (or bare column) to the panel DataFrame column."""
    raw = str(name).strip()
    if raw in _FEATURE_COLUMN_ALIASES:
        return _FEATURE_COLUMN_ALIASES[raw]
    if raw.startswith("$"):
        key = raw
        if key in _FEATURE_COLUMN_ALIASES:
            return _FEATURE_COLUMN_ALIASES[key]
        return key[1:]
    if raw == "amt":
        return "amount"
    return raw


def column_to_feature(column: str) -> str:
    """Map a panel DataFrame column name to the DSL leaf name."""
    raw = str(column).strip()
    if raw in _COLUMN_FEATURE_ALIASES:
        return _COLUMN_FEATURE_ALIASES[raw]
    if raw.startswith("$"):
        return normalize_feature_name(raw)
    return f"${raw}"


def get_features() -> list[str]:
    """Return a copy of the currently registered DSL feature leaves."""
    return list(FEATURES)


def get_feature_set() -> frozenset[str]:
    """Return the currently registered feature set (live view)."""
    return frozenset(FEATURES)


def get_feature_descriptions(
    features: list[str] | None = None,
) -> dict[str, str]:
    """Return descriptions for *features* (default: all registered)."""
    names = features if features is not None else get_features()
    return {name: FEATURE_DESCRIPTIONS.get(name, "") for name in names}


def register_features(
    extra_features: list[str] | tuple[str, ...] | None = None,
    *,
    descriptions: dict[str, str] | None = None,
    reset: bool = False,
) -> list[str]:
    """Register additional numeric leaf features on the global registry.

    Parameters
    ----------
    extra_features :
        DSL names (``$eps``) or bare column names (``eps``). Order is
        preserved; duplicates of already-registered names are skipped.
    descriptions :
        Optional human-readable descriptions merged into
        :data:`FEATURE_DESCRIPTIONS` for prompt rendering.
    reset :
        When True, restore the default OHLCV set before applying *extra_features*.

    Returns
    -------
    list[str]
        The full active feature list after registration.
    """
    global FEATURE_SET

    if reset:
        FEATURES[:] = list(DEFAULT_FEATURES)

    if descriptions:
        for key, value in descriptions.items():
            FEATURE_DESCRIPTIONS[normalize_feature_name(key)] = str(value)

    for raw in extra_features or ():
        name = normalize_feature_name(raw)
        if name not in FEATURES:
            FEATURES.append(name)

    FEATURE_SET = frozenset(FEATURES)
    return get_features()


def unregister_features(
    features: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Remove previously registered extra features (defaults are kept)."""
    global FEATURE_SET
    protected = set(DEFAULT_FEATURES)
    drop = {normalize_feature_name(name) for name in (features or ())}
    FEATURES[:] = [name for name in FEATURES if name in protected or name not in drop]
    FEATURE_SET = frozenset(FEATURES)
    return get_features()


def reset_features() -> list[str]:
    """Restore the registry to the eight default OHLCV leaves."""
    return register_features(None, reset=True)


# ---------------------------------------------------------------------------
# Complete operator library  (60+ operators)
# ---------------------------------------------------------------------------

def _window_params(
    default: int = 10,
    lo: int = 2,
    hi: int = 250,
) -> tuple[tuple[str, ...], dict[str, float], dict[str, tuple[float, float]]]:
    """Helper returning standard (window,) parameter triple."""
    return (
        ("window",),
        {"window": float(default)},
        {"window": (float(lo), float(hi))},
    )


def _build_operator_registry() -> dict[str, OperatorSpec]:
    """Construct the full operator registry.

    Returns a mapping from canonical operator name to its ``OperatorSpec``.
    """
    registry: dict[str, OperatorSpec] = {}

    def _reg(
        name: str,
        arity: int,
        cat: OperatorType,
        sig: SignatureType,
        param_names: tuple[str, ...] = (),
        param_defaults: dict[str, float] | None = None,
        param_ranges: dict[str, tuple[float, float]] | None = None,
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
    _reg("SignedPower", 1, A, EW,
         param_names=("exponent",),
         param_defaults={"exponent": 2.0},
         param_ranges={"exponent": (0.0, 10.0)},
         desc="paper alias: sign(x) * |x|^exponent")
    _reg("Power", 1, A, EW,
         param_names=("exponent",),
         param_defaults={"exponent": 2.0},
         param_ranges={"exponent": (0.0, 10.0)},
         desc="paper operator: x^exponent")
    _reg("Exp", 1, A, EW, desc="exp(x), clipped for numerical safety")
    _reg("Tanh", 1, A, EW, desc="tanh(x)")
    _reg("Max", 2, A, EW, desc="element-wise max(x, y)")
    _reg("Min", 2, A, EW, desc="element-wise min(x, y)")
    _reg("Max2", 2, A, EW, desc="paper alias: element-wise max(x, y)")
    _reg("Min2", 2, A, EW, desc="paper alias: element-wise min(x, y)")
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
    _reg("Med", 1, S, TS, *wp(10), desc="paper alias: rolling median")
    _reg("Sum", 1, S, TS, *wp(10), desc="rolling sum")
    _reg("Prod", 1, S, TS, *wp(10), desc="rolling product")
    _reg("Product", 1, S, TS, *wp(10), desc="paper alias: rolling product")
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
    _reg("TsDecay", 1, T, TS, *wp(10), desc="paper alias: time-series decay")
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
    _reg("Scale", 1, X, CS, desc="paper alias: cross-sectional scale")
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
    _reg("Slope", 1, R, TS, *wp(20), desc="paper alias: rolling linear-regression slope")
    _reg("Rsquare", 1, R, TS, *wp(20), desc="paper operator: rolling linear-regression R-squared")
    _reg("Resi", 1, R, TS, *wp(20), desc="paper alias: rolling linear-regression residual")

    # ---- Logical / conditional --------------------------------------------
    _reg("IfElse", 3, L, EW, desc="if cond > 0 then x else y")
    _reg("Greater", 2, L, EW, desc="1.0 where x > y else 0.0")
    _reg("GreaterEqual", 2, L, EW, desc="1.0 where x >= y else 0.0")
    _reg("Less", 2, L, EW, desc="1.0 where x < y else 0.0")
    _reg("LessEqual", 2, L, EW, desc="1.0 where x <= y else 0.0")
    _reg("Equal", 2, L, EW, desc="1.0 where x == y else 0.0")
    _reg("Eq", 2, L, EW, desc="paper alias: 1.0 where x == y else 0.0")
    _reg("Ne", 2, L, EW, desc="1.0 where x != y else 0.0")
    _reg("And", 2, L, EW, desc="logical and")
    _reg("Or", 2, L, EW, desc="logical or")
    _reg("Not", 1, L, EW, desc="logical not")

    return registry


OPERATOR_REGISTRY: dict[str, OperatorSpec] = _build_operator_registry()
"""Global mapping from operator name to its ``OperatorSpec``."""


def get_operator(name: str) -> OperatorSpec:
    """Look up an operator by name, raising ``KeyError`` if unknown."""
    if name not in OPERATOR_REGISTRY:
        raise KeyError(
            f"Unknown operator '{name}'. "
            f"Available: {sorted(OPERATOR_REGISTRY.keys())}"
        )
    return OPERATOR_REGISTRY[name]
