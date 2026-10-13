"""Reference factor sets expressed in the FactorMiner DSL.

``ALPHA158`` transcribes Qlib's ``Alpha158`` feature set (``Alpha158DL``
default configuration: 9 k-bar features, 4 price ratios, and 29 rolling
families over windows 5, 10, 20, 30 and 60) so it can serve as the reference
basis for marginal-value admission, and as a baseline, without a Qlib
installation. Operators follow FactorMiner semantics: ``Div`` is a safe
division (zero where the denominator vanishes), ``Std`` uses ``ddof=1``,
``TsRank`` counts ties as below, ``TsArgMax`` is zero-based and ``Corr``
scales the covariance with ``ddof=0``. These differ from Qlib by constants or
tie conventions that leave cross-sectional ranks, which is how a reference
basis is consumed, nearly unchanged; ``scripts/qlib_cn/alpha158_conformance.py``
measures the agreement with Qlib's own values.
"""

from __future__ import annotations

ALPHA158_WINDOWS: tuple[int, ...] = (5, 10, 20, 30, 60)

_KBAR: tuple[tuple[str, str], ...] = (
    ("KMID", "Div(Sub($close, $open), $open)"),
    ("KLEN", "Div(Sub($high, $low), $open)"),
    ("KMID2", "Div(Sub($close, $open), Add(Sub($high, $low), 1e-12))"),
    ("KUP", "Div(Sub($high, Max($open, $close)), $open)"),
    ("KUP2", "Div(Sub($high, Max($open, $close)), Add(Sub($high, $low), 1e-12))"),
    ("KLOW", "Div(Sub(Min($open, $close), $low), $open)"),
    ("KLOW2", "Div(Sub(Min($open, $close), $low), Add(Sub($high, $low), 1e-12))"),
    ("KSFT", "Div(Sub(Sub(Mul(2, $close), $high), $low), $open)"),
    ("KSFT2", "Div(Sub(Sub(Mul(2, $close), $high), $low), Add(Sub($high, $low), 1e-12))"),
)

_PRICE: tuple[tuple[str, str], ...] = (
    ("OPEN0", "Div($open, $close)"),
    ("HIGH0", "Div($high, $close)"),
    ("LOW0", "Div($low, $close)"),
    ("VWAP0", "Div($vwap, $close)"),
)

_UP = "Max(Sub($close, Delay($close, 1)), 0)"
_DOWN = "Max(Sub(Delay($close, 1), $close), 0)"
_MOVE = "Abs(Sub($close, Delay($close, 1)))"
_VUP = "Max(Sub($volume, Delay($volume, 1)), 0)"
_VDOWN = "Max(Sub(Delay($volume, 1), $volume), 0)"
_VMOVE = "Abs(Sub($volume, Delay($volume, 1)))"
_WEIGHTED = "Mul(Abs(Sub(Div($close, Delay($close, 1)), 1)), $volume)"

# Qlib family name -> template; "{d}" is the window.
_ROLLING: tuple[tuple[str, str], ...] = (
    ("ROC", "Div(Delay($close, {d}), $close)"),
    ("MA", "Div(Mean($close, {d}), $close)"),
    ("STD", "Div(Std($close, {d}), $close)"),
    ("BETA", "Div(TsLinRegSlope($close, {d}), $close)"),
    ("RSQR", "Rsquare($close, {d})"),
    ("RESI", "Div(TsLinRegResid($close, {d}), $close)"),
    ("MAX", "Div(TsMax($high, {d}), $close)"),
    ("MIN", "Div(TsMin($low, {d}), $close)"),
    ("QTLU", "Div(Quantile($close, {d}, 0.8), $close)"),
    ("QTLD", "Div(Quantile($close, {d}, 0.2), $close)"),
    ("RANK", "TsRank($close, {d})"),
    ("RSV", "Div(Sub($close, TsMin($low, {d})), Add(Sub(TsMax($high, {d}), TsMin($low, {d})), 1e-12))"),
    ("IMAX", "Div(TsArgMax($high, {d}), {d})"),
    ("IMIN", "Div(TsArgMin($low, {d}), {d})"),
    ("IMXD", "Div(Sub(TsArgMax($high, {d}), TsArgMin($low, {d})), {d})"),
    ("CORR", "Corr($close, Log($volume), {d})"),
    ("CORD", "Corr(Div($close, Delay($close, 1)), Log(Div($volume, Delay($volume, 1))), {d})"),
    ("CNTP", "Mean(Greater($close, Delay($close, 1)), {d})"),
    ("CNTN", "Mean(Less($close, Delay($close, 1)), {d})"),
    ("CNTD", "Sub(Mean(Greater($close, Delay($close, 1)), {d}), Mean(Less($close, Delay($close, 1)), {d}))"),
    ("SUMP", f"Div(Sum({_UP}, {{d}}), Add(Sum({_MOVE}, {{d}}), 1e-12))"),
    ("SUMN", f"Div(Sum({_DOWN}, {{d}}), Add(Sum({_MOVE}, {{d}}), 1e-12))"),
    ("SUMD", f"Div(Sub(Sum({_UP}, {{d}}), Sum({_DOWN}, {{d}})), Add(Sum({_MOVE}, {{d}}), 1e-12))"),
    ("VMA", "Div(Mean($volume, {d}), Add($volume, 1e-12))"),
    ("VSTD", "Div(Std($volume, {d}), Add($volume, 1e-12))"),
    ("WVMA", f"Div(Std({_WEIGHTED}, {{d}}), Add(Mean({_WEIGHTED}, {{d}}), 1e-12))"),
    ("VSUMP", f"Div(Sum({_VUP}, {{d}}), Add(Sum({_VMOVE}, {{d}}), 1e-12))"),
    ("VSUMN", f"Div(Sum({_VDOWN}, {{d}}), Add(Sum({_VMOVE}, {{d}}), 1e-12))"),
    ("VSUMD", f"Div(Sub(Sum({_VUP}, {{d}}), Sum({_VDOWN}, {{d}})), Add(Sum({_VMOVE}, {{d}}), 1e-12))"),
)


def _alpha158() -> tuple[tuple[str, str], ...]:
    rolling = tuple(
        (f"{family}{window}", template.format(d=window))
        for family, template in _ROLLING
        for window in ALPHA158_WINDOWS
    )
    return _KBAR + _PRICE + rolling


ALPHA158: tuple[tuple[str, str], ...] = _alpha158()

REFERENCE_CATALOGS: dict[str, tuple[tuple[str, str], ...]] = {"alpha158": ALPHA158}


def reference_catalog(name: str) -> tuple[tuple[str, str], ...]:
    """``(name, formula)`` pairs of a reference catalog."""
    try:
        return REFERENCE_CATALOGS[name]
    except KeyError:
        raise KeyError(
            f"Unknown reference catalog {name!r}; available: {sorted(REFERENCE_CATALOGS)}"
        ) from None
