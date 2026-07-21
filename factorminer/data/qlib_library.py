"""Best-effort FactorMiner-to-Qlib expression export adapter."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from factorminer.core.expression_tree import ConstantNode, LeafNode, Node, OperatorNode
from factorminer.core.factor_library import FactorLibrary
from factorminer.core.parser import try_parse

logger = logging.getLogger(__name__)

# Qlib Expression export (best-effort, honest translation)
# ======================================================================

# Maps FactorMiner DSL operator names to the equivalent Qlib ``Expression``
# operator name (see ``qlib/data/ops.py`` in microsoft/qlib). Only entries
# with a verified 1:1 positional-argument equivalent are listed here -- this
# is a best-effort translator, never a silently-wrong one. A handful of
# FactorMiner operators intentionally have *no* entry because Qlib has no
# matching Expression operator (e.g. ``Sqrt``, ``Square``, ``SignedPower``,
# ``CumSum``, the ``Cs*`` cross-sectional family, which Qlib instead handles
# via dataset processors rather than Expression operators). Formulas that use
# an unmapped operator are still emitted, just flagged as not translatable.
#
# Several FactorMiner/Qlib operators are deliberately *not* mapped despite
# sharing a name or an obvious-looking counterpart, because their semantics
# diverge in ways that would make the translation silently wrong rather
# than merely unsupported:
#   - ``TsRank`` -> Qlib's ``Rank`` uses a different percentile convention
#     (FactorMiner excludes the current bar from the rank denominator;
#     Qlib's rolling ``Rank(pct=True)`` includes it), so the two disagree
#     on every window.
#   - ``TsArgMax``/``TsArgMin`` -> Qlib's ``IdxMax``/``IdxMin`` are 1-based;
#     FactorMiner's are 0-based -- every translated value would be off by
#     exactly one.
#   - ``Log`` -> FactorMiner's is a safe, sign-preserving
#     ``sign(x) * log1p(|x|)``; Qlib's ``Log`` is a raw ``np.log`` (NaN on
#     non-positive input). They diverge on every negative or near-zero
#     value.
#   - ``And``/``Or``/``Not`` -> FactorMiner treats these as logical ops on
#     0/1-valued float arrays; Qlib's are bitwise (``np.bitwise_and`` etc.),
#     which do not apply cleanly to float alpha panels.
#   - ``IfElse`` -> Qlib's ``If`` selects via raw Python/NumPy truthiness
#     (``np.where(cond, ...)``), so NaN and negative condition values are
#     treated as *true*; FactorMiner's ``IfElse`` treats only strictly
#     positive values as true and propagates NaN. They disagree whenever
#     the condition isn't a clean 0/1.
# These operators are still emitted under their original FactorMiner name
# (so the output is visibly non-Qlib) and recorded in ``unsupported``,
# consistent with this translator's policy: never silently wrong.
#
# Two FactorMiner/Qlib name collisions are intentionally *not* a 1:1 name
# copy because the semantics differ:
#   - FactorMiner's element-wise ``Max``/``Max2`` (max(x, y)) maps to Qlib's
#     ``Greater`` (Qlib's ``Max`` is a *rolling* window max, matching
#     FactorMiner's ``TsMax`` instead).
#   - FactorMiner's element-wise ``Min``/``Min2`` maps to Qlib's ``Less``.
#   - FactorMiner's comparison ops (``Greater``, ``GreaterEqual``, ``Less``,
#     ``LessEqual``) map to Qlib's boolean-comparison ops (``Gt``, ``Ge``,
#     ``Lt``, ``Le``) -- Qlib's own ``Greater``/``Less`` names mean
#     element-wise max/min, not comparison.
_FACTORMINER_TO_QLIB_OPERATOR_MAP: dict[str, str] = {
    # Rolling statistics: Qlib signature is (feature, N), same arg order.
    "Mean": "Mean",
    "Std": "Std",
    "Var": "Var",
    "Skew": "Skew",
    "Kurt": "Kurt",
    "Median": "Med",
    "Med": "Med",
    "Sum": "Sum",
    "TsMax": "Max",
    "TsMin": "Min",
    "CountNotNaN": "Count",
    "Quantile": "Quantile",  # Qlib signature (feature, N, qscore) -- same order.
    # Time-series.
    "Delta": "Delta",
    "Delay": "Ref",
    "WMA": "WMA",
    "EMA": "EMA",
    # Pairwise rolling: Qlib signature is (feature_left, feature_right, N).
    "Corr": "Corr",
    "Cov": "Cov",
    # Regression.
    "Slope": "Slope",
    "TsLinRegSlope": "Slope",
    "Rsquare": "Rsquare",
    "Resi": "Resi",
    "TsLinRegResid": "Resi",
    # Element-wise arithmetic.
    "Abs": "Abs",
    "Sign": "Sign",
    "Add": "Add",
    "Sub": "Sub",
    "Mul": "Mul",
    "Div": "Div",
    "Pow": "Power",
    # See collision note above: element-wise max/min vs. comparison ops.
    "Max": "Greater",
    "Max2": "Greater",
    "Min": "Less",
    "Min2": "Less",
    "Greater": "Gt",
    "GreaterEqual": "Ge",
    "Less": "Lt",
    "LessEqual": "Le",
    "Equal": "Eq",
    "Eq": "Eq",
    "Ne": "Ne",
}


def _format_qlib_number(value: float) -> str:
    """Render a numeric parameter the same way ``OperatorNode.to_string`` does."""
    if value == int(value) and abs(value) < 1e12:
        return str(int(value))
    return f"{value:g}"


def _translate_node_to_qlib(node: Node, unsupported: set[str]) -> str:
    """Recursively render an expression-tree node as a Qlib Expression string.

    Feature leaves (``$close`` etc.) and numeric constants pass through
    unchanged -- FactorMiner's ``$``-prefixed feature syntax already matches
    Qlib's. Operator nodes are renamed via
    :data:`_FACTORMINER_TO_QLIB_OPERATOR_MAP`; any operator without a mapped
    entry is left under its original FactorMiner name (so the output is
    still valid-looking but visibly non-Qlib) and recorded in *unsupported*.
    """
    if isinstance(node, LeafNode):
        return node.feature_name
    if isinstance(node, ConstantNode):
        return node.to_string()
    if isinstance(node, OperatorNode):
        fm_name = node.operator.name
        qlib_name = _FACTORMINER_TO_QLIB_OPERATOR_MAP.get(fm_name)
        if qlib_name is None:
            unsupported.add(fm_name)
            qlib_name = fm_name
        child_strs = [_translate_node_to_qlib(c, unsupported) for c in node.children]
        param_strs = [
            _format_qlib_number(node.params[pname])
            for pname in node.operator.param_names
            if pname in node.params
        ]
        return f"{qlib_name}({', '.join(child_strs + param_strs)})"
    raise TypeError(f"Unknown expression-tree node type: {type(node)!r}")  # pragma: no cover


def export_formulas_qlib(library: FactorLibrary, path: str | Path) -> None:
    """Export the library's formulas translated into Qlib ``Expression`` syntax.

    Writes a JSON document with one entry per factor: the original
    FactorMiner formula, its best-effort Qlib ``$``-prefixed Expression
    translation (via :data:`_FACTORMINER_TO_QLIB_OPERATOR_MAP`), a
    ``qlib_translatable`` flag, and an ``unsupported_operators`` list. A
    formula is only marked translatable when every operator it uses has a
    verified Qlib equivalent -- this is a best-effort, honest translator:
    it never claims a formula is Qlib-ready when part of it silently
    couldn't be mapped.

    Formulas that fail to parse (should not normally happen for admitted
    factors, but formulas can be hand-edited between save/load) are
    reported with ``qlib_expression: null`` and ``qlib_translatable: false``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    n_translatable = 0
    for f in library.list_factors():
        tree = try_parse(f.formula)
        if tree is None:
            entries.append(
                {
                    "factor_id": f.id,
                    "name": f.name,
                    "formula": f.formula,
                    "qlib_expression": None,
                    "qlib_translatable": False,
                    "unsupported_operators": [],
                    "parse_error": True,
                }
            )
            continue
        unsupported: set[str] = set()
        qlib_expr = _translate_node_to_qlib(tree.root, unsupported)
        translatable = not unsupported
        if translatable:
            n_translatable += 1
        entries.append(
            {
                "factor_id": f.id,
                "name": f.name,
                "formula": f.formula,
                "qlib_expression": qlib_expr,
                "qlib_translatable": translatable,
                "unsupported_operators": sorted(unsupported),
                "parse_error": False,
            }
        )

    with open(path, "w") as fp:
        json.dump({"factors": entries}, fp, indent=2)

    logger.info(
        "Exported %d Qlib-translated formulas to %s (%d fully translatable)",
        library.size,
        path,
        n_translatable,
    )


# ======================================================================
