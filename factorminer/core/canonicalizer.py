"""SymPy-based formula canonicalization for duplicate detection.

Converts ``ExpressionTree`` objects into canonical SymPy expressions so that
algebraically equivalent formulas (e.g. ``Add($close, $open)`` vs
``Add($open, $close)``, or ``Neg(Neg($close))`` vs ``$close``) produce
identical hashes.

**Design principle**: Arithmetic operators map to native SymPy math so that
standard simplifications (commutativity, double-negation, x/x = 1, etc.) are
applied automatically.  Non-algebraic operators (rolling windows,
cross-sectional transforms, conditionals) are represented as opaque
``sympy.Function`` symbols so their structure is preserved without false
simplification.
"""

from __future__ import annotations

import hashlib

import sympy
from sympy import Abs, Float, Function, Symbol, log, sqrt

from factorminer.core.expression_tree import (
    ConstantNode,
    ExpressionTree,
    LeafNode,
    Node,
    OperatorNode,
)

# Arithmetic operator names that map directly to SymPy math.
_ALGEBRAIC_OPS = frozenset({
    "Add", "Sub", "Mul", "Div", "Neg", "Abs",
    "Square", "Sqrt", "Log", "Pow", "SignedPower",
})


class FormulaCanonicalizer:
    """Canonicalize expression trees via SymPy simplification.

    Maintains an internal cache so that repeated calls for the same formula
    string are fast.

    Examples
    --------
    >>> from factorminer.core.parser import parse
    >>> canon = FormulaCanonicalizer()
    >>> canon.is_duplicate(parse("Add($close, $open)"), parse("Add($open, $close)"))
    True
    >>> canon.is_duplicate(parse("Neg(Neg($close))"), parse("$close"))
    True
    """

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}  # formula string -> canonical MD5 hash

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def canonicalize(self, tree: ExpressionTree) -> str:
        """Return an MD5 hash of the canonical (simplified) form of *tree*.

        Parameters
        ----------
        tree : ExpressionTree
            The expression tree to canonicalize.

        Returns
        -------
        str
            Hex-encoded MD5 digest of the canonical string representation.
        """
        key = tree.to_string()
        if key in self._cache:
            return self._cache[key]

        sympy_expr = self._tree_to_sympy(tree.root)
        simplified = sympy.simplify(sympy_expr)
        canonical_str = str(simplified)
        digest = hashlib.md5(canonical_str.encode("utf-8")).hexdigest()
        self._cache[key] = digest
        return digest

    def is_duplicate(self, tree_a: ExpressionTree, tree_b: ExpressionTree) -> bool:
        """Return ``True`` if *tree_a* and *tree_b* are algebraically equivalent.

        Parameters
        ----------
        tree_a, tree_b : ExpressionTree
            Two expression trees to compare.

        Returns
        -------
        bool
        """
        return self.canonicalize(tree_a) == self.canonicalize(tree_b)

    def get_canonical_form(self, tree: ExpressionTree) -> str:
        """Return the simplified string representation (not hashed).

        Useful for debugging and display.

        Parameters
        ----------
        tree : ExpressionTree

        Returns
        -------
        str
            Human-readable simplified expression.
        """
        sympy_expr = self._tree_to_sympy(tree.root)
        simplified = sympy.simplify(sympy_expr)
        return str(simplified)

    def clear_cache(self) -> None:
        """Discard all cached canonical hashes."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Tree -> SymPy conversion
    # ------------------------------------------------------------------

    def _tree_to_sympy(self, node: Node) -> sympy.Expr:
        """Recursively convert an expression-tree node to a SymPy expression.

        Parameters
        ----------
        node : Node
            Any node in the expression tree hierarchy.

        Returns
        -------
        sympy.Expr
        """
        if isinstance(node, LeafNode):
            return Symbol(node.feature_name)

        if isinstance(node, ConstantNode):
            return Float(node.value)

        if isinstance(node, OperatorNode):
            children_sympy = [self._tree_to_sympy(c) for c in node.children]
            return self._map_operator(
                node.operator.name, children_sympy, node.params
            )

        raise TypeError(f"Unexpected node type: {type(node).__name__}")

    def _map_operator(
        self,
        name: str,
        children: list[sympy.Expr],
        params: dict[str, float],
    ) -> sympy.Expr:
        """Dispatch an operator to its SymPy equivalent.

        Arithmetic operators are mapped to native SymPy math so the
        simplifier can reason about them.  All other operators become opaque
        ``sympy.Function`` applications that preserve structure.

        Parameters
        ----------
        name : str
            Operator name from the registry (e.g. ``"Add"``, ``"CsRank"``).
        children : list[sympy.Expr]
            Already-converted child expressions.
        params : dict[str, float]
            Extra numeric parameters (e.g. ``{"window": 10}``).

        Returns
        -------
        sympy.Expr
        """
        # --- Arithmetic: map to SymPy math ------------------------------------
        if name == "Add":
            return children[0] + children[1]
        if name == "Sub":
            return children[0] - children[1]
        if name == "Mul":
            return children[0] * children[1]
        if name == "Div":
            return children[0] / children[1]
        if name == "Neg":
            return -children[0]
        if name == "Abs":
            return Abs(children[0])
        if name == "Square":
            return children[0] ** 2
        if name == "Sqrt":
            return sqrt(Abs(children[0]))
        if name == "Log":
            return log(1 + Abs(children[0]))
        if name in ("Pow", "SignedPower"):
            return children[0] ** children[1]

        # --- Non-algebraic: wrap as opaque Function ---------------------------
        func = Function(name)
        # Build argument list: children first, then params as Float values
        args: list[sympy.Expr] = list(children)
        # Append params in a deterministic order (sorted by param name).
        for pname in sorted(params):
            args.append(Float(params[pname]))
        return func(*args)
