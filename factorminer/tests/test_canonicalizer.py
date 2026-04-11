"""Tests for the SymPy-based formula canonicalizer (core/canonicalizer.py)."""

from __future__ import annotations

import pytest

from factorminer.core.canonicalizer import FormulaCanonicalizer
from factorminer.core.parser import parse


@pytest.fixture
def canon():
    return FormulaCanonicalizer()


# -----------------------------------------------------------------------
# Double negation: Neg(Neg($close)) == $close
# -----------------------------------------------------------------------

def test_double_negation(canon):
    tree_a = parse("Neg(Neg($close))")
    tree_b = parse("$close")
    assert canon.is_duplicate(tree_a, tree_b)


# -----------------------------------------------------------------------
# Commutativity: Add($close, $open) == Add($open, $close)
# -----------------------------------------------------------------------

def test_commutativity_add(canon):
    tree_a = parse("Add($close, $open)")
    tree_b = parse("Add($open, $close)")
    assert canon.is_duplicate(tree_a, tree_b)


# -----------------------------------------------------------------------
# Non-algebraic preserved: CsRank(Neg($close)) != Neg(CsRank($close))
# -----------------------------------------------------------------------

def test_non_algebraic_not_simplified(canon):
    tree_a = parse("CsRank(Neg($close))")
    tree_b = parse("Neg(CsRank($close))")
    assert not canon.is_duplicate(tree_a, tree_b)


# -----------------------------------------------------------------------
# is_duplicate method
# -----------------------------------------------------------------------

def test_is_duplicate_same_formula(canon):
    tree = parse("CsRank($close)")
    assert canon.is_duplicate(tree, tree)


def test_is_duplicate_different_formulas(canon):
    tree_a = parse("CsRank($close)")
    tree_b = parse("CsRank($volume)")
    assert not canon.is_duplicate(tree_a, tree_b)


# -----------------------------------------------------------------------
# Cache: second call should be faster (or at least not slower)
# -----------------------------------------------------------------------

def test_cache_works(canon):
    tree = parse("Add(Mul($close, $open), Neg($volume))")

    # First call populates cache
    h1 = canon.canonicalize(tree)

    # Second call should hit cache and return same hash
    h2 = canon.canonicalize(tree)
    assert h1 == h2

    # Verify cache is populated
    key = tree.to_string()
    assert key in canon._cache
