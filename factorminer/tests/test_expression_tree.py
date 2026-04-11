"""Tests for the expression tree and parser modules."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.core.parser import parse, tokenize, try_parse

# ---------------------------------------------------------------------------
# Parsing simple formulas
# ---------------------------------------------------------------------------

class TestParseSimple:
    """Test parsing of basic single-operator formulas."""

    def test_parse_neg_close(self):
        tree = parse("Neg($close)")
        assert tree.to_string() == "Neg($close)"

    def test_parse_add_open_close(self):
        tree = parse("Add($open, $close)")
        assert tree.to_string() == "Add($open, $close)"

    def test_parse_leaf_only(self):
        tree = parse("$close")
        assert tree.to_string() == "$close"
        assert tree.depth() == 1
        assert tree.size() == 1

    def test_parse_constant(self):
        tree = parse("0.0001")
        assert tree.depth() == 1

    def test_parse_div_with_two_features(self):
        tree = parse("Div($high, $low)")
        assert tree.to_string() == "Div($high, $low)"

    def test_parse_sub(self):
        tree = parse("Sub($close, $open)")
        assert tree.to_string() == "Sub($close, $open)"

    def test_parse_operator_with_window(self):
        tree = parse("Mean($close, 20)")
        assert tree.to_string() == "Mean($close, 20)"

    def test_parse_ema_with_window(self):
        tree = parse("EMA($close, 10)")
        assert tree.to_string() == "EMA($close, 10)"


# ---------------------------------------------------------------------------
# Parsing complex nested formulas from the paper
# ---------------------------------------------------------------------------

class TestParseComplex:
    """Test parsing of complex nested formulas (paper factors)."""

    def test_factor_006(self):
        """Neg(Div(Sub($close, $vwap), $vwap))"""
        formula = "Neg(Div(Sub($close, $vwap), $vwap))"
        tree = parse(formula)
        assert tree.to_string() == formula

    def test_factor_002(self):
        """Neg(Div(Sub($close, EMA($close, 10)), EMA($close, 18)))"""
        formula = "Neg(Div(Sub($close, EMA($close, 10)), EMA($close, 18)))"
        tree = parse(formula)
        assert tree.to_string() == formula

    def test_factor_046_ifelse(self):
        """Complex IfElse with Greater, Std, Mean, Neg, CsRank, Delta, Div, Sub, Add."""
        formula = (
            "IfElse(Greater(Std($returns, 12), Mean(Std($returns, 12), 48)), "
            "Neg(CsRank(Delta($close, 3))), "
            "Neg(CsRank(Div(Sub($close, $low), Add(Sub($high, $low), 0.0001)))))"
        )
        tree = parse(formula)
        roundtrip = tree.to_string()
        # Parse roundtrip should also succeed
        tree2 = parse(roundtrip)
        assert tree2.to_string() == roundtrip

    def test_nested_csrank_corr(self):
        formula = "CsRank(Corr($close, $volume, 20))"
        tree = parse(formula)
        assert tree.to_string() == formula

    def test_deeply_nested(self):
        formula = "CsRank(Neg(Div(Sub($close, Mean($close, 20)), Std($close, 20))))"
        tree = parse(formula)
        assert tree.to_string() == formula


# ---------------------------------------------------------------------------
# Roundtrip: parse -> to_string -> parse
# ---------------------------------------------------------------------------

class TestRoundtrip:
    """Test that parse -> to_string -> parse produces identical trees."""

    @pytest.mark.parametrize(
        "formula",
        [
            "Neg($close)",
            "Add($open, $close)",
            "Div(Sub($close, $vwap), $vwap)",
            "Mean($close, 20)",
            "EMA($close, 10)",
            "CsRank(Std($returns, 12))",
            "IfElse(Greater($close, $open), $high, $low)",
        ],
    )
    def test_roundtrip(self, formula):
        tree1 = parse(formula)
        s1 = tree1.to_string()
        tree2 = parse(s1)
        s2 = tree2.to_string()
        assert s1 == s2


# ---------------------------------------------------------------------------
# Expression tree evaluation with mock data
# ---------------------------------------------------------------------------

class TestEvaluate:
    """Test evaluate on known inputs."""

    def test_neg_evaluate(self, small_data):
        tree = parse("Neg($close)")
        result = tree.evaluate(small_data)
        np.testing.assert_array_almost_equal(result, -small_data["$close"])

    def test_add_evaluate(self, small_data):
        tree = parse("Add($open, $close)")
        result = tree.evaluate(small_data)
        expected = small_data["$open"] + small_data["$close"]
        np.testing.assert_array_almost_equal(result, expected)

    def test_sub_evaluate(self, small_data):
        tree = parse("Sub($close, $open)")
        result = tree.evaluate(small_data)
        expected = small_data["$close"] - small_data["$open"]
        np.testing.assert_array_almost_equal(result, expected)

    def test_div_evaluate(self, small_data):
        tree = parse("Div($high, $low)")
        result = tree.evaluate(small_data)
        assert result.shape == small_data["$high"].shape
        # Should be positive since high > low
        valid = ~np.isnan(result) & (result != 0)
        assert np.all(result[valid] > 0)

    def test_constant_in_formula(self, small_data):
        tree = parse("Add($close, 0.0001)")
        result = tree.evaluate(small_data)
        # The constant becomes a ConstantNode, which is treated as a trailing
        # parameter if arity is 2 and it becomes the second child.
        assert result.shape == small_data["$close"].shape

    def test_nested_evaluate_shape(self, small_data):
        tree = parse("Neg(Div(Sub($close, $vwap), $vwap))")
        result = tree.evaluate(small_data)
        assert result.shape == small_data["$close"].shape


# ---------------------------------------------------------------------------
# Tree depth and size
# ---------------------------------------------------------------------------

class TestTreeStructure:
    """Test depth() and size() computations."""

    def test_leaf_depth(self):
        tree = parse("$close")
        assert tree.depth() == 1

    def test_leaf_size(self):
        tree = parse("$close")
        assert tree.size() == 1

    def test_unary_depth(self):
        tree = parse("Neg($close)")
        assert tree.depth() == 2

    def test_unary_size(self):
        tree = parse("Neg($close)")
        assert tree.size() == 2

    def test_binary_depth(self):
        tree = parse("Add($open, $close)")
        assert tree.depth() == 2

    def test_binary_size(self):
        tree = parse("Add($open, $close)")
        assert tree.size() == 3  # Add + $open + $close

    def test_nested_depth(self):
        tree = parse("Neg(Div(Sub($close, $vwap), $vwap))")
        assert tree.depth() == 4  # Neg -> Div -> Sub -> $close

    def test_nested_size(self):
        tree = parse("Neg(Div(Sub($close, $vwap), $vwap))")
        assert tree.size() == 6  # Neg, Div, Sub, $close, $vwap, $vwap

    def test_leaf_features(self):
        tree = parse("Neg(Div(Sub($close, $vwap), $vwap))")
        feats = tree.leaf_features()
        assert feats == ["$close", "$vwap"]

    def test_clone_preserves_structure(self):
        tree = parse("Add($open, $close)")
        cloned = tree.clone()
        assert cloned.to_string() == tree.to_string()
        assert cloned.depth() == tree.depth()
        assert cloned.size() == tree.size()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test that invalid inputs raise appropriate errors."""

    def test_unknown_operator(self):
        with pytest.raises(SyntaxError, match="Unknown operator"):
            parse("FooBar($close)")

    def test_unknown_feature(self):
        with pytest.raises(SyntaxError, match="Unknown feature"):
            parse("Neg($foobar)")

    def test_wrong_arity_too_few(self):
        with pytest.raises(SyntaxError, match="expects"):
            parse("Add($close)")

    def test_wrong_arity_too_many_nodes(self):
        # Neg expects 1 expression arg; passing 2 should fail
        with pytest.raises(SyntaxError):
            parse("Neg($close, $open)")

    def test_empty_string(self):
        with pytest.raises((SyntaxError, IndexError)):
            parse("")

    def test_unmatched_paren(self):
        with pytest.raises(SyntaxError):
            parse("Neg($close")

    def test_trailing_content(self):
        with pytest.raises(SyntaxError, match="Unexpected trailing"):
            parse("Neg($close) extra")

    def test_try_parse_returns_none_on_failure(self):
        assert try_parse("InvalidOp($close)") is None
        assert try_parse("") is None

    def test_try_parse_returns_tree_on_success(self):
        result = try_parse("Neg($close)")
        assert result is not None
        assert result.to_string() == "Neg($close)"

    def test_missing_feature_in_data(self, small_data):
        tree = parse("Neg($close)")
        data_missing = {k: v for k, v in small_data.items() if k != "$close"}
        with pytest.raises(KeyError, match="\\$close"):
            tree.evaluate(data_missing)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TestTokenizer:
    """Test the tokenizer separately."""

    def test_simple_tokens(self):
        tokens = tokenize("Neg($close)")
        types = [t.type.name for t in tokens]
        assert types == ["IDENT", "LPAREN", "FEATURE", "RPAREN", "EOF"]

    def test_number_token(self):
        tokens = tokenize("0.0001")
        assert tokens[0].type.name == "NUMBER"
        assert tokens[0].value == "0.0001"

    def test_negative_number_token(self):
        tokens = tokenize("Mean($close, -3)")
        # -3 should be a number token after comma
        num_tokens = [t for t in tokens if t.type.name == "NUMBER"]
        assert len(num_tokens) == 1
        assert num_tokens[0].value == "-3"

    def test_whitespace_handling(self):
        tokens = tokenize("  Add(  $open ,  $close  ) ")
        ident_tokens = [t for t in tokens if t.type.name == "IDENT"]
        assert len(ident_tokens) == 1
        assert ident_tokens[0].value == "Add"
