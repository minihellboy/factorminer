"""Recursive-descent parser for the FactorMiner factor DSL.

Converts string formulas such as::

    Neg(CsRank(Div(Sub($close, $vwap), $vwap)))

into ``ExpressionTree`` objects backed by the operator registry defined in
:mod:`factorminer.core.types`.

Grammar (informal)
------------------

::

    expression  := function_call | feature_ref | number
    function_call := IDENTIFIER '(' arg_list ')'
    arg_list    := expression (',' expression)*
    feature_ref := '$' IDENTIFIER
    number      := ['-'] DIGITS ['.' DIGITS] [('e'|'E') ['-'|'+'] DIGITS]

Usage
-----

>>> from factorminer.core.parser import parse
>>> tree = parse("Neg(Div(Sub($close, $vwap), $vwap))")
>>> tree.to_string()
'Neg(Div(Sub($close, $vwap), $vwap))'
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from factorminer.core.expression_tree import (
    ConstantNode,
    ExpressionTree,
    LeafNode,
    Node,
    OperatorNode,
)
from factorminer.core.types import FEATURE_SET, OPERATOR_REGISTRY

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class TokenType(Enum):
    IDENT = auto()      # operator / function name
    FEATURE = auto()    # $close, $volume, ...
    NUMBER = auto()     # 0.0001, -3, 1e-6, ...
    LPAREN = auto()     # (
    RPAREN = auto()     # )
    COMMA = auto()      # ,
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    pos: int  # character position in the source string

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, pos={self.pos})"


# Regex fragments
_NUMBER_RE = re.compile(
    r"""
    -?                      # optional leading minus
    (?:\d+\.?\d*|\.\d+)     # integer or decimal
    (?:[eE][+-]?\d+)?       # optional exponent
    """,
    re.VERBOSE,
)

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_FEATURE_RE = re.compile(r"\$[A-Za-z_]\w*")
_WS_RE = re.compile(r"\s+")


def tokenize(source: str) -> list[Token]:
    """Convert a formula string into a list of ``Token`` objects.

    Raises
    ------
    SyntaxError
        If the string contains characters that cannot be tokenized.
    """
    tokens: list[Token] = []
    pos = 0
    length = len(source)

    while pos < length:
        # Skip whitespace
        m = _WS_RE.match(source, pos)
        if m:
            pos = m.end()
            continue

        ch = source[pos]

        if ch == "(":
            tokens.append(Token(TokenType.LPAREN, "(", pos))
            pos += 1
        elif ch == ")":
            tokens.append(Token(TokenType.RPAREN, ")", pos))
            pos += 1
        elif ch == ",":
            tokens.append(Token(TokenType.COMMA, ",", pos))
            pos += 1
        elif ch == "$":
            m = _FEATURE_RE.match(source, pos)
            if not m:
                raise SyntaxError(
                    f"Invalid feature reference at position {pos}: "
                    f"{source[pos:pos+20]!r}"
                )
            tokens.append(Token(TokenType.FEATURE, m.group(), pos))
            pos = m.end()
        elif ch == "-" or ch == "." or ch.isdigit():
            # Could be a negative number or just a number.
            # Disambiguate: a minus is part of a number only if
            #   (a) it's the very first token, OR
            #   (b) the preceding token is LPAREN or COMMA
            if ch == "-":
                prev_tok = tokens[-1] if tokens else None
                is_unary_minus = (
                    prev_tok is None
                    or prev_tok.type in (TokenType.LPAREN, TokenType.COMMA)
                )
                if not is_unary_minus:
                    raise SyntaxError(
                        f"Unexpected '-' at position {pos}. "
                        f"Subtraction should use the Sub() operator."
                    )
            m = _NUMBER_RE.match(source, pos)
            if not m:
                raise SyntaxError(
                    f"Invalid number at position {pos}: "
                    f"{source[pos:pos+20]!r}"
                )
            tokens.append(Token(TokenType.NUMBER, m.group(), pos))
            pos = m.end()
        elif ch.isalpha() or ch == "_":
            m = _IDENT_RE.match(source, pos)
            if not m:
                raise SyntaxError(
                    f"Invalid identifier at position {pos}: "
                    f"{source[pos:pos+20]!r}"
                )
            tokens.append(Token(TokenType.IDENT, m.group(), pos))
            pos = m.end()
        else:
            raise SyntaxError(
                f"Unexpected character {ch!r} at position {pos} in: "
                f"{source!r}"
            )

    tokens.append(Token(TokenType.EOF, "", length))
    return tokens


# ---------------------------------------------------------------------------
# Recursive descent parser
# ---------------------------------------------------------------------------

class Parser:
    """Recursive-descent parser that converts a token stream to a ``Node``.

    The parser consumes tokens left-to-right, building the expression tree
    in a single pass.
    """

    def __init__(self, tokens: list[Token], source: str) -> None:
        self.tokens = tokens
        self.source = source
        self.pos = 0

    # -- helpers ------------------------------------------------------------

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, tt: TokenType) -> Token:
        tok = self._advance()
        if tok.type != tt:
            raise SyntaxError(
                f"Expected {tt.name} but got {tok.type.name} ({tok.value!r}) "
                f"at position {tok.pos} in: {self.source!r}"
            )
        return tok

    # -- grammar rules ------------------------------------------------------

    def parse_expression(self) -> Node:
        """Parse a single expression (the start symbol)."""
        tok = self._peek()

        if tok.type == TokenType.FEATURE:
            return self._parse_feature()

        if tok.type == TokenType.NUMBER:
            return self._parse_number()

        if tok.type == TokenType.IDENT:
            return self._parse_function_call()

        raise SyntaxError(
            f"Unexpected token {tok.type.name} ({tok.value!r}) at position "
            f"{tok.pos} in: {self.source!r}"
        )

    def _parse_feature(self) -> LeafNode:
        tok = self._advance()
        if tok.value not in FEATURE_SET:
            raise SyntaxError(
                f"Unknown feature '{tok.value}' at position {tok.pos}. "
                f"Expected one of {sorted(FEATURE_SET)}."
            )
        return LeafNode(tok.value)

    def _parse_number(self) -> ConstantNode:
        tok = self._advance()
        try:
            return ConstantNode(float(tok.value))
        except ValueError:
            raise SyntaxError(
                f"Invalid numeric literal {tok.value!r} at position {tok.pos}."
            )

    def _parse_function_call(self) -> Node:
        """Parse ``Name(arg1, arg2, ..., paramN)``."""
        name_tok = self._advance()  # IDENT
        name = name_tok.value

        # Look up operator
        spec = OPERATOR_REGISTRY.get(name)
        if spec is None:
            raise SyntaxError(
                f"Unknown operator '{name}' at position {name_tok.pos}. "
                f"Available operators: {sorted(OPERATOR_REGISTRY.keys())}"
            )

        self._expect(TokenType.LPAREN)

        # Collect arguments (mix of sub-expressions and trailing numeric params)
        raw_args: list = []  # (Node | float) to separate children from params

        if self._peek().type != TokenType.RPAREN:
            raw_args.append(self._parse_arg())
            while self._peek().type == TokenType.COMMA:
                self._advance()  # consume comma
                raw_args.append(self._parse_arg())

        self._expect(TokenType.RPAREN)

        # Separate expression children from trailing numeric parameters.
        # Strategy: the first ``spec.arity`` arguments that are Nodes are
        # the children.  Remaining numeric values fill param slots in order.
        children: list[Node] = []
        trailing_numbers: list[float] = []

        for arg in raw_args:
            if isinstance(arg, Node) and len(children) < spec.arity:
                children.append(arg)
            elif isinstance(arg, (int, float)):
                trailing_numbers.append(float(arg))
            elif isinstance(arg, Node):
                # Extra node arguments beyond arity — could be a ConstantNode
                # that the user passed as a positional param (e.g. 0.0001).
                if isinstance(arg, ConstantNode):
                    trailing_numbers.append(arg.value)
                else:
                    children.append(arg)
            else:
                trailing_numbers.append(float(arg))

        # Validate arity
        if len(children) != spec.arity:
            raise SyntaxError(
                f"Operator '{name}' expects {spec.arity} expression "
                f"argument(s) but got {len(children)} at position "
                f"{name_tok.pos}."
            )

        # Map trailing numbers to parameter names
        params: dict[str, float] = {}
        for i, pname in enumerate(spec.param_names):
            if i < len(trailing_numbers):
                params[pname] = trailing_numbers[i]

        return OperatorNode(spec, children, params)

    def _parse_arg(self):
        """Parse a single argument inside a function call.

        Returns either a ``Node`` (for sub-expressions) or a bare ``float``
        for numeric literals that might be operator parameters.
        """
        tok = self._peek()

        if tok.type == TokenType.NUMBER:
            # Peek ahead: if this number is followed by COMMA or RPAREN it
            # could be a trailing parameter.  We still return a ConstantNode
            # and let the caller decide.
            num_tok = self._advance()
            val = float(num_tok.value)
            # If the next token is LPAREN, that's weird — just return as
            # constant.
            return ConstantNode(val)

        return self.parse_expression()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse(source: str) -> ExpressionTree:
    """Parse a factor formula string into an ``ExpressionTree``.

    Parameters
    ----------
    source : str
        A formula in the FactorMiner DSL, e.g.
        ``"Neg(CsRank(Div(Sub($close, $vwap), $vwap)))"``.

    Returns
    -------
    ExpressionTree

    Raises
    ------
    SyntaxError
        If the formula is malformed or references unknown operators / features.

    Examples
    --------
    >>> tree = parse("Neg($close)")
    >>> tree.to_string()
    'Neg($close)'
    """
    tokens = tokenize(source.strip())
    parser = Parser(tokens, source)
    root = parser.parse_expression()

    # Ensure we consumed everything
    remaining = parser._peek()
    if remaining.type != TokenType.EOF:
        raise SyntaxError(
            f"Unexpected trailing content at position {remaining.pos}: "
            f"{remaining.value!r} in: {source!r}"
        )

    return ExpressionTree(root)


def try_parse(source: str) -> ExpressionTree | None:
    """Like :func:`parse` but returns ``None`` on failure instead of raising."""
    try:
        return parse(source)
    except (SyntaxError, KeyError, ValueError):
        return None
