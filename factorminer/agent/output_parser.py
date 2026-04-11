"""Parse LLM output into structured CandidateFactor objects.

Handles various output formats from LLMs: numbered lists, JSON,
markdown code blocks, and raw text.  Validates each formula against
the expression tree parser.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from factorminer.core.expression_tree import ExpressionTree
from factorminer.core.parser import parse, try_parse

logger = logging.getLogger(__name__)


@dataclass
class CandidateFactor:
    """A candidate factor parsed from LLM output.

    Attributes
    ----------
    name : str
        Descriptive snake_case name.
    formula : str
        DSL formula string.
    expression_tree : ExpressionTree or None
        Parsed expression tree (None if parsing failed).
    category : str
        Inferred category based on outermost operators.
    parse_error : str
        Error message if formula failed to parse.
    """

    name: str
    formula: str
    expression_tree: ExpressionTree | None = None
    category: str = "unknown"
    parse_error: str = ""

    @property
    def is_valid(self) -> bool:
        return self.expression_tree is not None


def _infer_category(formula: str) -> str:
    """Infer a rough category from the outermost operators in the formula."""
    formula.lower()
    # Check for cross-sectional operators at the top
    if any(op in formula for op in ("CsRank", "CsZScore", "CsDemean", "CsScale", "CsNeutralize", "CsQuantile")):
        # Look deeper for sub-category
        if any(op in formula for op in ("Corr", "Cov", "Beta", "Resid")):
            return "cross_sectional_regression"
        if any(op in formula for op in ("Delta", "Delay", "Return", "LogReturn")):
            return "cross_sectional_momentum"
        if any(op in formula for op in ("Std", "Var", "Skew", "Kurt")):
            return "cross_sectional_volatility"
        if any(op in formula for op in ("Mean", "Sum", "EMA", "SMA", "WMA", "DEMA", "HMA", "KAMA")):
            return "cross_sectional_smoothing"
        if any(op in formula for op in ("TsLinReg", "TsLinRegSlope")):
            return "cross_sectional_trend"
        return "cross_sectional"
    if any(op in formula for op in ("Corr", "Cov", "Beta", "Resid")):
        return "regression"
    if any(op in formula for op in ("Delta", "Delay", "Return", "LogReturn")):
        return "momentum"
    if any(op in formula for op in ("Std", "Var", "Skew", "Kurt")):
        return "volatility"
    if any(op in formula for op in ("IfElse", "Greater", "Less")):
        return "conditional"
    return "general"


# ---------------------------------------------------------------------------
# Line parsing patterns
# ---------------------------------------------------------------------------

# Pattern: "1. name: formula" or "1) name: formula"
_NUMBERED_PATTERN = re.compile(
    r"^\s*\d+[\.\)]\s*"          # numbered prefix
    r"([a-zA-Z_][a-zA-Z0-9_]*)"  # factor name
    r"\s*:\s*"                    # colon separator
    r"(.+)$"                      # formula
)

# Pattern: "name: formula" (no number)
_PLAIN_PATTERN = re.compile(
    r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)"  # factor name
    r"\s*:\s*"                         # colon separator
    r"(.+)$"                           # formula
)

# Pattern: just a formula starting with an operator
_FORMULA_ONLY_PATTERN = re.compile(
    r"^\s*([A-Z][a-zA-Z]*\(.+\))\s*$"
)

# Pattern: JSON-like {"name": "...", "formula": "..."}
_JSON_PATTERN = re.compile(
    r'"name"\s*:\s*"([^"]+)"\s*,\s*"formula"\s*:\s*"([^"]+)"'
)


def _strip_markdown(text: str) -> str:
    """Remove markdown code block markers."""
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    return text


def _clean_formula(formula: str) -> str:
    """Clean up a formula string before parsing."""
    formula = formula.strip()
    # Remove trailing comments
    if " #" in formula:
        formula = formula[: formula.index(" #")]
    if " //" in formula:
        formula = formula[: formula.index(" //")]
    # Remove trailing punctuation
    formula = formula.rstrip(";,.")
    # Remove surrounding backticks
    formula = formula.strip("`")
    return formula.strip()


def parse_llm_output(raw_text: str) -> tuple[list[CandidateFactor], list[str]]:
    """Parse raw LLM text output into candidate factors.

    Parameters
    ----------
    raw_text : str
        Raw text from the LLM containing factor definitions.

    Returns
    -------
    tuple[list[CandidateFactor], list[str]]
        (successfully_parsed, failed_lines) where failed_lines are
        the original text lines that could not be parsed.
    """
    text = _strip_markdown(raw_text)

    candidates: list[CandidateFactor] = []
    failed: list[str] = []
    seen_names: set = set()

    # Try JSON pattern first (entire text)
    json_matches = _JSON_PATTERN.findall(text)
    if json_matches:
        for name, formula in json_matches:
            formula = _clean_formula(formula)
            candidate = _try_build_candidate(name, formula)
            if candidate.name not in seen_names:
                candidates.append(candidate)
                seen_names.add(candidate.name)
        if candidates:
            logger.debug("Parsed %d factors from JSON format", len(candidates))
            return candidates, failed

    # Line-by-line parsing
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue

        name: str | None = None
        formula: str | None = None

        # Try numbered pattern: "1. name: formula"
        m = _NUMBERED_PATTERN.match(line)
        if m:
            name, formula = m.group(1), m.group(2)
        else:
            # Try plain pattern: "name: formula"
            m = _PLAIN_PATTERN.match(line)
            if m:
                name, formula = m.group(1), m.group(2)
            else:
                # Try formula-only pattern
                m = _FORMULA_ONLY_PATTERN.match(line)
                if m:
                    formula = m.group(1)
                    # Generate name from formula
                    name = _generate_name_from_formula(formula, len(candidates))

        if name is None or formula is None:
            if any(c.isalpha() for c in line) and "(" in line:
                failed.append(line)
            continue

        formula = _clean_formula(formula)
        if not formula:
            failed.append(line)
            continue

        # Ensure unique name
        base_name = name.lower().replace("-", "_")
        unique_name = base_name
        counter = 2
        while unique_name in seen_names:
            unique_name = f"{base_name}_{counter}"
            counter += 1

        candidate = _try_build_candidate(unique_name, formula)
        candidates.append(candidate)
        seen_names.add(unique_name)

        if not candidate.is_valid:
            failed.append(line)

    logger.debug(
        "Parsed %d candidates (%d valid, %d failed lines)",
        len(candidates),
        sum(1 for c in candidates if c.is_valid),
        len(failed),
    )
    return candidates, failed


def _try_build_candidate(name: str, formula: str) -> CandidateFactor:
    """Attempt to parse a formula and build a CandidateFactor."""
    tree = try_parse(formula)
    if tree is not None:
        category = _infer_category(formula)
        return CandidateFactor(
            name=name,
            formula=tree.to_string(),  # canonicalize
            expression_tree=tree,
            category=category,
        )
    else:
        # Try to get a useful error message
        error_msg = ""
        try:
            parse(formula)
        except (SyntaxError, KeyError, ValueError) as e:
            error_msg = str(e)

        return CandidateFactor(
            name=name,
            formula=formula,
            expression_tree=None,
            category="unknown",
            parse_error=error_msg,
        )


def _generate_name_from_formula(formula: str, index: int) -> str:
    """Generate a descriptive name from a formula."""
    # Extract the outermost operator
    m = re.match(r"([A-Z][a-zA-Z]*)\(", formula)
    if m:
        outer_op = m.group(1).lower()
        return f"{outer_op}_factor_{index + 1}"
    return f"factor_{index + 1}"
