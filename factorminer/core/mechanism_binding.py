"""Deterministic checks binding economic narratives to formula structure.

The checker does not validate a causal story.  It only detects explicit claims
whose required observable inputs or transformations are absent from the parsed
formula AST.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from factorminer.core.expression_tree import OperatorNode
from factorminer.core.parser import try_parse


@dataclass(frozen=True)
class MechanismBindingResult:
    """Auditable result of a formula-to-narrative ingredient check."""

    status: str
    claims_checked: tuple[str, ...]
    contradictions: tuple[str, ...]
    formula_features: tuple[str, ...]
    formula_operators: tuple[str, ...]
    checker_version: str = "mechanism_binding_v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["claims_checked"] = list(self.claims_checked)
        payload["contradictions"] = list(self.contradictions)
        payload["formula_features"] = list(self.formula_features)
        payload["formula_operators"] = list(self.formula_operators)
        return payload


def check_mechanism_binding(
    formula: str,
    rationale: Mapping[str, Any] | Any | None,
) -> MechanismBindingResult:
    """Check explicit narrative ingredients against a parsed formula AST."""
    tree = try_parse(formula)
    if tree is None:
        return MechanismBindingResult(
            status="formula_parse_failure",
            claims_checked=(),
            contradictions=("Formula could not be parsed for mechanism binding",),
            formula_features=(),
            formula_operators=(),
        )

    features = tuple(sorted(tree.root.leaf_features()))
    operators = tuple(
        sorted(
            {
                node.operator.name
                for node in tree.root.iter_nodes()
                if isinstance(node, OperatorNode)
            }
        )
    )
    narrative = _rationale_text(rationale)
    claims: list[str] = []
    contradictions: list[str] = []

    def assess(name: str, mentioned: bool, present: bool, requirement: str) -> None:
        if not mentioned:
            return
        claims.append(name)
        if not present:
            contradictions.append(f"{name} claim requires {requirement}")

    volume_claim = bool(
        re.search(r"\b(volume|liquidity|order[ -]?flow|trading[ -]?flow|turnover)\b", narrative)
    )
    assess(
        "volume_or_liquidity_input",
        volume_claim,
        bool({"$volume", "$amt"} & set(features)),
        "$volume or $amt",
    )

    volatility_claim = bool(
        re.search(
            r"\b(volatil(?:ity|ity-normalized)|variance|standard deviation|"
            r"risk[ -]?(?:adjusted|normalized|scaled))\b",
            narrative,
        )
    )
    volatility_structure = bool({"Std", "Var"} & set(operators)) or (
        {"$high", "$low"}.issubset(features) and "Sub" in operators
    )
    assess(
        "volatility_normalization",
        volatility_claim,
        volatility_structure,
        "Std/Var or an explicit high-low range",
    )

    momentum_claim = bool(re.search(r"\b(momentum|trend|continuation)\b", narrative))
    momentum_structure = bool(
        {"Delta", "Return", "LogReturn", "TsLinRegSlope", "Slope"} & set(operators)
    ) or ({"Sub", "Delay"}.issubset(operators))
    assess(
        "temporal_momentum",
        momentum_claim,
        momentum_structure,
        "a temporal change/return/slope operator",
    )

    price_claim = bool(
        re.search(r"\b(price[ -]volume|price action|price movement|price return)\b", narrative)
    )
    assess(
        "price_input",
        price_claim,
        bool({"$open", "$high", "$low", "$close", "$vwap", "$returns"} & set(features)),
        "an OHLC, VWAP, or returns leaf",
    )

    if contradictions:
        status = "contradicted"
    elif claims:
        status = "consistent"
    else:
        status = "no_checkable_claims"
    return MechanismBindingResult(
        status=status,
        claims_checked=tuple(claims),
        contradictions=tuple(contradictions),
        formula_features=features,
        formula_operators=operators,
    )


def _rationale_text(rationale: Mapping[str, Any] | Any | None) -> str:
    if isinstance(rationale, Mapping):
        values = [
            rationale.get("financial_semantics", ""),
            rationale.get("market_logic", ""),
        ]
    else:
        values = [
            getattr(rationale, "financial_semantics", ""),
            getattr(rationale, "market_logic", ""),
        ]
    return " ".join(str(value or "") for value in values).lower()

