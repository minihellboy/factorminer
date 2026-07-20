"""Factor-family discovery and prompt-facing family diagnostics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

_FEATURE_PATTERN = re.compile(r"\$[a-zA-Z_]+")
_OPERATOR_PATTERN = re.compile(r"([A-Za-z][a-zA-Z]+)\(")


def extract_operators(formula: str) -> list[str]:
    return re.findall(_OPERATOR_PATTERN, formula)


def extract_features(formula: str) -> list[str]:
    return re.findall(_FEATURE_PATTERN, formula)


def infer_family(formula: str) -> str:
    """Infer a stable factor family from a formula string."""
    formula_upper = formula.upper()
    ops = {op.upper() for op in extract_operators(formula)}

    if ops & {"SKEW", "KURT"}:
        return "Higher-Moment"
    if ops & {"CORR", "COV", "BETA"} and "$VOLUME" in formula_upper:
        return "PV-Correlation"
    if ops & {"IFELSE", "GREATER", "LESS", "OR", "AND"}:
        return "Regime-Conditional"
    if ops & {"TSLINREG", "TSLINREGSLOPE", "TSLINREGRESID", "RESID"}:
        return "Regression"
    if ops & {"EMA", "DEMA", "KAMA", "HMA", "WMA", "SMA"}:
        return "Smoothing"
    if "$VWAP" in formula_upper:
        return "VWAP"
    if "$AMT" in formula_upper:
        return "Amount"
    if ops & {"DELTA", "DELAY", "RETURN", "LOGRETURN"}:
        return "Momentum"
    if ops & {"STD", "VAR"}:
        return "Volatility"
    if ops & {"TSMAX", "TSMIN", "TSARGMAX", "TSARGMIN", "TSRANK"}:
        return "Extrema"
    if ops & {"CSRANK", "CSZSCORE", "CSDEMEAN"}:
        return "Cross-Sectional"
    return "Other"

# Coarse "mechanism family" taxonomy (XAlpha-style Macro Brain routing, see
# docs/landscape-and-extensions.md §1 and docs/architecture.md "Family Discovery").
# Groups the 11 fine-grained `infer_family` categories (plus the `Other` fallback)
# into 6 broader mechanisms a research cycle can be themed around. This grouping is
# additive/local to macro-cycle planning; it does not replace or alter `infer_family`.
#
# - Trend/Momentum: directional/price-change signals and the moving-average style
#   smoothers typically used to follow them (Momentum, Smoothing).
# - Reversal/Mean-Reversion: signals framed around deviation from a local extreme or
#   a fitted trend, the two standard mean-reversion setups (Extrema, Regression).
# - Volatility/Risk: dispersion and higher-moment (tail) risk signals (Volatility,
#   Higher-Moment).
# - Price-Volume: signals that need volume/amount/VWAP alongside price (VWAP, Amount,
#   PV-Correlation).
# - Cross-Sectional/Structural: signals defined relative to the cross-section or a
#   conditioning/regime split rather than a single series' own history
#   (Cross-Sectional, Regime-Conditional).
# - Other/Unclassified: catch-all for `infer_family`'s own `Other` fallback.
_MECHANISM_FAMILY_MAP: dict[str, str] = {
    "Momentum": "Trend/Momentum",
    "Smoothing": "Trend/Momentum",
    "Extrema": "Reversal/Mean-Reversion",
    "Regression": "Reversal/Mean-Reversion",
    "Volatility": "Volatility/Risk",
    "Higher-Moment": "Volatility/Risk",
    "VWAP": "Price-Volume",
    "Amount": "Price-Volume",
    "PV-Correlation": "Price-Volume",
    "Cross-Sectional": "Cross-Sectional/Structural",
    "Regime-Conditional": "Cross-Sectional/Structural",
    "Other": "Other/Unclassified",
}

MECHANISM_FAMILIES: tuple[str, ...] = (
    "Trend/Momentum",
    "Reversal/Mean-Reversion",
    "Volatility/Risk",
    "Price-Volume",
    "Cross-Sectional/Structural",
    "Other/Unclassified",
)


def mechanism_family(fine_family: str) -> str:
    """Map a fine-grained `infer_family` category to a coarse mechanism family.

    This groups the 11 categories `infer_family` can return (plus its `Other`
    fallback) into the 6 `MECHANISM_FAMILIES` buckets documented above. Any
    unrecognized input (e.g. a raw formula, or a name that isn't one of
    `infer_family`'s outputs) safely falls back to ``"Other/Unclassified"`` rather
    than raising, so callers can pass either a fine family name or an
    already-coarse mechanism family name through unchanged.

    Parameters
    ----------
    fine_family:
        A category name as returned by `infer_family`, or already one of
        `MECHANISM_FAMILIES`.

    Returns
    -------
    str
        One of `MECHANISM_FAMILIES`.
    """
    if fine_family in MECHANISM_FAMILIES:
        return fine_family
    return _MECHANISM_FAMILY_MAP.get(fine_family, "Other/Unclassified")



@dataclass
class FactorFamily:
    name: str
    count: int = 0
    admitted_count: int = 0
    average_ic: float = 0.0
    operators: dict[str, int] = field(default_factory=dict)
    features: dict[str, int] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FactorFamilyDiscovery:
    """Discover family structure and prompt-facing gaps from formulas/library state."""

    def summarize(
        self,
        *,
        library_state: dict[str, Any],
        memory_signal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entries = list(library_state.get("recent_admissions", []) or [])
        entries.extend(
            {
                "name": pattern.get("name", ""),
                "formula": pattern.get("template", ""),
                "ic_mean": 0.0,
                "admitted": False,
            }
            for pattern in (memory_signal or {}).get("recommended_directions", [])
        )
        families = self.discover(entries)
        saturated = self._saturated_families(library_state, families)
        underexplored = self._underexplored_families(memory_signal, families)
        recommended = self._recommended_families(memory_signal)
        return {
            "families": [family.to_dict() for family in families],
            "saturated_families": saturated,
            "underexplored_families": underexplored,
            "recommended_families": recommended,
            "prompt_text": self._prompt_text(families, saturated, underexplored, recommended),
        }

    def discover(self, entries: list[dict[str, Any]]) -> list[FactorFamily]:
        family_map: dict[str, FactorFamily] = {}
        ic_totals: dict[str, float] = {}

        for entry in entries:
            formula = str(entry.get("formula", "") or "")
            if not formula:
                family_name = str(entry.get("category", "") or entry.get("name", "") or "Other")
            else:
                family_name = infer_family(formula)
            family = family_map.setdefault(family_name, FactorFamily(name=family_name))
            family.count += 1
            family.admitted_count += int(bool(entry.get("admitted", True)))
            ic_totals[family_name] = ic_totals.get(family_name, 0.0) + float(entry.get("ic_mean", 0.0))
            for op in extract_operators(formula):
                family.operators[op] = family.operators.get(op, 0) + 1
            for feature in extract_features(formula):
                family.features[feature] = family.features.get(feature, 0) + 1
            if formula and len(family.examples) < 3 and formula not in family.examples:
                family.examples.append(formula)

        for name, family in family_map.items():
            if family.count:
                family.average_ic = ic_totals.get(name, 0.0) / family.count

        return sorted(
            family_map.values(),
            key=lambda family: (family.admitted_count, family.count, family.average_ic),
            reverse=True,
        )

    def _saturated_families(
        self,
        library_state: dict[str, Any],
        families: list[FactorFamily],
    ) -> list[str]:
        category_counts = dict(library_state.get("categories", {}) or {})
        if not category_counts and families:
            category_counts = {family.name: family.count for family in families}
        if not category_counts:
            return []
        avg_count = sum(category_counts.values()) / max(len(category_counts), 1)
        return sorted(
            name for name, count in category_counts.items() if count >= max(2.0, avg_count * 1.5)
        )

    def _underexplored_families(
        self,
        memory_signal: dict[str, Any] | None,
        families: list[FactorFamily],
    ) -> list[str]:
        current = {family.name for family in families}
        recommended = set(self._recommended_families(memory_signal))
        missing = sorted(recommended - current)
        if missing:
            return missing
        low_count = [family.name for family in families if family.count <= 1]
        return sorted(low_count)

    def _recommended_families(self, memory_signal: dict[str, Any] | None) -> list[str]:
        families: set[str] = set()
        for pattern in (memory_signal or {}).get("recommended_directions", []):
            template = str(pattern.get("template", "") or "")
            name = str(pattern.get("name", "") or "")
            if template:
                families.add(infer_family(template))
            elif name:
                families.add(name)
        return sorted(families)

    def _prompt_text(
        self,
        families: list[FactorFamily],
        saturated: list[str],
        underexplored: list[str],
        recommended: list[str],
    ) -> str:
        lines = ["=== FACTOR FAMILY CONTEXT ==="]
        if families:
            top = ", ".join(f"{family.name} ({family.count})" for family in families[:5])
            lines.append(f"Current family mix: {top}")
        if saturated:
            lines.append(f"Saturated families: {', '.join(saturated)}")
        if underexplored:
            lines.append(f"Underexplored families: {', '.join(underexplored)}")
        if recommended:
            lines.append(f"Recommended families from memory: {', '.join(recommended)}")
        return "\n".join(lines)
