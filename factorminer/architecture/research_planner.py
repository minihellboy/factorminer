"""Macro research-cycle planner routed by coarse mechanism family (XAlpha Macro Brain).

XAlpha (arXiv, Jul 2026 -- see `docs/landscape-and-extensions.md` §1) plans research
cycles with a "Macro Brain" that names a cycle theme and routes generation toward a
mechanism family, in one of three modes: fixed-theme, coarse-guided, or memory-driven.
This module implements the equivalent surface for FactorMiner, reusing
`architecture.families.FactorFamilyDiscovery`'s existing saturation/gap diagnostics
rather than reimplementing family-saturation detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from factorminer.architecture.families import (
    MECHANISM_FAMILIES,
    FactorFamilyDiscovery,
    mechanism_family,
)

logger = logging.getLogger(__name__)

RoutingMode = Literal["fixed", "coarse_guided", "memory_driven"]

# Keyword buckets used by `coarse_guided` mode to map a free-text hint onto the
# nearest mechanism family. Deliberately simple (substring keyword match) -- this is
# a routing heuristic, not an NLP classifier.
_COARSE_HINT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Trend/Momentum": ("trend", "momentum", "breakout", "drift", "moving average", "smooth"),
    "Reversal/Mean-Reversion": (
        "revers",
        "mean revert",
        "mean-revert",
        "contrarian",
        "overbought",
        "oversold",
        "regression",
        "extrema",
    ),
    "Volatility/Risk": ("vol", "risk", "variance", "skew", "kurt", "tail", "dispersion"),
    "Price-Volume": ("volume", "vwap", "amount", "liquidity", "turnover", "price-volume", "price/volume"),
    "Cross-Sectional/Structural": (
        "cross-sectional",
        "cross sectional",
        "rank",
        "regime",
        "structural",
        "sector",
        "conditional",
    ),
}


@dataclass
class _MechanismStats:
    """Aggregated admission/rejection stats for one mechanism family."""

    attempts: int = 0
    admitted: int = 0
    saturated: bool = False
    underexplored: bool = False
    recommended: bool = False
    gap_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "admitted": self.admitted,
            "saturated": self.saturated,
            "underexplored": self.underexplored,
            "recommended": self.recommended,
            "gap_score": round(self.gap_score, 4),
        }


@dataclass
class CycleThemePlan:
    """A planned research-cycle theme: one primary mechanism family plus support.

    Mirrors XAlpha's Macro Brain output: a cycle is themed around one primary
    mechanism family, with 2-3 supporting families named alongside it so retrieval
    and generation aren't routed to a single point.
    """

    mode: str
    primary_family: str
    supporting_families: list[str] = field(default_factory=list)
    rationale: str = ""
    family_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    prompt_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this plan."""
        return {
            "mode": self.mode,
            "primary_family": self.primary_family,
            "supporting_families": list(self.supporting_families),
            "rationale": self.rationale,
            "family_stats": self.family_stats,
            "prompt_text": self.prompt_text,
        }


class ResearchCyclePlanner:
    """XAlpha-style Macro Brain: plans the next research cycle's mechanism theme.

    Routes to a primary mechanism family (plus supporting families) using one of
    three modes:

    - ``fixed``: caller names an explicit family (fine-grained `infer_family`
      category or an already-coarse mechanism family); used as-is.
    - ``coarse_guided``: caller supplies a rough natural-language hint (e.g. "look at
      volatility clustering"), mapped to the nearest mechanism family by keyword.
    - ``memory_driven``: the theme is derived purely from library-state gaps, reusing
      `FactorFamilyDiscovery`'s existing saturation/gap diagnostics. Mechanism
      families with the fewest admissions / most rejections relative to attempts are
      prioritized; already-saturated families are deprioritized.

    Family-saturation detection itself is never reimplemented here --
    `FactorFamilyDiscovery.summarize` remains the single source of truth for which
    fine-grained families are saturated, underexplored, or memory-recommended; this
    planner only re-aggregates that signal at the coarser mechanism-family level.
    """

    def __init__(self, family_discovery: FactorFamilyDiscovery | None = None) -> None:
        self._family_discovery = family_discovery or FactorFamilyDiscovery()

    def plan_cycle(
        self,
        library_state: dict[str, Any],
        memory_signal: dict[str, Any] | None = None,
        *,
        mode: RoutingMode = "memory_driven",
        fixed_family: str | None = None,
        coarse_hint: str | None = None,
        num_supporting: int = 2,
    ) -> CycleThemePlan:
        """Plan the next research cycle's mechanism-family theme.

        Parameters
        ----------
        library_state:
            Current factor-library state, in the shape consumed by
            `FactorFamilyDiscovery.summarize` (``recent_admissions``, ``categories``).
        memory_signal:
            Memory-policy retrieval signal, in the shape consumed by
            `FactorFamilyDiscovery.summarize` (``recommended_directions``).
        mode:
            Routing mode: ``"fixed"``, ``"coarse_guided"``, or ``"memory_driven"``.
        fixed_family:
            Required when ``mode="fixed"``. A fine-grained `infer_family` category or
            an already-coarse mechanism family name; mapped up via `mechanism_family`.
        coarse_hint:
            Required when ``mode="coarse_guided"``. Free-text hint mapped to the
            nearest mechanism family by keyword match.
        num_supporting:
            Requested number of supporting mechanism families (clamped to 2-3, the
            range XAlpha's Macro Brain uses).

        Returns
        -------
        CycleThemePlan

        Raises
        ------
        ValueError
            If ``mode`` is unrecognized, or a mode-required argument is missing.
        """
        num_supporting = max(2, min(3, num_supporting))
        ranked = self._rank_mechanism_families(library_state, memory_signal)
        stats_by_name = {name: stats for name, stats in ranked}

        if mode == "fixed":
            if not fixed_family:
                raise ValueError("mode='fixed' requires fixed_family")
            primary = mechanism_family(fixed_family)
            rationale = f"Fixed theme requested explicitly: '{fixed_family}' -> {primary}."
        elif mode == "coarse_guided":
            if not coarse_hint:
                raise ValueError("mode='coarse_guided' requires coarse_hint")
            primary = self._nearest_mechanism_family(coarse_hint)
            rationale = f"Coarse hint {coarse_hint!r} routed to nearest mechanism family: {primary}."
        elif mode == "memory_driven":
            if ranked:
                top_name, top_stats = ranked[0]
                primary = top_name
                rationale = (
                    f"Memory-driven: {primary} has the largest admission gap "
                    f"(admitted={top_stats.admitted}/{top_stats.attempts} attempts, "
                    f"saturated={top_stats.saturated})."
                )
            else:
                primary = MECHANISM_FAMILIES[0]
                rationale = "Memory-driven: no library/memory signal available; defaulted to first mechanism family."
        else:
            raise ValueError(f"Unknown routing mode: {mode!r}")

        supporting = [name for name, _ in ranked if name != primary][:num_supporting]
        if len(supporting) < num_supporting:
            for name in MECHANISM_FAMILIES:
                if name != primary and name not in supporting:
                    supporting.append(name)
                if len(supporting) >= num_supporting:
                    break

        plan = CycleThemePlan(
            mode=mode,
            primary_family=primary,
            supporting_families=supporting,
            rationale=rationale,
            family_stats={name: stats.to_dict() for name, stats in stats_by_name.items()},
        )
        plan.prompt_text = self._prompt_text(plan)
        return plan

    def _rank_mechanism_families(
        self,
        library_state: dict[str, Any],
        memory_signal: dict[str, Any] | None,
    ) -> list[tuple[str, _MechanismStats]]:
        """Aggregate `FactorFamilyDiscovery`'s fine-grained signal to mechanism level.

        Reuses `FactorFamilyDiscovery.summarize` (attempts/admissions per fine
        family, saturated families, underexplored families, memory-recommended
        families) and re-groups it by `mechanism_family`, then ranks mechanism
        families by an admission-gap score: highest for families with the fewest
        admissions relative to attempts (most rejections), lowest (heavily
        penalized) for families already flagged saturated.
        """
        summary = self._family_discovery.summarize(
            library_state=library_state,
            memory_signal=memory_signal,
        )
        saturated_raw = set(summary.get("saturated_families", []) or [])
        underexplored_raw = set(summary.get("underexplored_families", []) or [])
        recommended_raw = set(summary.get("recommended_families", []) or [])

        stats: dict[str, _MechanismStats] = {name: _MechanismStats() for name in MECHANISM_FAMILIES}

        for entry in summary.get("families", []) or []:
            raw_name = str(entry.get("name", "Other") or "Other")
            mech = mechanism_family(raw_name)
            mech_stats = stats.setdefault(mech, _MechanismStats())
            mech_stats.attempts += int(entry.get("count", 0) or 0)
            mech_stats.admitted += int(entry.get("admitted_count", 0) or 0)
            if raw_name in saturated_raw:
                mech_stats.saturated = True
            if raw_name in underexplored_raw:
                mech_stats.underexplored = True
            if raw_name in recommended_raw:
                mech_stats.recommended = True

        for raw_name in underexplored_raw:
            stats.setdefault(mechanism_family(raw_name), _MechanismStats()).underexplored = True
        for raw_name in recommended_raw:
            stats.setdefault(mechanism_family(raw_name), _MechanismStats()).recommended = True

        for mech_stats in stats.values():
            attempts = mech_stats.attempts
            rejected = max(attempts - mech_stats.admitted, 0)
            # No attempts at all is treated as maximal gap (rejection_rate=1.0,
            # admission_rate=0.0), same as a family that was tried and always rejected.
            rejection_rate = (rejected / attempts) if attempts else 1.0
            admission_rate = (mech_stats.admitted / attempts) if attempts else 0.0
            gap = rejection_rate - admission_rate
            if mech_stats.recommended:
                gap += 0.5
            if mech_stats.underexplored:
                gap += 0.25
            if mech_stats.saturated:
                gap -= 1.0
            mech_stats.gap_score = gap

        return sorted(
            stats.items(),
            key=lambda kv: (kv[1].gap_score, -kv[1].attempts),
            reverse=True,
        )

    def _nearest_mechanism_family(self, hint: str) -> str:
        """Map a free-text hint to the nearest mechanism family by keyword overlap."""
        hint_lower = hint.lower()
        best_family = "Other/Unclassified"
        best_score = 0
        for family_name, keywords in _COARSE_HINT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in hint_lower)
            if score > best_score:
                best_score = score
                best_family = family_name
        return best_family

    def _prompt_text(self, plan: CycleThemePlan) -> str:
        lines = ["=== RESEARCH CYCLE THEME (Macro Brain) ==="]
        lines.append(f"Mode: {plan.mode}")
        lines.append(f"Primary mechanism family: {plan.primary_family}")
        if plan.supporting_families:
            lines.append(f"Supporting families: {', '.join(plan.supporting_families)}")
        lines.append(plan.rationale)
        return "\n".join(lines)
