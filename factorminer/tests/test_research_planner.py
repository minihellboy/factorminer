"""Tests for the mechanism-family taxonomy and the macro research-cycle planner."""

from __future__ import annotations

import pytest

from factorminer.architecture.families import MECHANISM_FAMILIES, infer_family, mechanism_family
from factorminer.architecture.paper_protocol import PaperProtocol
from factorminer.architecture.prompt_context import PromptContextBuilder
from factorminer.architecture.research_planner import CycleThemePlan, ResearchCyclePlanner
from factorminer.utils.config import load_config

# One representative formula per fine-grained `infer_family` branch, covering all 11
# non-"Other" categories plus the "Other" fallback (12 outputs total).
_FINE_FAMILY_FORMULAS: dict[str, str] = {
    "Higher-Moment": "Skew($close, 10)",
    "PV-Correlation": "Corr($close, $volume, 10)",
    "Regime-Conditional": "IfElse(Greater($close, $open), 1, 0)",
    "Regression": "TSLinRegSlope($close, 10)",
    "Smoothing": "EMA($close, 10)",
    "VWAP": "$vwap - $close",
    "Amount": "$amt / $volume",
    "Momentum": "Delta($close, 5)",
    "Volatility": "Std($close, 10)",
    "Extrema": "TSMax($close, 10)",
    "Cross-Sectional": "CSRank($close)",
    "Other": "1 + 1",
}

_EXPECTED_MECHANISM: dict[str, str] = {
    "Higher-Moment": "Volatility/Risk",
    "PV-Correlation": "Price-Volume",
    "Regime-Conditional": "Cross-Sectional/Structural",
    "Regression": "Reversal/Mean-Reversion",
    "Smoothing": "Trend/Momentum",
    "VWAP": "Price-Volume",
    "Amount": "Price-Volume",
    "Momentum": "Trend/Momentum",
    "Volatility": "Volatility/Risk",
    "Extrema": "Reversal/Mean-Reversion",
    "Cross-Sectional": "Cross-Sectional/Structural",
    "Other": "Other/Unclassified",
}


def test_all_infer_family_outputs_are_produced_by_the_fixtures():
    """Sanity check: the fixture formulas actually exercise every infer_family branch."""
    produced = {infer_family(formula) for formula in _FINE_FAMILY_FORMULAS.values()}
    assert produced == set(_FINE_FAMILY_FORMULAS.keys())


def test_mechanism_family_maps_every_fine_family_with_no_unmapped_case():
    for fine_family, expected_mechanism in _EXPECTED_MECHANISM.items():
        mapped = mechanism_family(fine_family)
        assert mapped == expected_mechanism
        assert mapped in MECHANISM_FAMILIES


def test_mechanism_family_is_idempotent_on_coarse_names():
    for name in MECHANISM_FAMILIES:
        assert mechanism_family(name) == name


def test_mechanism_family_falls_back_for_unknown_input():
    assert mechanism_family("Not-A-Real-Family") == "Other/Unclassified"


def _library_state_with_saturated_and_gap_family() -> dict:
    momentum_entries = [
        {"name": f"mom_{i}", "formula": "Delta($close, 5)", "ic_mean": 0.05, "admitted": True}
        for i in range(8)
    ]
    volatility_entry = [
        {"name": "vol_0", "formula": "Std($close, 10)", "ic_mean": 0.0, "admitted": False},
    ]
    return {
        "library_size": 8,
        "recent_admissions": momentum_entries + volatility_entry,
        "categories": {"Momentum": 8, "Volatility": 1},
    }


def test_memory_driven_mode_prioritizes_under_represented_family():
    planner = ResearchCyclePlanner()
    library_state = _library_state_with_saturated_and_gap_family()

    plan = planner.plan_cycle(library_state, mode="memory_driven")

    assert isinstance(plan, CycleThemePlan)
    # Momentum (8/8 admitted) collapses into the saturated "Trend/Momentum" mechanism
    # family; Volatility (0/1 admitted) collapses into "Volatility/Risk", which has
    # both zero admissions and is not saturated -> it must win as the primary theme.
    assert plan.primary_family == "Volatility/Risk"
    assert plan.family_stats["Trend/Momentum"]["saturated"] is True
    assert plan.family_stats["Volatility/Risk"]["admitted"] == 0
    assert "Trend/Momentum" not in plan.supporting_families or plan.primary_family != "Trend/Momentum"
    assert 2 <= len(plan.supporting_families) <= 3
    assert plan.primary_family not in plan.supporting_families
    assert "Volatility/Risk" in plan.prompt_text


def test_memory_driven_mode_with_no_signal_defaults_without_crashing():
    planner = ResearchCyclePlanner()
    plan = planner.plan_cycle(
        library_state={"library_size": 0, "recent_admissions": [], "categories": {}},
        mode="memory_driven",
    )
    assert plan.primary_family in MECHANISM_FAMILIES
    assert 2 <= len(plan.supporting_families) <= 3


def test_fixed_mode_maps_fine_family_to_mechanism_family():
    planner = ResearchCyclePlanner()
    plan = planner.plan_cycle(
        library_state={"library_size": 0, "recent_admissions": [], "categories": {}},
        mode="fixed",
        fixed_family="Momentum",
    )
    assert plan.mode == "fixed"
    assert plan.primary_family == "Trend/Momentum"


def test_fixed_mode_accepts_an_already_coarse_family_unchanged():
    planner = ResearchCyclePlanner()
    plan = planner.plan_cycle(
        library_state={"library_size": 0, "recent_admissions": [], "categories": {}},
        mode="fixed",
        fixed_family="Price-Volume",
    )
    assert plan.primary_family == "Price-Volume"


def test_fixed_mode_requires_fixed_family():
    planner = ResearchCyclePlanner()
    with pytest.raises(ValueError):
        planner.plan_cycle(
            library_state={"library_size": 0, "recent_admissions": [], "categories": {}},
            mode="fixed",
        )


def test_coarse_guided_mode_routes_hint_to_nearest_mechanism_family():
    planner = ResearchCyclePlanner()
    library_state = {"library_size": 0, "recent_admissions": [], "categories": {}}

    volatility_plan = planner.plan_cycle(
        library_state,
        mode="coarse_guided",
        coarse_hint="let's dig into volatility clustering and tail risk",
    )
    assert volatility_plan.primary_family == "Volatility/Risk"

    cross_sectional_plan = planner.plan_cycle(
        library_state,
        mode="coarse_guided",
        coarse_hint="explore cross-sectional rank effects across sectors",
    )
    assert cross_sectional_plan.primary_family == "Cross-Sectional/Structural"


def test_coarse_guided_mode_requires_coarse_hint():
    planner = ResearchCyclePlanner()
    with pytest.raises(ValueError):
        planner.plan_cycle(
            library_state={"library_size": 0, "recent_admissions": [], "categories": {}},
            mode="coarse_guided",
        )


def test_unknown_mode_raises():
    planner = ResearchCyclePlanner()
    with pytest.raises(ValueError):
        planner.plan_cycle(
            library_state={"library_size": 0, "recent_admissions": [], "categories": {}},
            mode="not-a-mode",  # type: ignore[arg-type]
        )


def test_cycle_theme_plan_to_dict_is_json_safe():
    planner = ResearchCyclePlanner()
    plan = planner.plan_cycle(
        library_state=_library_state_with_saturated_and_gap_family(),
        mode="memory_driven",
    )
    payload = plan.to_dict()
    assert payload["primary_family"] == plan.primary_family
    assert payload["mode"] == "memory_driven"
    assert isinstance(payload["supporting_families"], list)
    assert isinstance(payload["family_stats"], dict)


def test_prompt_context_builder_includes_research_cycle_theme():
    cfg = load_config()
    protocol = PaperProtocol.from_config(cfg)
    builder = PromptContextBuilder(
        protocol=protocol,
        cycle_planner=ResearchCyclePlanner(),
    )

    payload = builder.build(
        memory_signal={"recommended_directions": []},
        library_state=_library_state_with_saturated_and_gap_family(),
        batch_size=8,
    )

    assert "research_cycle_theme" in payload
    assert "research_cycle_prompt_text" in payload
    assert payload["research_cycle_theme"]["primary_family"] == "Volatility/Risk"
    # Additive alongside (never replacing) the family-discovery field this builder
    # already exposes, and independent of whatever ResearchAbsorption's own field adds.
    assert "family_prompt_text" not in payload or isinstance(payload["family_prompt_text"], str)


def test_prompt_context_builder_without_cycle_planner_omits_the_field():
    cfg = load_config()
    protocol = PaperProtocol.from_config(cfg)
    builder = PromptContextBuilder(protocol=protocol)

    payload = builder.build(
        memory_signal={"recommended_directions": []},
        library_state={"library_size": 0, "recent_admissions": [], "categories": {}},
        batch_size=8,
    )

    assert "research_cycle_theme" not in payload
    assert "research_cycle_prompt_text" not in payload
