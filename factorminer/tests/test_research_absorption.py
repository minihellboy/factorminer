"""Tests for the XAlpha-style Report-to-Memory Absorption (RMA) service."""

from __future__ import annotations

from factorminer.agent.llm_interface import MockProvider
from factorminer.architecture.families import MECHANISM_FAMILIES
from factorminer.architecture.paper_protocol import PaperProtocol
from factorminer.architecture.prompt_context import PromptContextBuilder
from factorminer.architecture.research_absorption import (
    ResearchAbsorptionService,
    ResearchArchetype,
    ResearchNote,
)
from factorminer.utils.config import load_config

INELIGIBLE_NOTE = (
    "Recent sell-side analyst EPS revisions and consensus estimate changes suggest "
    "earnings momentum for this stock is deteriorating quarter over quarter, driven "
    "by weaker forward guidance from management."
)

ELIGIBLE_NOTE = (
    "After several consecutive days of heavy selling volume, price shows signs of "
    "selling exhaustion near the recent low, often followed by a short-term reversal "
    "bounce over the next few sessions."
)


def _service() -> ResearchAbsorptionService:
    return ResearchAbsorptionService(llm_provider=MockProvider())


# ---------------------------------------------------------------------------
# ResearchNote / ResearchArchetype dataclasses
# ---------------------------------------------------------------------------

def test_research_note_and_archetype_are_json_safe_dataclasses():
    note = ResearchNote(text=ELIGIBLE_NOTE, source="unit-test")
    assert note.to_dict() == {"text": ELIGIBLE_NOTE, "source": "unit-test"}

    archetype = ResearchArchetype(
        name="test_archetype",
        mechanism_family="Reversal/Mean-Reversion",
        fine_family="Extrema",
        mechanism_role="role",
        research_paths=["cue 1", "cue 2"],
        source_text=ELIGIBLE_NOTE,
        eligibility_reason="reason",
    )
    payload = archetype.to_dict()
    assert payload["mechanism_family"] == "Reversal/Mean-Reversion"
    assert payload["research_paths"] == ["cue 1", "cue 2"]


# ---------------------------------------------------------------------------
# (a) A-layer eligibility gate under the mock provider
# ---------------------------------------------------------------------------

def test_screen_eligibility_drops_analyst_eps_note_under_mock():
    service = _service()
    keep, reason = service.screen_eligibility(INELIGIBLE_NOTE)
    assert keep is False
    assert reason


def test_screen_eligibility_keeps_price_reversal_note_under_mock():
    service = _service()
    keep, reason = service.screen_eligibility(ELIGIBLE_NOTE)
    assert keep is True
    assert reason


def test_screen_eligibility_alt_enabled_keeps_known_alt_leaf_note_standalone():
    """`alt_enabled` must be useful when called standalone (no prior EDGAR/futures
    loader has run register_features() in this process) -- it should default to the
    known alt-data leaf catalog, not just whatever happens to be live in the global
    registry. Regression test for a bug caught in Round 2 CLI verification where
    `ingest-research --eligibility-mode alt_enabled` never actually kept anything
    because the process-local feature registry was always just the 8 OHLCV defaults."""
    service = ResearchAbsorptionService(llm_provider=MockProvider(), eligibility_mode="alt_enabled")
    keep, reason = service.screen_eligibility(INELIGIBLE_NOTE)
    assert keep is True
    assert "$eps" in reason

    # ohlcv_only (the default) must still drop the exact same note.
    default_service = _service()
    keep_default, _ = default_service.screen_eligibility(INELIGIBLE_NOTE)
    assert keep_default is False


def test_classify_mechanism_assigns_mechanism_family_from_shared_taxonomy():
    service = _service()
    archetype = service.classify_mechanism(ELIGIBLE_NOTE)
    assert isinstance(archetype, ResearchArchetype)
    assert archetype.mechanism_family in MECHANISM_FAMILIES
    # A price-reversal-after-selling-exhaustion note should land in the
    # Extrema fine family -> Reversal/Mean-Reversion mechanism family.
    assert archetype.fine_family == "Extrema"
    assert archetype.mechanism_family == "Reversal/Mean-Reversion"
    assert archetype.research_paths
    assert archetype.name


# ---------------------------------------------------------------------------
# (b) absorb() batch pipeline
# ---------------------------------------------------------------------------

def test_absorb_produces_one_archetype_per_eligible_note():
    service = _service()
    other_eligible_note = (
        "Stocks with unusually high turnover relative to their own 20-day average "
        "volume tend to show elevated short-term volatility and momentum continuation "
        "in the following sessions."
    )
    notes = [INELIGIBLE_NOTE, ELIGIBLE_NOTE, other_eligible_note]

    archetypes = service.absorb(notes)

    assert len(archetypes) == 2
    assert all(isinstance(a, ResearchArchetype) for a in archetypes)
    assert all(a.mechanism_family in MECHANISM_FAMILIES for a in archetypes)
    # eligibility_reason is threaded through from the A-layer gate.
    assert all(a.eligibility_reason for a in archetypes)


def test_absorb_on_empty_batch_returns_empty_list():
    service = _service()
    assert service.absorb([]) == []


# ---------------------------------------------------------------------------
# (c) PromptContextBuilder: additive, default-empty, byte-identical when unused
# ---------------------------------------------------------------------------

def _builder() -> PromptContextBuilder:
    protocol = PaperProtocol.from_config(load_config())
    return PromptContextBuilder(protocol=protocol)


def test_prompt_context_unchanged_when_no_archetypes_supplied():
    builder_without = _builder()
    builder_with = _builder()

    memory_signal = {"recommended_directions": []}
    library_state = {"size": 3, "target_size": 10}

    payload_without_kw = builder_without.build(memory_signal, library_state, batch_size=8)
    payload_with_explicit_none = builder_with.build(
        memory_signal, library_state, batch_size=8, research_archetypes=None
    )

    assert payload_without_kw == payload_with_explicit_none
    assert "research_archetypes" not in payload_without_kw
    assert "research_prompt_text" not in payload_without_kw


def test_prompt_context_includes_research_path_text_when_archetypes_supplied():
    service = _service()
    archetypes = service.absorb([ELIGIBLE_NOTE])
    assert archetypes

    builder = _builder()
    memory_signal = {"recommended_directions": []}
    library_state = {"size": 3, "target_size": 10}

    payload = builder.build(
        memory_signal,
        library_state,
        batch_size=8,
        research_archetypes=archetypes,
    )

    assert "research_archetypes" in payload
    assert payload["research_archetypes"][0]["name"] == archetypes[0].name
    assert "research_prompt_text" in payload
    assert "RESEARCH ARCHETYPE CONTEXT" in payload["research_prompt_text"]
    for path in archetypes[0].research_paths:
        assert path in payload["research_prompt_text"]
