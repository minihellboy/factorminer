"""Persistent research knowledge and attribution tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from factorminer.agent.llm_interface import MockProvider
from factorminer.application.research_knowledge import ResearchKnowledgeStore
from factorminer.architecture.research_absorption import (
    ResearchAbsorptionService,
    ResearchNote,
)

ELIGIBLE_NOTE = (
    "Heavy selling volume near a recent price low can indicate exhaustion and a "
    "short-term reversal bounce in subsequent daily bars."
)
INELIGIBLE_NOTE = "Analyst EPS revisions after quarterly earnings predict returns."


def _service() -> ResearchAbsorptionService:
    return ResearchAbsorptionService(MockProvider())


def test_ingestion_persists_eligible_and_ineligible_sources(tmp_path):
    store = ResearchKnowledgeStore(tmp_path)
    source, hypothesis = store.ingest(
        ResearchNote(ELIGIBLE_NOTE, "paper-a"),
        _service(),
    )
    dropped, dropped_hypothesis = store.ingest(
        ResearchNote(INELIGIBLE_NOTE, "paper-b"),
        _service(),
    )

    assert source.eligible is True
    assert hypothesis is not None
    assert hypothesis.source_id == source.source_id
    assert dropped.eligible is False
    assert dropped_hypothesis is None
    assert (store.sources_dir / f"{source.source_id}.json").exists()
    assert (store.sources_dir / f"{dropped.source_id}.json").exists()
    assert (store.hypotheses_dir / f"{hypothesis.hypothesis_id}.json").exists()

    repeated_source, repeated_hypothesis = store.ingest(
        ResearchNote(ELIGIBLE_NOTE, "paper-a"),
        _service(),
    )
    assert repeated_source == source
    assert repeated_hypothesis == hypothesis


def test_retrieval_is_bounded_and_carries_exact_ids(tmp_path):
    store = ResearchKnowledgeStore(tmp_path)
    _, first = store.ingest(ResearchNote(ELIGIBLE_NOTE, "paper-a"), _service())
    _, second = store.ingest(
        ResearchNote(
            "High volume price momentum and trend continuation can persist across daily bars.",
            "paper-b",
        ),
        _service(),
    )
    assert first is not None and second is not None

    retrieval = store.retrieve({"category_counts": {}}, limit=1)
    signal = retrieval.enrich({"recommended_directions": []})

    assert len(retrieval.hypothesis_ids) == 1
    assert len(signal["research_archetypes"]) == 1
    assert signal["source_ids"] == list(retrieval.source_ids)
    assert signal["hypothesis_ids"] == list(retrieval.hypothesis_ids)


def test_candidate_outcomes_are_attributed_once(tmp_path):
    store = ResearchKnowledgeStore(tmp_path)
    source, hypothesis = store.ingest(
        ResearchNote(ELIGIBLE_NOTE, "paper-a"),
        _service(),
    )
    assert hypothesis is not None
    signal = {
        "source_ids": [source.source_id],
        "hypothesis_ids": [hypothesis.hypothesis_id],
    }
    result = SimpleNamespace(
        factor_name="reversal_1",
        formula="Neg($close)",
        admitted=True,
        rejection_reason="",
        ic_mean=0.05,
        ic_paper_mean=0.05,
        icir=0.7,
        max_correlation=0.2,
        stage_passed=4,
    )

    records = store.record_results([result], signal, iteration=3)
    repeated = store.record_results([result], signal, iteration=3)

    assert len(records) == 1
    assert repeated == ()
    payload = json.loads(
        (store.outcomes_dir / f"{records[0].outcome_id}.json").read_text(encoding="utf-8")
    )
    assert payload["admitted"] is True
    assert payload["hypothesis_ids"] == [hypothesis.hypothesis_id]
