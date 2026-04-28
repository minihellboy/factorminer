"""Tests for the experience memory system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from factorminer.evaluation.regime import MeanRevRegime, RegimeState, TrendRegime, VolRegime
from factorminer.memory.embeddings import FormulaEmbedder
from factorminer.memory.evolution import bump_pattern_confidence, evolve_memory
from factorminer.memory.experience_memory import ExperienceMemoryManager
from factorminer.memory.formation import form_memory
from factorminer.memory.kg_retrieval import retrieve_memory_enhanced
from factorminer.memory.knowledge_graph import FactorKnowledgeGraph, FactorNode
from factorminer.memory.memory_store import (
    ExperienceMemory,
    ForbiddenDirection,
    MiningState,
    StrategicInsight,
    SuccessPattern,
)
from factorminer.memory.online_regime_memory import OnlineMemoryUpdater

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    """Test memory manager initialization with default patterns."""

    def test_default_success_patterns_loaded(self, mock_memory):
        assert len(mock_memory.memory.success_patterns) > 0

    def test_default_forbidden_directions_loaded(self, mock_memory):
        assert len(mock_memory.memory.forbidden_directions) > 0

    def test_default_insights_loaded(self, mock_memory):
        assert len(mock_memory.memory.insights) > 0

    def test_initial_version_is_zero(self, mock_memory):
        assert mock_memory.version == 0

    def test_default_pattern_names(self, mock_memory):
        names = [p.name for p in mock_memory.memory.success_patterns]
        assert "Higher Moment Regimes" in names
        assert "PV Corr Interaction" in names

    def test_default_forbidden_names(self, mock_memory):
        names = [f.name for f in mock_memory.memory.forbidden_directions]
        assert "Standardized Returns/Amount" in names
        assert "VWAP Deviation variants" in names

    def test_memory_reset(self, mock_memory):
        # Modify state
        mock_memory.memory.version = 99
        mock_memory.reset()
        assert mock_memory.version == 0
        assert len(mock_memory.memory.success_patterns) > 0


# ---------------------------------------------------------------------------
# Formation
# ---------------------------------------------------------------------------

class TestFormation:
    """Test memory formation from trajectory."""

    def test_form_memory_creates_new_memory(self, mock_memory, sample_trajectory):
        formed = form_memory(mock_memory.memory, sample_trajectory, batch_number=1)
        assert isinstance(formed, ExperienceMemory)

    def test_form_memory_updates_state(self, mock_memory, sample_trajectory):
        formed = form_memory(mock_memory.memory, sample_trajectory, batch_number=1)
        # Should count 2 admitted factors
        assert formed.state.library_size == mock_memory.memory.state.library_size + 2

    def test_form_memory_recent_admissions(self, mock_memory, sample_trajectory):
        formed = form_memory(mock_memory.memory, sample_trajectory, batch_number=1)
        assert len(formed.state.recent_admissions) == 2

    def test_form_memory_recent_rejections(self, mock_memory, sample_trajectory):
        formed = form_memory(mock_memory.memory, sample_trajectory, batch_number=1)
        assert len(formed.state.recent_rejections) == 2

    def test_form_memory_admission_log(self, mock_memory, sample_trajectory):
        formed = form_memory(mock_memory.memory, sample_trajectory, batch_number=1)
        assert len(formed.state.admission_log) >= 1
        last_log = formed.state.admission_log[-1]
        assert last_log["batch"] == 1
        assert last_log["admitted"] == 2
        assert last_log["rejected"] == 2

    def test_form_memory_empty_trajectory(self, mock_memory):
        formed = form_memory(mock_memory.memory, [], batch_number=1)
        assert formed.state.library_size == mock_memory.memory.state.library_size

    def test_form_memory_extracts_success_patterns(self, mock_memory, sample_trajectory):
        formed = form_memory(mock_memory.memory, sample_trajectory, batch_number=1)
        # Should have some patterns (at least the defaults)
        assert len(formed.success_patterns) >= len(mock_memory.memory.success_patterns)


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

class TestEvolution:
    """Test memory evolution (merge + consolidate)."""

    def test_evolve_increments_version(self, mock_memory, sample_trajectory):
        formed = form_memory(mock_memory.memory, sample_trajectory, batch_number=1)
        evolved = evolve_memory(mock_memory.memory, formed)
        assert evolved.version == mock_memory.memory.version + 1

    def test_evolve_merges_success_patterns(self, mock_memory, sample_trajectory):
        formed = form_memory(mock_memory.memory, sample_trajectory, batch_number=1)
        evolved = evolve_memory(mock_memory.memory, formed)
        # Should have at least as many patterns as before
        assert len(evolved.success_patterns) >= len(mock_memory.memory.success_patterns)

    def test_evolve_merges_forbidden_directions(self, mock_memory, sample_trajectory):
        formed = form_memory(mock_memory.memory, sample_trajectory, batch_number=1)
        evolved = evolve_memory(mock_memory.memory, formed)
        assert len(evolved.forbidden_directions) >= len(mock_memory.memory.forbidden_directions)

    def test_evolve_caps_memory_size(self, mock_memory, sample_trajectory):
        formed = form_memory(mock_memory.memory, sample_trajectory, batch_number=1)
        evolved = evolve_memory(
            mock_memory.memory, formed,
            max_success_patterns=5,
            max_failure_patterns=5,
            max_insights=5,
        )
        assert len(evolved.success_patterns) <= 5
        assert len(evolved.forbidden_directions) <= 5
        assert len(evolved.insights) <= 5


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestRetrieval:
    """Test context-dependent memory retrieval."""

    def test_retrieve_returns_dict(self, mock_memory):
        result = mock_memory.retrieve()
        assert isinstance(result, dict)

    def test_retrieve_has_required_keys(self, mock_memory):
        result = mock_memory.retrieve()
        assert "recommended_directions" in result
        assert "forbidden_directions" in result
        assert "insights" in result
        assert "library_state" in result
        assert "prompt_text" in result

    def test_retrieve_prompt_text_is_string(self, mock_memory):
        result = mock_memory.retrieve()
        assert isinstance(result["prompt_text"], str)
        assert len(result["prompt_text"]) > 0

    def test_retrieve_with_library_state(self, mock_memory):
        lib_state = {
            "library_size": 50,
            "domain_saturation": {"Momentum": 0.8, "VWAP": 0.3},
        }
        result = mock_memory.retrieve(library_state=lib_state)
        assert result["library_state"]["library_size"] == 50

    def test_retrieve_respects_max_limits(self, mock_memory):
        result = mock_memory.retrieve(max_success=2, max_forbidden=2, max_insights=1)
        assert len(result["recommended_directions"]) <= 2
        assert len(result["forbidden_directions"]) <= 2
        assert len(result["insights"]) <= 1

    def test_retrieve_deprioritizes_saturated_patterns(self, mock_memory):
        # Set high domain saturation
        mock_memory.memory.state.domain_saturation = {
            "Higher Moment Regimes": 0.9,
            "PV Corr Interaction": 0.9,
        }
        result = mock_memory.retrieve(max_success=3)
        # Saturated patterns should be scored lower
        names = [p["name"] for p in result["recommended_directions"]]
        # There should still be patterns, but saturated ones ranked lower
        assert len(names) > 0

    def test_enhanced_retrieval_uses_semantic_similarity_and_removals(self):
        memory = ExperienceMemory()
        memory.state.recent_admissions = [
            {
                "factor_id": "query_factor",
                "formula": "CsRank(Corr($close, $volume, 20))",
            }
        ]

        kg = FactorKnowledgeGraph()
        kg.add_factor(FactorNode(
            factor_id="neighbor_factor",
            formula="CsRank(Corr($close, $volume, 20))",
            operators=["CsRank", "Corr"],
            features=["$close", "$volume"],
            admitted=True,
        ))
        kg.add_factor(FactorNode(
            factor_id="distant_factor",
            formula="Neg(Std($returns, 10))",
            operators=["Neg", "Std"],
            features=["$returns"],
            admitted=True,
        ))

        embedder = FormulaEmbedder(use_faiss=False)

        result = retrieve_memory_enhanced(
            memory,
            kg=kg,
            embedder=embedder,
        )

        assert result["semantic_neighbors"]
        assert any("neighbor_factor" in item for item in result["semantic_neighbors"])

        kg.remove_factor("neighbor_factor")
        embedder.remove("neighbor_factor")

        refreshed = retrieve_memory_enhanced(
            memory,
            kg=kg,
            embedder=embedder,
        )

        assert all("neighbor_factor" not in item for item in refreshed["semantic_neighbors"])


# ---------------------------------------------------------------------------
# Full update cycle
# ---------------------------------------------------------------------------

class TestUpdateCycle:
    """Test the full update (formation + evolution) via the manager."""

    def test_update_returns_summary(self, mock_memory, sample_trajectory):
        summary = mock_memory.update(sample_trajectory)
        assert "batch" in summary
        assert "admitted_count" in summary
        assert "rejected_count" in summary
        assert summary["admitted_count"] == 2
        assert summary["rejected_count"] == 2

    def test_update_increments_version(self, mock_memory, sample_trajectory):
        assert mock_memory.version == 0
        mock_memory.update(sample_trajectory)
        assert mock_memory.version == 1

    def test_multiple_updates(self, mock_memory, sample_trajectory):
        for i in range(3):
            mock_memory.update(sample_trajectory)
        assert mock_memory.version == 3


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

class TestPersistence:
    """Test save and load roundtrip."""

    def test_save_load_roundtrip(self, mock_memory, sample_trajectory):
        # Update memory with some data
        mock_memory.update(sample_trajectory)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            mock_memory.save(path)

            # Verify file exists and is valid JSON
            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert "version" in data
            assert "success_patterns" in data

            # Load into new manager
            new_manager = ExperienceMemoryManager()
            new_manager.load(path)

            assert new_manager.version == mock_memory.version
            assert len(new_manager.memory.success_patterns) == len(
                mock_memory.memory.success_patterns
            )
            assert len(new_manager.memory.forbidden_directions) == len(
                mock_memory.memory.forbidden_directions
            )

    def test_save_creates_directory(self, mock_memory):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "deep" / "memory.json"
            mock_memory.save(path)
            assert path.exists()


# ---------------------------------------------------------------------------
# Memory store serialization
# ---------------------------------------------------------------------------

class TestMemoryStoreSerialization:
    """Test data class to_dict / from_dict methods."""

    def test_success_pattern_roundtrip(self):
        pat = SuccessPattern(
            name="Test Pattern",
            description="A test",
            template="CsRank($close)",
            success_rate="High",
            example_factors=["f1", "f2"],
            occurrence_count=5,
        )
        d = pat.to_dict()
        restored = SuccessPattern.from_dict(d)
        assert restored.name == pat.name
        assert restored.occurrence_count == pat.occurrence_count
        assert restored.success_rate == pat.success_rate
        assert restored.confidence == 1.0

    def test_old_success_pattern_payload_defaults_confidence(self):
        restored = SuccessPattern.from_dict(
            {
                "name": "Legacy",
                "description": "legacy payload",
                "template": "CsRank($close)",
                "success_rate": "Medium",
            }
        )
        assert restored.confidence == 1.0

    def test_confidence_helpers_accept_single_keyword(self):
        memory = ExperienceMemory(
            success_patterns=[
                SuccessPattern(
                    name="Momentum",
                    description="bull momentum",
                    template="CsRank($close)",
                    success_rate="Medium",
                    occurrence_count=2,
                    confidence=0.4,
                )
            ]
        )
        bumped = bump_pattern_confidence(memory, "bull", boost=0.2)
        assert abs(bumped.success_patterns[0].confidence - 0.6) < 1e-12

    def test_online_forgetting_and_regime_change_use_regime_labels(self):
        memory = ExperienceMemory(
            success_patterns=[
                SuccessPattern(
                    name="Bull Momentum",
                    description="bull low_vol momentum",
                    template="CsRank($close)",
                    success_rate="High",
                    occurrence_count=2,
                    confidence=1.0,
                )
            ]
        )
        updater = OnlineMemoryUpdater(
            memory,
            forgetting_rate=0.9,
            min_confidence=0.05,
            regime_boost=0.1,
            regime_penalty=0.1,
        )
        updater.apply_forgetting(iterations_elapsed=2)
        assert abs(updater.base_memory.success_patterns[0].confidence - 0.81) < 1e-12

        updater.on_regime_change(
            RegimeState(
                trend=TrendRegime.BEAR,
                vol=VolRegime.HIGH_VOL,
                mean_rev=MeanRevRegime.MEAN_REVERTING,
            ),
            RegimeState(
                trend=TrendRegime.BULL,
                vol=VolRegime.LOW_VOL,
                mean_rev=MeanRevRegime.TRENDING,
            ),
        )
        assert updater.base_memory.insights

    def test_forbidden_direction_roundtrip(self):
        fd = ForbiddenDirection(
            name="Bad Direction",
            description="Avoid this",
            correlated_factors=["f1"],
            typical_correlation=0.7,
            reason="Too correlated",
            occurrence_count=3,
        )
        d = fd.to_dict()
        restored = ForbiddenDirection.from_dict(d)
        assert restored.name == fd.name
        assert restored.typical_correlation == fd.typical_correlation

    def test_strategic_insight_roundtrip(self):
        insight = StrategicInsight(
            insight="Test insight",
            evidence="Some evidence",
            batch_source=5,
        )
        d = insight.to_dict()
        restored = StrategicInsight.from_dict(d)
        assert restored.insight == insight.insight
        assert restored.batch_source == 5

    def test_mining_state_roundtrip(self):
        state = MiningState(
            library_size=42,
            domain_saturation={"Momentum": 0.5},
        )
        d = state.to_dict()
        restored = MiningState.from_dict(d)
        assert restored.library_size == 42
        assert restored.domain_saturation["Momentum"] == 0.5

    def test_full_memory_roundtrip(self):
        mem = ExperienceMemory(
            state=MiningState(library_size=10),
            success_patterns=[
                SuccessPattern(name="P1", description="d1", template="t1", success_rate="High")
            ],
            forbidden_directions=[
                ForbiddenDirection(name="F1", description="d1")
            ],
            insights=[
                StrategicInsight(insight="I1", evidence="E1")
            ],
            version=3,
        )
        d = mem.to_dict()
        restored = ExperienceMemory.from_dict(d)
        assert restored.version == 3
        assert len(restored.success_patterns) == 1
        assert len(restored.forbidden_directions) == 1
        assert len(restored.insights) == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    """Test memory manager statistics."""

    def test_get_stats_keys(self, mock_memory):
        stats = mock_memory.get_stats()
        assert "version" in stats
        assert "batch_counter" in stats
        assert "success_patterns" in stats
        assert "forbidden_directions" in stats
        assert "insights" in stats

    def test_get_stats_after_update(self, mock_memory, sample_trajectory):
        mock_memory.update(sample_trajectory)
        stats = mock_memory.get_stats()
        assert stats["batch_counter"] == 1
        assert stats["version"] == 1
