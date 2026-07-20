"""Regression tests for KGMemoryPolicy embedder wiring (roadmap item 2).

Before the fix, ``KGMemoryPolicy.retrieve`` called
``retrieve_memory_enhanced(..., kg=...)`` without ``embedder=``, so
``semantic_neighbors`` was always empty under the ``kg`` policy even when
embeddings were enabled elsewhere. These tests pin the before/after contract.
"""

from __future__ import annotations

from factorminer.architecture.memory_policy import KGMemoryPolicy, build_memory_policy
from factorminer.architecture.paper_protocol import PaperProtocol
from factorminer.memory.embeddings import FormulaEmbedder
from factorminer.memory.knowledge_graph import FactorNode
from factorminer.memory.memory_store import ExperienceMemory, SuccessPattern
from factorminer.memory.retrieval import HybridRetrievalConfig
from factorminer.utils.config import load_config


def _protocol() -> PaperProtocol:
    return PaperProtocol.from_config(load_config())


def _seeded_memory_and_graph(policy: KGMemoryPolicy) -> ExperienceMemory:
    """Populate KG + memory so semantic neighbors have something to match."""
    memory = ExperienceMemory()
    memory.success_patterns = [
        SuccessPattern(
            name="Ranked Close-Volume Corr",
            description="Cross-sectional rank of close/volume correlation",
            template="CsRank(Corr($close, $volume, 20))",
            success_rate="High",
            example_factors=["CsRank(Corr($close, $volume, 20))"],
            occurrence_count=5,
        )
    ]
    memory.state.recent_admissions = [
        {
            "factor_id": "query_factor",
            "formula": "CsRank(Corr($close, $volume, 20))",
        }
    ]
    memory.state.library_size = 2
    memory.state.domain_saturation = {"Ranked Close-Volume Corr": 0.2}

    policy.knowledge_graph.add_factor(
        FactorNode(
            factor_id="neighbor_factor",
            formula="CsRank(Corr($close, $volume, 20))",
            operators=["CsRank", "Corr"],
            features=["$close", "$volume"],
            admitted=True,
            ic_mean=0.05,
        )
    )
    policy.knowledge_graph.add_factor(
        FactorNode(
            factor_id="distant_factor",
            formula="Neg(Std($returns, 10))",
            operators=["Neg", "Std"],
            features=["$returns"],
            admitted=True,
            ic_mean=0.01,
        )
    )
    return memory


def test_kg_policy_without_embeddings_has_empty_semantic_neighbors() -> None:
    """Baseline: enable_embeddings=False → no dense neighbor context."""
    policy = KGMemoryPolicy(_protocol(), enable_embeddings=False)
    memory = _seeded_memory_and_graph(policy)

    signal = policy.retrieve(
        memory,
        library_state={
            "library_size": 2,
            "domain_saturation": memory.state.domain_saturation,
        },
    )

    assert signal.get("embeddings_enabled") is False
    assert signal.get("semantic_neighbors") == []
    assert policy.embedder is None


def test_kg_policy_with_embeddings_populates_semantic_neighbors() -> None:
    """After fix: enable_embeddings=True wires embedder → neighbors populated."""
    embedder = FormulaEmbedder(use_faiss=False)
    policy = KGMemoryPolicy(
        _protocol(),
        enable_embeddings=True,
        embedder=embedder,
        hybrid_config=HybridRetrievalConfig(enabled=True, enable_rerank=False),
    )
    memory = _seeded_memory_and_graph(policy)

    signal = policy.retrieve(
        memory,
        library_state={
            "library_size": 2,
            "domain_saturation": memory.state.domain_saturation,
            "query_formula": "CsRank(Corr($close, $volume, 20))",
        },
    )

    assert signal.get("embeddings_enabled") is True
    neighbors = signal.get("semantic_neighbors") or []
    assert neighbors, "expected semantic_neighbors after embedder wiring fix"
    assert any("neighbor_factor" in item for item in neighbors)
    # Prompt text must surface the neighbor section (downstream consumers).
    assert "SEMANTIC NEIGHBORS" in signal.get("prompt_text", "")


def test_kg_policy_respects_enable_embeddings_toggle() -> None:
    """Even with an inject embedder, toggle off must not pass it through."""
    embedder = FormulaEmbedder(use_faiss=False)
    policy = KGMemoryPolicy(
        _protocol(),
        enable_embeddings=False,
        embedder=embedder,  # present but toggle off
    )
    memory = _seeded_memory_and_graph(policy)
    # Warm cache manually so a leak would definitely produce neighbors.
    embedder.embed("neighbor_factor", "CsRank(Corr($close, $volume, 20))")
    embedder.embed("distant_factor", "Neg(Std($returns, 10))")

    signal = policy.retrieve(
        memory,
        library_state={"library_size": 2, "domain_saturation": {}},
    )
    assert signal.get("embeddings_enabled") is False
    assert signal.get("semantic_neighbors") == []


def test_build_memory_policy_kg_honors_memory_enable_embeddings() -> None:
    cfg = load_config()
    cfg.memory.policy = "kg"
    cfg.memory.enable_embeddings = True
    policy = build_memory_policy(cfg, _protocol())
    assert isinstance(policy, KGMemoryPolicy)
    assert policy.enable_embeddings is True
    assert policy.embedder is not None
    assert policy.schema()["enable_embeddings"] is True


def test_kg_policy_serialize_does_not_persist_embedder_weights() -> None:
    policy = KGMemoryPolicy(_protocol(), enable_embeddings=True)
    memory = ExperienceMemory()
    payload = policy.serialize(memory)
    assert payload.get("enable_embeddings") is True
    assert "embedder" not in payload
    assert "model_weights" not in payload
