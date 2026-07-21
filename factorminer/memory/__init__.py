"""Experience-memory APIs with optional implementations loaded on demand."""

from __future__ import annotations

from factorminer._lazy_exports import public_dir, resolve_export

_EXPORTS = {
    "form_memory": "factorminer.memory.formation",
    "evolve_memory": "factorminer.memory.evolution",
    **{
        name: "factorminer.memory.memory_store"
        for name in (
            "ExperienceMemory",
            "MiningState",
            "SuccessPattern",
            "ForbiddenDirection",
            "StrategicInsight",
        )
    },
    **{
        name: "factorminer.memory.retrieval"
        for name in ("retrieve_memory", "HybridRetrievalConfig", "retrieval_quality_smoke")
    },
    "create_default_memory": "factorminer.memory.defaults",
    **{
        name: "factorminer.memory.knowledge_graph"
        for name in ("FactorKnowledgeGraph", "FactorNode", "EdgeType")
    },
    "FormulaEmbedder": "factorminer.memory.embeddings",
    "retrieve_memory_enhanced": "factorminer.memory.kg_retrieval",
    **{
        name: "factorminer.memory.online_regime_memory"
        for name in (
            "OnlineRegimeMemory",
            "OnlineMemoryUpdater",
            "RegimeSpecificPatternStore",
            "RegimeTransitionForecaster",
            "MemoryForgetCurve",
        )
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), name, _EXPORTS)


def __dir__() -> list[str]:
    return public_dir(globals(), _EXPORTS)
