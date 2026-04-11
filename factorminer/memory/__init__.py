"""Experience memory system for mining loop feedback.

Implements the memory M = {S, P_succ, P_fail, I} with operators:
- F(M, tau): Memory Formation - extract experience from mining trajectory
- E(M, M_form): Memory Evolution - consolidate and prune memory
- R(M, L): Memory Retrieval - context-dependent retrieval for LLM prompts

Phase 2 additions:
- Knowledge Graph: factor lineage and structural analysis
- Embeddings: semantic formula similarity and deduplication
- Enhanced Retrieval: KG + embedding augmented retrieval
"""

from factorminer.memory.evolution import evolve_memory
from factorminer.memory.experience_memory import ExperienceMemoryManager
from factorminer.memory.formation import form_memory
from factorminer.memory.memory_store import (
    ExperienceMemory,
    ForbiddenDirection,
    MiningState,
    StrategicInsight,
    SuccessPattern,
)
from factorminer.memory.retrieval import retrieve_memory

# Phase 2: Optional imports (graceful if dependencies missing)
try:
    from factorminer.memory.knowledge_graph import (
        EdgeType,
        FactorKnowledgeGraph,
        FactorNode,
    )
except ImportError:
    pass

try:
    from factorminer.memory.embeddings import FormulaEmbedder
except ImportError:
    pass

try:
    from factorminer.memory.kg_retrieval import retrieve_memory_enhanced
except ImportError:
    pass

try:
    from factorminer.memory.online_regime_memory import (
        MemoryForgetCurve,
        OnlineMemoryUpdater,
        OnlineRegimeMemory,
        RegimeSpecificPatternStore,
        RegimeTransitionForecaster,
    )
except ImportError:
    pass

__all__ = [
    # Data structures
    "ExperienceMemory",
    "MiningState",
    "SuccessPattern",
    "ForbiddenDirection",
    "StrategicInsight",
    # Operators
    "form_memory",
    "evolve_memory",
    "retrieve_memory",
    # Manager
    "ExperienceMemoryManager",
    # Phase 2: Knowledge Graph
    "FactorKnowledgeGraph",
    "FactorNode",
    "EdgeType",
    # Phase 2: Embeddings
    "FormulaEmbedder",
    # Phase 2: Enhanced Retrieval
    "retrieve_memory_enhanced",
    # Phase 2: Online Regime Memory
    "OnlineRegimeMemory",
    "OnlineMemoryUpdater",
    "RegimeSpecificPatternStore",
    "RegimeTransitionForecaster",
    "MemoryForgetCurve",
]
