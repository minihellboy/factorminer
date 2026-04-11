"""Enhanced memory retrieval combining Knowledge Graph + Embeddings + flat memory."""

from __future__ import annotations

from typing import Any

from factorminer.memory.memory_store import ExperienceMemory
from factorminer.memory.retrieval import retrieve_memory

# Optional imports -- presence checked at call time
try:
    from factorminer.memory.knowledge_graph import FactorKnowledgeGraph
except ImportError:
    FactorKnowledgeGraph = None  # type: ignore[assignment,misc]

try:
    from factorminer.memory.embeddings import FormulaEmbedder
except ImportError:
    FormulaEmbedder = None  # type: ignore[assignment,misc]


def retrieve_memory_enhanced(
    memory: ExperienceMemory,
    library_state: dict[str, Any] | None = None,
    max_success: int = 8,
    max_forbidden: int = 10,
    max_insights: int = 10,
    kg: FactorKnowledgeGraph | None = None,  # type: ignore[type-arg]
    embedder: FormulaEmbedder | None = None,  # type: ignore[type-arg]
) -> dict[str, Any]:
    """Enhanced memory retrieval operator R+(M, L, KG, E).

    Calls the base :func:`retrieve_memory` first, then augments the
    returned dict with additional prompt-oriented keys derived from the
    knowledge graph and embedder.

    Parameters
    ----------
    memory : ExperienceMemory
        The flat experience memory.
    library_state : dict, optional
        Current library diagnostics.
    max_success, max_forbidden, max_insights : int
        Limits forwarded to the base retrieval.
    kg : FactorKnowledgeGraph, optional
        Knowledge graph instance.
    embedder : FormulaEmbedder, optional
        Formula embedder instance.

    Returns
    -------
    dict
        Base memory signal plus the four additional keys above.
    """
    # Base retrieval
    result = retrieve_memory(
        memory,
        library_state=library_state,
        max_success=max_success,
        max_forbidden=max_forbidden,
        max_insights=max_insights,
    )

    # Default augmented keys
    result["complementary_patterns"] = []
    result["conflict_warnings"] = []
    result["operator_cooccurrence"] = []
    result["semantic_neighbors"] = []
    result["semantic_duplicates"] = []
    result["semantic_gaps"] = []

    # ----------------------------------------------------------------
    # Knowledge-graph augmentations
    # ----------------------------------------------------------------
    if kg is not None:
        # Complementary patterns: for each recently admitted factor,
        # find structurally complementary neighbours.
        complementary: list[str] = []
        seen: set[str] = set()
        for admission in memory.state.recent_admissions[-5:]:
            fid = admission.get("factor_id", "")
            if not fid:
                continue
            for comp in kg.find_complementary_patterns(fid, max_hops=2):
                if comp not in seen:
                    seen.add(comp)
                    complementary.append(_describe_factor_node(kg, comp))
        result["complementary_patterns"] = complementary

        # Conflict warnings: saturated regions
        saturated_regions = kg.find_saturated_regions(threshold=0.5)
        result["conflict_warnings"] = [
            _describe_conflict_cluster(kg, region) for region in saturated_regions
        ]

        # Operator co-occurrence
        cooc = kg.get_operator_cooccurrence()
        # Sort by count descending, take top 20
        top_cooc = sorted(cooc.items(), key=lambda x: x[1], reverse=True)[:20]
        result["operator_cooccurrence"] = [
            f"{a} + {b} (seen {count} times)" for (a, b), count in top_cooc
        ]

    # ----------------------------------------------------------------
    # Embedding-based augmentations
    # ----------------------------------------------------------------
    if embedder is not None:
        _seed_embedder_from_memory(memory, kg, embedder)
        semantic_neighbors, semantic_duplicates = _collect_semantic_context(
            memory=memory,
            kg=kg,
            embedder=embedder,
        )
        result["semantic_neighbors"] = semantic_neighbors
        result["semantic_duplicates"] = semantic_duplicates
        result["semantic_gaps"] = _find_semantic_gaps(memory, kg, embedder)

    # ----------------------------------------------------------------
    # Augment prompt text
    # ----------------------------------------------------------------
    extra_sections: list[str] = []

    if result["complementary_patterns"]:
        extra_sections.append("=== COMPLEMENTARY PATTERNS (explore) ===")
        extra_sections.append(
            "Factors structurally complementary to recent admissions:"
        )
        for fid in result["complementary_patterns"][:8]:
            extra_sections.append(f"  - {fid}")
        extra_sections.append("")

    if result["conflict_warnings"]:
        extra_sections.append("=== SATURATION WARNINGS ===")
        extra_sections.append(
            "The following factor clusters are highly correlated -- "
            "avoid generating variants:"
        )
        for cluster in result["conflict_warnings"][:5]:
            extra_sections.append(f"  Cluster: {', '.join(cluster[:6])}")
        extra_sections.append("")

    if result["semantic_gaps"]:
        extra_sections.append("=== SEMANTIC GAPS (underexplored) ===")
        extra_sections.append(
            "Operators present in success patterns but underused in the library:"
        )
        for op in result["semantic_gaps"][:10]:
            extra_sections.append(f"  - {op}")
        extra_sections.append("")

    if result["semantic_neighbors"]:
        extra_sections.append("=== SEMANTIC NEIGHBORS (similar library factors) ===")
        for item in result["semantic_neighbors"][:8]:
            extra_sections.append(f"  - {item}")
        extra_sections.append("")

    if result["semantic_duplicates"]:
        extra_sections.append("=== SEMANTIC DUPLICATES (near-duplicate risk) ===")
        for item in result["semantic_duplicates"][:5]:
            extra_sections.append(f"  - {item}")
        extra_sections.append("")

    if extra_sections:
        result["prompt_text"] += "\n" + "\n".join(extra_sections)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_semantic_gaps(
    memory: ExperienceMemory,
    kg: FactorKnowledgeGraph | None,  # type: ignore[type-arg]
    embedder: FormulaEmbedder | None,  # type: ignore[type-arg]
) -> list[str]:
    """Identify success-pattern operators with poor semantic coverage."""
    import re

    template_ops: set[str] = set()
    op_pattern = re.compile(r"\b([A-Z][a-zA-Z]+)\(")

    for pat in memory.success_patterns:
        for match in op_pattern.finditer(pat.template):
            template_ops.add(match.group(1))

    if not template_ops:
        return []

    if embedder is None:
        return sorted(template_ops)

    # A pattern is considered underexplored when it has no close semantic
    # neighbors in the current library representation.
    uncovered_ops: set[str] = set()
    anchors = list(memory.success_patterns[:10])
    if not anchors:
        return sorted(template_ops)

    for pat in anchors:
        nearest = embedder.find_nearest(pat.template, k=1)
        best_similarity = nearest[0][1] if nearest else 0.0
        if best_similarity < 0.72:
            for match in op_pattern.finditer(pat.template):
                uncovered_ops.add(match.group(1))

    if not uncovered_ops and kg is None:
        return sorted(template_ops)

    if not uncovered_ops:
        # Fall back to the operators that are entirely absent from the admitted set.
        used_ops: set[str] = set()
        if kg is not None:
            for node in kg.list_factor_nodes(admitted_only=True):
                used_ops.update(node.operators)
        uncovered_ops = template_ops - used_ops

    return sorted(uncovered_ops or template_ops)


def _seed_embedder_from_memory(
    memory: ExperienceMemory,
    kg: FactorKnowledgeGraph | None,  # type: ignore[type-arg]
    embedder: FormulaEmbedder,  # type: ignore[type-arg]
) -> None:
    """Ensure the embedder cache reflects the current known factors."""
    seen: set[str] = set()

    if kg is not None:
        for node in kg.list_factor_nodes(admitted_only=True):
            if node.factor_id and node.formula and node.factor_id not in seen:
                embedder.embed(node.factor_id, node.formula)
                seen.add(node.factor_id)

    for admission in memory.state.recent_admissions[-10:]:
        fid = admission.get("factor_id", "")
        formula = admission.get("formula", "")
        if fid and formula and fid not in seen:
            embedder.embed(fid, formula)
            seen.add(fid)


def _collect_semantic_context(
    memory: ExperienceMemory,
    kg: FactorKnowledgeGraph | None,  # type: ignore[type-arg]
    embedder: FormulaEmbedder,  # type: ignore[type-arg]
    max_neighbors: int = 8,
    similarity_threshold: float = 0.72,
) -> tuple[list[str], list[str]]:
    """Collect semantically similar neighbors and duplicate warnings."""
    anchors: list[tuple[str, str, str]] = []
    for admission in memory.state.recent_admissions[-5:]:
        fid = admission.get("factor_id", "")
        formula = admission.get("formula", "")
        if fid and formula:
            anchors.append(("recent admission", fid, formula))

    if not anchors:
        for pattern in memory.success_patterns[:5]:
            if pattern.template:
                anchors.append(("success pattern", pattern.name, pattern.template))

    semantic_neighbors: list[str] = []
    semantic_duplicates: list[str] = []
    seen_matches: set[tuple[str, str]] = set()

    if embedder.cache_size == 0:
        return semantic_neighbors, semantic_duplicates

    for anchor_kind, anchor_id, formula in anchors:
        nearest = embedder.find_nearest(formula, k=min(5, embedder.cache_size))
        for match_id, similarity in nearest:
            if anchor_id == match_id:
                continue
            if similarity < similarity_threshold:
                continue
            match_key = (anchor_id, match_id)
            if match_key in seen_matches:
                continue
            seen_matches.add(match_key)
            match_desc = _describe_factor_node(kg, match_id)
            if match_desc == match_id:
                semantic_neighbors.append(
                    f"{anchor_kind} {anchor_id} -> {match_id} (sim={similarity:.2f})"
                )
            else:
                semantic_neighbors.append(
                    f"{anchor_kind} {anchor_id} -> {match_desc} (sim={similarity:.2f})"
                )
            if similarity >= 0.90:
                semantic_duplicates.append(
                    f"{anchor_kind} {anchor_id} is very close to {match_id} "
                    f"(sim={similarity:.2f})"
                )
            if len(semantic_neighbors) >= max_neighbors:
                return semantic_neighbors, semantic_duplicates

    return semantic_neighbors, semantic_duplicates


def _describe_factor_node(
    kg: FactorKnowledgeGraph,  # type: ignore[type-arg]
    factor_id: str,
) -> str:
    """Render a factor node into short prompt-friendly text."""
    if kg is None:
        return factor_id

    node = kg.get_factor_node(factor_id)
    if node is None:
        return factor_id

    category = node.category or "unknown"
    ic_mean = node.ic_mean
    formula = node.formula
    summary = factor_id
    if category:
        summary += f" [{category}]"
    if ic_mean is not None:
        summary += f" IC={float(ic_mean):.4f}"
    if formula:
        summary += f": {formula[:80]}"
        if len(formula) > 80:
            summary += "..."
    return summary


def _describe_conflict_cluster(
    kg: FactorKnowledgeGraph,  # type: ignore[type-arg]
    cluster: set[str],
) -> str:
    """Render one saturated cluster into short text."""
    described = [_describe_factor_node(kg, factor_id) for factor_id in sorted(cluster)]
    return " | ".join(described[:3])
