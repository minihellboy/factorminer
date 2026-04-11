"""Tests for the factor knowledge graph (memory/knowledge_graph.py)."""

from __future__ import annotations

from factorminer.memory.knowledge_graph import (
    FactorKnowledgeGraph,
    FactorNode,
)

# -----------------------------------------------------------------------
# Basic node and edge operations
# -----------------------------------------------------------------------

def test_add_factor():
    kg = FactorKnowledgeGraph()
    node = FactorNode(
        factor_id="f1",
        formula="CsRank($close)",
        ic_mean=0.05,
        operators=["CsRank"],
        features=["$close"],
        admitted=True,
    )
    kg.add_factor(node)
    assert kg.get_factor_count() == 1
    # Operator node should also be created
    assert kg.get_edge_count() >= 1
    assert kg.get_factor_node("f1") is not None


def test_list_factor_nodes_filters_admitted():
    kg = FactorKnowledgeGraph()
    kg.add_factor(FactorNode(
        factor_id="f1", formula="CsRank($close)", operators=["CsRank"], admitted=True,
    ))
    kg.add_factor(FactorNode(
        factor_id="f2", formula="Neg($volume)", operators=["Neg"], admitted=False,
    ))

    admitted_ids = [node.factor_id for node in kg.list_factor_nodes(admitted_only=True)]
    all_ids = [node.factor_id for node in kg.list_factor_nodes()]

    assert admitted_ids == ["f1"]
    assert set(all_ids) == {"f1", "f2"}


def test_add_correlation_edge():
    kg = FactorKnowledgeGraph()
    kg.add_factor(FactorNode(factor_id="f1", formula="A", operators=["CsRank"]))
    kg.add_factor(FactorNode(factor_id="f2", formula="B", operators=["Neg"]))

    # Below threshold -> no edge
    kg.add_correlation_edge("f1", "f2", rho=0.3, threshold=0.4)
    initial_edges = kg.get_edge_count()

    # Above threshold -> edge added (bidirectional = 2 edges)
    kg.add_correlation_edge("f1", "f2", rho=0.6, threshold=0.4)
    assert kg.get_edge_count() >= initial_edges + 2


# -----------------------------------------------------------------------
# find_saturated_regions
# -----------------------------------------------------------------------

def test_find_saturated_regions():
    kg = FactorKnowledgeGraph()
    for i in range(3):
        kg.add_factor(FactorNode(
            factor_id=f"f{i}", formula=f"Op{i}($close)", operators=[f"Op{i}"],
        ))
    # High correlation between f0 and f1
    kg.add_correlation_edge("f0", "f1", rho=0.8, threshold=0.4)
    # Low correlation with f2
    kg.add_correlation_edge("f0", "f2", rho=0.2, threshold=0.4)

    regions = kg.find_saturated_regions(threshold=0.5)
    assert len(regions) >= 1
    # f0 and f1 should be in the same cluster
    found = any({"f0", "f1"}.issubset(r) for r in regions)
    assert found


# -----------------------------------------------------------------------
# find_complementary_patterns
# -----------------------------------------------------------------------

def test_find_complementary_patterns():
    kg = FactorKnowledgeGraph()
    kg.add_factor(FactorNode(
        factor_id="f1", formula="CsRank($close)", operators=["CsRank"],
    ))
    kg.add_factor(FactorNode(
        factor_id="f2", formula="Neg($volume)", operators=["Neg"],
    ))
    # Connect them via a shared operator node (indirectly)
    # f1 uses CsRank, f2 uses Neg -- different operators
    # Add a derivation edge so they are reachable
    kg.add_derivation_edge("f2", "f1", mutation_type="test")

    complementary = kg.find_complementary_patterns("f1", max_hops=2)
    # f2 uses a different operator set and is not correlated -> complementary
    assert "f2" in complementary


# -----------------------------------------------------------------------
# Serialization roundtrip
# -----------------------------------------------------------------------

def test_save_load_roundtrip():
    kg = FactorKnowledgeGraph()
    kg.add_factor(FactorNode(
        factor_id="f1", formula="CsRank($close)",
        ic_mean=0.05, operators=["CsRank"], admitted=True,
    ))
    kg.add_factor(FactorNode(
        factor_id="f2", formula="Neg($volume)",
        operators=["Neg"], admitted=True,
    ))
    kg.add_correlation_edge("f1", "f2", rho=0.5)

    data = kg.to_dict()
    kg2 = FactorKnowledgeGraph.from_dict(data)
    assert kg2.get_factor_count() == 2
    assert kg2.get_edge_count() == kg.get_edge_count()


def test_remove_factor_prunes_graph_state():
    kg = FactorKnowledgeGraph()
    kg.add_factor(FactorNode(
        factor_id="f1",
        formula="CsRank(Neg($close))",
        operators=["CsRank", "Neg"],
        features=["$close"],
        admitted=True,
    ))

    assert kg.remove_factor("f1") is True
    assert kg.get_factor_count() == 0
    assert kg.get_factor_node("f1") is None
    assert kg.get_edge_count() == 0
    assert kg.remove_factor("f1") is False


# -----------------------------------------------------------------------
# get_operator_cooccurrence
# -----------------------------------------------------------------------

def test_get_operator_cooccurrence():
    kg = FactorKnowledgeGraph()
    kg.add_factor(FactorNode(
        factor_id="f1", formula="CsRank(Neg($close))",
        operators=["CsRank", "Neg"], admitted=True,
    ))
    kg.add_factor(FactorNode(
        factor_id="f2", formula="CsRank(Mean($close, 10))",
        operators=["CsRank", "Mean"], admitted=True,
    ))

    cooc = kg.get_operator_cooccurrence()
    assert ("CsRank", "Neg") in cooc
    assert ("CsRank", "Mean") in cooc
    assert cooc[("CsRank", "Neg")] == 1
