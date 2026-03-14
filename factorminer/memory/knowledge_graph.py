"""Factor Knowledge Graph for lineage tracking and structural analysis.

Uses a NetworkX DiGraph to model relationships between factors, operators,
and feature inputs. Supports:
- Factor derivation lineage (parent -> child mutations)
- Correlation-based edges for saturation detection
- Operator co-occurrence analysis for diversity guidance
- Complementary pattern discovery via BFS
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore[assignment]


class EdgeType(Enum):
    """Types of edges in the factor knowledge graph."""

    DERIVED_FROM = "derived_from"
    CORRELATED_WITH = "correlated_with"
    USES_OPERATOR = "uses_operator"
    COMPLEMENTARY = "complementary"
    CONFLICTS = "conflicts"


@dataclass
class FactorNode:
    """A node in the factor knowledge graph representing a single factor.

    Attributes
    ----------
    factor_id : str
        Unique identifier for the factor.
    formula : str
        DSL formula string.
    ic_mean : float
        Mean information coefficient.
    category : str
        Factor category (e.g., "momentum", "mean_reversion").
    operators : list[str]
        List of operator names used in the formula.
    features : list[str]
        List of input features (e.g., "$close", "$volume").
    batch_number : int
        Batch in which the factor was generated.
    admitted : bool
        Whether the factor was admitted to the library.
    embedding : ndarray or None
        Optional semantic embedding vector.
    """

    factor_id: str
    formula: str
    ic_mean: float = 0.0
    category: str = ""
    operators: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    batch_number: int = 0
    admitted: bool = False
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        else:
            d["embedding"] = None
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FactorNode:
        embedding = d.get("embedding")
        if embedding is not None:
            embedding = np.array(embedding, dtype=np.float32)
        return cls(
            factor_id=d["factor_id"],
            formula=d.get("formula", ""),
            ic_mean=d.get("ic_mean", 0.0),
            category=d.get("category", ""),
            operators=d.get("operators", []),
            features=d.get("features", []),
            batch_number=d.get("batch_number", 0),
            admitted=d.get("admitted", False),
            embedding=embedding,
        )


def _ensure_networkx() -> None:
    """Raise a clear error if networkx is not installed."""
    if nx is None:
        raise ImportError(
            "networkx is required for FactorKnowledgeGraph. "
            "Install it with: pip install networkx"
        )


class FactorKnowledgeGraph:
    """Directed graph tracking factor lineage and relationships.

    Uses ``networkx.DiGraph`` internally. Factor nodes store a
    :class:`FactorNode` dataclass; operator nodes are prefixed with
    ``op:`` and feature nodes with ``feat:``.
    """

    def __init__(self) -> None:
        _ensure_networkx()
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_factor(self, node: FactorNode) -> None:
        """Add a factor node and auto-create USES_OPERATOR edges.

        For each operator in ``node.operators``, an ``op:{name}`` node
        is created (if absent) and a USES_OPERATOR edge is drawn from
        the factor to that operator node.
        """
        self._graph.add_node(
            node.factor_id,
            node_type="factor",
            data=node.to_dict(),
        )

        for op in node.operators:
            op_id = f"op:{op}"
            if not self._graph.has_node(op_id):
                self._graph.add_node(op_id, node_type="operator")
            self._graph.add_edge(
                node.factor_id,
                op_id,
                edge_type=EdgeType.USES_OPERATOR.value,
            )

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_correlation_edge(
        self,
        a: str,
        b: str,
        rho: float,
        threshold: float = 0.4,
    ) -> None:
        """Add a CORRELATED_WITH edge if ``|rho| >= threshold``."""
        if abs(rho) >= threshold:
            self._graph.add_edge(
                a,
                b,
                edge_type=EdgeType.CORRELATED_WITH.value,
                rho=rho,
            )
            self._graph.add_edge(
                b,
                a,
                edge_type=EdgeType.CORRELATED_WITH.value,
                rho=rho,
            )

    def add_derivation_edge(
        self,
        child: str,
        parent: str,
        mutation_type: str = "",
    ) -> None:
        """Add a DERIVED_FROM edge from *child* to *parent*."""
        self._graph.add_edge(
            child,
            parent,
            edge_type=EdgeType.DERIVED_FROM.value,
            mutation_type=mutation_type,
        )

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def find_complementary_patterns(
        self,
        factor_id: str,
        max_hops: int = 2,
    ) -> List[str]:
        """Find factors complementary to *factor_id* via BFS.

        A complementary factor is one that:
        1. Is reachable within *max_hops* in the undirected view, and
        2. Is NOT directly correlated with the source factor, and
        3. Uses at least one different operator.

        Returns a list of factor IDs.
        """
        if not self._graph.has_node(factor_id):
            return []

        # Collect correlated neighbours (direct CORRELATED_WITH edges)
        correlated: Set[str] = set()
        for _, nbr, data in self._graph.edges(factor_id, data=True):
            if data.get("edge_type") == EdgeType.CORRELATED_WITH.value:
                correlated.add(nbr)
        for pred, _, data in self._graph.in_edges(factor_id, data=True):
            if data.get("edge_type") == EdgeType.CORRELATED_WITH.value:
                correlated.add(pred)

        # Source operators
        source_ops = self._get_operators(factor_id)

        # BFS on undirected view
        undirected = self._graph.to_undirected()
        visited: Set[str] = {factor_id}
        frontier: List[str] = [factor_id]
        complementary: List[str] = []

        for _ in range(max_hops):
            next_frontier: List[str] = []
            for node in frontier:
                for nbr in undirected.neighbors(node):
                    if nbr in visited:
                        continue
                    visited.add(nbr)
                    next_frontier.append(nbr)

                    # Only consider factor nodes
                    if self._graph.nodes[nbr].get("node_type") != "factor":
                        continue
                    # Skip if correlated
                    if nbr in correlated:
                        continue
                    # Must use at least one different operator
                    nbr_ops = self._get_operators(nbr)
                    if nbr_ops and source_ops and not nbr_ops.issubset(source_ops):
                        complementary.append(nbr)
            frontier = next_frontier

        return complementary

    def find_saturated_regions(
        self,
        threshold: float = 0.5,
    ) -> List[Set[str]]:
        """Find clusters of highly correlated factors.

        Builds a subgraph of CORRELATED_WITH edges where
        ``|rho| > threshold``, then returns connected components.
        Each component is a set of factor IDs.
        """
        sub = nx.Graph()
        for u, v, data in self._graph.edges(data=True):
            if data.get("edge_type") != EdgeType.CORRELATED_WITH.value:
                continue
            rho = abs(data.get("rho", 0.0))
            if rho > threshold:
                # Only include factor nodes
                if (
                    self._graph.nodes.get(u, {}).get("node_type") == "factor"
                    and self._graph.nodes.get(v, {}).get("node_type") == "factor"
                ):
                    sub.add_edge(u, v)

        components = list(nx.connected_components(sub))
        # Filter out singletons
        return [c for c in components if len(c) > 1]

    def get_operator_cooccurrence(self) -> Dict[Tuple[str, str], int]:
        """Count operator pair co-occurrences across admitted factors.

        Returns a dict mapping ``(op_a, op_b)`` (sorted tuple) to count.
        """
        cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        for node_id, attrs in self._graph.nodes(data=True):
            if attrs.get("node_type") != "factor":
                continue
            node_data = attrs.get("data", {})
            if not node_data.get("admitted", False):
                continue

            ops = sorted(set(node_data.get("operators", [])))
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    pair = (ops[i], ops[j])
                    cooccurrence[pair] += 1

        return dict(cooccurrence)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_factor_count(self) -> int:
        """Return the number of factor nodes in the graph."""
        return sum(
            1
            for _, d in self._graph.nodes(data=True)
            if d.get("node_type") == "factor"
        )

    def get_edge_count(self) -> int:
        """Return total number of edges in the graph."""
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict via ``nx.node_link_data``."""
        return nx.node_link_data(self._graph, edges="links")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FactorKnowledgeGraph:
        """Deserialize from a dict produced by :meth:`to_dict`."""
        kg = cls()
        kg._graph = nx.node_link_graph(data, edges="links")
        return kg

    def save(self, path: str | Path) -> None:
        """Persist the graph to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> FactorKnowledgeGraph:
        """Load a graph from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_operators(self, factor_id: str) -> Set[str]:
        """Return the set of operator names used by a factor."""
        ops: Set[str] = set()
        for _, nbr, data in self._graph.edges(factor_id, data=True):
            if data.get("edge_type") == EdgeType.USES_OPERATOR.value:
                # Strip "op:" prefix
                ops.add(nbr.removeprefix("op:"))
        return ops
