"""Main ExperienceMemory manager class.

Provides the high-level API for the experience memory system:
- Initializes with default patterns from the paper (Tables 4 and 5)
- Persists to/from JSON
- update(trajectory) orchestrates formation + evolution
- retrieve(library_state) performs context-dependent retrieval
- Optional knowledge graph and embedding support for Phase 2
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from factorminer.memory.evolution import evolve_memory
from factorminer.memory.formation import form_memory
from factorminer.memory.memory_store import (
    ExperienceMemory,
    ForbiddenDirection,
    MiningState,
    StrategicInsight,
    SuccessPattern,
)
from factorminer.memory.retrieval import retrieve_memory

# Optional Phase 2 imports
try:
    from factorminer.memory.knowledge_graph import FactorKnowledgeGraph, FactorNode
except ImportError:
    FactorKnowledgeGraph = None  # type: ignore[assignment,misc]
    FactorNode = None  # type: ignore[assignment,misc]

try:
    from factorminer.memory.embeddings import FormulaEmbedder
except ImportError:
    FormulaEmbedder = None  # type: ignore[assignment,misc]

try:
    from factorminer.memory.kg_retrieval import retrieve_memory_enhanced
except ImportError:
    retrieve_memory_enhanced = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default knowledge base from the paper
# ---------------------------------------------------------------------------

def _default_success_patterns() -> list[SuccessPattern]:
    """Initial success patterns from FactorMiner Table 4."""
    return [
        SuccessPattern(
            name="Higher Moment Regimes",
            description=(
                "Use Skew/Kurt as IfElse conditions to route between different "
                "factor computations. High-moment regime switching captures "
                "non-linear market states effectively."
            ),
            template="IfElse(Skew($close, 20), <factor_a>, <factor_b>)",
            success_rate="High",
            example_factors=["HMR_001", "HMR_002"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="PV Corr Interaction",
            description=(
                "Price-volume correlation interaction: use rolling Corr($close, $volume) "
                "as a signal or conditioning variable. Captures supply-demand imbalance "
                "through price-volume divergence."
            ),
            template="CsRank(Corr($close, $volume, 20))",
            success_rate="High",
            example_factors=["PVC_001", "PVC_002"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Robust Efficiency",
            description=(
                "Use Median for noise filtering instead of Mean. Rolling median "
                "is more robust to outliers in intraday data, producing factors "
                "with higher ICIR."
            ),
            template="CsRank(Div(Median($close, 10), Median($close, 60)))",
            success_rate="High",
            example_factors=["RE_001"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Smoothed Efficiency Rank",
            description=(
                "Combine EMA smoothing with CsRank cross-sectional normalization. "
                "EMA reduces noise while CsRank ensures cross-sectional comparability."
            ),
            template="CsRank(EMA(Div($close, Mean($close, 20)), 10))",
            success_rate="High",
            example_factors=["SER_001", "SER_002"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Trend Regression Adaptive",
            description=(
                "Use TsLinRegSlope, TsLinRegResid, or rolling R-squared to capture "
                "trend strength and mean reversion. Regression residuals identify "
                "deviations from local trends."
            ),
            template="CsRank(TsLinRegSlope($close, 20))",
            success_rate="High",
            example_factors=["TRA_001", "TRA_002"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Logical Or Extreme Regimes",
            description=(
                "Use Or/And with Greater/Less to combine multiple extreme-value "
                "conditions. Captures compound regime states that single indicators miss."
            ),
            template="IfElse(Or(Greater(Skew($returns, 20), 1), Less(Kurt($returns, 20), -1)), <a>, <b>)",
            success_rate="Medium",
            example_factors=["LOR_001"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Kurtosis Regime",
            description=(
                "Use rolling kurtosis to detect fat-tail regimes and switch "
                "factor behavior accordingly. High kurtosis indicates regime "
                "changes and trend breaks."
            ),
            template="IfElse(Kurt($returns, 20), CsRank(Std($returns, 10)), CsRank(Mean($returns, 10)))",
            success_rate="Medium",
            example_factors=["KR_001"],
            occurrence_count=0,
        ),
        SuccessPattern(
            name="Amt Efficiency Rank Interaction",
            description=(
                "Combine $amt (turnover) with efficiency ratios and CsRank. "
                "Amount-weighted efficiency captures liquidity-adjusted momentum."
            ),
            template="CsRank(Div(EMA($amt, 5), EMA($amt, 20)))",
            success_rate="Medium",
            example_factors=["AER_001"],
            occurrence_count=0,
        ),
    ]


def _default_forbidden_directions() -> list[ForbiddenDirection]:
    """Initial forbidden directions from FactorMiner Table 5."""
    return [
        ForbiddenDirection(
            name="Standardized Returns/Amount",
            description=(
                "CsZScore or Std-normalized $returns and $amt variants. "
                "These produce a cluster of highly correlated factors."
            ),
            correlated_factors=["std_ret_cluster"],
            typical_correlation=0.6,
            reason="Standardized return/amount variants cluster with rho > 0.6",
            occurrence_count=0,
        ),
        ForbiddenDirection(
            name="VWAP Deviation variants",
            description=(
                "Factors based on deviation from VWAP (Sub($close, $vwap) or "
                "Delta($vwap)). All VWAP deviation variants converge to the "
                "same signal."
            ),
            correlated_factors=["vwap_dev_cluster"],
            typical_correlation=0.5,
            reason="VWAP deviation variants produce highly correlated factors (rho > 0.5)",
            occurrence_count=0,
        ),
        ForbiddenDirection(
            name="Simple Delta Reversal",
            description=(
                "Simple price-change reversal factors using Delta($close) or "
                "Neg(Return($close)). These are well-known and already "
                "saturated in most factor libraries."
            ),
            correlated_factors=["delta_rev_cluster"],
            typical_correlation=0.5,
            reason="Simple delta-based reversal factors are redundant (rho > 0.5)",
            occurrence_count=0,
        ),
        ForbiddenDirection(
            name="WMA/EMA Smoothed Efficiency",
            description=(
                "Smoothing the same base signal with WMA, EMA, SMA, DEMA "
                "produces nearly identical factors. Different smoothing methods "
                "on the same input do not add diversity."
            ),
            correlated_factors=["smoothed_eff_cluster"],
            typical_correlation=0.9,
            reason="WMA/EMA/SMA smoothed efficiency variants nearly identical (rho > 0.9)",
            occurrence_count=0,
        ),
    ]


def _default_insights() -> list[StrategicInsight]:
    """Initial strategic insights from the paper."""
    return [
        StrategicInsight(
            insight="Non-linear transformations (IfElse, Skew, Kurt) outperform linear ones",
            evidence="Paper finding: regime-switching factors consistently achieve higher IC",
            batch_source=0,
        ),
        StrategicInsight(
            insight="Cross-sectional ranking (CsRank) as final layer improves factor stability",
            evidence="CsRank normalization reduces outlier sensitivity and improves ICIR",
            batch_source=0,
        ),
        StrategicInsight(
            insight="Combining operators from different categories produces more diverse factors",
            evidence="Multi-category composition (e.g., Statistical + Logical + CrossSectional) "
                     "reduces correlation with existing library members",
            batch_source=0,
        ),
    ]


# ---------------------------------------------------------------------------
# Manager class
# ---------------------------------------------------------------------------

class ExperienceMemoryManager:
    """High-level manager for the experience memory system.

    Orchestrates formation, evolution, retrieval, and persistence of the
    experience memory M across mining sessions.

    Parameters
    ----------
    max_success_patterns : int
        Maximum number of success patterns to retain.
    max_failure_patterns : int
        Maximum number of forbidden directions to retain.
    max_insights : int
        Maximum number of strategic insights to retain.
    """

    def __init__(
        self,
        max_success_patterns: int = 50,
        max_failure_patterns: int = 100,
        max_insights: int = 30,
        enable_knowledge_graph: bool = False,
        enable_embeddings: bool = False,
    ) -> None:
        self.max_success_patterns = max_success_patterns
        self.max_failure_patterns = max_failure_patterns
        self.max_insights = max_insights
        self._batch_counter = 0

        # Initialize with default knowledge base
        self.memory = ExperienceMemory(
            state=MiningState(),
            success_patterns=_default_success_patterns(),
            forbidden_directions=_default_forbidden_directions(),
            insights=_default_insights(),
            version=0,
        )

        # Phase 2: Optional knowledge graph
        self.kg: FactorKnowledgeGraph | None = None  # type: ignore[type-arg]
        if enable_knowledge_graph:
            if FactorKnowledgeGraph is not None:
                self.kg = FactorKnowledgeGraph()
            else:
                logger.warning(
                    "Knowledge graph requested but networkx is not installed. "
                    "Install with: pip install networkx"
                )

        # Phase 2: Optional formula embedder
        self.embedder: FormulaEmbedder | None = None  # type: ignore[type-arg]
        if enable_embeddings:
            if FormulaEmbedder is not None:
                self.embedder = FormulaEmbedder()
            else:
                logger.warning(
                    "Embeddings requested but required packages are not installed. "
                    "Install with: pip install sentence-transformers"
                )

    @property
    def version(self) -> int:
        return self.memory.version

    def update(self, trajectory: list[dict]) -> dict[str, Any]:
        """Process a batch trajectory: formation + evolution.

        Parameters
        ----------
        trajectory : list[dict]
            Batch of evaluated candidates. Each dict should contain:
            - formula: str - the DSL formula
            - factor_id: str - unique identifier
            - ic: float - information coefficient
            - icir: float - IC information ratio
            - max_correlation: float - max correlation with existing factors
            - correlated_with: str - ID of most correlated existing factor
            - admitted: bool - whether the factor was admitted
            - rejection_reason: str - reason for rejection (if rejected)

        Returns
        -------
        dict
            Summary of the update: admitted_count, rejected_count,
            new_patterns, new_forbidden, new_insights, version.
        """
        self._batch_counter += 1

        # Formation: extract experience from trajectory
        formed = form_memory(self.memory, trajectory, self._batch_counter)

        # Evolution: merge formed experience into persistent memory
        self.memory = evolve_memory(
            self.memory,
            formed,
            max_success_patterns=self.max_success_patterns,
            max_failure_patterns=self.max_failure_patterns,
            max_insights=self.max_insights,
        )

        admitted_count = sum(1 for c in trajectory if c.get("admitted", False))
        rejected_count = len(trajectory) - admitted_count

        # Phase 2: Update knowledge graph with new factors
        if self.kg is not None and FactorNode is not None:
            self._update_knowledge_graph(trajectory)

        return {
            "batch": self._batch_counter,
            "admitted_count": admitted_count,
            "rejected_count": rejected_count,
            "success_patterns": len(self.memory.success_patterns),
            "forbidden_directions": len(self.memory.forbidden_directions),
            "insights": len(self.memory.insights),
            "version": self.memory.version,
        }

    def retrieve(
        self,
        library_state: dict[str, Any] | None = None,
        max_success: int = 8,
        max_forbidden: int = 10,
        max_insights: int = 10,
    ) -> dict[str, Any]:
        """Retrieve context-dependent memory signal for LLM prompt.

        Parameters
        ----------
        library_state : dict, optional
            Current library diagnostics. Keys: library_size,
            domain_saturation, etc.
        max_success : int
            Maximum number of success patterns to include.
        max_forbidden : int
            Maximum number of forbidden directions to include.
        max_insights : int
            Maximum number of insights to include.

        Returns
        -------
        dict
            Memory signal m with keys: recommended_directions,
            forbidden_directions, insights, library_state, prompt_text.
        """
        # Use enhanced retrieval if KG or embedder is available
        if (self.kg is not None or self.embedder is not None) and retrieve_memory_enhanced is not None:
            return retrieve_memory_enhanced(
                self.memory,
                library_state=library_state,
                max_success=max_success,
                max_forbidden=max_forbidden,
                max_insights=max_insights,
                kg=self.kg,
                embedder=self.embedder,
            )

        return retrieve_memory(
            self.memory,
            library_state=library_state,
            max_success=max_success,
            max_forbidden=max_forbidden,
            max_insights=max_insights,
        )

    def save(self, path: str | Path) -> None:
        """Persist memory to a JSON file.

        Also saves the knowledge graph to a sibling file
        ``<stem>_kg.json`` if enabled.

        Parameters
        ----------
        path : str or Path
            Output file path (will be created/overwritten).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.memory.to_dict()
        data["_batch_counter"] = self._batch_counter
        data["_config"] = {
            "max_success_patterns": self.max_success_patterns,
            "max_failure_patterns": self.max_failure_patterns,
            "max_insights": self.max_insights,
            "enable_knowledge_graph": self.kg is not None,
            "enable_embeddings": self.embedder is not None,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Phase 2: Save knowledge graph alongside
        if self.kg is not None:
            kg_path = path.with_name(f"{path.stem}_kg.json")
            self.kg.save(kg_path)

    def load(self, path: str | Path) -> None:
        """Load memory from a JSON file.

        Also loads the knowledge graph from ``<stem>_kg.json`` if
        the file exists and the KG feature is enabled.

        Parameters
        ----------
        path : str or Path
            Path to a previously saved memory file.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        self.memory = ExperienceMemory.from_dict(data)
        self._batch_counter = data.get("_batch_counter", 0)

        config = data.get("_config", {})
        if config:
            self.max_success_patterns = config.get(
                "max_success_patterns", self.max_success_patterns
            )
            self.max_failure_patterns = config.get(
                "max_failure_patterns", self.max_failure_patterns
            )
            self.max_insights = config.get(
                "max_insights", self.max_insights
            )

        # Phase 2: Load knowledge graph if available
        kg_path = path.with_name(f"{path.stem}_kg.json")
        if kg_path.exists() and FactorKnowledgeGraph is not None:
            if self.kg is None:
                # Enable KG if saved config says so, or if file exists
                if config.get("enable_knowledge_graph", False):
                    self.kg = FactorKnowledgeGraph.load(kg_path)
            else:
                self.kg = FactorKnowledgeGraph.load(kg_path)

        # Re-enable embedder if config says so
        if config.get("enable_embeddings", False) and self.embedder is None:
            if FormulaEmbedder is not None:
                self.embedder = FormulaEmbedder()

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics about the current memory state.

        Returns
        -------
        dict
            Keys: version, batch_counter, library_size, success_patterns,
            forbidden_directions, insights, domain_saturation,
            recent_admission_rate, plus kg_* keys when KG is enabled.
        """
        recent_logs = self.memory.state.admission_log[-5:]
        avg_rate = 0.0
        if recent_logs:
            avg_rate = sum(
                log.get("admission_rate", 0) for log in recent_logs
            ) / len(recent_logs)

        stats: dict[str, Any] = {
            "version": self.memory.version,
            "batch_counter": self._batch_counter,
            "library_size": self.memory.state.library_size,
            "success_patterns": len(self.memory.success_patterns),
            "forbidden_directions": len(self.memory.forbidden_directions),
            "insights": len(self.memory.insights),
            "domain_saturation": dict(self.memory.state.domain_saturation),
            "recent_admission_rate": round(avg_rate, 4),
            "top_success_patterns": [
                {"name": p.name, "rate": p.success_rate, "count": p.occurrence_count}
                for p in sorted(
                    self.memory.success_patterns,
                    key=lambda p: p.occurrence_count,
                    reverse=True,
                )[:5]
            ],
            "top_forbidden_directions": [
                {"name": f.name, "corr": f.typical_correlation, "count": f.occurrence_count}
                for f in sorted(
                    self.memory.forbidden_directions,
                    key=lambda f: f.occurrence_count,
                    reverse=True,
                )[:5]
            ],
        }

        # Phase 2: KG stats
        if self.kg is not None:
            stats["kg_factor_count"] = self.kg.get_factor_count()
            stats["kg_edge_count"] = self.kg.get_edge_count()
            saturated = self.kg.find_saturated_regions()
            stats["kg_saturated_clusters"] = len(saturated)

        return stats

    def reset(self) -> None:
        """Reset memory to initial state with default knowledge base."""
        self._batch_counter = 0
        self.memory = ExperienceMemory(
            state=MiningState(),
            success_patterns=_default_success_patterns(),
            forbidden_directions=_default_forbidden_directions(),
            insights=_default_insights(),
            version=0,
        )

        # Phase 2: Reset KG and embedder
        if self.kg is not None and FactorKnowledgeGraph is not None:
            self.kg = FactorKnowledgeGraph()
        if self.embedder is not None and FormulaEmbedder is not None:
            self.embedder = FormulaEmbedder()

    # ------------------------------------------------------------------
    # Phase 2: Knowledge graph helpers
    # ------------------------------------------------------------------

    def _update_knowledge_graph(self, trajectory: list[dict]) -> None:
        """Add factors from a trajectory to the knowledge graph.

        Extracts operators from formulas, creates FactorNode instances,
        and registers correlation edges between co-evaluated candidates.
        """
        import re

        if self.kg is None or FactorNode is None:
            return

        op_pattern = re.compile(r"\b([A-Z][a-zA-Z]+)\(")
        feat_pattern = re.compile(r"\$\w+")

        factor_ids: list[str] = []

        for candidate in trajectory:
            fid = candidate.get("factor_id", "")
            formula = candidate.get("formula", "")
            if not fid or not formula:
                continue

            # Parse operators and features from formula
            operators = op_pattern.findall(formula)
            features = feat_pattern.findall(formula)

            node = FactorNode(
                factor_id=fid,
                formula=formula,
                ic_mean=candidate.get("ic", 0.0),
                category=candidate.get("category", ""),
                operators=operators,
                features=features,
                batch_number=self._batch_counter,
                admitted=candidate.get("admitted", False),
            )

            # Embed if embedder is available
            if self.embedder is not None:
                node.embedding = self.embedder.embed(fid, formula)

            self.kg.add_factor(node)
            factor_ids.append(fid)

            # Add correlation edge to existing library member
            correlated_with = candidate.get("correlated_with", "")
            max_corr = candidate.get("max_correlation", 0.0)
            if correlated_with and max_corr > 0:
                self.kg.add_correlation_edge(fid, correlated_with, max_corr)
