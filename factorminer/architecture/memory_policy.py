"""Formal policy boundary for memory retrieval, formation, and persistence."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from factorminer.architecture.families import FactorFamilyDiscovery, infer_family
from factorminer.architecture.paper_protocol import PaperProtocol
from factorminer.core.expression_tree import ConstantNode, LeafNode, Node, OperatorNode
from factorminer.core.parser import try_parse
from factorminer.evaluation.regime import MarketRegime, RegimeConfig, RegimeDetector
from factorminer.memory.evolution import evolve_memory
from factorminer.memory.formation import form_memory
from factorminer.memory.kg_retrieval import retrieve_memory_enhanced
from factorminer.memory.knowledge_graph import FactorKnowledgeGraph, FactorNode
from factorminer.memory.memory_store import ExperienceMemory
from factorminer.memory.retrieval import HybridRetrievalConfig, retrieve_memory

logger = logging.getLogger(__name__)


class MemoryPolicy(ABC):
    """Policy interface for memory state, retrieval, and evolution."""

    @abstractmethod
    def schema(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        memory: ExperienceMemory,
        *,
        library_state: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def form(
        self,
        memory: ExperienceMemory,
        trajectory: list[dict[str, Any]],
        *,
        iteration: int,
    ) -> ExperienceMemory:
        raise NotImplementedError

    @abstractmethod
    def evolve(
        self,
        memory: ExperienceMemory,
        formed: ExperienceMemory,
    ) -> ExperienceMemory:
        raise NotImplementedError

    @abstractmethod
    def serialize(self, memory: ExperienceMemory) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def restore(self, payload: dict[str, Any]) -> ExperienceMemory:
        raise NotImplementedError


class PaperMemoryPolicy(MemoryPolicy):
    """Default paper-faithful memory policy using the F/E/R operators."""

    def __init__(
        self,
        protocol: PaperProtocol,
        *,
        max_success_patterns: int = 50,
        max_failure_patterns: int = 100,
        max_insights: int = 30,
    ) -> None:
        self.protocol = protocol
        self.max_success_patterns = max_success_patterns
        self.max_failure_patterns = max_failure_patterns
        self.max_insights = max_insights

    def schema(self) -> dict[str, Any]:
        return {
            "policy": "paper",
            "versioning": "monotonic_integer",
            "state_schema": {
                "library_size": "int",
                "recent_admissions": "list[dict]",
                "recent_rejections": "list[dict]",
                "domain_saturation": "dict[str,float]",
                "admission_log": "list[dict]",
            },
            "formation_rules": "factorminer.memory.formation.form_memory",
            "retrieval_ranking": "factorminer.memory.retrieval.retrieve_memory",
            "reclassification_rules": "factorminer.memory.evolution.evolve_memory",
            "persistence": "ExperienceMemory.to_dict()/from_dict()",
            "limits": {
                "max_success_patterns": self.max_success_patterns,
                "max_failure_patterns": self.max_failure_patterns,
                "max_insights": self.max_insights,
            },
        }

    def retrieve(
        self,
        memory: ExperienceMemory,
        *,
        library_state: dict[str, Any],
    ) -> dict[str, Any]:
        signal = retrieve_memory(
            memory,
            library_state=library_state,
            max_success=min(8, self.max_success_patterns),
            max_forbidden=min(10, self.max_failure_patterns),
            max_insights=min(10, self.max_insights),
        )
        signal["memory_policy"] = self.schema()
        signal["protocol_mode"] = self.protocol.benchmark_mode
        return signal

    def form(
        self,
        memory: ExperienceMemory,
        trajectory: list[dict[str, Any]],
        *,
        iteration: int,
    ) -> ExperienceMemory:
        return form_memory(memory, trajectory, iteration)

    def evolve(
        self,
        memory: ExperienceMemory,
        formed: ExperienceMemory,
    ) -> ExperienceMemory:
        return evolve_memory(
            memory,
            formed,
            max_success_patterns=self.max_success_patterns,
            max_failure_patterns=self.max_failure_patterns,
        )

    def serialize(self, memory: ExperienceMemory) -> dict[str, Any]:
        payload = memory.to_dict()
        payload["memory_policy"] = self.schema()
        return payload

    def restore(self, payload: dict[str, Any]) -> ExperienceMemory:
        return ExperienceMemory.from_dict(payload)


class NoMemoryPolicy(MemoryPolicy):
    """Ablation policy that disables retrieval and distillation."""

    def __init__(self, protocol: PaperProtocol) -> None:
        self.protocol = protocol

    def schema(self) -> dict[str, Any]:
        return {
            "policy": "none",
            "versioning": "passthrough",
            "state_schema": {},
            "formation_rules": "disabled",
            "retrieval_ranking": "disabled",
            "reclassification_rules": "disabled",
            "persistence": "ExperienceMemory.to_dict()/from_dict()",
        }

    def retrieve(
        self,
        memory: ExperienceMemory,
        *,
        library_state: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "recommended_directions": [],
            "forbidden_directions": [],
            "insights": [],
            "library_state": {
                "library_size": int(library_state.get("library_size", memory.state.library_size)),
            },
            "prompt_text": "",
            "memory_policy": self.schema(),
            "protocol_mode": self.protocol.benchmark_mode,
            "memory_disabled": True,
        }

    def form(
        self,
        memory: ExperienceMemory,
        trajectory: list[dict[str, Any]],
        *,
        iteration: int,
    ) -> ExperienceMemory:
        return memory

    def evolve(
        self,
        memory: ExperienceMemory,
        formed: ExperienceMemory,
    ) -> ExperienceMemory:
        return memory

    def serialize(self, memory: ExperienceMemory) -> dict[str, Any]:
        payload = memory.to_dict()
        payload["memory_policy"] = self.schema()
        return payload

    def restore(self, payload: dict[str, Any]) -> ExperienceMemory:
        return ExperienceMemory.from_dict(payload)


class RegimeAwareMemoryPolicy(PaperMemoryPolicy):
    """Paper memory with regime-conditioned retrieval context and ranking."""

    _REGIME_KEYWORDS = {
        MarketRegime.BULL: ("momentum", "trend", "breakout", "strength", "volume"),
        MarketRegime.BEAR: ("reversal", "defensive", "quality", "volatility", "liquidity"),
        MarketRegime.SIDEWAYS: ("mean", "reversion", "range", "oscillator", "spread"),
    }

    def __init__(
        self,
        protocol: PaperProtocol,
        returns: Any,
        *,
        lookback_window: int = 60,
        max_success_patterns: int = 50,
        max_failure_patterns: int = 100,
        max_insights: int = 30,
    ) -> None:
        super().__init__(
            protocol,
            max_success_patterns=max_success_patterns,
            max_failure_patterns=max_failure_patterns,
            max_insights=max_insights,
        )
        self.lookback_window = lookback_window
        self._regime = None
        returns_arr = None if returns is None else getattr(returns, "copy", lambda: returns)()
        if returns_arr is not None:
            detector = RegimeDetector(RegimeConfig(lookback_window=max(5, lookback_window)))
            self._regime = detector.classify(returns_arr)

    def schema(self) -> dict[str, Any]:
        schema = super().schema()
        schema["policy"] = "regime_aware"
        schema["regime_conditioning"] = {
            "enabled": self._regime is not None,
            "lookback_window": self.lookback_window,
        }
        return schema

    def retrieve(
        self,
        memory: ExperienceMemory,
        *,
        library_state: dict[str, Any],
    ) -> dict[str, Any]:
        signal = super().retrieve(memory, library_state=library_state)
        regime_context = self._regime_context()
        signal["regime_context"] = regime_context
        if regime_context["active_regime"] == "unknown":
            return signal

        active_regime = MarketRegime[regime_context["active_regime"]]
        recommended = sorted(
            signal["recommended_directions"],
            key=lambda item: self._bias_score(item, active_regime),
            reverse=True,
        )
        signal["recommended_directions"] = recommended
        signal["prompt_text"] = (
            f"{signal['prompt_text']}\n\n=== ACTIVE REGIME ===\n"
            f"Current regime: {regime_context['active_regime']}\n"
            f"Recent regime share: {regime_context['recent_regime_share']:.2f}\n"
            "Prefer directions aligned with this regime and discount stale priors."
        ).strip()
        return signal

    def _regime_context(self) -> dict[str, Any]:
        if self._regime is None:
            return {"active_regime": "unknown", "recent_regime_share": 0.0}

        labels = self._regime.labels
        active = MarketRegime(int(labels[-1]))
        recent = labels[-min(len(labels), self.lookback_window) :]
        share = float(np.mean(recent == active.value)) if len(recent) else 0.0
        stats = self._regime.stats.get(active, {})
        return {
            "active_regime": active.name,
            "recent_regime_share": share,
            "mean_return": float(stats.get("mean_return", 0.0)),
            "volatility": float(stats.get("volatility", 0.0)),
            "n_periods": int(stats.get("n_periods", 0)),
        }

    def _bias_score(self, pattern: dict[str, Any], regime: MarketRegime) -> float:
        text = " ".join(
            str(pattern.get(key, "")).lower()
            for key in ("name", "description", "template")
        )
        keywords = self._REGIME_KEYWORDS.get(regime, ())
        return float(sum(1.0 for keyword in keywords if keyword in text))


class KGMemoryPolicy(PaperMemoryPolicy):
    """Paper memory augmented with a persistent factor knowledge graph.

    When ``enable_embeddings`` is True, a :class:`FormulaEmbedder` is created
    (lazily, same MiniLM/TF-IDF/hash fallback chain Helix uses) and passed into
    :func:`retrieve_memory_enhanced` so ``kg`` policy mining runs receive the
    same semantic-neighbor context Helix already gets. Embeddings stay off by
    default -- the toggle is respected and never forced on.
    """

    def __init__(
        self,
        protocol: PaperProtocol,
        *,
        max_success_patterns: int = 50,
        max_failure_patterns: int = 100,
        max_insights: int = 30,
        enable_embeddings: bool = False,
        hybrid_config: HybridRetrievalConfig | None = None,
        embedder: Any | None = None,
    ) -> None:
        super().__init__(
            protocol,
            max_success_patterns=max_success_patterns,
            max_failure_patterns=max_failure_patterns,
            max_insights=max_insights,
        )
        self.knowledge_graph = FactorKnowledgeGraph()
        self.enable_embeddings = bool(enable_embeddings)
        self.hybrid_config = hybrid_config or HybridRetrievalConfig()
        # Operator-supplied embedder wins; otherwise build only when toggled on.
        self.embedder = embedder
        if self.embedder is None and self.enable_embeddings:
            self.embedder = self._build_embedder()

    @staticmethod
    def _build_embedder() -> Any | None:
        """Construct a FormulaEmbedder using the same lazy import Helix uses.

        Never forces a HuggingFace download during construction -- MiniLM is
        loaded on first encode only, and TF-IDF/hash backends need no network.
        """
        try:
            from factorminer.memory.embeddings import FormulaEmbedder
        except ImportError:
            logger.warning(
                "KGMemoryPolicy: enable_embeddings=True but embeddings module unavailable"
            )
            return None
        try:
            # use_faiss=False keeps tests deterministic and avoids optional FAISS.
            return FormulaEmbedder(use_faiss=False)
        except Exception as exc:  # noqa: BLE001 - embeddings are best-effort
            logger.warning("KGMemoryPolicy: failed to init embedder: %s", exc)
            return None

    def schema(self) -> dict[str, Any]:
        schema = super().schema()
        schema["policy"] = "kg"
        schema["retrieval_ranking"] = "factorminer.memory.kg_retrieval.retrieve_memory_enhanced"
        schema["knowledge_graph"] = {
            "enabled": True,
            "factor_nodes": self.knowledge_graph.get_factor_count(),
            "edges": self.knowledge_graph.get_edge_count(),
        }
        schema["enable_embeddings"] = bool(self.enable_embeddings and self.embedder is not None)
        schema["hybrid_retrieval"] = self.hybrid_config.to_dict()
        return schema

    def retrieve(
        self,
        memory: ExperienceMemory,
        *,
        library_state: dict[str, Any],
    ) -> dict[str, Any]:
        # Bug fix (roadmap item 2): previously kg= was passed but embedder=
        # was omitted, so semantic_neighbors were always empty under the kg
        # policy even when embeddings were enabled elsewhere in the stack.
        signal = retrieve_memory_enhanced(
            memory=memory,
            library_state=library_state,
            max_success=min(8, self.max_success_patterns),
            max_forbidden=min(10, self.max_failure_patterns),
            max_insights=min(10, self.max_insights),
            kg=self.knowledge_graph,
            embedder=self.embedder if self.enable_embeddings else None,
            hybrid_config=self.hybrid_config,
        )
        signal["memory_policy"] = self.schema()
        signal["protocol_mode"] = self.protocol.benchmark_mode
        signal["knowledge_graph"] = {
            "factor_nodes": self.knowledge_graph.get_factor_count(),
            "edges": self.knowledge_graph.get_edge_count(),
        }
        signal["embeddings_enabled"] = bool(
            self.enable_embeddings and self.embedder is not None
        )
        return signal

    def form(
        self,
        memory: ExperienceMemory,
        trajectory: list[dict[str, Any]],
        *,
        iteration: int,
    ) -> ExperienceMemory:
        formed = super().form(memory, trajectory, iteration=iteration)
        self._update_knowledge_graph(trajectory, iteration)
        return formed

    def serialize(self, memory: ExperienceMemory) -> dict[str, Any]:
        payload = super().serialize(memory)
        payload["knowledge_graph"] = self.knowledge_graph.to_dict()
        payload["enable_embeddings"] = self.enable_embeddings
        # Never persist raw model weights or API secrets -- only the toggle.
        return payload

    def restore(self, payload: dict[str, Any]) -> ExperienceMemory:
        if payload.get("knowledge_graph"):
            self.knowledge_graph = FactorKnowledgeGraph.from_dict(payload["knowledge_graph"])
        if "enable_embeddings" in payload:
            self.enable_embeddings = bool(payload["enable_embeddings"])
            if self.enable_embeddings and self.embedder is None:
                self.embedder = self._build_embedder()
        return ExperienceMemory.from_dict(payload)

    def _update_knowledge_graph(self, trajectory: list[dict[str, Any]], iteration: int) -> None:
        known_formulas = {node.formula for node in self.knowledge_graph.list_factor_nodes()}
        for entry in trajectory:
            factor_id = str(entry.get("factor_id", "") or "")
            formula = str(entry.get("formula", "") or "")
            if not factor_id or not formula or formula in known_formulas:
                continue
            node = FactorNode(
                factor_id=factor_id,
                formula=formula,
                ic_mean=float(entry.get("ic", 0.0)),
                category=infer_family(formula),
                operators=[],
                features=[],
                batch_number=iteration,
                admitted=bool(entry.get("admitted", False)),
            )
            self.knowledge_graph.add_factor(node)
            # Keep the embedder cache warm so the next retrieve() sees neighbors.
            if self.enable_embeddings and self.embedder is not None and node.admitted:
                try:
                    self.embedder.embed(factor_id, formula)
                except Exception:  # noqa: BLE001 - cache warm is best-effort
                    logger.debug("KGMemoryPolicy: embed cache warm failed", exc_info=True)
            correlated_with = str(entry.get("correlated_with", "") or "")
            if correlated_with:
                self.knowledge_graph.add_correlation_edge(
                    factor_id,
                    correlated_with,
                    rho=float(entry.get("max_correlation", 0.0)),
                    threshold=min(self.protocol.correlation_threshold, 0.4),
                )
            known_formulas.add(formula)


class FamilyAwareMemoryPolicy(PaperMemoryPolicy):
    """Paper memory reranked by family saturation and family gaps."""

    def __init__(
        self,
        protocol: PaperProtocol,
        *,
        max_success_patterns: int = 50,
        max_failure_patterns: int = 100,
        max_insights: int = 30,
        family_discovery: FactorFamilyDiscovery | None = None,
    ) -> None:
        super().__init__(
            protocol,
            max_success_patterns=max_success_patterns,
            max_failure_patterns=max_failure_patterns,
            max_insights=max_insights,
        )
        self.family_discovery = family_discovery or FactorFamilyDiscovery()

    def schema(self) -> dict[str, Any]:
        schema = super().schema()
        schema["policy"] = "family_aware"
        schema["family_discovery"] = {"enabled": True}
        return schema

    def retrieve(
        self,
        memory: ExperienceMemory,
        *,
        library_state: dict[str, Any],
    ) -> dict[str, Any]:
        signal = super().retrieve(memory, library_state=library_state)
        family_context = self.family_discovery.summarize(
            library_state=library_state,
            memory_signal=signal,
        )
        saturated = set(family_context.get("saturated_families", []))
        recommended = set(family_context.get("recommended_families", []))
        signal["recommended_directions"] = sorted(
            signal["recommended_directions"],
            key=lambda item: self._family_bias(item, saturated, recommended),
            reverse=True,
        )
        signal["family_context"] = family_context
        signal["prompt_text"] = (
            f"{signal['prompt_text']}\n\n{family_context.get('prompt_text', '')}"
        ).strip()
        return signal

    def _family_bias(
        self,
        pattern: dict[str, Any],
        saturated: set[str],
        recommended: set[str],
    ) -> float:
        family = infer_family(str(pattern.get("template", "") or pattern.get("name", "")))
        score = 1.0
        if family in recommended:
            score += 1.0
        if family in saturated:
            score -= 1.0
        return score

# ---------------------------------------------------------------------------
# AlphaMemo-style edit-motif memory (arXiv:2606.20625, Yu et al. 2026)
# ---------------------------------------------------------------------------
#
# The remainder of this module implements ``EditAwareMemoryPolicy``: a
# ``PaperMemoryPolicy`` subclass that adds AlphaMemo's Structured Search-Process
# Memory (SSPM) -- edge-level (parent formula -> child formula) credit assignment
# on top of the existing pattern/family-level memory. See the class docstring below
# for the mapping from paper mechanism to this implementation.

_COMMUTATIVE_OPS = frozenset(
    {"Add", "Mul", "Corr", "Cov", "Max", "Min", "Max2", "Min2", "And", "Or", "Equal", "Eq", "Ne"}
)
_SMOOTHING_OPS = frozenset({"EMA", "DEMA", "SMA", "KAMA", "HMA", "WMA", "Decay", "TsDecay"})
_CONDITIONAL_OPS = frozenset(
    {
        "IfElse", "Greater", "GreaterEqual", "Less", "LessEqual",
        "Equal", "Eq", "Ne", "And", "Or", "Not",
    }
)
_INTERACTION_OPS = frozenset({"Mul", "Corr", "Cov", "Beta"})
_RANK_OPS = frozenset({"TsRank", "CsRank", "TsArgMax", "TsArgMin", "CsZScore", "CsQuantile", "Quantile"})
_SIGN_OP = "Neg"

EDIT_MOTIF_LABELS: tuple[str, ...] = (
    "operator_swap",
    "window_rescale",
    "add_conditional",
    "feature_swap",
    "wrap_smoothing",
    "sign_flip",
    "add_interaction",
    "rank_swap",
    "structural_grow",
    "other",
)
"""Fixed AST-diff edit-motif taxonomy (AlphaMemo, arXiv:2606.20625 Sec 4.3 uses "ten
motif labels: nine named edit types plus an 'other' bucket"). Each label names a
distinct, structurally-detectable class of parent -> child formula edit:

- ``operator_swap``: root operator changed, same arity, structurally identical
  children (e.g. ``Greater(a, b)`` -> ``Less(a, b)``).
- ``window_rescale``: identical operator shape and feature references; only numeric
  parameters (e.g. a rolling window length) changed.
- ``add_conditional``: child introduces a new logical/conditional operator
  (``IfElse``, ``Greater``, ``And``, ...) that directly gates a subtree carried over
  unchanged from the parent.
- ``feature_swap``: identical operator shape and parameters; only leaf feature
  references differ (e.g. ``$close`` -> ``$open``).
- ``wrap_smoothing``: child wraps the *entire* parent expression in one new
  smoothing operator (``EMA``/``WMA``/``SMA``/``DEMA``/``KAMA``/``HMA``/``Decay``).
- ``sign_flip``: child is ``Neg(parent)`` or parent is ``Neg(child)`` with the
  wrapped subtree otherwise unchanged.
- ``add_interaction``: child introduces a new multiplicative/co-movement operator
  (``Mul``/``Corr``/``Cov``/``Beta``) that directly combines a subtree carried over
  unchanged from the parent with a new sibling subtree.
- ``rank_swap``: root operator changed between two ranking/ordering operators
  (``TsRank``/``CsRank``/``TsArgMax``/``TsArgMin``/``CsZScore``/``CsQuantile``/
  ``Quantile``) with identical children -- a more specific case of operator_swap.
- ``structural_grow``: child is structurally larger than the parent but matches
  none of the more specific categories above.
- ``other``: catch-all, including no detected change and net-shrinking edits.
"""


def _structural_key(node: Node) -> Any:
    """Shape signature ignoring leaf feature identity, constant value, and operator
    parameters -- two subtrees share a structural key iff every operator in the tree
    matches (same name, same position). Used to detect "this edit only changed a
    feature reference or a numeric parameter, the surrounding structure is
    identical" (-> ``feature_swap`` / ``window_rescale``).
    """
    if isinstance(node, LeafNode):
        return ("leaf",)
    if isinstance(node, ConstantNode):
        return ("const",)
    assert isinstance(node, OperatorNode)
    children = tuple(_structural_key(c) for c in node.children)
    if node.operator.name in _COMMUTATIVE_OPS:
        children = tuple(sorted(children, key=repr))
    return (node.operator.name, children)


def _canonical_string(node: Node) -> str:
    """Serialize *node* with commutative-operator children sorted, so structurally
    identical subtrees compare equal regardless of argument order.
    """
    if isinstance(node, LeafNode):
        return node.feature_name
    if isinstance(node, ConstantNode):
        return node.to_string()
    assert isinstance(node, OperatorNode)
    parts = [_canonical_string(c) for c in node.children]
    if node.operator.name in _COMMUTATIVE_OPS:
        parts = sorted(parts)
    params = ",".join(
        f"{name}={node.params[name]:g}"
        for name in node.operator.param_names
        if name in node.params
    )
    args = ",".join(parts)
    if params:
        args = f"{args},{params}" if args else params
    return f"{node.operator.name}({args})"


def _param_sequence(node: Node) -> tuple[tuple[tuple[str, float], ...], ...]:
    """Pre-order sequence of every operator node's parameter dict (sorted by key).

    Used to detect a pure ``window_rescale`` edit: identical shape and feature
    references, but a different sequence of numeric parameters.
    """
    seq = []
    for n in node.iter_nodes():
        if isinstance(n, OperatorNode):
            seq.append(tuple(sorted(n.params.items())))
    return tuple(seq)


def _operator_names(node: Node) -> set[str]:
    return {n.operator.name for n in node.iter_nodes() if isinstance(n, OperatorNode)}


def _classify_edit(parent: Node, child: Node) -> str:
    """Classify the structural edit from *parent* to *child* (both AST roots)."""
    p_shape = _structural_key(parent)
    c_shape = _structural_key(child)
    p_canon = _canonical_string(parent)
    c_canon = _canonical_string(child)

    if p_shape == c_shape:
        if p_canon == c_canon:
            return "other"
        if sorted(parent.leaf_features()) != sorted(child.leaf_features()):
            return "feature_swap"
        if _param_sequence(parent) != _param_sequence(child):
            return "window_rescale"
        return "other"

    # sign flip: one side wraps the other, verbatim, in a bare Neg(...).
    if (
        isinstance(child, OperatorNode)
        and child.operator.name == _SIGN_OP
        and len(child.children) == 1
        and _canonical_string(child.children[0]) == p_canon
    ):
        return "sign_flip"
    if (
        isinstance(parent, OperatorNode)
        and parent.operator.name == _SIGN_OP
        and len(parent.children) == 1
        and _canonical_string(parent.children[0]) == c_canon
    ):
        return "sign_flip"

    # smoothing wrap: child wraps the whole parent in one new smoothing operator.
    if (
        isinstance(child, OperatorNode)
        and child.operator.name in _SMOOTHING_OPS
        and len(child.children) == 1
        and _canonical_string(child.children[0]) == p_canon
    ):
        return "wrap_smoothing"

    # root operator swap: same arity, same (multiset of) children, different op.
    if (
        isinstance(parent, OperatorNode)
        and isinstance(child, OperatorNode)
        and parent.operator.arity == child.operator.arity
        and parent.operator.name != child.operator.name
        and sorted(_canonical_string(c) for c in parent.children)
        == sorted(_canonical_string(c) for c in child.children)
    ):
        if parent.operator.name in _RANK_OPS and child.operator.name in _RANK_OPS:
            return "rank_swap"
        return "operator_swap"

    # add_conditional / add_interaction: child introduces a brand-new operator (not
    # present anywhere in the parent) that directly wraps a subtree carried over
    # unchanged from the parent.
    parent_ops = _operator_names(parent)
    new_ops = _operator_names(child) - parent_ops
    parent_subtrees = {_canonical_string(sub) for sub in parent.iter_nodes()}

    if new_ops & _CONDITIONAL_OPS:
        for n in child.iter_nodes():
            if (
                isinstance(n, OperatorNode)
                and n.operator.name in _CONDITIONAL_OPS
                and n.operator.name in new_ops
                and any(_canonical_string(c) in parent_subtrees for c in n.children)
            ):
                return "add_conditional"

    if new_ops & _INTERACTION_OPS:
        for n in child.iter_nodes():
            if (
                isinstance(n, OperatorNode)
                and n.operator.name in _INTERACTION_OPS
                and n.operator.name in new_ops
                and any(_canonical_string(c) in parent_subtrees for c in n.children)
            ):
                return "add_interaction"

    if child.size() > parent.size():
        return "structural_grow"
    return "other"


def extract_edit_motif(parent_formula: str, child_formula: str) -> str:
    """Classify the structural edit from *parent_formula* to *child_formula*.

    Parses both formulas with :func:`factorminer.core.parser.try_parse` into
    :class:`~factorminer.core.expression_tree.ExpressionTree` objects (AlphaMemo
    Sec 4.3: "parse each formula with the same typed operator grammar used by the
    evaluator") and diffs their canonicalized shapes to assign one of
    :data:`EDIT_MOTIF_LABELS`. Falls back to ``"other"`` if either formula fails to
    parse.
    """
    parent_tree = try_parse(parent_formula)
    child_tree = try_parse(child_formula)
    if parent_tree is None or child_tree is None:
        return "other"
    return _classify_edit(parent_tree.root, child_tree.root)


def _quality_bucket(value: float) -> str:
    """Bucket a validation-quality score (Q(p) ~ ``ic_paper_mean``) into a coarse
    tier -- a discretized ``b_q(p)`` (AlphaMemo Eq. 2). Thresholds are intentionally
    coarse, matching the paper's discretization intent rather than its exact cuts:

    - ``value < 0.0``        -> ``"poor"``     (worse than a null factor)
    - ``0.0 <= value < 0.02`` -> ``"weak"``
    - ``0.02 <= value < 0.05`` -> ``"moderate"``
    - ``value >= 0.05``        -> ``"strong"``
    """
    if value < 0.0:
        return "poor"
    if value < 0.02:
        return "weak"
    if value < 0.05:
        return "moderate"
    return "strong"


def _depth_bucket(depth: int) -> str:
    """Bucket expression-tree depth into a coarse tier -- a discretized ``b_d(p)``
    (AlphaMemo Eq. 2's search-depth bucket). FactorMiner has no separate generation
    counter attached to a formula, so tree depth is used as a self-contained proxy:
    deeper trees tend to reflect more accumulated edits from a seed factor.
    """
    if depth <= 2:
        return "shallow"
    if depth <= 4:
        return "medium"
    return "deep"


@dataclass(frozen=True)
class ParentContext:
    """Coarse discretization z(p) of the state around a parent factor p.

    Mirrors AlphaMemo Eq. 2: ``z(p) = (g(p), b_q(p), b_d(p))``. FactorMiner
    substitutes :func:`factorminer.architecture.families.infer_family` for the
    paper's semantic factor category ``g(p)``.
    """

    family: str
    quality_bucket: str
    depth_bucket: str

    @property
    def key(self) -> str:
        """Stable string key for use as a dict/JSON key."""
        return f"{self.family}|{self.quality_bucket}|{self.depth_bucket}"


@dataclass
class MotifStats:
    """Sufficient statistics for one Structured Search-Process Memory cell
    ``M_t[z, m]`` (AlphaMemo Eq. 6): an online (Welford) estimate of the residual
    mean/variance, plus a ``Beta(failure_alpha, failure_beta)`` posterior over "this
    (context, motif) produces a negative residual," which drives the asymmetric veto.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    failure_alpha: float = 1.0
    failure_beta: float = 1.0

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def update(self, residual: float) -> None:
        """Fold one new residual observation into the running statistics."""
        self.count += 1
        delta = residual - self.mean
        self.mean += delta / self.count
        delta2 = residual - self.mean
        self.m2 += delta * delta2
        if residual < 0.0:
            self.failure_alpha += 1.0
        else:
            self.failure_beta += 1.0

    def confidence(self, kappa: float, *, eps: float = 1e-6) -> float:
        """Confidence gate ``c_t(z, m) in [0, 1]`` (AlphaMemo Eq. 7).

        Grows with the observation count (``count / (count + kappa)``, so *kappa*
        observations yield 0.5) and with the residual's signal-to-noise ratio
        (``min(1, |mean| / (std + eps))``), so a handful of noisy early observations
        cannot dominate while many consistent observations saturate toward 1.
        """
        if self.count == 0:
            return 0.0
        volume_term = self.count / (self.count + kappa)
        snr_term = min(1.0, abs(self.mean) / (self.std + eps))
        return float(volume_term * snr_term)

    def failure_probability(self) -> float:
        """Posterior mean ``P(residual < 0 | history)`` under the Beta model."""
        return self.failure_alpha / (self.failure_alpha + self.failure_beta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "failure_alpha": self.failure_alpha,
            "failure_beta": self.failure_beta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MotifStats:
        return cls(
            count=int(d.get("count", 0)),
            mean=float(d.get("mean", 0.0)),
            m2=float(d.get("m2", 0.0)),
            failure_alpha=float(d.get("failure_alpha", 1.0)),
            failure_beta=float(d.get("failure_beta", 1.0)),
        )


@dataclass(frozen=True)
class ActionScore:
    """Result of confidence-gated residual fusion + asymmetric veto for one
    (parent, motif) action (AlphaMemo Eq. 4-5, 9-10).
    """

    parent_formula: str
    motif: str
    context_key: str
    base_score: float
    residual: float
    confidence: float
    failure_probability: float
    vetoed: bool
    adjusted_score: float


class EditAwareMemoryPolicy(PaperMemoryPolicy):
    """Paper memory augmented with AlphaMemo-style Structured Search-Process Memory.

    Implements the edit-level credit-assignment mechanism from AlphaMemo (Yu, Zheng,
    Pan, Liu, Wang & He, "AlphaMemo: Structured Search-Process Memory for
    Self-Evolving Alpha Mining Agents", arXiv:2606.20625, 2026): rather than storing
    pattern/family-level statistics only, this policy attaches a residual-quality
    observation to every parent -> child formula edge, keyed by
    ``(parent context, AST-diff edit motif)`` (see :func:`extract_edit_motif`,
    :class:`ParentContext`). Residuals are folded into a confidence-gated statistic
    (:class:`MotifStats`) that:

    1. Extracts a small, fixed-taxonomy edit motif from the AST difference between
       parent and child (:data:`EDIT_MOTIF_LABELS`).
    2. Stores ``(parent_context, edit_motif) -> residual statistics`` where the
       residual is the child's quality minus a context-local ledger-prior baseline
       (:meth:`_context_baseline`, AlphaMemo Eq. 3), falling back to the parent's own
       quality when no context history exists yet.
    3. Gates that residual's influence by an explicit confidence formula
       (:meth:`MotifStats.confidence`) that vanishes for sparse/noisy evidence and
       saturates toward full strength once evidence is plentiful and consistent.
    4. Applies an *asymmetric* veto (:meth:`score_action`): high-confidence,
       reliably negative ``(context, motif)`` residuals strongly suppress that
       action, while positive residuals only ever contribute a small, explicitly
       bounded soft boost -- never a symmetric signed weight.

    Real ``RalphLoop``/``HelixLoop`` mining trajectories do not currently attach
    parent lineage to trajectory entries, so trajectory entries that omit
    ``parent_formula`` are simply skipped for edge extraction (see
    :meth:`_observe_edge`) and this policy degrades gracefully to plain
    ``PaperMemoryPolicy`` behavior with zero motif history -- mirroring the paper's
    own "fall back to the base search prior when evidence is sparse" design.
    Trajectory entries opt in by adding ``parent_formula`` (and, optionally,
    ``parent_ic_paper_mean``) alongside the existing ``formula``/``ic_paper_mean``
    (or ``paper_ic``/``ic``) fields already produced by the mining loop.
    """

    def __init__(
        self,
        protocol: PaperProtocol,
        *,
        max_success_patterns: int = 50,
        max_failure_patterns: int = 100,
        max_insights: int = 30,
        confidence_kappa: float = 8.0,
        memory_weight: float = 1.0,
        positive_boost_cap: float = 0.25,
        veto_confidence_threshold: float = 0.6,
        veto_failure_threshold: float = 0.75,
        veto_penalty: float = 10.0,
    ) -> None:
        super().__init__(
            protocol,
            max_success_patterns=max_success_patterns,
            max_failure_patterns=max_failure_patterns,
            max_insights=max_insights,
        )
        self.confidence_kappa = confidence_kappa
        self.memory_weight = memory_weight
        self.positive_boost_cap = positive_boost_cap
        self.veto_confidence_threshold = veto_confidence_threshold
        self.veto_failure_threshold = veto_failure_threshold
        self.veto_penalty = veto_penalty
        self._motif_stats: dict[tuple[str, str], MotifStats] = {}
        self._quality_ledger: dict[str, float] = {}
        self._context_quality: dict[str, tuple[float, int]] = {}

    # -- schema / persistence ------------------------------------------------

    def schema(self) -> dict[str, Any]:
        schema = super().schema()
        schema["policy"] = "edit_aware"
        schema["edit_motif_taxonomy"] = list(EDIT_MOTIF_LABELS)
        schema["sspm"] = {
            "confidence_kappa": self.confidence_kappa,
            "memory_weight": self.memory_weight,
            "positive_boost_cap": self.positive_boost_cap,
            "veto_confidence_threshold": self.veto_confidence_threshold,
            "veto_failure_threshold": self.veto_failure_threshold,
            "veto_penalty": self.veto_penalty,
            "tracked_edges": len(self._motif_stats),
        }
        return schema

    def serialize(self, memory: ExperienceMemory) -> dict[str, Any]:
        payload = super().serialize(memory)
        payload["edit_motif_memory"] = self._export_state()
        return payload

    def restore(self, payload: dict[str, Any]) -> ExperienceMemory:
        state = payload.get("edit_motif_memory")
        if state:
            self._import_state(state)
        return super().restore(payload)

    def _export_state(self) -> dict[str, Any]:
        return {
            "motif_stats": [
                {"context_key": ctx, "motif": motif, **stats.to_dict()}
                for (ctx, motif), stats in self._motif_stats.items()
            ],
            "quality_ledger": dict(self._quality_ledger),
            "context_quality": [
                {"context_key": ctx, "sum": total, "count": count}
                for ctx, (total, count) in self._context_quality.items()
            ],
        }

    def _import_state(self, state: dict[str, Any]) -> None:
        self._motif_stats = {
            (str(row["context_key"]), str(row["motif"])): MotifStats.from_dict(row)
            for row in state.get("motif_stats", [])
        }
        self._quality_ledger = {
            str(k): float(v) for k, v in dict(state.get("quality_ledger", {})).items()
        }
        self._context_quality = {
            str(row["context_key"]): (float(row["sum"]), int(row["count"]))
            for row in state.get("context_quality", [])
        }

    # -- formation -------------------------------------------------------

    def form(
        self,
        memory: ExperienceMemory,
        trajectory: list[dict[str, Any]],
        *,
        iteration: int,
    ) -> ExperienceMemory:
        formed = super().form(memory, trajectory, iteration=iteration)
        for entry in trajectory:
            self._observe_edge(entry)
        return formed

    def _observe_edge(self, entry: dict[str, Any]) -> None:
        """Extract and record one parent -> child edge observation from a
        trajectory entry, if it carries enough information to do so.
        """
        child_formula = str(entry.get("formula", "") or "")
        if not child_formula:
            return
        child_quality = self._extract_quality(entry)
        if child_quality is not None:
            self._quality_ledger[child_formula] = child_quality

        parent_formula = str(entry.get("parent_formula", "") or "")
        if not parent_formula or child_quality is None:
            return

        motif = extract_edit_motif(parent_formula, child_formula)
        parent_quality = self._resolve_parent_quality(entry, parent_formula)
        context = self._context_for(parent_formula, parent_quality)
        baseline = self._context_baseline(context.key, fallback=parent_quality)
        residual = child_quality - baseline

        stats = self._motif_stats.setdefault((context.key, motif), MotifStats())
        stats.update(residual)

        total, count = self._context_quality.get(context.key, (0.0, 0))
        self._context_quality[context.key] = (total + child_quality, count + 1)

    @staticmethod
    def _extract_quality(entry: dict[str, Any]) -> float | None:
        """Read a factor's paper-target IC from a trajectory entry.

        Tries ``ic_paper_mean`` first (the task's canonical field name / the name on
        ``Factor``), then ``paper_ic`` (the field name ``RalphLoop._build_trajectory``
        actually emits today), then plain ``ic`` as a last resort.
        """
        for key in ("ic_paper_mean", "paper_ic", "ic"):
            value = entry.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _resolve_parent_quality(entry: dict[str, Any], parent_formula: str) -> float:
        for key in ("parent_ic_paper_mean", "parent_paper_ic", "parent_ic"):
            value = entry.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    def _context_for(self, parent_formula: str, parent_quality: float) -> ParentContext:
        family = infer_family(parent_formula)
        tree = try_parse(parent_formula)
        depth = tree.depth() if tree is not None else 1
        return ParentContext(
            family=family,
            quality_bucket=_quality_bucket(parent_quality),
            depth_bucket=_depth_bucket(depth),
        )

    def _context_baseline(self, context_key: str, *, fallback: float) -> float:
        """Estimate ``Q_hat_ledger(p)`` (AlphaMemo Eq. 3): the mean observed child
        quality across every prior edge whose parent fell in the same context
        bucket, falling back to *fallback* (the parent's own quality) when no such
        history exists yet -- this keeps the residual target zero-centered from the
        very first observation in a new context.
        """
        total, count = self._context_quality.get(context_key, (0.0, 0))
        if count == 0:
            return fallback
        return total / count

    # -- scoring / veto ----------------------------------------------------

    def score_action(
        self,
        parent_formula: str,
        motif: str,
        *,
        base_score: float = 0.0,
        parent_quality: float | None = None,
    ) -> ActionScore:
        """Fuse *base_score* (a caller-supplied ledger/search-prior score for the
        parent, e.g. its quality) with the confidence-gated SSPM residual for the
        ``(parent, motif)`` action, applying the asymmetric process veto.

        Mirrors AlphaMemo Eq. 4-5's action score ``A_t(p, m) = log(S_ledger(p) + eps)
        + lambda_t * c_t(z, m) * Delta_t(z, m)`` and Eq. 9's veto rule
        ``Veto(z, m) = I[c_t(z, m) > tau_c AND pi-_z,m > tau_v]``. The base ledger
        score ``S_ledger(p)`` itself lives outside this policy (it is FactorMiner's
        ``EvaluationKernel``/library-geometry scoring, not duplicated here); callers
        pass whatever prior score they already have via *base_score*.

        Asymmetry (task requirement 4, AlphaMemo Sec 4.4): a *positive* residual's
        contribution is always capped at ``self.positive_boost_cap`` regardless of
        confidence (a small, bounded soft boost). A *negative* residual's
        contribution is uncapped and proportional to confidence, and once both the
        confidence and the failure posterior clear their thresholds the action is
        vetoed outright (score collapsed by ``self.veto_penalty``) rather than
        merely nudged.
        """
        if parent_quality is None:
            parent_quality = self._quality_ledger.get(parent_formula, 0.0)
        context = self._context_for(parent_formula, parent_quality)
        stats = self._motif_stats.get((context.key, motif))
        if stats is None or stats.count == 0:
            return ActionScore(
                parent_formula=parent_formula,
                motif=motif,
                context_key=context.key,
                base_score=base_score,
                residual=0.0,
                confidence=0.0,
                failure_probability=0.5,
                vetoed=False,
                adjusted_score=base_score,
            )

        confidence = stats.confidence(self.confidence_kappa)
        residual = stats.mean
        failure_p = stats.failure_probability()
        vetoed = (
            residual < 0.0
            and confidence >= self.veto_confidence_threshold
            and failure_p >= self.veto_failure_threshold
        )

        if vetoed:
            adjusted = base_score - self.veto_penalty
        else:
            influence = self.memory_weight * confidence * residual
            if residual >= 0.0:
                influence = min(influence, self.positive_boost_cap)
            adjusted = base_score + influence

        return ActionScore(
            parent_formula=parent_formula,
            motif=motif,
            context_key=context.key,
            base_score=base_score,
            residual=residual,
            confidence=confidence,
            failure_probability=failure_p,
            vetoed=vetoed,
            adjusted_score=adjusted,
        )

    # -- retrieval --------------------------------------------------------

    def retrieve(
        self,
        memory: ExperienceMemory,
        *,
        library_state: dict[str, Any],
    ) -> dict[str, Any]:
        signal = super().retrieve(memory, library_state=library_state)
        candidates = list(library_state.get("recent_admissions", []) or [])
        guidance: dict[str, dict[str, Any]] = {}
        vetoed_summary: list[str] = []
        for item in candidates:
            formula = str(item.get("formula", "") or "")
            if not formula:
                continue
            try:
                quality = float(item.get("ic_paper_mean", item.get("ic_mean", 0.0)) or 0.0)
            except (TypeError, ValueError):
                quality = 0.0
            per_motif: dict[str, Any] = {}
            for motif in EDIT_MOTIF_LABELS:
                action = self.score_action(
                    formula, motif, base_score=quality, parent_quality=quality
                )
                per_motif[motif] = {
                    "score": action.adjusted_score,
                    "confidence": action.confidence,
                    "residual": action.residual,
                    "vetoed": action.vetoed,
                }
                if action.vetoed:
                    vetoed_summary.append(f"{motif} on '{formula}'")
            guidance[formula] = per_motif

        signal["edit_motif_guidance"] = guidance
        signal["motif_taxonomy"] = list(EDIT_MOTIF_LABELS)
        if vetoed_summary:
            veto_text = (
                "\n\n=== EDIT-MOTIF VETOES ===\n"
                "Avoid these edits -- high-confidence history of underperformance:\n"
                + "\n".join(f"- {item}" for item in vetoed_summary)
            )
            signal["prompt_text"] = f"{signal['prompt_text']}{veto_text}".strip()
        return signal


def _resolve_enable_embeddings(config: Any, memory_cfg: Any) -> bool:
    """Resolve the embeddings toggle without forcing it on.

    Precedence: explicit ``memory.enable_embeddings`` → flat
    ``config.enable_embeddings`` → ``phase2.helix.enable_embeddings`` → False.
    """
    if memory_cfg is not None and hasattr(memory_cfg, "enable_embeddings"):
        return bool(getattr(memory_cfg, "enable_embeddings"))
    if hasattr(config, "enable_embeddings"):
        return bool(getattr(config, "enable_embeddings"))
    phase2 = getattr(config, "phase2", None)
    helix = getattr(phase2, "helix", None) if phase2 is not None else None
    if helix is not None and hasattr(helix, "enable_embeddings"):
        return bool(helix.enable_embeddings)
    return False


def _resolve_hybrid_config(memory_cfg: Any) -> HybridRetrievalConfig:
    """Build HybridRetrievalConfig from optional memory.* nested fields."""
    if memory_cfg is None:
        return HybridRetrievalConfig()
    hybrid = getattr(memory_cfg, "hybrid_retrieval", None)
    if isinstance(hybrid, HybridRetrievalConfig):
        return hybrid
    if isinstance(hybrid, dict):
        return HybridRetrievalConfig(
            enabled=bool(hybrid.get("enabled", True)),
            rrf_k=int(hybrid.get("rrf_k", 60)),
            bm25_k1=float(hybrid.get("bm25_k1", 1.5)),
            bm25_b=float(hybrid.get("bm25_b", 0.75)),
            enable_dense=bool(hybrid.get("enable_dense", True)),
            enable_bm25=bool(hybrid.get("enable_bm25", True)),
            enable_heuristic=bool(hybrid.get("enable_heuristic", True)),
            enable_rerank=bool(hybrid.get("enable_rerank", False)),
            rerank_pool_size=int(hybrid.get("rerank_pool_size", 16)),
        )
    # Flat optional attrs on MemoryConfig (all default-preserving).
    return HybridRetrievalConfig(
        enabled=bool(getattr(memory_cfg, "hybrid_enabled", True)),
        enable_rerank=bool(getattr(memory_cfg, "hybrid_enable_rerank", False)),
        rrf_k=int(getattr(memory_cfg, "hybrid_rrf_k", 60)),
    )


def build_memory_policy(
    config: Any,
    protocol: PaperProtocol,
    *,
    returns: Any = None,
) -> MemoryPolicy:
    """Construct the configured memory policy from flat or hierarchical config."""

    memory_cfg = getattr(config, "memory", None)
    policy_name = str(
        getattr(memory_cfg, "policy", getattr(config, "memory_policy", "paper"))
    ).strip().lower()
    max_success_patterns = int(
        getattr(memory_cfg, "max_success_patterns", getattr(config, "max_success_patterns", 50))
    )
    max_failure_patterns = int(
        getattr(memory_cfg, "max_failure_patterns", getattr(config, "max_failure_patterns", 100))
    )
    max_insights = int(
        getattr(memory_cfg, "max_insights", getattr(config, "max_insights", 30))
    )
    regime_lookback_window = int(
        getattr(
            memory_cfg,
            "regime_lookback_window",
            getattr(config, "memory_regime_lookback_window", 60),
        )
    )

    if policy_name in {"paper", "default"}:
        return PaperMemoryPolicy(
            protocol,
            max_success_patterns=max_success_patterns,
            max_failure_patterns=max_failure_patterns,
            max_insights=max_insights,
        )
    if policy_name in {"none", "no_memory"}:
        return NoMemoryPolicy(protocol)
    if policy_name in {"kg", "knowledge_graph"}:
        enable_embeddings = _resolve_enable_embeddings(config, memory_cfg)
        hybrid_config = _resolve_hybrid_config(memory_cfg)
        return KGMemoryPolicy(
            protocol,
            max_success_patterns=max_success_patterns,
            max_failure_patterns=max_failure_patterns,
            max_insights=max_insights,
            enable_embeddings=enable_embeddings,
            hybrid_config=hybrid_config,
        )
    if policy_name in {"family", "family_aware", "family-aware"}:
        return FamilyAwareMemoryPolicy(
            protocol,
            max_success_patterns=max_success_patterns,
            max_failure_patterns=max_failure_patterns,
            max_insights=max_insights,
        )
    if policy_name in {"regime_aware", "regime-aware"}:
        return RegimeAwareMemoryPolicy(
            protocol,
            returns,
            lookback_window=regime_lookback_window,
            max_success_patterns=max_success_patterns,
            max_failure_patterns=max_failure_patterns,
            max_insights=max_insights,
        )
    if policy_name in {"edit_aware", "edit-aware"}:
        return EditAwareMemoryPolicy(
            protocol,
            max_success_patterns=max_success_patterns,
            max_failure_patterns=max_failure_patterns,
            max_insights=max_insights,
        )
    raise ValueError(
        "Unsupported memory policy "
        f"'{policy_name}'. Expected one of: paper, none, kg, family_aware, "
        "regime_aware, edit_aware"
    )
