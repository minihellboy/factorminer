"""Formal policy boundary for memory retrieval, formation, and persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from factorminer.architecture.families import FactorFamilyDiscovery, infer_family
from factorminer.architecture.paper_protocol import PaperProtocol
from factorminer.evaluation.regime import MarketRegime, RegimeConfig, RegimeDetector
from factorminer.memory.evolution import evolve_memory
from factorminer.memory.formation import form_memory
from factorminer.memory.memory_store import ExperienceMemory
from factorminer.memory.kg_retrieval import retrieve_memory_enhanced
from factorminer.memory.knowledge_graph import FactorKnowledgeGraph, FactorNode
from factorminer.memory.retrieval import retrieve_memory


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
    """Paper memory augmented with a persistent factor knowledge graph."""

    def __init__(
        self,
        protocol: PaperProtocol,
        *,
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
        self.knowledge_graph = FactorKnowledgeGraph()

    def schema(self) -> dict[str, Any]:
        schema = super().schema()
        schema["policy"] = "kg"
        schema["retrieval_ranking"] = "factorminer.memory.kg_retrieval.retrieve_memory_enhanced"
        schema["knowledge_graph"] = {
            "enabled": True,
            "factor_nodes": self.knowledge_graph.get_factor_count(),
            "edges": self.knowledge_graph.get_edge_count(),
        }
        return schema

    def retrieve(
        self,
        memory: ExperienceMemory,
        *,
        library_state: dict[str, Any],
    ) -> dict[str, Any]:
        signal = retrieve_memory_enhanced(
            memory=memory,
            library_state=library_state,
            max_success=min(8, self.max_success_patterns),
            max_forbidden=min(10, self.max_failure_patterns),
            max_insights=min(10, self.max_insights),
            kg=self.knowledge_graph,
        )
        signal["memory_policy"] = self.schema()
        signal["protocol_mode"] = self.protocol.benchmark_mode
        signal["knowledge_graph"] = {
            "factor_nodes": self.knowledge_graph.get_factor_count(),
            "edges": self.knowledge_graph.get_edge_count(),
        }
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
        return payload

    def restore(self, payload: dict[str, Any]) -> ExperienceMemory:
        if payload.get("knowledge_graph"):
            self.knowledge_graph = FactorKnowledgeGraph.from_dict(payload["knowledge_graph"])
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
        return KGMemoryPolicy(
            protocol,
            max_success_patterns=max_success_patterns,
            max_failure_patterns=max_failure_patterns,
            max_insights=max_insights,
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
    raise ValueError(
        "Unsupported memory policy "
        f"'{policy_name}'. Expected one of: paper, none, kg, family_aware, regime_aware"
    )
