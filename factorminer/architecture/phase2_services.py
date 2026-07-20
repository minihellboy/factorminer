"""Reusable Phase 2 services extracted from Helix loop logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from factorminer.architecture.families import extract_features, extract_operators, infer_family

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Phase2Components:
    """Optional runtime components consumed by ``HelixLoop``."""

    debate_generator: Any = None
    canonicalizer: Any = None
    causal_validator: Any = None
    regime_detector: Any = None
    regime_evaluator: Any = None
    regime_classification: Any = None
    capacity_estimator: Any = None
    bootstrap_tester: Any = None
    fdr_controller: Any = None
    knowledge_graph: Any = None
    embedder: Any = None
    auto_inventor: Any = None
    custom_operator_store: Any = None


class Phase2ComponentFactory:
    """Resolve and construct optional Helix components outside the loop.

    Imports remain lazy so base Ralph runs and installations without optional
    dependencies do not pay for, or fail on, Phase-2 modules.
    """

    @staticmethod
    def resolve(module_name: str, *attribute_names: str) -> tuple[Any | None, ...]:
        try:
            module = import_module(module_name)
        except ImportError:
            return tuple(None for _ in attribute_names)
        return tuple(getattr(module, name, None) for name in attribute_names)

    @staticmethod
    def _construct(label: str, constructor, *args, **kwargs):
        if constructor is None:
            logger.warning("Helix: %s module unavailable", label)
            return None
        try:
            return constructor(*args, **kwargs)
        except Exception as exc:
            logger.warning("Helix: failed to initialize %s: %s", label, exc)
            return None

    def build(
        self,
        *,
        llm_provider: Any,
        data_tensor: Any,
        returns: Any,
        output_dir: str,
        debate_config: Any = None,
        canonicalize: bool = False,
        causal_config: Any = None,
        regime_config: Any = None,
        capacity_config: Any = None,
        significance_config: Any = None,
        enable_knowledge_graph: bool = False,
        enable_embeddings: bool = False,
        enable_auto_inventor: bool = False,
        volume: Any = None,
    ) -> Phase2Components:
        """Build only the components enabled by the supplied configuration."""
        components = Phase2Components()

        if debate_config is not None:
            (generator_cls,) = self.resolve("factorminer.agent.debate", "DebateGenerator")
            components.debate_generator = self._construct(
                "debate generator",
                generator_cls,
                llm_provider=llm_provider,
                debate_config=debate_config,
            )

        if canonicalize:
            (canonicalizer_cls,) = self.resolve(
                "factorminer.core.canonicalizer", "FormulaCanonicalizer"
            )
            components.canonicalizer = self._construct(
                "canonicalizer",
                canonicalizer_cls,
            )

        if causal_config is not None:
            (components.causal_validator,) = self.resolve(
                "factorminer.evaluation.causal", "CausalValidator"
            )
            if components.causal_validator is None:
                logger.warning("Helix: causal validator module unavailable")

        if regime_config is not None:
            detector_cls, evaluator_cls = self.resolve(
                "factorminer.evaluation.regime",
                "RegimeDetector",
                "RegimeAwareEvaluator",
            )
            components.regime_detector = self._construct(
                "regime detector",
                detector_cls,
                regime_config,
            )
            if components.regime_detector is not None:
                try:
                    components.regime_classification = components.regime_detector.classify(
                        returns
                    )
                    components.regime_evaluator = evaluator_cls(
                        returns=returns,
                        regime=components.regime_classification,
                        config=regime_config,
                    )
                except Exception as exc:
                    logger.warning("Helix: failed to initialize regime evaluator: %s", exc)
                    components.regime_evaluator = None

        if capacity_config is not None:
            if volume is None:
                logger.warning("Helix: capacity config provided without volume data")
            else:
                (estimator_cls,) = self.resolve(
                    "factorminer.evaluation.capacity", "CapacityEstimator"
                )
                components.capacity_estimator = self._construct(
                    "capacity estimator",
                    estimator_cls,
                    returns=returns,
                    volume=volume,
                    config=capacity_config,
                )

        if significance_config is not None:
            bootstrap_cls, fdr_cls = self.resolve(
                "factorminer.evaluation.significance",
                "BootstrapICTester",
                "FDRController",
            )
            components.bootstrap_tester = self._construct(
                "bootstrap tester",
                bootstrap_cls,
                significance_config,
            )
            components.fdr_controller = self._construct(
                "FDR controller",
                fdr_cls,
                significance_config,
            )

        if enable_knowledge_graph:
            (graph_cls,) = self.resolve(
                "factorminer.memory.knowledge_graph", "FactorKnowledgeGraph"
            )
            components.knowledge_graph = self._construct(
                "knowledge graph",
                graph_cls,
            )

        if enable_embeddings:
            (embedder_cls,) = self.resolve("factorminer.memory.embeddings", "FormulaEmbedder")
            components.embedder = self._construct("formula embedder", embedder_cls)

        if enable_auto_inventor:
            inventor_cls, store_cls = (
                self.resolve("factorminer.operators.auto_inventor", "OperatorInventor")[0],
                self.resolve("factorminer.operators.custom", "CustomOperatorStore")[0],
            )
            components.auto_inventor = self._construct(
                "operator inventor",
                inventor_cls,
                llm_provider=llm_provider,
                data_tensor=data_tensor,
                returns=returns,
            )
            components.custom_operator_store = self._construct(
                "custom operator store",
                store_cls,
                store_dir=str(Path(output_dir) / "custom_operators"),
            )

        return components


@dataclass
class OnlineForgettingService:
    """Applies online forgetting to memory state after dry spells."""

    forgetting_lambda: float = 0.95
    no_admission_demote_after: int = 20

    def apply(self, memory: Any, no_admission_streak: int) -> None:
        for pattern in memory.success_patterns:
            if hasattr(pattern, "occurrence_count"):
                pattern.occurrence_count = int(pattern.occurrence_count * self.forgetting_lambda)

        if no_admission_streak < self.no_admission_demote_after:
            return

        for pattern in memory.success_patterns:
            if not hasattr(pattern, "success_rate"):
                continue
            if pattern.success_rate == "High":
                pattern.success_rate = "Medium"
            elif pattern.success_rate == "Medium":
                pattern.success_rate = "Low"


class KnowledgeGraphService:
    """Owns factor-node construction, correlation edges, and derivation edges."""

    def add_result(
        self,
        *,
        kg: Any,
        result: Any,
        library: Any,
        iteration: int,
        embedder: Any = None,
    ) -> None:
        if kg is None or not getattr(result, "admitted", False):
            return

        from factorminer.memory.knowledge_graph import FactorNode

        node = FactorNode(
            factor_id=result.factor_name,
            formula=result.formula,
            ic_mean=result.ic_mean,
            category=infer_family(result.formula),
            operators=extract_operators(result.formula),
            features=extract_features(result.formula),
            batch_number=iteration,
            admitted=True,
        )
        if embedder is not None:
            try:
                node.embedding = embedder.embed(result.factor_name, result.formula)
            except Exception:
                pass
        kg.add_factor(node)

        if result.signals is not None:
            for factor in library.list_factors():
                if factor.name == result.factor_name or factor.signals is None:
                    continue
                try:
                    corr = library.compute_correlation(result.signals, factor.signals)
                    kg.add_correlation_edge(
                        result.factor_name,
                        factor.name,
                        rho=corr,
                        threshold=0.4,
                    )
                except Exception:
                    continue

        self._detect_derivation(kg=kg, result=result, library=library)

    def remove_factor(self, *, kg: Any, embedder: Any, factor_id: str) -> None:
        if kg is not None:
            try:
                kg.remove_factor(factor_id)
            except Exception:
                pass
        if embedder is not None:
            try:
                embedder.remove(factor_id)
            except Exception:
                pass

    def _detect_derivation(self, *, kg: Any, result: Any, library: Any) -> None:
        if kg is None:
            return
        new_ops = set(extract_operators(result.formula))
        if not new_ops:
            return
        for factor in library.list_factors():
            if factor.name == result.factor_name:
                continue
            existing_ops = set(extract_operators(factor.formula))
            if not existing_ops:
                continue
            shared = new_ops & existing_ops
            if not shared:
                continue
            overlap = len(shared) / max(len(new_ops), len(existing_ops))
            if 0.5 <= overlap < 1.0:
                diff_ops = (new_ops - existing_ops) | (existing_ops - new_ops)
                try:
                    kg.add_derivation_edge(
                        child=result.factor_name,
                        parent=factor.name,
                        mutation_type=f"operator_change:{','.join(sorted(diff_ops))}",
                    )
                except Exception:
                    pass
