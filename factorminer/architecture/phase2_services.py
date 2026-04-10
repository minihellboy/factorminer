"""Reusable Phase 2 services extracted from Helix loop logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from factorminer.architecture.families import extract_features, extract_operators, infer_family


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
