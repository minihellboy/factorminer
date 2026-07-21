"""Application-layer contracts for constructing and running mining workflows."""

from __future__ import annotations

from factorminer._lazy_exports import public_dir, resolve_export

_EXPORTS = {
    "BudgetTracker": "factorminer.application.mining_budget",
    "EvidenceStore": "factorminer.application.evidence_service",
    "EvaluationResult": "factorminer.application.mining_budget",
    "FactorEvidenceService": "factorminer.application.evidence_service",
    "HelixGenerationService": "factorminer.application.helix_generation",
    "HelixValidationService": "factorminer.application.helix_validation",
    "MiningReporter": "factorminer.application.mining_reporting",
    "MiningArtifactService": "factorminer.application.run_artifacts",
    **{
        name: "factorminer.application.research_knowledge"
        for name in (
            "ResearchHypothesisRecord",
            "ResearchKnowledgeStore",
            "ResearchOutcomeRecord",
            "ResearchRetrieval",
            "ResearchSourceRecord",
        )
    },
    "ValidationPipeline": "factorminer.application.validation_pipeline",
    **{
        name: "factorminer.application.runtime_context"
        for name in ("MiningRunContext", "MiningSettings", "build_run_context")
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), name, _EXPORTS)


def __dir__() -> list[str]:
    return public_dir(globals(), _EXPORTS)
