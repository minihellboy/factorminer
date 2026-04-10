"""Canonical protocol and runtime architecture contracts for FactorMiner."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DatasetContract",
    "DependenceMetric",
    "DistillStage",
    "DistanceCorrelationMetric",
    "EvaluateStage",
    "EvaluationKernel",
    "FamilyAwareMemoryPolicy",
    "FactorFamily",
    "FactorFamilyDiscovery",
    "FactorLifecycleEvent",
    "FactorLifecycleStore",
    "GenerateStage",
    "IterationPayload",
    "LibraryGeometry",
    "FactorAdmissionService",
    "LibraryUpdateStage",
    "LoopStage",
    "KGMemoryPolicy",
    "MemoryPolicy",
    "NoMemoryPolicy",
    "PaperMemoryPolicy",
    "PaperProtocol",
    "PearsonDependenceMetric",
    "PromptContextBuilder",
    "KnowledgeGraphService",
    "OnlineForgettingService",
    "RegimeAwareMemoryPolicy",
    "RetrieveStage",
    "SpearmanDependenceMetric",
    "build_dependence_metric",
    "build_memory_policy",
]

_ATTRIBUTE_MAP = {
    "DatasetContract": ("factorminer.architecture.dataset_contract", "DatasetContract"),
    "DependenceMetric": ("factorminer.architecture.dependence", "DependenceMetric"),
    "DistillStage": ("factorminer.architecture.stages", "DistillStage"),
    "DistanceCorrelationMetric": (
        "factorminer.architecture.dependence",
        "DistanceCorrelationMetric",
    ),
    "EvaluateStage": ("factorminer.architecture.stages", "EvaluateStage"),
    "EvaluationKernel": ("factorminer.architecture.evaluation_kernel", "EvaluationKernel"),
    "FamilyAwareMemoryPolicy": ("factorminer.architecture.memory_policy", "FamilyAwareMemoryPolicy"),
    "FactorFamily": ("factorminer.architecture.families", "FactorFamily"),
    "FactorFamilyDiscovery": ("factorminer.architecture.families", "FactorFamilyDiscovery"),
    "FactorLifecycleEvent": ("factorminer.architecture.lifecycle", "FactorLifecycleEvent"),
    "FactorLifecycleStore": ("factorminer.architecture.lifecycle", "FactorLifecycleStore"),
    "GenerateStage": ("factorminer.architecture.stages", "GenerateStage"),
    "IterationPayload": ("factorminer.architecture.stages", "IterationPayload"),
    "LibraryGeometry": ("factorminer.architecture.geometry", "LibraryGeometry"),
    "FactorAdmissionService": (
        "factorminer.architecture.library_services",
        "FactorAdmissionService",
    ),
    "LibraryUpdateStage": ("factorminer.architecture.stages", "LibraryUpdateStage"),
    "LoopStage": ("factorminer.architecture.stages", "LoopStage"),
    "KGMemoryPolicy": ("factorminer.architecture.memory_policy", "KGMemoryPolicy"),
    "MemoryPolicy": ("factorminer.architecture.memory_policy", "MemoryPolicy"),
    "NoMemoryPolicy": ("factorminer.architecture.memory_policy", "NoMemoryPolicy"),
    "PaperMemoryPolicy": ("factorminer.architecture.memory_policy", "PaperMemoryPolicy"),
    "PaperProtocol": ("factorminer.architecture.paper_protocol", "PaperProtocol"),
    "PearsonDependenceMetric": ("factorminer.architecture.dependence", "PearsonDependenceMetric"),
    "PromptContextBuilder": ("factorminer.architecture.prompt_context", "PromptContextBuilder"),
    "KnowledgeGraphService": ("factorminer.architecture.phase2_services", "KnowledgeGraphService"),
    "OnlineForgettingService": ("factorminer.architecture.phase2_services", "OnlineForgettingService"),
    "RegimeAwareMemoryPolicy": ("factorminer.architecture.memory_policy", "RegimeAwareMemoryPolicy"),
    "RetrieveStage": ("factorminer.architecture.stages", "RetrieveStage"),
    "SpearmanDependenceMetric": ("factorminer.architecture.dependence", "SpearmanDependenceMetric"),
    "build_dependence_metric": ("factorminer.architecture.dependence", "build_dependence_metric"),
    "build_memory_policy": ("factorminer.architecture.memory_policy", "build_memory_policy"),
}


def __getattr__(name: str):
    if name not in _ATTRIBUTE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _ATTRIBUTE_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
