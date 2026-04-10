"""Tests for the canonical protocol/contracts architecture."""

from __future__ import annotations

import numpy as np

from factorminer.architecture import (
    DatasetContract,
    FamilyAwareMemoryPolicy,
    FactorFamilyDiscovery,
    KGMemoryPolicy,
    LibraryGeometry,
    NoMemoryPolicy,
    PaperMemoryPolicy,
    PaperProtocol,
    PromptContextBuilder,
    RegimeAwareMemoryPolicy,
    build_dependence_metric,
)
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import load_library, save_library
from factorminer.memory.memory_store import ExperienceMemory
from factorminer.operators.c_backend import backend_available
from factorminer.operators.registry import execute_operator
from factorminer.utils.config import load_config


def test_paper_protocol_builds_from_config():
    cfg = load_config()
    protocol = PaperProtocol.from_config(cfg)

    assert protocol.default_target == cfg.data.default_target
    assert protocol.ic_threshold == cfg.mining.ic_threshold
    assert protocol.train_test_protocol.freeze_top_k == cfg.benchmark.freeze_top_k
    assert "paper" in protocol.target_definitions
    assert protocol.redundancy_metric == cfg.evaluation.redundancy_metric


def test_dataset_contract_tracks_shapes_and_targets():
    cfg = load_config()
    data_tensor = np.zeros((3, 5, 8), dtype=np.float64)
    returns = np.zeros((3, 5), dtype=np.float64)

    contract = DatasetContract.from_arrays(
        cfg,
        data_tensor=data_tensor,
        returns=returns,
        target_panels={"paper": returns},
        target_horizons={"paper": 1},
    )

    assert contract.data_shape == (3, 5, 8)
    assert contract.returns_shape == (3, 5)
    assert contract.default_target == cfg.data.default_target
    assert contract.target_horizons["paper"] == 1


def test_memory_policy_exposes_schema_and_serialization():
    cfg = load_config()
    protocol = PaperProtocol.from_config(cfg)
    policy = PaperMemoryPolicy(protocol)

    schema = policy.schema()
    assert "state_schema" in schema
    assert "retrieval_ranking" in schema

    payload = policy.serialize(ExperienceMemory())
    assert payload["version"] == 0
    assert "memory_policy" in payload


def test_no_memory_policy_disables_retrieval():
    cfg = load_config()
    protocol = PaperProtocol.from_config(cfg)
    policy = NoMemoryPolicy(protocol)

    signal = policy.retrieve(ExperienceMemory(), library_state={"library_size": 3})
    assert signal["memory_disabled"] is True
    assert signal["recommended_directions"] == []
    assert signal["prompt_text"] == ""


def test_regime_aware_memory_policy_adds_regime_context():
    cfg = load_config()
    protocol = PaperProtocol.from_config(cfg)
    returns = np.array(
        [
            np.linspace(-0.02, 0.03, 12),
            np.linspace(-0.01, 0.02, 12),
        ],
        dtype=np.float64,
    )
    memory = ExperienceMemory()
    memory.success_patterns = []
    policy = RegimeAwareMemoryPolicy(protocol, returns, lookback_window=5)

    signal = policy.retrieve(memory, library_state={"library_size": 0})
    assert "regime_context" in signal
    assert "active_regime" in signal["regime_context"]


def test_kg_memory_policy_serializes_knowledge_graph():
    cfg = load_config()
    protocol = PaperProtocol.from_config(cfg)
    policy = KGMemoryPolicy(protocol)
    memory = ExperienceMemory()

    formed = policy.form(
        memory,
        [
            {
                "factor_id": "factor_1",
                "formula": "Mean($close, 5)",
                "ic": 0.08,
                "admitted": True,
            }
        ],
        iteration=1,
    )
    payload = policy.serialize(formed)

    assert "knowledge_graph" in payload
    restored = policy.restore(payload)
    assert restored.version == formed.version
    assert policy.knowledge_graph.get_factor_count() == 1


def test_family_aware_memory_policy_includes_family_context():
    cfg = load_config()
    protocol = PaperProtocol.from_config(cfg)
    policy = FamilyAwareMemoryPolicy(protocol)
    memory = ExperienceMemory()
    memory.success_patterns = []

    signal = policy.retrieve(
        memory,
        library_state={
            "library_size": 2,
            "recent_admissions": [
                {
                    "name": "f1",
                    "formula": "Mean($close, 5)",
                    "category": "Smoothing",
                    "ic_mean": 0.06,
                }
            ],
            "categories": {"Smoothing": 2},
        },
    )

    assert "family_context" in signal
    assert "prompt_text" in signal["family_context"]


def test_prompt_context_builder_includes_family_prompt_text():
    cfg = load_config()
    protocol = PaperProtocol.from_config(cfg)
    builder = PromptContextBuilder(
        protocol=protocol,
        family_discovery=FactorFamilyDiscovery(),
    )

    payload = builder.build(
        memory_signal={"recommended_directions": []},
        library_state={"library_size": 0, "recent_admissions": [], "categories": {}},
        batch_size=8,
    )

    assert payload["generation_batch_size"] == 8
    assert "protocol_contract" in payload
    assert "family_context" in payload
    assert "family_prompt_text" in payload


def test_library_geometry_detects_single_replacement_target():
    library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)
    signals = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.5, 1.7, 1.9, 2.1],
        ],
        dtype=np.float64,
    )
    library.admit_factor(
        Factor(
            id=0,
            name="f1",
            formula="Neg($close)",
            category="test",
            ic_mean=0.06,
            icir=0.8,
            ic_win_rate=0.6,
            max_correlation=0.0,
            batch_number=1,
            signals=signals,
        )
    )

    geometry = LibraryGeometry(library)
    candidate = signals.copy()
    candidate[:, -1] += 0.01

    snapshot = geometry.candidate_geometry(candidate)
    assert snapshot.max_correlation >= 0.5
    assert snapshot.dependence_metric == "spearman"
    assert geometry.replacement_target(candidate) == 1


def test_dependence_metric_builder_supports_pearson():
    metric = build_dependence_metric("pearson")
    x = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0], [3.0, 9.0, 27.0]])
    value = metric.compute(x, x * 2.0)
    assert metric.name == "pearson"
    assert value > 0.99


def test_library_io_preserves_dependence_metric(tmp_path):
    library = FactorLibrary(
        correlation_threshold=0.5,
        ic_threshold=0.04,
        dependence_metric="distance_correlation",
    )
    path = tmp_path / "library"
    save_library(library, path, save_signals=False)
    restored = load_library(path)
    assert restored.dependence_metric.name == "distance_correlation"


def test_evaluation_kernel_c_backend_executes_when_available():
    x = np.arange(20, dtype=np.float64).reshape(2, 10)
    result = execute_operator("Mean", x, params={"window": 3}, backend="c")
    assert result.shape == x.shape
    if backend_available():
        assert np.isfinite(result[:, 2:]).all()
