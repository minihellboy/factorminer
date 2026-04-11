"""Tests for mining run manifests and factor provenance."""

from __future__ import annotations

import json

import numpy as np

from factorminer.agent.llm_interface import MockProvider
from factorminer.core.config import MiningConfig
from factorminer.core.factor_library import Factor
from factorminer.core.helix_loop import HelixLoop
from factorminer.core.library_io import load_library
from factorminer.core.ralph_loop import EvaluationResult
from factorminer.core.session import MiningSession


def test_factor_provenance_roundtrip():
    factor = Factor(
        id=7,
        name="alpha_7",
        formula="Neg($close)",
        category="test",
        ic_mean=0.12,
        icir=1.4,
        ic_win_rate=0.6,
        max_correlation=0.1,
        batch_number=3,
        provenance={
            "run_id": "run_123",
            "loop_type": "helix",
            "memory_summary": {"insight_count": 2},
        },
    )

    restored = Factor.from_dict(factor.to_dict())

    assert restored.provenance["run_id"] == "run_123"
    assert restored.provenance["loop_type"] == "helix"
    assert restored.provenance["memory_summary"]["insight_count"] == 2


def test_helix_run_writes_manifest_and_factor_provenance(tmp_path, small_data, monkeypatch):
    data = np.stack(
        [
            small_data["$open"],
            small_data["$high"],
            small_data["$low"],
            small_data["$close"],
            small_data["$volume"],
            small_data["$amt"],
            small_data["$vwap"],
        ],
        axis=2,
    )
    returns = small_data["$returns"]
    config = MiningConfig(
        target_library_size=1,
        max_iterations=1,
        batch_size=1,
        output_dir=str(tmp_path / "helix-output"),
    )
    provider = MockProvider()

    loop = HelixLoop(
        config=config,
        data_tensor=data,
        returns=returns,
        llm_provider=provider,
        canonicalize=False,
        enable_knowledge_graph=False,
        enable_embeddings=False,
        enable_auto_inventor=False,
    )

    monkeypatch.setattr(
        loop.generator,
        "generate_batch",
        lambda *args, **kwargs: [("alpha_1", "Neg($close)")],
    )
    monkeypatch.setattr(
        loop.pipeline,
        "evaluate_batch",
        lambda candidates: [
            EvaluationResult(
                factor_name="alpha_1",
                formula="Neg($close)",
                parse_ok=True,
                ic_mean=0.12,
                icir=1.3,
                ic_win_rate=0.6,
                max_correlation=0.0,
                admitted=True,
                stage_passed=3,
                signals=np.ones_like(returns),
                score_vector={"primary_score": 0.12},
            )
        ],
    )

    library = loop.run(target_size=1, max_iterations=1)

    output_dir = tmp_path / "helix-output"
    run_manifest_path = output_dir / "run_manifest.json"
    checkpoint_manifest_path = output_dir / "checkpoint" / "run_manifest.json"
    session_path = output_dir / "session.json"
    library_path = output_dir / "factor_library.json"
    checkpoint_library_path = output_dir / "checkpoint" / "library.json"

    assert run_manifest_path.exists()
    assert checkpoint_manifest_path.exists()
    assert session_path.exists()
    assert library_path.exists()
    assert checkpoint_library_path.exists()

    manifest = json.loads(run_manifest_path.read_text())
    assert manifest["loop_type"] == "helix"
    assert manifest["library_size"] >= 1
    assert manifest["artifact_paths"]["run_manifest"] == str(run_manifest_path)

    session = MiningSession.load(session_path)
    assert session.run_manifest_path == str(run_manifest_path)
    assert session.run_manifest["loop_type"] == "helix"

    loaded_library = load_library(output_dir / "factor_library")
    factor = loaded_library.list_factors()[0]
    assert factor.provenance["run_id"] == manifest["run_id"]
    assert factor.provenance["loop_type"] == "helix"
    assert factor.provenance["admission"]["admitted"] is True
    assert factor.provenance["evaluation"]["ic_mean"] == 0.12
    assert library.size == 1
