"""Immutable evidence-pack and content-addressed storage coverage."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
from click.testing import CliRunner

from factorminer.application.evidence_service import EvidenceStore
from factorminer.cli import main
from factorminer.domain.evidence import EvidencePack


def _pack() -> EvidencePack:
    return EvidencePack.create(
        content_hash="content",
        dataset_hash="dataset",
        config_hash="config",
        code_hash="code",
        source_ids=["paper:1"],
        hypothesis_ids=["hypothesis:1"],
        factor_name="alpha",
        formula="Neg($close)",
        formula_ast={"type": "operator", "name": "Neg"},
        lineage={"edit_type": "fresh"},
        split_metrics={"test": {"ic": 0.1}},
        admission_decision={"admitted": True},
        replacement_decision={"performed": False},
        created_at="2026-07-21T00:00:00+00:00",
    )


def test_evidence_pack_is_immutable_and_deterministic():
    left = _pack()
    right = _pack()

    assert left.evidence_id == right.evidence_id
    assert left.verify()
    with pytest.raises(FrozenInstanceError):
        left.approval_state = "approved"  # type: ignore[misc]


def test_evidence_pack_detects_tampering():
    payload = _pack().to_dict()
    payload["split_metrics"]["test"]["ic"] = 0.9

    with pytest.raises(ValueError, match="content hash mismatch"):
        EvidencePack.from_dict(payload)


def test_evidence_store_roundtrip_and_verification(tmp_path):
    store = EvidenceStore(tmp_path)
    pack = _pack()
    path = store.put(pack)

    assert path.name == f"{pack.evidence_id}.json"
    assert store.get(pack.evidence_id) == pack
    assert store.verify_all() == {"ok": True, "checked": 1, "failures": []}

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["formula"] = "$open"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = store.verify_all()
    assert result["ok"] is False
    assert result["checked"] == 1


def test_verify_evidence_cli_reports_tampering(tmp_path):
    store = EvidenceStore(tmp_path)
    path = store.put(_pack())
    runner = CliRunner()

    valid = runner.invoke(main, ["--output-dir", str(tmp_path), "verify-evidence"])
    assert valid.exit_code == 0
    assert '"ok": true' in valid.output

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["formula"] = "$open"
    path.write_text(json.dumps(payload), encoding="utf-8")
    invalid = runner.invoke(main, ["--output-dir", str(tmp_path), "verify-evidence"])
    assert invalid.exit_code != 0
    assert "Evidence verification failed" in invalid.output
