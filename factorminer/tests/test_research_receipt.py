"""Adversarial coverage for published research receipts."""

from __future__ import annotations

import json
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from factorminer.architecture.research_receipt import (
    DatasetCommitment,
    EvidenceTier,
    ExternalResearchReceipt,
    RunStatus,
    attest_research_receipt,
    derive_release_id,
)
from factorminer.benchmark.receipt import (
    build_research_receipt,
    generate_commitment_key,
    publish_portable_bundle,
    verify_research_receipt,
    write_receipt,
)


def _base_kwargs(**overrides):
    kwargs = dict(
        release_id="",
        evidence_tier=EvidenceTier.SIMULATED,
        run_status=RunStatus.COMPLETED,
        generated_at="2026-07-21T00:00:00",
        code_sha="deadbeef",
        config_sha256="a" * 64,
        environment_lock_sha256="b" * 64,
        seed=42,
        dataset_descriptor={"kind": "synthetic", "identity": "fixture"},
        dataset_commitment=DatasetCommitment(scheme="none", digest="mock"),
        data_license_class="synthetic",
        factor_library_sha256="c" * 64,
    )
    kwargs.update(overrides)
    return kwargs


def _build_fixture(
    tmp_path,
    *,
    evidence_tier=EvidenceTier.SIMULATED,
    data_license_class="synthetic",
    data_path=None,
    commitment_key=None,
    missing_artifact=False,
    runtime_manifest=False,
):
    artifact_path = tmp_path / "report.html"
    if not missing_artifact:
        artifact_path.write_text("<html>report</html>")
    manifest = {
        "run_parameters": {"seed": 1},
        "artifact_paths": {"html_report": str(artifact_path)},
        "runtime_manifest_refs": [],
    }
    nested_path = None
    if runtime_manifest:
        nested_path = tmp_path / "runtime_manifest.json"
        nested_path.write_text(json.dumps({"dataset_hashes": {"test": "original"}}))
        from factorminer.benchmark.reporting import file_sha256

        manifest["runtime_manifest_refs"] = [
            {
                "path": str(nested_path),
                "sha256": file_sha256(nested_path),
                "baseline": "fixture",
                "artifact_paths": {},
            }
        ]
    manifest_path = tmp_path / "phase2_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    receipt = build_research_receipt(
        phase2_manifest=manifest,
        phase2_manifest_path=manifest_path,
        evidence_tier=evidence_tier,
        run_status=RunStatus.COMPLETED,
        seed=1,
        config_path=manifest_path,
        data_path=data_path,
        dataset_descriptor={"kind": "fixture", "identity": "test-data"},
        protocol_admission_contract={},
        protocol_replacement_contract={},
        memory_policy_schema=None,
        ic_metric="paper_abs_mean_spearman_ic",
        metric_version="paper_ic_v2",
        data_license_class=data_license_class,
        commitment_key=commitment_key,
    )
    return receipt, manifest_path, artifact_path, nested_path


def test_derive_release_id_changes_with_cost_assumptions() -> None:
    receipt_a = ExternalResearchReceipt(
        **_base_kwargs(cost_and_capacity_assumptions={"cost_bps": [1.0]})
    )
    receipt_b = ExternalResearchReceipt(
        **_base_kwargs(cost_and_capacity_assumptions={"cost_bps": [5.0]})
    )
    assert derive_release_id(receipt_a) != derive_release_id(receipt_b)

    receipt_c = ExternalResearchReceipt(**_base_kwargs(generated_at="2026-07-21T00:00:00"))
    receipt_d = ExternalResearchReceipt(**_base_kwargs(generated_at="2099-01-01T00:00:00"))
    assert derive_release_id(receipt_c) != derive_release_id(receipt_d)


def test_construction_rejects_mock_labeled_as_public_reproducible() -> None:
    with pytest.raises(ValueError, match="requires a real dataset_commitment"):
        ExternalResearchReceipt(
            **_base_kwargs(
                evidence_tier=EvidenceTier.PUBLIC_REPRODUCIBLE,
                dataset_commitment=DatasetCommitment(scheme="none", digest="mock"),
            )
        )


def test_construction_enforces_tier_license_and_commitment_scheme() -> None:
    with pytest.raises(ValueError, match="publicly obtainable"):
        ExternalResearchReceipt(
            **_base_kwargs(
                evidence_tier=EvidenceTier.PUBLIC_REPRODUCIBLE,
                dataset_commitment=DatasetCommitment(scheme="sha256", digest="d" * 64),
                data_license_class="proprietary_licensed",
            )
        )

    public = ExternalResearchReceipt(
        **_base_kwargs(
            evidence_tier=EvidenceTier.PUBLIC_REPRODUCIBLE,
            dataset_commitment=DatasetCommitment(scheme="sha256", digest="d" * 64),
            data_license_class="publicly_retrievable",
        )
    )
    assert public.data_license_class == "publicly_retrievable"
    with pytest.raises(ValueError, match="hmac-sha256"):
        ExternalResearchReceipt(
            **_base_kwargs(
                evidence_tier=EvidenceTier.PRIVATE_PARTNER_OBSERVED,
                dataset_commitment=DatasetCommitment(scheme="sha256", digest="d" * 64),
                data_license_class="proprietary_licensed",
            )
        )


def test_construction_rejects_invalid_data_license_class() -> None:
    with pytest.raises(ValueError, match="data_license_class must be one of"):
        ExternalResearchReceipt(**_base_kwargs(data_license_class="not_a_real_license"))


def test_attest_research_receipt_is_explicit_and_immutable() -> None:
    receipt = ExternalResearchReceipt(**_base_kwargs())
    attested = attest_research_receipt(receipt, attestor="alice")
    assert attested.reviewer_state.attested is True
    assert attested.reviewer_state.source == "human"
    assert receipt.reviewer_state.attested is False
    with pytest.raises(FrozenInstanceError):
        receipt.seed = 9  # type: ignore[misc]


def test_write_receipt_is_idempotent_and_refuses_same_id_divergence(tmp_path) -> None:
    receipt, _, _, _ = _build_fixture(tmp_path)
    releases_root = tmp_path / "releases"
    path = write_receipt(receipt, releases_root)
    assert write_receipt(receipt, releases_root) == path

    divergent = replace(receipt, cost_and_capacity_assumptions={"cost_bps": [99]})
    with pytest.raises(ValueError, match="release_id is stale"):
        write_receipt(divergent, releases_root)


def test_verify_receipt_recomputes_release_id_from_content(tmp_path) -> None:
    receipt, _, _, _ = _build_fixture(tmp_path)
    receipt_path = write_receipt(receipt, tmp_path / "releases")
    payload = json.loads(receipt_path.read_text())
    payload["code_sha"] = "tampered"
    receipt_path.write_text(json.dumps(payload))

    result = verify_research_receipt(receipt_path.parent)
    assert result.passed is False
    assert any("content mismatch" in mismatch for mismatch in result.mismatches)


def test_verify_receipt_rejects_invalid_json_without_crashing(tmp_path) -> None:
    release_dir = tmp_path / ("a" * 64)
    release_dir.mkdir()
    (release_dir / "receipt.json").write_text("{not-json")

    result = verify_research_receipt(release_dir)
    assert result.passed is False
    assert "cannot be read as JSON" in result.mismatches[0]


def test_builder_refuses_missing_artifact(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="phase2/html_report"):
        _build_fixture(tmp_path, missing_artifact=True)


def test_verify_receipt_detects_post_hoc_artifact_tampering(tmp_path) -> None:
    receipt, _, artifact_path, _ = _build_fixture(tmp_path)
    releases_root = tmp_path / "releases"
    write_receipt(receipt, releases_root)
    artifact_path.write_text("<html>TAMPERED</html>")

    result = verify_research_receipt(releases_root / receipt.release_id)
    assert result.passed is False
    assert any("phase2/html_report" in mismatch for mismatch in result.mismatches)


def test_verify_receipt_detects_nested_manifest_tampering(tmp_path) -> None:
    receipt, _, _, nested_path = _build_fixture(tmp_path, runtime_manifest=True)
    releases_root = tmp_path / "releases"
    write_receipt(receipt, releases_root)
    assert nested_path is not None
    nested_path.write_text(json.dumps({"dataset_hashes": {"test": "tampered"}}))

    result = verify_research_receipt(releases_root / receipt.release_id)
    assert result.passed is False
    assert any("runtime manifest" in mismatch for mismatch in result.mismatches)


def test_public_commitment_requires_and_verifies_input(tmp_path) -> None:
    data_path = tmp_path / "public.csv"
    data_path.write_text("date,value\n2026-01-01,1\n")
    receipt, _, _, _ = _build_fixture(
        tmp_path,
        evidence_tier=EvidenceTier.PUBLIC_REPRODUCIBLE,
        data_license_class="public_domain",
        data_path=data_path,
    )
    releases_root = tmp_path / "releases"
    write_receipt(receipt, releases_root)
    release_dir = releases_root / receipt.release_id

    assert verify_research_receipt(release_dir).passed is False
    assert verify_research_receipt(release_dir, commitment_input=data_path).passed is True


def test_private_commitment_requires_witness_without_exposing_key(tmp_path) -> None:
    data_path = tmp_path / "private.csv"
    data_path.write_text("date,value\n2026-01-01,1\n")
    key = generate_commitment_key()
    receipt, _, _, _ = _build_fixture(
        tmp_path,
        evidence_tier=EvidenceTier.PRIVATE_PARTNER_OBSERVED,
        data_license_class="proprietary_licensed",
        data_path=data_path,
        commitment_key=key,
    )
    dumped = json.dumps(receipt.to_dict())
    assert key not in dumped
    assert receipt.dataset_commitment.nonce is None

    releases_root = tmp_path / "releases"
    write_receipt(receipt, releases_root)
    release_dir = releases_root / receipt.release_id
    assert verify_research_receipt(release_dir).passed is False
    assert (
        verify_research_receipt(
            release_dir,
            commitment_key=key,
            commitment_input=data_path,
        ).passed
        is True
    )
    assert (
        verify_research_receipt(
            release_dir,
            commitment_key="00" * 32,
            commitment_input=data_path,
        ).passed
        is False
    )


def test_portable_bundle_verifies_after_relocation_and_source_removal(tmp_path) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    receipt, _, _, _ = _build_fixture(producer, runtime_manifest=True)
    receipt_path = publish_portable_bundle(
        receipt,
        phase2_manifest=json.loads((producer / "phase2_manifest.json").read_text()),
        releases_root=tmp_path / "releases",
    )
    payload = json.loads(receipt_path.read_text())
    manifest = json.loads((receipt_path.parent / "manifest.json").read_text())
    assert payload["source_manifest"]["path"] == "manifest.json"
    assert all(not Path(path).is_absolute() for path in manifest["artifact_paths"].values())
    assert all(
        not Path(path).is_absolute()
        for ref in manifest["runtime_manifest_refs"]
        for path in ref["artifact_paths"].values()
    )
    serialized_json = "\n".join(
        path.read_text() for path in receipt_path.parent.rglob("*.json")
    )
    assert str(producer) not in serialized_json

    relocated = tmp_path / "relocated" / receipt_path.parent.name
    relocated.parent.mkdir()
    shutil.move(receipt_path.parent, relocated)
    shutil.rmtree(producer)

    result = verify_research_receipt(relocated)
    assert result.passed is True, result.mismatches


def test_portable_public_bundle_can_verify_bundled_dataset(tmp_path) -> None:
    producer = tmp_path / "producer"
    producer.mkdir()
    data_path = producer / "public.csv"
    data_path.write_text("date,value\n2026-01-01,1\n")
    receipt, manifest_path, _, _ = _build_fixture(
        producer,
        evidence_tier=EvidenceTier.PUBLIC_REPRODUCIBLE,
        data_license_class="public_domain",
        data_path=data_path,
    )
    receipt_path = publish_portable_bundle(
        receipt,
        phase2_manifest=json.loads(manifest_path.read_text()),
        releases_root=tmp_path / "releases",
        commitment_input=data_path,
        include_commitment_input=True,
    )
    shutil.rmtree(producer)

    result = verify_research_receipt(receipt_path.parent)
    assert result.passed is True, result.mismatches
    payload = json.loads(receipt_path.read_text())
    bundled = receipt_path.parent / payload["dataset_descriptor"]["bundle_path"]
    assert bundled.read_text() == "date,value\n2026-01-01,1\n"


def test_portable_public_bundle_refuses_unlicensed_redistribution(tmp_path) -> None:
    data_path = tmp_path / "publicly-retrievable.csv"
    data_path.write_text("date,value\n2026-01-01,1\n")
    receipt, manifest_path, _, _ = _build_fixture(
        tmp_path,
        evidence_tier=EvidenceTier.PUBLIC_REPRODUCIBLE,
        data_license_class="publicly_retrievable",
        data_path=data_path,
    )
    with pytest.raises(ValueError, match="does not permit public bundling"):
        publish_portable_bundle(
            receipt,
            phase2_manifest=json.loads(manifest_path.read_text()),
            releases_root=tmp_path / "releases",
            commitment_input=data_path,
            include_commitment_input=True,
        )


def test_portable_bundle_refuses_private_dataset_copy(tmp_path) -> None:
    data_path = tmp_path / "private.csv"
    data_path.write_text("date,value\n2026-01-01,1\n")
    receipt, manifest_path, _, _ = _build_fixture(
        tmp_path,
        evidence_tier=EvidenceTier.PRIVATE_PARTNER_OBSERVED,
        data_license_class="proprietary_licensed",
        data_path=data_path,
        commitment_key=generate_commitment_key(),
    )
    with pytest.raises(ValueError, match="only public_reproducible"):
        publish_portable_bundle(
            receipt,
            phase2_manifest=json.loads(manifest_path.read_text()),
            releases_root=tmp_path / "releases",
            commitment_input=data_path,
            include_commitment_input=True,
        )


def test_portable_bundle_detects_bundled_artifact_tampering(tmp_path) -> None:
    receipt, manifest_path, _, _ = _build_fixture(tmp_path)
    receipt_path = publish_portable_bundle(
        receipt,
        phase2_manifest=json.loads(manifest_path.read_text()),
        releases_root=tmp_path / "releases",
    )
    manifest = json.loads((receipt_path.parent / "manifest.json").read_text())
    artifact = receipt_path.parent / manifest["artifact_paths"]["html_report"]
    artifact.write_text("tampered")

    result = verify_research_receipt(receipt_path.parent)
    assert result.passed is False
    assert any("phase2/html_report" in mismatch for mismatch in result.mismatches)


def test_portable_bundle_verifier_rejects_manifest_path_escape(tmp_path) -> None:
    receipt, manifest_path, _, _ = _build_fixture(tmp_path)
    receipt_path = publish_portable_bundle(
        receipt,
        phase2_manifest=json.loads(manifest_path.read_text()),
        releases_root=tmp_path / "releases",
    )
    bundled_manifest_path = receipt_path.parent / "manifest.json"
    bundled_manifest = json.loads(bundled_manifest_path.read_text())
    bundled_manifest["artifact_paths"]["html_report"] = "../../outside.html"
    bundled_manifest_path.write_text(json.dumps(bundled_manifest))

    result = verify_research_receipt(receipt_path.parent)
    assert result.passed is False
    assert any("invalid artifact path" in mismatch for mismatch in result.mismatches)


def test_portable_bundle_verifier_rejects_absolute_artifact_path(tmp_path) -> None:
    receipt, manifest_path, _, _ = _build_fixture(tmp_path)
    receipt_path = publish_portable_bundle(
        receipt,
        phase2_manifest=json.loads(manifest_path.read_text()),
        releases_root=tmp_path / "releases",
    )
    bundled_manifest_path = receipt_path.parent / "manifest.json"
    bundled_manifest = json.loads(bundled_manifest_path.read_text())
    bundled_manifest["artifact_paths"]["html_report"] = str(tmp_path / "outside.html")
    bundled_manifest_path.write_text(json.dumps(bundled_manifest))

    result = verify_research_receipt(receipt_path.parent)
    assert result.passed is False
    assert any("portable bundle paths must be relative" in item for item in result.mismatches)
