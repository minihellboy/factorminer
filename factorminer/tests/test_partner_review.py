"""External partner acknowledgments must be explicit, bounded, and tamper-evident."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from factorminer.architecture.research_receipt import (
    DatasetCommitment,
    EvidenceTier,
    ExternalResearchReceipt,
    RunStatus,
    derive_release_id,
)
from factorminer.benchmark.partner_review import (
    acknowledge_partner_review,
    prepare_partner_review,
    verify_partner_acknowledgment,
)
from factorminer.benchmark.receipt import write_receipt


def _release(tmp_path, *, tier=EvidenceTier.PRIVATE_PARTNER_OBSERVED):
    commitment = (
        DatasetCommitment(scheme="hmac-sha256", digest="d" * 64)
        if tier == EvidenceTier.PRIVATE_PARTNER_OBSERVED
        else DatasetCommitment(scheme="sha256", digest="d" * 64)
    )
    license_class = (
        "proprietary_licensed" if tier == EvidenceTier.PRIVATE_PARTNER_OBSERVED else "public_domain"
    )
    draft = ExternalResearchReceipt(
        release_id="",
        evidence_tier=tier,
        run_status=RunStatus.COMPLETED,
        generated_at="2026-07-21T00:00:00Z",
        code_sha="deadbeef",
        config_sha256="a" * 64,
        environment_lock_sha256="b" * 64,
        seed=7,
        dataset_descriptor={"identity": "partner-fixture"},
        dataset_commitment=commitment,
        data_license_class=license_class,
        factor_library_sha256="c" * 64,
    )
    receipt = replace(draft, release_id=derive_release_id(draft))
    path = write_receipt(receipt, tmp_path / "releases")
    return path.parent


def test_partner_acknowledgment_binds_receipt_and_assertion_subset(tmp_path) -> None:
    release_dir = _release(tmp_path)
    now = datetime(2026, 7, 21, tzinfo=UTC)
    request = prepare_partner_review(
        release_dir,
        partner_pseudonym="partner-01",
        requested_assertions=(
            "artifacts_reviewed",
            "limitations_acknowledged",
            "protocol_observed",
        ),
        now=now,
    )
    key = "11" * 32
    acknowledgment = acknowledge_partner_review(
        request,
        reviewer_pseudonym="reviewer-01",
        assertions=("limitations_acknowledged", "protocol_observed"),
        publication_consent="anonymous",
        key_hex=key,
        structured_feedback={
            "setup_minutes": 18,
            "campaign_completed": True,
            "completion_rate": 1.0,
            "trust_rating": 4,
            "decision_outcome": "pilot_more",
            "adoption_blockers": ["data_integration"],
            "repeated_use_intent": "yes",
            "useful_outputs": ["evidence_report", "receipt"],
            "failure_modes": ["none"],
            "requested_integrations": ["market_data"],
        },
        now=now + timedelta(hours=2),
    )

    passed, mismatches = verify_partner_acknowledgment(
        request, acknowledgment, release_dir=release_dir, key_hex=key
    )
    assert passed is True, mismatches
    dumped = json.dumps(acknowledgment.to_dict())
    assert key not in dumped
    assert acknowledgment.assertions == (
        "limitations_acknowledged",
        "protocol_observed",
    )


def test_partner_acknowledgment_rejects_wrong_key_and_tampering(tmp_path) -> None:
    release_dir = _release(tmp_path)
    now = datetime(2026, 7, 21, tzinfo=UTC)
    request = prepare_partner_review(
        release_dir,
        partner_pseudonym="partner-01",
        requested_assertions=("protocol_observed",),
        now=now,
    )
    acknowledgment = acknowledge_partner_review(
        request,
        reviewer_pseudonym="reviewer-01",
        assertions=("protocol_observed",),
        publication_consent="private",
        key_hex="22" * 32,
        now=now,
    )

    passed, mismatches = verify_partner_acknowledgment(
        request, acknowledgment, release_dir=release_dir, key_hex="33" * 32
    )
    assert passed is False
    assert "partner acknowledgment signature mismatch" in mismatches

    tampered = replace(acknowledgment, structured_feedback={"campaign_completed": False})
    passed, mismatches = verify_partner_acknowledgment(
        request, tampered, release_dir=release_dir, key_hex="22" * 32
    )
    assert passed is False
    assert "acknowledgment ID does not match its content" in mismatches
    assert "partner acknowledgment signature mismatch" in mismatches


def test_partner_review_cannot_be_prepared_for_public_receipt(tmp_path) -> None:
    release_dir = _release(tmp_path, tier=EvidenceTier.PUBLIC_REPRODUCIBLE)
    with pytest.raises(ValueError, match="private_partner_observed"):
        prepare_partner_review(
            release_dir,
            partner_pseudonym="partner-01",
            requested_assertions=("protocol_observed",),
        )


def test_partner_cannot_assert_unrequested_or_expired_claim(tmp_path) -> None:
    release_dir = _release(tmp_path)
    now = datetime(2026, 7, 21, tzinfo=UTC)
    request = prepare_partner_review(
        release_dir,
        partner_pseudonym="partner-01",
        requested_assertions=("protocol_observed",),
        valid_days=1,
        now=now,
    )
    with pytest.raises(ValueError, match="not requested"):
        acknowledge_partner_review(
            request,
            reviewer_pseudonym="reviewer-01",
            assertions=("results_reproduced",),
            publication_consent="private",
            key_hex="44" * 32,
            now=now,
        )
    with pytest.raises(ValueError, match="expired"):
        acknowledge_partner_review(
            request,
            reviewer_pseudonym="reviewer-01",
            assertions=("protocol_observed",),
            publication_consent="private",
            key_hex="44" * 32,
            now=now + timedelta(days=2),
        )


def test_receipt_change_after_request_invalidates_acknowledgment(tmp_path) -> None:
    release_dir = _release(tmp_path)
    now = datetime(2026, 7, 21, tzinfo=UTC)
    request = prepare_partner_review(
        release_dir,
        partner_pseudonym="partner-01",
        requested_assertions=("protocol_observed",),
        now=now,
    )
    acknowledgment = acknowledge_partner_review(
        request,
        reviewer_pseudonym="reviewer-01",
        assertions=("protocol_observed",),
        publication_consent="private",
        key_hex="55" * 32,
        now=now,
    )
    receipt_path = release_dir / "receipt.json"
    payload = json.loads(receipt_path.read_text())
    payload["limitations"] = ["changed after request"]
    receipt_path.write_text(json.dumps(payload))

    passed, mismatches = verify_partner_acknowledgment(
        request, acknowledgment, release_dir=release_dir, key_hex="55" * 32
    )
    assert passed is False
    assert "receipt content changed after the review request" in mismatches


def test_partner_review_feedback_is_bounded_and_non_narrative(tmp_path) -> None:
    release_dir = _release(tmp_path)
    request = prepare_partner_review(
        release_dir,
        partner_pseudonym="partner-07",
        requested_assertions=("artifacts_reviewed",),
    )
    with pytest.raises(ValueError, match="unknown structured feedback"):
        acknowledge_partner_review(
            request,
            reviewer_pseudonym="reviewer-01",
            assertions=("artifacts_reviewed",),
            publication_consent="private",
            key_hex="ab" * 32,
            structured_feedback={"free_text": "confidential narrative"},
        )


def test_partner_review_rejects_receipt_substitution_and_out_of_window_time(tmp_path) -> None:
    release_dir = _release(tmp_path)
    now = datetime(2026, 7, 21, tzinfo=UTC)
    request = prepare_partner_review(
        release_dir,
        partner_pseudonym="partner-01",
        requested_assertions=("protocol_observed",),
        valid_days=1,
        now=now,
    )
    acknowledgment = acknowledge_partner_review(
        request,
        reviewer_pseudonym="reviewer-01",
        assertions=("protocol_observed",),
        publication_consent="private",
        key_hex="cd" * 32,
        now=now,
    )

    substituted = replace(
        acknowledgment,
        receipt_release_id="f" * 64,
        acknowledgment_id=acknowledgment.acknowledgment_id,
    )
    passed, mismatches = verify_partner_acknowledgment(
        request, substituted, release_dir=release_dir, key_hex="cd" * 32
    )
    assert passed is False
    assert "acknowledgment references a different receipt release" in mismatches

    outside_window = replace(
        acknowledgment,
        reviewed_at=(now + timedelta(days=2)).isoformat(),
        acknowledgment_id=acknowledgment.acknowledgment_id,
    )
    passed, mismatches = verify_partner_acknowledgment(
        request, outside_window, release_dir=release_dir, key_hex="cd" * 32
    )
    assert passed is False
    assert "acknowledgment review time is outside the request window" in mismatches
