"""Prepare and verify explicit human review of private research receipts."""

from __future__ import annotations

import hmac
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from factorminer.architecture.partner_review import (
    PARTNER_REVIEW_SCHEMA_VERSION,
    PUBLICATION_CONSENT_VALUES,
    PartnerReviewAcknowledgment,
    PartnerReviewRequest,
    derive_acknowledgment_id,
    derive_review_id,
    validate_review_assertions,
    validate_structured_feedback,
)
from factorminer.architecture.research_receipt import EvidenceTier, ExternalResearchReceipt
from factorminer.benchmark.reporting import file_sha256


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _parse_key(key_hex: str) -> bytes:
    try:
        key = bytes.fromhex(key_hex)
    except ValueError as exc:
        raise ValueError("partner review key must be hexadecimal") from exc
    if len(key) < 32:
        raise ValueError("partner review key must contain at least 32 bytes")
    return key


def _parse_timestamp(value: str, *, field: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return timestamp


def _normalize_pseudonym(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 128 or any(char in "\r\n" for char in normalized):
        raise ValueError(f"{field} must contain 1-128 characters on one line")
    return normalized


def _load_private_receipt(release_dir: Path) -> tuple[ExternalResearchReceipt, Path]:
    receipt_path = release_dir / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"receipt is missing: {receipt_path}")
    receipt = ExternalResearchReceipt.from_dict(json.loads(receipt_path.read_text()))
    if receipt.evidence_tier != EvidenceTier.PRIVATE_PARTNER_OBSERVED:
        raise ValueError("partner review requests require a private_partner_observed receipt")
    return receipt, receipt_path


def prepare_partner_review(
    release_dir: Path,
    *,
    partner_pseudonym: str,
    requested_assertions: tuple[str, ...],
    valid_days: int = 30,
    now: datetime | None = None,
) -> PartnerReviewRequest:
    """Create a bounded request. This never marks a receipt as reviewed."""
    receipt, receipt_path = _load_private_receipt(release_dir)
    assertions = tuple(sorted(set(requested_assertions)))
    validate_review_assertions(assertions)
    normalized_pseudonym = _normalize_pseudonym(partner_pseudonym, field="partner_pseudonym")
    if valid_days < 1 or valid_days > 365:
        raise ValueError("valid_days must be between 1 and 365")
    requested = now or datetime.now(UTC)
    if requested.tzinfo is None or requested.utcoffset() is None:
        raise ValueError("review request time must include a timezone")
    draft = PartnerReviewRequest(
        schema_version=PARTNER_REVIEW_SCHEMA_VERSION,
        review_id="",
        receipt_release_id=receipt.release_id,
        receipt_sha256=file_sha256(receipt_path),
        partner_pseudonym=normalized_pseudonym,
        requested_assertions=assertions,
        requested_at=requested.isoformat(),
        expires_at=(requested + timedelta(days=valid_days)).isoformat(),
    )
    return replace(draft, review_id=derive_review_id(draft))


def acknowledge_partner_review(
    request: PartnerReviewRequest,
    *,
    reviewer_pseudonym: str,
    assertions: tuple[str, ...],
    publication_consent: str,
    key_hex: str,
    structured_feedback: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> PartnerReviewAcknowledgment:
    """Sign only assertions explicitly selected by the external reviewer."""
    if request.review_id != derive_review_id(request):
        raise ValueError("partner review request ID does not match its content")
    reviewed = now or datetime.now(UTC)
    requested_at = _parse_timestamp(request.requested_at, field="requested_at")
    expires_at = _parse_timestamp(request.expires_at, field="expires_at")
    if expires_at <= requested_at:
        raise ValueError("partner review request expiry must be after its request time")
    if reviewed.tzinfo is None or reviewed.utcoffset() is None:
        raise ValueError("reviewed_at must include a timezone")
    if reviewed < requested_at:
        raise ValueError("partner review cannot predate its request")
    if reviewed > expires_at:
        raise ValueError("partner review request has expired")
    selected = tuple(sorted(set(assertions)))
    validate_review_assertions(selected)
    if not set(selected).issubset(request.requested_assertions):
        raise ValueError("acknowledgment contains assertions not requested by the producer")
    if publication_consent not in PUBLICATION_CONSENT_VALUES:
        raise ValueError(
            f"publication_consent must be one of: {sorted(PUBLICATION_CONSENT_VALUES)}"
        )
    normalized_reviewer = _normalize_pseudonym(reviewer_pseudonym, field="reviewer_pseudonym")
    draft = PartnerReviewAcknowledgment(
        schema_version=PARTNER_REVIEW_SCHEMA_VERSION,
        acknowledgment_id="",
        review_id=request.review_id,
        receipt_release_id=request.receipt_release_id,
        receipt_sha256=request.receipt_sha256,
        reviewer_pseudonym=normalized_reviewer,
        assertions=selected,
        reviewed_at=reviewed.isoformat(),
        publication_consent=publication_consent,
        structured_feedback=validate_structured_feedback(dict(structured_feedback or {})),
    )
    signature = hmac.new(_parse_key(key_hex), _canonical_bytes(draft.unsigned_dict()), sha256)
    signed = replace(draft, signature=signature.hexdigest())
    return replace(signed, acknowledgment_id=derive_acknowledgment_id(signed))


def parse_partner_review_request(payload: dict[str, Any]) -> PartnerReviewRequest:
    request = PartnerReviewRequest(
        schema_version=int(payload["schema_version"]),
        review_id=str(payload["review_id"]),
        receipt_release_id=str(payload["receipt_release_id"]),
        receipt_sha256=str(payload["receipt_sha256"]),
        partner_pseudonym=str(payload["partner_pseudonym"]),
        requested_assertions=tuple(payload["requested_assertions"]),
        requested_at=str(payload["requested_at"]),
        expires_at=str(payload["expires_at"]),
        instructions=tuple(payload.get("instructions", ())),
    )
    validate_review_assertions(request.requested_assertions)
    if request.schema_version != PARTNER_REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported partner review request version")
    return request


def parse_partner_acknowledgment(payload: dict[str, Any]) -> PartnerReviewAcknowledgment:
    acknowledgment = PartnerReviewAcknowledgment(
        schema_version=int(payload["schema_version"]),
        acknowledgment_id=str(payload["acknowledgment_id"]),
        review_id=str(payload["review_id"]),
        receipt_release_id=str(payload["receipt_release_id"]),
        receipt_sha256=str(payload["receipt_sha256"]),
        reviewer_pseudonym=str(payload["reviewer_pseudonym"]),
        assertions=tuple(payload["assertions"]),
        reviewed_at=str(payload["reviewed_at"]),
        publication_consent=str(payload["publication_consent"]),
        structured_feedback=dict(payload.get("structured_feedback", {})),
        signature_scheme=str(payload["signature_scheme"]),
        signature=str(payload["signature"]),
    )
    validate_review_assertions(acknowledgment.assertions)
    validate_structured_feedback(acknowledgment.structured_feedback)
    return acknowledgment


def verify_partner_acknowledgment(
    request: PartnerReviewRequest,
    acknowledgment: PartnerReviewAcknowledgment,
    *,
    release_dir: Path,
    key_hex: str,
) -> tuple[bool, tuple[str, ...]]:
    mismatches: list[str] = []
    try:
        receipt, receipt_path = _load_private_receipt(release_dir)
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, (str(exc),)
    if request.schema_version != PARTNER_REVIEW_SCHEMA_VERSION:
        mismatches.append("unsupported partner review request version")
    if acknowledgment.schema_version != PARTNER_REVIEW_SCHEMA_VERSION:
        mismatches.append("unsupported partner acknowledgment version")
    if request.review_id != derive_review_id(request):
        mismatches.append("review request ID does not match its content")
    if acknowledgment.acknowledgment_id != derive_acknowledgment_id(acknowledgment):
        mismatches.append("acknowledgment ID does not match its content")
    if acknowledgment.review_id != request.review_id:
        mismatches.append("acknowledgment references a different review request")
    if receipt.release_id != request.receipt_release_id:
        mismatches.append("review request references a different receipt release")
    if acknowledgment.receipt_release_id != request.receipt_release_id:
        mismatches.append("acknowledgment references a different receipt release")
    if file_sha256(receipt_path) != request.receipt_sha256:
        mismatches.append("receipt content changed after the review request")
    if acknowledgment.receipt_sha256 != request.receipt_sha256:
        mismatches.append("acknowledgment references a different receipt digest")
    if not set(acknowledgment.assertions).issubset(request.requested_assertions):
        mismatches.append("acknowledgment contains an unrequested assertion")
    if acknowledgment.publication_consent not in PUBLICATION_CONSENT_VALUES:
        mismatches.append("acknowledgment has an invalid publication consent value")
    if acknowledgment.signature_scheme != "hmac-sha256":
        mismatches.append("unsupported acknowledgment signature scheme")
    try:
        requested_at = _parse_timestamp(request.requested_at, field="requested_at")
        expires_at = _parse_timestamp(request.expires_at, field="expires_at")
        reviewed_at = _parse_timestamp(acknowledgment.reviewed_at, field="reviewed_at")
    except ValueError as exc:
        mismatches.append(str(exc))
    else:
        if expires_at <= requested_at:
            mismatches.append("partner review request expiry is not after its request time")
        if reviewed_at < requested_at or reviewed_at > expires_at:
            mismatches.append("acknowledgment review time is outside the request window")
    try:
        _normalize_pseudonym(request.partner_pseudonym, field="partner_pseudonym")
        _normalize_pseudonym(acknowledgment.reviewer_pseudonym, field="reviewer_pseudonym")
        validate_review_assertions(acknowledgment.assertions)
    except ValueError as exc:
        mismatches.append(str(exc))
    try:
        validate_structured_feedback(acknowledgment.structured_feedback)
    except ValueError as exc:
        mismatches.append(str(exc))
    try:
        expected = hmac.new(
            _parse_key(key_hex),
            _canonical_bytes(acknowledgment.unsigned_dict()),
            sha256,
        ).hexdigest()
    except ValueError as exc:
        mismatches.append(str(exc))
    else:
        if not hmac.compare_digest(expected, acknowledgment.signature):
            mismatches.append("partner acknowledgment signature mismatch")
    return not mismatches, tuple(mismatches)


def write_partner_review_artifact(payload: dict[str, Any], output: Path) -> Path:
    content = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output.exists() and output.read_text() != content:
        raise FileExistsError(f"refusing to overwrite divergent partner review artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    return output
