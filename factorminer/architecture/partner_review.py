"""Signed, receipt-bound design-partner review contracts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from factorminer.core.provenance import _json_safe, stable_digest

PARTNER_REVIEW_SCHEMA_VERSION = 1
ALLOWED_REVIEW_ASSERTIONS = frozenset(
    {
        "protocol_observed",
        "dataset_commitment_verified",
        "artifacts_reviewed",
        "limitations_acknowledged",
        "results_reproduced",
    }
)
PUBLICATION_CONSENT_VALUES = frozenset({"private", "anonymous", "public"})
PARTNER_DECISION_VALUES = frozenset({"adopt", "pilot_more", "not_now", "reject", "undisclosed"})
PARTNER_BLOCKER_VALUES = frozenset(
    {
        "cost",
        "data_integration",
        "evidence_quality",
        "governance",
        "missing_feature",
        "runtime",
        "security_review",
        "workflow_fit",
        "other",
    }
)
PARTNER_REPEATED_USE_VALUES = frozenset({"yes", "no", "undecided"})
PARTNER_USEFUL_OUTPUT_VALUES = frozenset(
    {
        "benchmark_comparison",
        "cost_stress",
        "evidence_report",
        "factor_library",
        "provenance",
        "receipt",
    }
)
PARTNER_FAILURE_MODE_VALUES = frozenset(
    {
        "data_validation",
        "formula_evaluation",
        "none",
        "other",
        "provider_error",
        "report_generation",
        "resource_limit",
        "timeout",
    }
)
PARTNER_INTEGRATION_VALUES = frozenset(
    {"lean", "market_data", "notebook", "object_storage", "other", "qlib", "sso", "webhook"}
)
ALLOWED_FEEDBACK_FIELDS = frozenset(
    {
        "setup_minutes",
        "campaign_completed",
        "completion_rate",
        "trust_rating",
        "decision_outcome",
        "adoption_blockers",
        "repeated_use_intent",
        "useful_outputs",
        "failure_modes",
        "requested_integrations",
    }
)


@dataclass(frozen=True)
class PartnerReviewRequest:
    schema_version: int
    review_id: str
    receipt_release_id: str
    receipt_sha256: str
    partner_pseudonym: str
    requested_assertions: tuple[str, ...]
    requested_at: str
    expires_at: str
    instructions: tuple[str, ...] = (
        "Verify the referenced private-partner receipt and only assert actions personally observed.",
        "Keep the commitment key and proprietary input outside the review artifact.",
        "Acknowledgement is bounded to this receipt digest and does not certify future performance.",
    )

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class PartnerReviewAcknowledgment:
    schema_version: int
    acknowledgment_id: str
    review_id: str
    receipt_release_id: str
    receipt_sha256: str
    reviewer_pseudonym: str
    assertions: tuple[str, ...]
    reviewed_at: str
    publication_consent: str
    structured_feedback: dict[str, Any] = field(default_factory=dict)
    signature_scheme: str = "hmac-sha256"
    signature: str = ""

    def unsigned_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("signature", None)
        payload.pop("acknowledgment_id", None)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def derive_review_id(request: PartnerReviewRequest) -> str:
    payload = request.to_dict()
    payload.pop("review_id", None)
    return stable_digest(payload)


def derive_acknowledgment_id(acknowledgment: PartnerReviewAcknowledgment) -> str:
    payload = acknowledgment.to_dict()
    payload.pop("acknowledgment_id", None)
    return stable_digest(payload)


def validate_review_assertions(assertions: tuple[str, ...]) -> None:
    if not assertions:
        raise ValueError("at least one review assertion is required")
    if tuple(sorted(set(assertions))) != assertions:
        raise ValueError("review assertions must be unique and sorted")
    unknown = set(assertions) - ALLOWED_REVIEW_ASSERTIONS
    if unknown:
        raise ValueError(f"unknown review assertions: {sorted(unknown)}")


def validate_structured_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    """Normalize a bounded, non-narrative partner feedback record."""
    unknown = set(feedback) - ALLOWED_FEEDBACK_FIELDS
    if unknown:
        raise ValueError(f"unknown structured feedback fields: {sorted(unknown)}")
    normalized: dict[str, Any] = {}
    if "setup_minutes" in feedback:
        setup_minutes = feedback["setup_minutes"]
        if not isinstance(setup_minutes, int) or isinstance(setup_minutes, bool):
            raise ValueError("setup_minutes must be an integer between 0 and 10080")
        if not 0 <= setup_minutes <= 10080:
            raise ValueError("setup_minutes must be an integer between 0 and 10080")
        normalized["setup_minutes"] = setup_minutes
    if "campaign_completed" in feedback:
        if not isinstance(feedback["campaign_completed"], bool):
            raise ValueError("campaign_completed must be boolean")
        normalized["campaign_completed"] = feedback["campaign_completed"]
    if "completion_rate" in feedback:
        completion_rate = feedback["completion_rate"]
        if (
            not isinstance(completion_rate, (int, float))
            or isinstance(completion_rate, bool)
            or not math.isfinite(float(completion_rate))
            or not 0 <= float(completion_rate) <= 1
        ):
            raise ValueError("completion_rate must be finite and between 0 and 1")
        normalized["completion_rate"] = float(completion_rate)
    if "trust_rating" in feedback:
        trust_rating = feedback["trust_rating"]
        if not isinstance(trust_rating, int) or isinstance(trust_rating, bool):
            raise ValueError("trust_rating must be an integer between 1 and 5")
        if not 1 <= trust_rating <= 5:
            raise ValueError("trust_rating must be an integer between 1 and 5")
        normalized["trust_rating"] = trust_rating
    if "decision_outcome" in feedback:
        decision = str(feedback["decision_outcome"])
        if decision not in PARTNER_DECISION_VALUES:
            raise ValueError(f"unsupported decision_outcome: {decision}")
        normalized["decision_outcome"] = decision
    if "adoption_blockers" in feedback:
        blockers = feedback["adoption_blockers"]
        if not isinstance(blockers, list) or len(blockers) > len(PARTNER_BLOCKER_VALUES):
            raise ValueError("adoption_blockers must be a bounded list")
        selected = sorted({str(item) for item in blockers})
        unknown_blockers = set(selected) - PARTNER_BLOCKER_VALUES
        if unknown_blockers:
            raise ValueError(f"unsupported adoption blockers: {sorted(unknown_blockers)}")
        normalized["adoption_blockers"] = selected
    if "repeated_use_intent" in feedback:
        repeated_use = str(feedback["repeated_use_intent"])
        if repeated_use not in PARTNER_REPEATED_USE_VALUES:
            raise ValueError(f"unsupported repeated_use_intent: {repeated_use}")
        normalized["repeated_use_intent"] = repeated_use
    for field_name, allowed_values in (
        ("useful_outputs", PARTNER_USEFUL_OUTPUT_VALUES),
        ("failure_modes", PARTNER_FAILURE_MODE_VALUES),
        ("requested_integrations", PARTNER_INTEGRATION_VALUES),
    ):
        if field_name not in feedback:
            continue
        values = feedback[field_name]
        if not isinstance(values, list) or len(values) > len(allowed_values):
            raise ValueError(f"{field_name} must be a bounded list")
        selected = sorted({str(item) for item in values})
        unknown_values = set(selected) - allowed_values
        if unknown_values:
            raise ValueError(f"unsupported {field_name}: {sorted(unknown_values)}")
        normalized[field_name] = selected
    return normalized
