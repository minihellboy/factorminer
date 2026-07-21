"""Content-addressed, offline-verifiable evidence receipt for a published run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from string import hexdigits
from typing import Any

from factorminer.core.provenance import _json_safe, stable_digest


class EvidenceTier(StrEnum):
    """How much of the underlying data a third party can independently check."""

    SIMULATED = "simulated"
    PUBLIC_REPRODUCIBLE = "public_reproducible"
    PRIVATE_PARTNER_OBSERVED = "private_partner_observed"
    UNVERIFIED = "unverified"


class RunStatus(StrEnum):
    """This run's own completion status. Never mutated after publication."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


_DATA_LICENSE_CLASSES = frozenset(
    {
        "proprietary_licensed",
        "public_domain",
        "publicly_retrievable",
        "redistributable_with_attribution",
        "synthetic",
        "unknown",
        "vendor_redistributable_sample",
    }
)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in hexdigits for char in value)


@dataclass(frozen=True)
class DatasetCommitment:
    """Cryptographic commitment to the input dataset.

    ``scheme="none"`` means no real data was hashed (mock/simulated runs).
    ``scheme="sha256"`` is a plain, independently-reproducible file digest.
    ``scheme="hmac-sha256"`` re-keys that digest with a withheld nonce so a
    small, guessable universe of candidate datasets cannot be brute-forced
    from the published receipt alone; ``nonce`` stays ``None`` in every
    receipt that is ever written to disk or published.
    """

    scheme: str
    digest: str
    nonce: str | None = None


@dataclass(frozen=True)
class ReviewerState:
    """Mirrors ``EconomicRationale.attested`` exactly: never auto-set True."""

    attested: bool = False
    attestor: str = ""
    attested_at: str = ""
    source: str = "system"  # system | human


@dataclass(frozen=True)
class SourceManifestRef:
    """Pointer to the manifest this receipt was built from."""

    path: str
    sha256: str


# Mirrors the RFT_EXPORT_HONESTY banner convention in architecture/rft_export.py.
RESEARCH_RECEIPT_LIMITATIONS: tuple[str, ...] = (
    "This receipt describes a research pipeline's recomputed output, not a live "
    "trading track record. No capital was put at risk to produce these numbers.",
    "evidence_tier=simulated or unverified receipts must not be represented as "
    "external validation.",
    "Verifying artifact hashes confirms the published files were not altered "
    "after release; it does not confirm the underlying strategy is profitable "
    "out-of-sample beyond what the cited metrics report.",
)


@dataclass(frozen=True)
class ExternalResearchReceipt:
    """Falsifiable, content-addressed evidence artifact for one published run."""

    release_id: str
    evidence_tier: EvidenceTier
    run_status: RunStatus
    generated_at: str
    code_sha: str
    config_sha256: str
    environment_lock_sha256: str
    seed: int
    dataset_descriptor: dict[str, Any]
    dataset_commitment: DatasetCommitment
    data_license_class: str
    factor_library_sha256: str
    baseline_provenance: dict[str, Any] = field(default_factory=dict)
    split_and_freeze_contract: dict[str, Any] = field(default_factory=dict)
    cost_and_capacity_assumptions: dict[str, Any] = field(default_factory=dict)
    metric_definitions: dict[str, str] = field(default_factory=dict)
    selection_policy: dict[str, Any] = field(default_factory=dict)
    source_manifest: SourceManifestRef | None = None
    artifact_sha256s: dict[str, str] = field(default_factory=dict)
    limitations: tuple[str, ...] = RESEARCH_RECEIPT_LIMITATIONS
    reviewer_state: ReviewerState = field(default_factory=ReviewerState)
    supersedes_release_id: str | None = None

    def __post_init__(self) -> None:
        if self.data_license_class not in _DATA_LICENSE_CLASSES:
            raise ValueError(f"data_license_class must be one of: {sorted(_DATA_LICENSE_CLASSES)}")
        if (
            self.evidence_tier
            in (EvidenceTier.PUBLIC_REPRODUCIBLE, EvidenceTier.PRIVATE_PARTNER_OBSERVED)
            and self.dataset_commitment.scheme == "none"
        ):
            raise ValueError(
                f"evidence_tier={self.evidence_tier.value} requires a real "
                "dataset_commitment (scheme != 'none'); mock/simulated runs must "
                "use evidence_tier=simulated"
            )
        if self.evidence_tier == EvidenceTier.PUBLIC_REPRODUCIBLE:
            if self.dataset_commitment.scheme != "sha256":
                raise ValueError(
                    "evidence_tier=public_reproducible requires a sha256 dataset commitment"
                )
            if self.data_license_class not in {
                "public_domain",
                "publicly_retrievable",
                "redistributable_with_attribution",
                "vendor_redistributable_sample",
            }:
                raise ValueError(
                    "evidence_tier=public_reproducible requires a publicly obtainable "
                    "data license class"
                )
        if self.evidence_tier == EvidenceTier.PRIVATE_PARTNER_OBSERVED:
            if self.dataset_commitment.scheme != "hmac-sha256":
                raise ValueError(
                    "evidence_tier=private_partner_observed requires an hmac-sha256 "
                    "dataset commitment"
                )
            if self.data_license_class != "proprietary_licensed":
                raise ValueError(
                    "evidence_tier=private_partner_observed requires "
                    "data_license_class=proprietary_licensed"
                )
        if self.dataset_commitment.scheme in {"sha256", "hmac-sha256"} and not _is_sha256(
            self.dataset_commitment.digest
        ):
            raise ValueError("dataset_commitment.digest must be a 64-character SHA-256 hex value")
        for field_name, digest in (
            ("config_sha256", self.config_sha256),
            ("environment_lock_sha256", self.environment_lock_sha256),
            ("factor_library_sha256", self.factor_library_sha256),
        ):
            if digest != "unknown" and not _is_sha256(digest):
                raise ValueError(f"{field_name} must be a 64-character SHA-256 hex value")
        if not self.dataset_descriptor:
            raise ValueError("dataset_descriptor must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExternalResearchReceipt:
        """Parse and validate an untrusted serialized receipt."""
        values = dict(payload)
        values["evidence_tier"] = EvidenceTier(values["evidence_tier"])
        values["run_status"] = RunStatus(values["run_status"])
        values["dataset_commitment"] = DatasetCommitment(**dict(values["dataset_commitment"]))
        reviewer = values.get("reviewer_state") or {}
        values["reviewer_state"] = ReviewerState(**dict(reviewer))
        source = values.get("source_manifest")
        values["source_manifest"] = (
            SourceManifestRef(**dict(source)) if source is not None else None
        )
        values["limitations"] = tuple(values.get("limitations") or ())
        return cls(**values)


def derive_release_id(receipt: ExternalResearchReceipt) -> str:
    """Content-address every serialized field except ``release_id`` itself."""
    payload = receipt.to_dict()
    payload.pop("release_id", None)
    return stable_digest(payload)


def derive_release_id_from_payload(payload: dict[str, Any]) -> str:
    """Recompute a release ID from an untrusted serialized receipt payload."""
    content = dict(payload)
    content.pop("release_id", None)
    return stable_digest(content)


def attest_research_receipt(
    receipt: ExternalResearchReceipt, *, attestor: str
) -> ExternalResearchReceipt:
    """The only supported path that sets ``reviewer_state.attested=True``.

    Returns a NEW receipt (frozen dataclass); this changes the receipt's
    content, so it also changes derive_release_id's output -- callers that
    need the attested version at a stable address must call
    derive_release_id again and use write_receipt to publish it separately
    from the pre-attestation release.
    """
    from datetime import datetime

    return replace(
        receipt,
        reviewer_state=ReviewerState(
            attested=True,
            attestor=str(attestor or "human"),
            attested_at=datetime.now().isoformat(timespec="seconds"),
            source="human",
        ),
    )
