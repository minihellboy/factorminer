"""Frozen contracts for the hosted-pilot control plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class TenantStatus(StrEnum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class JobKind(StrEnum):
    VALIDATE_DATA = "validate_data"
    MINE = "mine"
    BENCHMARK = "benchmark"
    GENERATE_REPORT = "generate_report"
    VERIFY_RECEIPT = "verify_receipt"


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


TERMINAL_JOB_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED})

ALLOWED_SCOPES = frozenset(
    {
        "jobs:submit",
        "jobs:read",
        "jobs:cancel",
        "artifacts:read",
        "usage:read",
        "consent:write",
    }
)

ALLOWED_CONSENT_PURPOSES = frozenset(
    {
        "aggregate_failure_taxonomy",
        "aggregate_workflow_metrics",
        "aggregate_baseline_selection",
    }
)

ALLOWED_AGGREGATE_METRICS = frozenset(
    {
        "setup_minutes",
        "campaign_completion",
        "runtime_seconds",
        "admitted_factor_count",
        "failed_candidate_count",
    }
)

ALLOWED_AGGREGATE_CONTEXT = {
    "asset_class": frozenset({"crypto", "equities", "futures", "fx", "multi_asset", "other"}),
    "deployment_mode": frozenset({"hosted", "local"}),
    "evidence_tier": frozenset(
        {"private_partner_observed", "public_reproducible", "simulated", "unverified"}
    ),
}


@dataclass(frozen=True)
class TenantQuota:
    max_active_jobs: int = 2
    max_queued_jobs: int = 10
    max_payload_bytes: int = 64 * 1024
    max_runtime_seconds: int = 1800
    max_log_bytes: int = 8 * 1024 * 1024
    max_input_bytes: int = 2 * 1024 * 1024 * 1024
    max_storage_bytes: int = 5 * 1024 * 1024 * 1024
    requests_per_minute: int = 60
    compute_seconds_per_day: int = 4 * 3600

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if int(value) <= 0:
                raise ValueError(f"tenant quota {name} must be positive")

    def to_dict(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> TenantQuota:
        return cls(**{key: int(value) for key, value in dict(payload or {}).items()})


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    credential_id: str
    scopes: tuple[str, ...]
    expires_at: int | None = None

    def require(self, scope: str) -> None:
        if scope not in self.scopes:
            raise PermissionError(f"credential lacks required scope: {scope}")


@dataclass(frozen=True)
class CredentialIssue:
    credential_id: str
    tenant_id: str
    token: str
    scopes: tuple[str, ...]
    expires_at: str | None


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    tenant_id: str
    kind: JobKind
    state: JobState
    parameters: dict[str, Any]
    idempotency_key: str
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    timeout_seconds: int
    attempt: int
    cancel_requested: bool
    result: dict[str, Any] | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["state"] = self.state.value
        return payload
