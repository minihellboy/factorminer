"""SQLite-backed durable state for the hosted research pilot."""

from __future__ import annotations

import hmac
import json
import math
import secrets
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from factorminer.hosted.models import (
    ALLOWED_AGGREGATE_CONTEXT,
    ALLOWED_AGGREGATE_METRICS,
    ALLOWED_CONSENT_PURPOSES,
    ALLOWED_SCOPES,
    CredentialIssue,
    JobKind,
    JobRecord,
    JobState,
    Principal,
    TenantQuota,
    TenantStatus,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _token_digest(token: str) -> str:
    return sha256(token.encode()).hexdigest()


def _validate_identifier(value: str, *, field: str) -> str:
    clean = value.strip()
    if not clean or len(clean) > 128:
        raise ValueError(f"{field} must contain 1-128 characters")
    if any(
        char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        for char in clean
    ):
        raise ValueError(f"{field} contains unsupported characters")
    return clean


class HostedStore:
    """Small durable control-plane store with append-only chained audit events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tenants (
                    tenant_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    quota_json TEXT NOT NULL,
                    retention_days INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS credentials (
                    credential_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                    token_hash TEXT NOT NULL UNIQUE,
                    label TEXT NOT NULL,
                    scopes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    revoked_at TEXT
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    timeout_seconds INTEGER NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    result_json TEXT,
                    error TEXT,
                    UNIQUE(tenant_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS jobs_queue_idx ON jobs(state, created_at);
                CREATE INDEX IF NOT EXISTS jobs_tenant_idx ON jobs(tenant_id, created_at);

                CREATE TABLE IF NOT EXISTS usage_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                    operation TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    unit TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS usage_tenant_time_idx
                    ON usage_events(tenant_id, created_at);

                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    tenant_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE INDEX IF NOT EXISTS audit_tenant_seq_idx
                    ON audit_events(tenant_id, sequence);

                CREATE TABLE IF NOT EXISTS consent_grants (
                    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                    purpose TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    state TEXT NOT NULL,
                    granted_at TEXT,
                    revoked_at TEXT,
                    PRIMARY KEY(tenant_id, purpose)
                );

                CREATE TABLE IF NOT EXISTS aggregate_observations (
                    observation_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL REFERENCES tenants(tenant_id),
                    purpose TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    context_json TEXT NOT NULL,
                    source_receipt_id TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS aggregate_metric_idx
                    ON aggregate_observations(purpose, metric_name);
                """
            )
            aggregate_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(aggregate_observations)")
            }
            if "policy_version" not in aggregate_columns:
                connection.execute(
                    "ALTER TABLE aggregate_observations "
                    "ADD COLUMN policy_version TEXT NOT NULL DEFAULT 'legacy-unversioned'"
                )

    def _append_audit(
        self,
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        action: str,
        subject: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        previous = connection.execute(
            "SELECT event_hash FROM audit_events WHERE tenant_id = ? ORDER BY sequence DESC LIMIT 1",
            (tenant_id,),
        ).fetchone()
        previous_hash = str(previous["event_hash"]) if previous else "0" * 64
        event_id = str(uuid.uuid4())
        created_at = _iso()
        metadata_json = _canonical_json(metadata or {})
        event_hash = sha256(
            _canonical_json(
                {
                    "event_id": event_id,
                    "tenant_id": tenant_id,
                    "action": action,
                    "subject": subject,
                    "metadata": json.loads(metadata_json),
                    "created_at": created_at,
                    "previous_hash": previous_hash,
                }
            ).encode()
        ).hexdigest()
        connection.execute(
            """
            INSERT INTO audit_events(
                event_id, tenant_id, action, subject, metadata_json,
                created_at, previous_hash, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                tenant_id,
                action,
                subject,
                metadata_json,
                created_at,
                previous_hash,
                event_hash,
            ),
        )
        return event_id

    def record_audit(
        self,
        tenant_id: str,
        *,
        action: str,
        subject: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self._transaction() as connection:
            return self._append_audit(
                connection,
                tenant_id=tenant_id,
                action=action,
                subject=subject,
                metadata=metadata,
            )

    def create_tenant(
        self,
        tenant_id: str,
        *,
        display_name: str,
        quota: TenantQuota | None = None,
        retention_days: int = 30,
    ) -> dict[str, Any]:
        tenant_id = _validate_identifier(tenant_id, field="tenant_id")
        if not display_name.strip() or len(display_name) > 200:
            raise ValueError("display_name must contain 1-200 characters")
        if retention_days < 1 or retention_days > 3650:
            raise ValueError("retention_days must be between 1 and 3650")
        timestamp = _iso()
        selected_quota = quota or TenantQuota()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO tenants(
                    tenant_id, display_name, status, quota_json,
                    retention_days, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    display_name.strip(),
                    TenantStatus.ACTIVE.value,
                    _canonical_json(selected_quota.to_dict()),
                    retention_days,
                    timestamp,
                    timestamp,
                ),
            )
            self._append_audit(
                connection,
                tenant_id=tenant_id,
                action="tenant.created",
                subject=tenant_id,
                metadata={"retention_days": retention_days},
            )
        return self.get_tenant(tenant_id)

    def get_tenant(self, tenant_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"tenant not found: {tenant_id}")
        return {
            "tenant_id": row["tenant_id"],
            "display_name": row["display_name"],
            "status": row["status"],
            "quota": json.loads(row["quota_json"]),
            "retention_days": int(row["retention_days"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def issue_credential(
        self,
        tenant_id: str,
        *,
        label: str,
        scopes: tuple[str, ...],
        expires_at: datetime | None = None,
    ) -> CredentialIssue:
        normalized_scopes = tuple(sorted(set(scopes)))
        if not normalized_scopes:
            raise ValueError("credential requires at least one scope")
        unknown = set(normalized_scopes) - ALLOWED_SCOPES
        if unknown:
            raise ValueError(f"unknown credential scopes: {sorted(unknown)}")
        tenant = self.get_tenant(tenant_id)
        if tenant["status"] != TenantStatus.ACTIVE.value:
            raise ValueError("credentials may only be issued to active tenants")
        credential_id = str(uuid.uuid4())
        token = f"fm_{tenant_id}_{secrets.token_urlsafe(32)}"
        expires_iso = expires_at.astimezone(UTC).isoformat() if expires_at else None
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO credentials(
                    credential_id, tenant_id, token_hash, label, scopes_json,
                    created_at, expires_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    credential_id,
                    tenant_id,
                    _token_digest(token),
                    label.strip() or "credential",
                    _canonical_json(normalized_scopes),
                    _iso(),
                    expires_iso,
                ),
            )
            self._append_audit(
                connection,
                tenant_id=tenant_id,
                action="credential.issued",
                subject=credential_id,
                metadata={"label": label, "scopes": normalized_scopes, "expires_at": expires_iso},
            )
        return CredentialIssue(
            credential_id=credential_id,
            tenant_id=tenant_id,
            token=token,
            scopes=normalized_scopes,
            expires_at=expires_iso,
        )

    def verify_token(self, token: str) -> Principal | None:
        digest = _token_digest(token)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT c.*, t.status AS tenant_status
                FROM credentials c JOIN tenants t ON t.tenant_id = c.tenant_id
                WHERE c.token_hash = ?
                """,
                (digest,),
            ).fetchone()
        if row is None or not hmac.compare_digest(str(row["token_hash"]), digest):
            return None
        if row["tenant_status"] != TenantStatus.ACTIVE.value or row["revoked_at"]:
            return None
        expires = datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
        if expires is not None and expires <= _now():
            return None
        return Principal(
            tenant_id=str(row["tenant_id"]),
            credential_id=str(row["credential_id"]),
            scopes=tuple(json.loads(row["scopes_json"])),
            expires_at=int(expires.timestamp()) if expires else None,
        )

    def revoke_credential(self, tenant_id: str, credential_id: str) -> None:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE credentials SET revoked_at = ?
                WHERE tenant_id = ? AND credential_id = ? AND revoked_at IS NULL
                """,
                (_iso(), tenant_id, credential_id),
            )
            if cursor.rowcount != 1:
                raise KeyError("active credential not found")
            self._append_audit(
                connection,
                tenant_id=tenant_id,
                action="credential.revoked",
                subject=credential_id,
            )

    def record_usage(
        self,
        tenant_id: str,
        *,
        operation: str,
        quantity: float = 1.0,
        unit: str = "request",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        event_id = str(uuid.uuid4())
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO usage_events(
                    event_id, tenant_id, operation, quantity, unit, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    tenant_id,
                    operation,
                    float(quantity),
                    unit,
                    _canonical_json(metadata or {}),
                    _iso(),
                ),
            )
        return event_id

    def recent_request_count(self, tenant_id: str, *, since: datetime) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM usage_events
                WHERE tenant_id = ? AND unit = 'request' AND created_at >= ?
                """,
                (tenant_id, since.astimezone(UTC).isoformat()),
            ).fetchone()
        return int(row["count"])

    def record_rate_limited_request(
        self,
        tenant_id: str,
        *,
        operation: str,
        limit: int,
        since: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Atomically enforce and record a tenant request-rate limit."""
        event_id = str(uuid.uuid4())
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM usage_events
                WHERE tenant_id = ? AND unit = 'request' AND created_at >= ?
                """,
                (tenant_id, since.astimezone(UTC).isoformat()),
            ).fetchone()
            if int(row["count"]) >= limit:
                raise RuntimeError("tenant request rate limit exceeded")
            connection.execute(
                """
                INSERT INTO usage_events(
                    event_id, tenant_id, operation, quantity, unit,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, 1, 'request', ?, ?)
                """,
                (
                    event_id,
                    tenant_id,
                    operation,
                    _canonical_json(metadata or {}),
                    _iso(),
                ),
            )
        return event_id

    def compute_seconds_since(self, tenant_id: str, *, since: datetime) -> float:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(SUM(quantity), 0) AS total FROM usage_events
                WHERE tenant_id = ? AND unit = 'compute_second' AND created_at >= ?
                """,
                (tenant_id, since.astimezone(UTC).isoformat()),
            ).fetchone()
        return float(row["total"])

    def submit_job(
        self,
        tenant_id: str,
        *,
        kind: JobKind,
        parameters: dict[str, Any],
        idempotency_key: str,
        timeout_seconds: int,
        quota: TenantQuota,
    ) -> JobRecord:
        idempotency_key = _validate_identifier(idempotency_key, field="idempotency_key")
        payload = _canonical_json(parameters)
        if len(payload.encode()) > quota.max_payload_bytes:
            raise ValueError("job payload exceeds tenant quota")
        if timeout_seconds < 1 or timeout_seconds > quota.max_runtime_seconds:
            raise ValueError("job timeout exceeds tenant quota")
        timestamp = _iso()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? AND idempotency_key = ?",
                (tenant_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["kind"] != kind.value or existing["parameters_json"] != payload:
                    raise ValueError("idempotency key was already used with different input")
                return self._job_from_row(existing)
            start_of_day = _now().replace(hour=0, minute=0, second=0, microsecond=0)
            used_compute = connection.execute(
                """
                SELECT COALESCE(SUM(quantity), 0) AS total FROM usage_events
                WHERE tenant_id = ? AND unit = 'compute_second' AND created_at >= ?
                """,
                (tenant_id, start_of_day.isoformat()),
            ).fetchone()
            reserved_compute = connection.execute(
                """
                SELECT COALESCE(SUM(timeout_seconds), 0) AS total FROM jobs
                WHERE tenant_id = ? AND state IN ('queued', 'running')
                """,
                (tenant_id,),
            ).fetchone()
            projected_compute = (
                float(used_compute["total"])
                + float(reserved_compute["total"])
                + timeout_seconds
            )
            if projected_compute > quota.compute_seconds_per_day:
                raise RuntimeError("tenant daily compute quota reservation exceeded")
            counts = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN state = 'queued' THEN 1 ELSE 0 END) AS queued,
                    SUM(CASE WHEN state = 'running' THEN 1 ELSE 0 END) AS active
                FROM jobs WHERE tenant_id = ?
                """,
                (tenant_id,),
            ).fetchone()
            if int(counts["queued"] or 0) >= quota.max_queued_jobs:
                raise RuntimeError("tenant queued-job quota exceeded")
            if int(counts["active"] or 0) >= quota.max_active_jobs:
                raise RuntimeError("tenant active-job quota exceeded")
            job_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, tenant_id, kind, state, parameters_json, idempotency_key,
                    created_at, updated_at, timeout_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    tenant_id,
                    kind.value,
                    JobState.QUEUED.value,
                    payload,
                    idempotency_key,
                    timestamp,
                    timestamp,
                    timeout_seconds,
                ),
            )
            self._append_audit(
                connection,
                tenant_id=tenant_id,
                action="job.submitted",
                subject=job_id,
                metadata={"kind": kind.value, "idempotency_key": idempotency_key},
            )
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        assert row is not None
        return self._job_from_row(row)

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=str(row["job_id"]),
            tenant_id=str(row["tenant_id"]),
            kind=JobKind(row["kind"]),
            state=JobState(row["state"]),
            parameters=json.loads(row["parameters_json"]),
            idempotency_key=str(row["idempotency_key"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            timeout_seconds=int(row["timeout_seconds"]),
            attempt=int(row["attempt"]),
            cancel_requested=bool(row["cancel_requested"]),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
        )

    def get_job(self, tenant_id: str, job_id: str) -> JobRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? AND job_id = ?",
                (tenant_id, job_id),
            ).fetchone()
        if row is None:
            raise KeyError("job not found")
        return self._job_from_row(row)

    def list_jobs(self, tenant_id: str, *, limit: int = 100) -> list[JobRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? ORDER BY created_at DESC LIMIT ?",
                (tenant_id, max(1, min(limit, 500))),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def list_retention_jobs(
        self, tenant_id: str, *, finished_before: datetime, limit: int = 1000
    ) -> list[JobRecord]:
        """Return one bounded oldest-first batch of expired terminal jobs."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE tenant_id = ?
                  AND state IN ('succeeded', 'failed', 'canceled')
                  AND finished_at IS NOT NULL
                  AND finished_at < ?
                ORDER BY finished_at ASC
                LIMIT ?
                """,
                (
                    tenant_id,
                    finished_before.astimezone(UTC).isoformat(),
                    max(1, min(limit, 1000)),
                ),
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def cancel_job(self, tenant_id: str, job_id: str) -> JobRecord:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? AND job_id = ?",
                (tenant_id, job_id),
            ).fetchone()
            if row is None:
                raise KeyError("job not found")
            state = JobState(row["state"])
            if state == JobState.QUEUED:
                connection.execute(
                    """
                    UPDATE jobs SET state = ?, cancel_requested = 1,
                        updated_at = ?, finished_at = ? WHERE job_id = ?
                    """,
                    (JobState.CANCELED.value, _iso(), _iso(), job_id),
                )
            elif state == JobState.RUNNING:
                connection.execute(
                    "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE job_id = ?",
                    (_iso(), job_id),
                )
            self._append_audit(
                connection,
                tenant_id=tenant_id,
                action="job.cancel_requested",
                subject=job_id,
                metadata={"previous_state": state.value},
            )
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        assert updated is not None
        return self._job_from_row(updated)

    def recover_expired_leases(self, *, max_attempts: int = 3) -> int:
        now = _iso()
        with self._transaction() as connection:
            rows = connection.execute(
                """
                SELECT * FROM jobs WHERE state = 'running'
                AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                """,
                (now,),
            ).fetchall()
            for row in rows:
                failed = int(row["attempt"]) >= max_attempts or bool(row["cancel_requested"])
                state = JobState.FAILED if failed else JobState.QUEUED
                error = "worker lease expired" if failed else None
                connection.execute(
                    """
                    UPDATE jobs SET state = ?, lease_owner = NULL, lease_expires_at = NULL,
                        updated_at = ?, finished_at = ?, error = ? WHERE job_id = ?
                    """,
                    (
                        state.value,
                        now,
                        now if failed else None,
                        error,
                        row["job_id"],
                    ),
                )
                self._append_audit(
                    connection,
                    tenant_id=str(row["tenant_id"]),
                    action="job.lease_recovered",
                    subject=str(row["job_id"]),
                    metadata={"new_state": state.value},
                )
        return len(rows)

    def claim_job(self, *, worker_id: str, lease_seconds: int = 30) -> JobRecord | None:
        with self._transaction() as connection:
            candidates = connection.execute(
                "SELECT * FROM jobs WHERE state = 'queued' ORDER BY created_at LIMIT 100"
            ).fetchall()
            row = None
            for candidate in candidates:
                quota_row = connection.execute(
                    "SELECT quota_json, status FROM tenants WHERE tenant_id = ?",
                    (candidate["tenant_id"],),
                ).fetchone()
                if quota_row is None or quota_row["status"] != TenantStatus.ACTIVE.value:
                    continue
                quota = TenantQuota.from_dict(json.loads(quota_row["quota_json"]))
                active = connection.execute(
                    "SELECT COUNT(*) AS count FROM jobs WHERE tenant_id = ? AND state = 'running'",
                    (candidate["tenant_id"],),
                ).fetchone()
                if int(active["count"]) < quota.max_active_jobs:
                    row = candidate
                    break
            if row is None:
                return None
            now = _now()
            connection.execute(
                """
                UPDATE jobs SET state = ?, started_at = COALESCE(started_at, ?),
                    updated_at = ?, attempt = attempt + 1, lease_owner = ?,
                    lease_expires_at = ? WHERE job_id = ? AND state = 'queued'
                """,
                (
                    JobState.RUNNING.value,
                    now.isoformat(),
                    now.isoformat(),
                    worker_id,
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    row["job_id"],
                ),
            )
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            assert claimed is not None
            self._append_audit(
                connection,
                tenant_id=str(claimed["tenant_id"]),
                action="job.started",
                subject=str(claimed["job_id"]),
                metadata={"worker_id": worker_id, "attempt": int(claimed["attempt"])},
            )
        return self._job_from_row(claimed)

    def heartbeat_job(self, job_id: str, *, worker_id: str, lease_seconds: int = 30) -> bool:
        now = _now()
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET lease_expires_at = ?, updated_at = ?
                WHERE job_id = ? AND state = 'running' AND lease_owner = ?
                """,
                (
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                    now.isoformat(),
                    job_id,
                    worker_id,
                ),
            )
        return cursor.rowcount == 1

    def finish_job(
        self,
        job_id: str,
        *,
        worker_id: str,
        state: JobState,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> JobRecord:
        if state not in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELED}:
            raise ValueError("finish_job requires a terminal state")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ? AND lease_owner = ?",
                (job_id, worker_id),
            ).fetchone()
            if row is None:
                raise KeyError("worker does not own this job lease")
            now = _iso()
            connection.execute(
                """
                UPDATE jobs SET state = ?, updated_at = ?, finished_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL,
                    result_json = ?, error = ? WHERE job_id = ?
                """,
                (
                    state.value,
                    now,
                    now,
                    _canonical_json(result) if result is not None else None,
                    error,
                    job_id,
                ),
            )
            self._append_audit(
                connection,
                tenant_id=str(row["tenant_id"]),
                action=f"job.{state.value}",
                subject=job_id,
                metadata={"error": error} if error else {},
            )
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        assert updated is not None
        return self._job_from_row(updated)

    def list_usage(self, tenant_id: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, operation, quantity, unit, metadata_json, created_at
                FROM usage_events WHERE tenant_id = ?
                ORDER BY sequence DESC LIMIT ?
                """,
                (tenant_id, max(1, min(limit, 10000))),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "operation": row["operation"],
                "quantity": float(row["quantity"]),
                "unit": row["unit"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def set_consent(
        self,
        tenant_id: str,
        *,
        purpose: str,
        granted: bool,
        policy_version: str,
    ) -> None:
        if purpose not in ALLOWED_CONSENT_PURPOSES:
            raise ValueError(f"unsupported consent purpose: {purpose}")
        if not policy_version.strip() or len(policy_version) > 100:
            raise ValueError("policy_version must contain 1 to 100 characters")
        now = _iso()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO consent_grants(
                    tenant_id, purpose, policy_version, state, granted_at, revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, purpose) DO UPDATE SET
                    policy_version = excluded.policy_version,
                    state = excluded.state,
                    granted_at = excluded.granted_at,
                    revoked_at = excluded.revoked_at
                """,
                (
                    tenant_id,
                    purpose,
                    policy_version,
                    "granted" if granted else "revoked",
                    now if granted else None,
                    None if granted else now,
                ),
            )
            if not granted:
                connection.execute(
                    "DELETE FROM aggregate_observations WHERE tenant_id = ? AND purpose = ?",
                    (tenant_id, purpose),
                )
            self._append_audit(
                connection,
                tenant_id=tenant_id,
                action="consent.granted" if granted else "consent.revoked",
                subject=purpose,
                metadata={"policy_version": policy_version},
            )

    def record_aggregate_observation(
        self,
        tenant_id: str,
        *,
        purpose: str,
        metric_name: str,
        metric_value: float,
        context: dict[str, Any],
        source_receipt_id: str,
    ) -> str:
        if purpose not in ALLOWED_CONSENT_PURPOSES:
            raise ValueError(f"unsupported consent purpose: {purpose}")
        if metric_name not in ALLOWED_AGGREGATE_METRICS:
            raise ValueError(f"unsupported aggregate metric: {metric_name}")
        if not math.isfinite(float(metric_value)):
            raise ValueError("aggregate metric values must be finite")
        unknown_context = set(context) - set(ALLOWED_AGGREGATE_CONTEXT)
        if unknown_context:
            raise ValueError(f"unsupported aggregate context fields: {sorted(unknown_context)}")
        normalized_context = {key: str(value) for key, value in sorted(context.items())}
        for key, value in normalized_context.items():
            if value not in ALLOWED_AGGREGATE_CONTEXT[key]:
                raise ValueError(f"unsupported {key} aggregate context: {value}")
        if len(source_receipt_id) != 64 or any(
            character not in "0123456789abcdef" for character in source_receipt_id
        ):
            raise ValueError("source_receipt_id must be a lowercase SHA-256 identifier")
        with self._transaction() as connection:
            consent = connection.execute(
                """
                SELECT state, policy_version FROM consent_grants
                WHERE tenant_id = ? AND purpose = ?
                """,
                (tenant_id, purpose),
            ).fetchone()
            if consent is None or consent["state"] != "granted":
                raise PermissionError("active consent is required before recording observations")
            observation_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO aggregate_observations(
                    observation_id, tenant_id, purpose, metric_name, metric_value,
                    context_json, source_receipt_id, policy_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation_id,
                    tenant_id,
                    purpose,
                    metric_name,
                    float(metric_value),
                    _canonical_json(normalized_context),
                    source_receipt_id,
                    str(consent["policy_version"]),
                    _iso(),
                ),
            )
            self._append_audit(
                connection,
                tenant_id=tenant_id,
                action="aggregate.recorded",
                subject=observation_id,
                metadata={"purpose": purpose, "metric_name": metric_name},
            )
        return observation_id

    def aggregate_snapshot(self, *, purpose: str, min_tenants: int = 5) -> dict[str, Any]:
        if purpose not in ALLOWED_CONSENT_PURPOSES:
            raise ValueError(f"unsupported consent purpose: {purpose}")
        if min_tenants < 2:
            raise ValueError("aggregate snapshots require at least two tenants")
        with self._connect() as connection:
            rows = connection.execute(
                """
                WITH per_tenant AS (
                    SELECT tenant_id, metric_name, context_json, policy_version,
                        COUNT(*) AS tenant_observation_count,
                        AVG(metric_value) AS tenant_mean
                    FROM aggregate_observations
                    WHERE purpose = ?
                    GROUP BY tenant_id, metric_name, context_json, policy_version
                )
                SELECT metric_name, context_json, policy_version,
                    COUNT(*) AS tenant_count,
                    SUM(tenant_observation_count) AS observation_count,
                    AVG(tenant_mean) AS mean_value,
                    MIN(tenant_mean) AS min_value,
                    MAX(tenant_mean) AS max_value
                FROM per_tenant
                GROUP BY metric_name, context_json, policy_version
                HAVING COUNT(*) >= ?
                ORDER BY metric_name, context_json
                """,
                (purpose, min_tenants),
            ).fetchall()
        aggregates = [
            {
                "metric_name": row["metric_name"],
                "context": json.loads(row["context_json"]),
                "policy_version": row["policy_version"],
                "tenant_count": int(row["tenant_count"]),
                "observation_count": int(row["observation_count"]),
                "mean": float(row["mean_value"]),
                "min": float(row["min_value"]),
                "max": float(row["max_value"]),
            }
            for row in rows
        ]
        payload = {
            "schema_version": 2,
            "purpose": purpose,
            "min_tenants": min_tenants,
            "aggregates": aggregates,
        }
        payload["snapshot_id"] = sha256(_canonical_json(payload).encode()).hexdigest()
        return payload

    def verify_audit_chain(self, tenant_id: str) -> tuple[bool, tuple[str, ...]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events WHERE tenant_id = ? ORDER BY sequence",
                (tenant_id,),
            ).fetchall()
        mismatches: list[str] = []
        previous_hash = "0" * 64
        for row in rows:
            if row["previous_hash"] != previous_hash:
                mismatches.append(f"audit sequence {row['sequence']} previous hash mismatch")
            expected = sha256(
                _canonical_json(
                    {
                        "event_id": row["event_id"],
                        "tenant_id": row["tenant_id"],
                        "action": row["action"],
                        "subject": row["subject"],
                        "metadata": json.loads(row["metadata_json"]),
                        "created_at": row["created_at"],
                        "previous_hash": row["previous_hash"],
                    }
                ).encode()
            ).hexdigest()
            if expected != row["event_hash"]:
                mismatches.append(f"audit sequence {row['sequence']} event hash mismatch")
            previous_hash = str(row["event_hash"])
        return not mismatches, tuple(mismatches)

    def delete_tenant_records(self, tenant_id: str) -> None:
        """Delete tenant content while retaining only a pseudonymous audit tombstone."""
        with self._transaction() as connection:
            tenant = connection.execute(
                "SELECT status FROM tenants WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
            if tenant is None:
                raise KeyError("tenant not found")
            connection.execute(
                "DELETE FROM aggregate_observations WHERE tenant_id = ?", (tenant_id,)
            )
            connection.execute("DELETE FROM consent_grants WHERE tenant_id = ?", (tenant_id,))
            connection.execute("DELETE FROM usage_events WHERE tenant_id = ?", (tenant_id,))
            connection.execute("DELETE FROM jobs WHERE tenant_id = ?", (tenant_id,))
            connection.execute("DELETE FROM credentials WHERE tenant_id = ?", (tenant_id,))
            connection.execute(
                """
                UPDATE tenants SET display_name = '[deleted]', status = ?,
                    quota_json = '{}', updated_at = ? WHERE tenant_id = ?
                """,
                (TenantStatus.DELETED.value, _iso(), tenant_id),
            )
            self._append_audit(
                connection,
                tenant_id=tenant_id,
                action="tenant.data_deleted",
                subject=tenant_id,
                metadata={"retained": "pseudonymous audit chain only"},
            )
