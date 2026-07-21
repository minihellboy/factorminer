"""Adversarial isolation and durability coverage for the hosted pilot."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from factorminer.hosted.models import (
    ALLOWED_SCOPES,
    JobKind,
    JobState,
    TenantQuota,
)
from factorminer.hosted.service import HostedPilotService
from factorminer.hosted.store import HostedStore
from factorminer.hosted.worker import HostedWorker


def _service(tmp_path: Path, *, quota: TenantQuota | None = None):
    service = HostedPilotService(HostedStore(tmp_path / "control.sqlite3"), tmp_path / "storage")
    service.initialize()
    service.create_tenant("tenant-a", display_name="Tenant A", quota=quota)
    service.create_tenant("tenant-b", display_name="Tenant B", quota=quota)
    issued_a = service.issue_credential("tenant-a", label="a", scopes=tuple(sorted(ALLOWED_SCOPES)))
    issued_b = service.issue_credential("tenant-b", label="b", scopes=tuple(sorted(ALLOWED_SCOPES)))
    principal_a = service.authenticate(issued_a.token)
    principal_b = service.authenticate(issued_b.token)
    assert principal_a is not None and principal_b is not None
    return service, issued_a, issued_b, principal_a, principal_b


def _market_csv(path: Path, *, asset: str = "BTCUSDT") -> Path:
    path.write_text(
        "datetime,asset_id,open,high,low,close,volume,amount\n"
        f"2024-01-01T00:00:00Z,{asset},1,2,0.5,1.5,10,15\n"
        f"2024-01-02T00:00:00Z,{asset},1.5,2.5,1,2,11,22\n"
    )
    return path


def test_two_tenants_cannot_cross_read_jobs_inputs_or_artifacts(tmp_path: Path) -> None:
    service, _, _, principal_a, principal_b = _service(tmp_path)
    source_a = _market_csv(tmp_path / "a.csv", asset="BTCUSDT")
    source_b = _market_csv(tmp_path / "b.csv", asset="ETHUSDT")
    service.put_input("tenant-a", source_a, destination_name="market.csv")
    service.put_input("tenant-b", source_b, destination_name="market.csv")
    assert (
        service.resolve_tenant_path("tenant-a", "inputs/market.csv").read_text()
        != service.resolve_tenant_path("tenant-b", "inputs/market.csv").read_text()
    )
    job_a = service.submit_job(
        principal_a,
        kind=JobKind.VALIDATE_DATA,
        parameters={"input_path": "inputs/market.csv"},
        idempotency_key="validate-a",
    )
    job_b = service.submit_job(
        principal_b,
        kind=JobKind.VALIDATE_DATA,
        parameters={"input_path": "inputs/market.csv"},
        idempotency_key="validate-b",
    )
    with pytest.raises(KeyError, match="job not found"):
        service.get_job(principal_a, job_b.job_id)
    with pytest.raises(KeyError, match="job not found"):
        service.list_artifacts(principal_a, job_b.job_id)
    assert service.get_job(principal_a, job_a.job_id).tenant_id == "tenant-a"
    with pytest.raises(ValueError, match="tenant-relative"):
        service.resolve_tenant_path("tenant-a", "../tenant-b/inputs/market.csv")
    with pytest.raises(ValueError, match="tenant-relative"):
        service.resolve_tenant_path("tenant-a", "/etc/passwd")


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    service, *_ = _service(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = service.tenant_root("tenant-a") / "inputs" / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="Symbolic|symbolic"):
        service.resolve_tenant_path("tenant-a", "inputs/link.txt", must_exist=True)


def test_scoped_expiring_and_revocable_credentials(tmp_path: Path) -> None:
    service, _, _, _, _ = _service(tmp_path)
    limited = service.issue_credential("tenant-a", label="read-only", scopes=("jobs:read",))
    principal = service.authenticate(limited.token)
    assert principal is not None
    with pytest.raises(PermissionError, match="jobs:submit"):
        service.submit_job(
            principal,
            kind=JobKind.VALIDATE_DATA,
            parameters={"input_path": "inputs/missing.csv"},
            idempotency_key="nope",
        )
    service.store.revoke_credential("tenant-a", limited.credential_id)
    assert service.authenticate(limited.token) is None

    expired = service.issue_credential(
        "tenant-a",
        label="expired",
        scopes=("jobs:read",),
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    assert service.authenticate(expired.token) is None


def test_idempotency_quota_and_cancel_are_fail_closed(tmp_path: Path) -> None:
    quota = TenantQuota(max_queued_jobs=1, requests_per_minute=20)
    service, _, _, principal_a, _ = _service(tmp_path, quota=quota)
    service.put_input("tenant-a", _market_csv(tmp_path / "a.csv"))
    first = service.submit_job(
        principal_a,
        kind="validate_data",
        parameters={"input_path": "inputs/a.csv"},
        idempotency_key="same-key",
    )
    repeated = service.submit_job(
        principal_a,
        kind="validate_data",
        parameters={"input_path": "inputs/a.csv"},
        idempotency_key="same-key",
    )
    assert repeated.job_id == first.job_id
    with pytest.raises(ValueError, match="different input"):
        service.submit_job(
            principal_a,
            kind="validate_data",
            parameters={"input_path": "inputs/a.csv", "strict": False},
            idempotency_key="same-key",
        )
    with pytest.raises(RuntimeError, match="queued-job quota"):
        service.submit_job(
            principal_a,
            kind="validate_data",
            parameters={"input_path": "inputs/a.csv"},
            idempotency_key="second-key",
        )
    canceled = service.cancel_job(principal_a, first.job_id)
    assert canceled.state == JobState.CANCELED


def test_compute_reservations_and_request_rate_limits_are_fail_closed(tmp_path: Path) -> None:
    quota = TenantQuota(
        max_runtime_seconds=10,
        compute_seconds_per_day=10,
        max_queued_jobs=5,
        requests_per_minute=3,
    )
    service, _, _, principal_a, _ = _service(tmp_path, quota=quota)
    service.put_input("tenant-a", _market_csv(tmp_path / "a.csv"))
    first = service.submit_job(
        principal_a,
        kind="validate_data",
        parameters={"input_path": "inputs/a.csv"},
        idempotency_key="reserved-six",
        timeout_seconds=6,
    )
    with pytest.raises(RuntimeError, match="compute quota reservation"):
        service.submit_job(
            principal_a,
            kind="validate_data",
            parameters={"input_path": "inputs/a.csv"},
            idempotency_key="reserved-five",
            timeout_seconds=5,
        )
    assert service.get_job(principal_a, first.job_id).job_id == first.job_id
    with pytest.raises(RuntimeError, match="request rate limit"):
        service.get_job(principal_a, first.job_id)


def test_worker_runs_allowlisted_job_and_records_artifacts_usage_and_audit(tmp_path: Path) -> None:
    service, _, _, principal_a, _ = _service(tmp_path)
    service.put_input("tenant-a", _market_csv(tmp_path / "a.csv"))
    submitted = service.submit_job(
        principal_a,
        kind=JobKind.VALIDATE_DATA,
        parameters={"input_path": "inputs/a.csv", "strict": True},
        idempotency_key="worker-validation",
        timeout_seconds=60,
    )
    result = HostedWorker(service, worker_id="worker-test", lease_seconds=10).run_once()
    assert result is not None
    assert result.job_id == submitted.job_id
    assert result.state == JobState.SUCCEEDED, result.error
    artifacts = service.list_artifacts(principal_a, submitted.job_id)
    paths = {item["path"] for item in artifacts}
    assert f"jobs/{submitted.job_id}/stdout.log" in paths
    stdout = service.read_text_artifact(
        principal_a, submitted.job_id, f"jobs/{submitted.job_id}/stdout.log"
    )
    assert '"status": "valid"' in stdout["text"]
    usage = service.usage(principal_a)
    assert any(item["unit"] == "compute_second" for item in usage)
    passed, mismatches = service.store.verify_audit_chain("tenant-a")
    assert passed is True, mismatches


def test_mining_job_has_bounded_schema_and_allowlisted_argv(tmp_path: Path) -> None:
    service, _, _, principal_a, _ = _service(tmp_path)
    service.put_input("tenant-a", _market_csv(tmp_path / "a.csv"))
    submitted = service.submit_job(
        principal_a,
        kind=JobKind.MINE,
        parameters={
            "input_path": "inputs/a.csv",
            "loop": "helix",
            "iterations": 5,
            "batch_size": 4,
            "target": 2,
        },
        idempotency_key="mining-job",
    )
    claimed = service.store.claim_job(worker_id="worker-mining", lease_seconds=30)
    assert claimed is not None and claimed.job_id == submitted.job_id
    worker = HostedWorker(service, worker_id="worker-mining")
    command = worker._command(claimed, worker._job_directory(claimed))
    assert "helix" in command
    assert "--data" in command
    assert command[command.index("--iterations") + 1] == "5"
    assert all(";" not in argument for argument in command)

    with pytest.raises(ValueError, match="between 1 and 200"):
        service.submit_job(
            principal_a,
            kind=JobKind.MINE,
            parameters={"input_path": "inputs/a.csv", "loop": "ralph", "iterations": 1000},
            idempotency_key="unbounded-mining-job",
        )


def test_worker_claim_respects_per_tenant_active_job_quota(tmp_path: Path) -> None:
    quota = TenantQuota(max_active_jobs=1, max_queued_jobs=5)
    (
        service,
        _,
        _,
        principal_a,
        _,
    ) = _service(tmp_path, quota=quota)
    service.put_input("tenant-a", _market_csv(tmp_path / "a.csv"))
    for index in range(2):
        service.submit_job(
            principal_a,
            kind="validate_data",
            parameters={"input_path": "inputs/a.csv"},
            idempotency_key=f"job-{index}",
        )
    claimed = service.store.claim_job(worker_id="worker-a", lease_seconds=30)
    assert claimed is not None
    assert service.store.claim_job(worker_id="worker-b", lease_seconds=30) is None


def test_consent_required_aggregate_is_k_anonymous_and_revocable(tmp_path: Path) -> None:
    service, _, _, principal_a, principal_b = _service(tmp_path)
    with pytest.raises(PermissionError, match="active consent"):
        service.record_aggregate_observation(
            principal_a,
            purpose="aggregate_workflow_metrics",
            metric_name="setup_minutes",
            metric_value=10,
            context={"asset_class": "crypto"},
            source_receipt_id="a" * 64,
        )
    for principal, value, receipt in (
        (principal_a, 10.0, "a" * 64),
        (principal_b, 20.0, "b" * 64),
    ):
        service.set_consent(
            principal,
            purpose="aggregate_workflow_metrics",
            granted=True,
            policy_version="v1",
        )
        service.record_aggregate_observation(
            principal,
            purpose="aggregate_workflow_metrics",
            metric_name="setup_minutes",
            metric_value=value,
            context={"asset_class": "crypto"},
            source_receipt_id=receipt,
        )
    snapshot = service.store.aggregate_snapshot(purpose="aggregate_workflow_metrics", min_tenants=2)
    assert snapshot["aggregates"][0]["mean"] == 15.0
    assert snapshot["aggregates"][0]["policy_version"] == "v1"
    assert snapshot["schema_version"] == 2
    assert "tenant-a" not in json.dumps(snapshot)

    service.set_consent(
        principal_a,
        purpose="aggregate_workflow_metrics",
        granted=False,
        policy_version="v1",
    )
    after_revoke = service.store.aggregate_snapshot(
        purpose="aggregate_workflow_metrics", min_tenants=2
    )
    assert after_revoke["aggregates"] == []


def test_aggregate_learning_rejects_unbounded_or_identifying_fields(tmp_path: Path) -> None:
    service, _, _, principal_a, _ = _service(tmp_path)
    service.set_consent(
        principal_a,
        purpose="aggregate_workflow_metrics",
        granted=True,
        policy_version="v1",
    )
    with pytest.raises(ValueError, match="context fields"):
        service.record_aggregate_observation(
            principal_a,
            purpose="aggregate_workflow_metrics",
            metric_name="setup_minutes",
            metric_value=10,
            context={"customer_name": "identifying"},
            source_receipt_id="a" * 64,
        )
    with pytest.raises(ValueError, match="finite"):
        service.record_aggregate_observation(
            principal_a,
            purpose="aggregate_workflow_metrics",
            metric_name="setup_minutes",
            metric_value=float("nan"),
            context={"asset_class": "crypto"},
            source_receipt_id="a" * 64,
        )


def test_tenant_deletion_removes_workspace_tokens_and_records(tmp_path: Path) -> None:
    service, issued_a, _, _, _ = _service(tmp_path)
    service.put_input("tenant-a", _market_csv(tmp_path / "a.csv"))
    with pytest.raises(ValueError, match="exactly match"):
        service.delete_tenant("tenant-a", confirmation="wrong")
    service.delete_tenant("tenant-a", confirmation="tenant-a")
    assert not service.tenant_root("tenant-a").exists()
    assert service.authenticate(issued_a.token) is None
    tenant = service.store.get_tenant("tenant-a")
    assert tenant["status"] == "deleted"
    passed, mismatches = service.store.verify_audit_chain("tenant-a")
    assert passed is True, mismatches


def test_audit_chain_detects_database_tampering(tmp_path: Path) -> None:
    service, *_ = _service(tmp_path)
    with sqlite3.connect(service.store.path) as connection:
        connection.execute(
            "UPDATE audit_events SET metadata_json = ? WHERE tenant_id = ? AND sequence = (SELECT MIN(sequence) FROM audit_events WHERE tenant_id = ?)",
            ('{"tampered":true}', "tenant-a", "tenant-a"),
        )
    passed, mismatches = service.store.verify_audit_chain("tenant-a")
    assert passed is False
    assert any("event hash mismatch" in mismatch for mismatch in mismatches)


def test_hosted_token_verifier_binds_scopes_and_resource(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    from factorminer.hosted.mcp_server import HostedTokenVerifier, build_hosted_mcp

    service, issued_a, _, _, _ = _service(tmp_path)
    verifier = HostedTokenVerifier(service, resource_url="https://pilot.example/mcp")
    access = asyncio.run(verifier.verify_token(issued_a.token))
    assert access is not None
    assert access.client_id == "tenant-a"
    assert access.resource == "https://pilot.example/mcp"
    assert set(access.scopes) == ALLOWED_SCOPES
    assert asyncio.run(verifier.verify_token("wrong")) is None

    server = build_hosted_mcp(
        service,
        issuer_url="https://auth.example/",
        resource_url="https://pilot.example/mcp",
    )
    tool_names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert "submit_job" in tool_names
    assert "read_text_artifact" in tool_names


def test_retention_purge_is_dry_run_first_and_audited(tmp_path: Path) -> None:
    service, _, _, principal_a, _ = _service(tmp_path)
    service.put_input("tenant-a", _market_csv(tmp_path / "a.csv"))
    submitted = service.submit_job(
        principal_a,
        kind=JobKind.VALIDATE_DATA,
        parameters={"input_path": "inputs/a.csv"},
        idempotency_key="expired-job",
    )
    result = HostedWorker(service, worker_id="retention-test").run_once()
    assert result is not None and result.state == JobState.SUCCEEDED
    expired = (datetime.now(UTC) - timedelta(days=31)).isoformat()
    with sqlite3.connect(service.store.path) as connection:
        connection.execute(
            "UPDATE jobs SET finished_at = ? WHERE job_id = ?",
            (expired, submitted.job_id),
        )
    expected = [f"jobs/{submitted.job_id}"]
    assert service.purge_retention("tenant-a") == expected
    assert service.resolve_tenant_path("tenant-a", expected[0]).exists()
    assert service.purge_retention("tenant-a", apply=True) == expected
    assert not service.resolve_tenant_path("tenant-a", expected[0]).exists()
    assert service.purge_retention("tenant-a") == []
    passed, mismatches = service.store.verify_audit_chain("tenant-a")
    assert passed is True, mismatches


def test_worker_fails_closed_when_output_exceeds_storage_quota(tmp_path: Path) -> None:
    quota = TenantQuota(max_storage_bytes=500, max_input_bytes=300)
    service, _, _, principal_a, _ = _service(tmp_path, quota=quota)
    service.put_input("tenant-a", _market_csv(tmp_path / "a.csv"))
    remaining = quota.max_storage_bytes - service.tenant_storage_bytes("tenant-a") - 1
    (service.tenant_root("tenant-a") / "inputs" / "filler.bin").write_bytes(b"x" * remaining)
    submitted = service.submit_job(
        principal_a,
        kind=JobKind.VALIDATE_DATA,
        parameters={"input_path": "inputs/a.csv"},
        idempotency_key="storage-limited",
    )
    result = HostedWorker(service, worker_id="storage-test").run_once()
    assert result is not None
    assert result.job_id == submitted.job_id
    assert result.state == JobState.FAILED
    assert result.error == "job exceeded tenant storage quota"
