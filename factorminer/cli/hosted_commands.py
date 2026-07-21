"""Operator commands for the isolated hosted-pilot control plane."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click

from factorminer.cli.context import main


def _service(db: Path, storage_root: Path):
    from factorminer.hosted.service import HostedPilotService
    from factorminer.hosted.store import HostedStore

    service = HostedPilotService(HostedStore(db), storage_root)
    service.initialize()
    return service


def _control_paths(function):
    function = click.option(
        "--storage-root",
        required=True,
        type=click.Path(file_okay=False, path_type=Path),
    )(function)
    function = click.option(
        "--db",
        required=True,
        type=click.Path(dir_okay=False, path_type=Path),
    )(function)
    return function


@main.group("hosted-pilot")
def hosted_pilot() -> None:
    """Operate the tenant-isolated hosted research pilot."""


@hosted_pilot.command("init")
@_control_paths
def init_hosted(db: Path, storage_root: Path) -> None:
    """Initialize the durable DB and storage root."""
    _service(db, storage_root)
    click.echo("Hosted pilot initialized")


@hosted_pilot.command("tenant-create")
@click.argument("tenant_id")
@click.option("--display-name", required=True)
@click.option("--retention-days", type=click.IntRange(1, 3650), default=30)
@click.option(
    "--quota-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
@_control_paths
def tenant_create(
    tenant_id: str,
    display_name: str,
    retention_days: int,
    quota_json: Path | None,
    db: Path,
    storage_root: Path,
) -> None:
    """Create an isolated tenant workspace."""
    from factorminer.hosted.models import TenantQuota

    quota = TenantQuota.from_dict(json.loads(quota_json.read_text())) if quota_json else None
    tenant = _service(db, storage_root).create_tenant(
        tenant_id,
        display_name=display_name,
        quota=quota,
        retention_days=retention_days,
    )
    click.echo(json.dumps(tenant, indent=2, sort_keys=True))


@hosted_pilot.command("credential-create")
@click.argument("tenant_id")
@click.option("--label", required=True)
@click.option("--scope", "scopes", multiple=True, required=True)
@click.option("--expires-days", type=click.IntRange(1, 3650), default=90)
@_control_paths
def credential_create(
    tenant_id: str,
    label: str,
    scopes: tuple[str, ...],
    expires_days: int,
    db: Path,
    storage_root: Path,
) -> None:
    """Issue a revocable scoped token. The token is printed only now."""
    issued = _service(db, storage_root).issue_credential(
        tenant_id,
        label=label,
        scopes=scopes,
        expires_at=datetime.now(UTC) + timedelta(days=expires_days),
    )
    click.echo(json.dumps(issued.__dict__, indent=2, sort_keys=True))


@hosted_pilot.command("credential-revoke")
@click.argument("tenant_id")
@click.argument("credential_id")
@_control_paths
def credential_revoke(tenant_id: str, credential_id: str, db: Path, storage_root: Path) -> None:
    """Revoke a tenant credential immediately."""
    _service(db, storage_root).store.revoke_credential(tenant_id, credential_id)
    click.echo("Credential revoked")


@hosted_pilot.command("put-input")
@click.argument("tenant_id")
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--name", default=None)
@_control_paths
def put_input(
    tenant_id: str,
    source: Path,
    name: str | None,
    db: Path,
    storage_root: Path,
) -> None:
    """Provision an operator-reviewed file into one tenant input namespace."""
    result = _service(db, storage_root).put_input(tenant_id, source, destination_name=name)
    click.echo(json.dumps(result, indent=2, sort_keys=True))


@hosted_pilot.command("worker")
@click.option("--once", is_flag=True, help="Process at most one queued job.")
@click.option("--poll-seconds", type=click.FloatRange(min=0.1), default=1.0)
@click.option("--worker-id", default=None)
@click.option(
    "--provider-secret-env",
    "provider_secret_env",
    multiple=True,
    type=click.Choice(["ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]),
    help="Explicitly forward one operator-managed provider credential by environment name.",
)
@_control_paths
def worker(
    once: bool,
    poll_seconds: float,
    worker_id: str | None,
    provider_secret_env: tuple[str, ...],
    db: Path,
    storage_root: Path,
) -> None:
    """Run a crash-recoverable allow-listed job worker."""
    from factorminer.hosted.worker import HostedWorker

    hosted_worker = HostedWorker(
        _service(db, storage_root),
        worker_id=worker_id,
        provider_secret_env=provider_secret_env,
    )
    if once:
        result = hosted_worker.run_once()
        click.echo(json.dumps(result.to_dict() if result else None, indent=2, sort_keys=True))
        return
    hosted_worker.run_forever(poll_seconds=poll_seconds)


@hosted_pilot.command("usage-export")
@click.argument("tenant_id")
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@_control_paths
def usage_export(tenant_id: str, output: Path, db: Path, storage_root: Path) -> None:
    """Export append-only billing-neutral usage records as JSON Lines."""
    click.echo(str(_service(db, storage_root).export_usage(tenant_id, output)))


@hosted_pilot.command("audit-verify")
@click.argument("tenant_id")
@_control_paths
def audit_verify(tenant_id: str, db: Path, storage_root: Path) -> None:
    """Verify the tenant audit hash chain."""
    passed, mismatches = _service(db, storage_root).store.verify_audit_chain(tenant_id)
    for mismatch in mismatches:
        click.echo(mismatch, err=True)
    if not passed:
        raise SystemExit(1)
    click.echo("Audit chain verification passed")


@hosted_pilot.command("consensus-snapshot")
@click.argument("purpose")
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--min-tenants", type=click.IntRange(2, 1000), default=5, show_default=True)
@_control_paths
def consensus_snapshot(
    purpose: str,
    output: Path,
    min_tenants: int,
    db: Path,
    storage_root: Path,
) -> None:
    """Publish a k-anonymous aggregate snapshot without tenant identifiers."""
    payload = _service(db, storage_root).store.aggregate_snapshot(
        purpose=purpose, min_tenants=min_tenants
    )
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != content:
        raise click.ClickException("refusing to overwrite divergent consensus snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content)
    click.echo(str(output))


@hosted_pilot.command("tenant-delete")
@click.argument("tenant_id")
@click.option(
    "--confirm",
    required=True,
    help="Must exactly equal tenant_id. Deletes tenant content and retains only an audit tombstone.",
)
@_control_paths
def tenant_delete(tenant_id: str, confirm: str, db: Path, storage_root: Path) -> None:
    """Irreversibly delete one tenant workspace and control-plane records."""
    _service(db, storage_root).delete_tenant(tenant_id, confirmation=confirm)
    click.echo("Tenant data deleted; pseudonymous audit tombstone retained")


@hosted_pilot.command("retention-purge")
@click.argument("tenant_id")
@click.option(
    "--apply",
    is_flag=True,
    help="Delete the listed expired terminal-job artifacts. The default is a dry run.",
)
@_control_paths
def retention_purge(
    tenant_id: str,
    apply: bool,
    db: Path,
    storage_root: Path,
) -> None:
    """Preview or enforce one tenant's configured artifact retention period."""
    paths = _service(db, storage_root).purge_retention(tenant_id, apply=apply)
    click.echo(json.dumps({"applied": apply, "paths": paths}, indent=2, sort_keys=True))


@hosted_pilot.command("serve")
@click.option("--issuer-url", required=True)
@click.option("--resource-url", required=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=click.IntRange(1, 65535), default=8766, show_default=True)
@_control_paths
def serve(
    issuer_url: str,
    resource_url: str,
    host: str,
    port: int,
    db: Path,
    storage_root: Path,
) -> None:
    """Run the scoped stateless MCP resource server."""
    from factorminer.hosted.mcp_server import run_hosted_mcp

    run_hosted_mcp(
        _service(db, storage_root),
        issuer_url=issuer_url,
        resource_url=resource_url,
        host=host,
        port=port,
    )


__all__ = ["hosted_pilot"]
