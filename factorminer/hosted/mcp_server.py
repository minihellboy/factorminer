"""Scoped MCP resource server for the tenant-isolated hosted pilot."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from factorminer.hosted.models import JobKind, Principal
from factorminer.hosted.service import HostedPilotService


def _mcp_imports():
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token
        from mcp.server.auth.provider import AccessToken
        from mcp.server.auth.settings import AuthSettings
        from mcp.server.fastmcp import FastMCP
        from pydantic import AnyHttpUrl
    except ModuleNotFoundError as exc:  # pragma: no cover - operator-facing
        raise ModuleNotFoundError(
            "The hosted MCP server requires the mcp extra: pip install 'factorminer[mcp]'"
        ) from exc
    return get_access_token, AccessToken, AuthSettings, FastMCP, AnyHttpUrl


class HostedTokenVerifier:
    """Validate high-entropy opaque pilot tokens against the control-plane DB."""

    def __init__(self, service: HostedPilotService, *, resource_url: str) -> None:
        self.service = service
        self.resource_url = resource_url

    async def verify_token(self, token: str):
        _, AccessToken, _, _, _ = _mcp_imports()
        principal = self.service.authenticate(token)
        if principal is None:
            return None
        return AccessToken(
            token=token,
            client_id=principal.tenant_id,
            scopes=list(principal.scopes),
            expires_at=principal.expires_at,
            resource=self.resource_url,
        )


def _request_principal(service: HostedPilotService) -> Principal:
    get_access_token, _, _, _, _ = _mcp_imports()
    access_token = get_access_token()
    if access_token is None:
        raise PermissionError("hosted MCP tools require an authenticated access token")
    principal = service.authenticate(access_token.token)
    if principal is None:
        raise PermissionError("hosted MCP access token is invalid, expired, or revoked")
    if principal.tenant_id != access_token.client_id:
        raise PermissionError("authenticated tenant binding mismatch")
    return principal


def build_hosted_mcp(
    service: HostedPilotService,
    *,
    issuer_url: str,
    resource_url: str,
    host: str = "127.0.0.1",
    port: int = 8766,
):
    """Build a stateless resource server containing only tenant-safe tools."""
    _, _, AuthSettings, FastMCP, AnyHttpUrl = _mcp_imports()
    server = FastMCP(
        "factorminer-hosted-pilot",
        instructions=(
            "Tenant-isolated research job control. Jobs produce research artifacts only; "
            "the service never routes orders or operates an autonomous account."
        ),
        token_verifier=HostedTokenVerifier(service, resource_url=resource_url),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(issuer_url),
            resource_server_url=AnyHttpUrl(resource_url),
            required_scopes=None,
        ),
        host=host,
        port=port,
        stateless_http=True,
        json_response=True,
    )

    @server.tool()
    def submit_job(
        kind: str,
        parameters: dict[str, Any],
        idempotency_key: str,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Submit one allow-listed durable research job."""
        principal = _request_principal(service)
        return service.submit_job(
            principal,
            kind=JobKind(kind),
            parameters=parameters,
            idempotency_key=idempotency_key,
            timeout_seconds=timeout_seconds,
        ).to_dict()

    @server.tool()
    def get_job(job_id: str) -> dict[str, Any]:
        """Read one job belonging to the authenticated tenant."""
        return service.get_job(_request_principal(service), job_id).to_dict()

    @server.tool()
    def list_jobs(limit: int = 100) -> list[dict[str, Any]]:
        """List jobs belonging to the authenticated tenant."""
        return [
            job.to_dict() for job in service.list_jobs(_request_principal(service), limit=limit)
        ]

    @server.tool()
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel a queued job or request termination of a running job."""
        return service.cancel_job(_request_principal(service), job_id).to_dict()

    @server.tool()
    def list_artifacts(job_id: str) -> list[dict[str, Any]]:
        """List content hashes and sizes for artifacts from one tenant job."""
        return service.list_artifacts(_request_principal(service), job_id)

    @server.tool()
    def read_text_artifact(job_id: str, path: str) -> dict[str, Any]:
        """Read a bounded UTF-8 artifact belonging to one tenant job."""
        return service.read_text_artifact(_request_principal(service), job_id, path)

    @server.tool()
    def get_usage(limit: int = 1000) -> list[dict[str, Any]]:
        """Return billing-neutral usage events for the authenticated tenant."""
        return service.usage(_request_principal(service), limit=limit)

    @server.tool()
    def set_learning_consent(purpose: str, granted: bool, policy_version: str) -> dict[str, Any]:
        """Explicitly grant or revoke one aggregate-learning purpose."""
        service.set_consent(
            _request_principal(service),
            purpose=purpose,
            granted=granted,
            policy_version=policy_version,
        )
        return {"ok": True, "purpose": purpose, "granted": granted}

    @server.tool()
    def record_aggregate_observation(
        purpose: str,
        metric_name: str,
        metric_value: float,
        context: dict[str, Any],
        source_receipt_id: str,
    ) -> dict[str, Any]:
        """Record an allow-listed aggregate only when active consent exists."""
        observation_id = service.record_aggregate_observation(
            _request_principal(service),
            purpose=purpose,
            metric_name=metric_name,
            metric_value=metric_value,
            context=context,
            source_receipt_id=source_receipt_id,
        )
        return {"observation_id": observation_id}

    return server


def run_hosted_mcp(
    service: HostedPilotService,
    *,
    issuer_url: str,
    resource_url: str,
    host: str = "127.0.0.1",
    port: int = 8766,
) -> None:
    parsed_resource = urlparse(resource_url)
    if host not in {"127.0.0.1", "::1", "localhost"} and parsed_resource.scheme != "https":
        raise ValueError("non-loopback hosted MCP requires an https resource URL")
    service.initialize()
    server = build_hosted_mcp(
        service,
        issuer_url=issuer_url,
        resource_url=resource_url,
        host=host,
        port=port,
    )
    server.run(transport="streamable-http")
