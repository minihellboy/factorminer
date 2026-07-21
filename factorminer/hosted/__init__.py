"""Tenant-isolated hosted-pilot control plane.

The hosted surface is deliberately separate from ``factorminer mcp-serve``.
The local MCP server exposes synchronous filesystem tools and must never be
reused as a public multi-tenant endpoint.
"""

from factorminer.hosted.models import (
    ALLOWED_SCOPES,
    JobKind,
    JobState,
    TenantQuota,
    TenantStatus,
)
from factorminer.hosted.service import HostedPilotService
from factorminer.hosted.store import HostedStore

__all__ = [
    "ALLOWED_SCOPES",
    "HostedPilotService",
    "HostedStore",
    "JobKind",
    "JobState",
    "TenantQuota",
    "TenantStatus",
]
