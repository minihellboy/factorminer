"""Model Context Protocol surface for FactorMiner.

This subpackage exposes the FactorMiner research engine as MCP tools so that any
Claude agent -- Claude Code, a Cowork plugin, or a Managed Agent deployed via
``POST /v1/agents`` -- can mine, evaluate, backtest, benchmark, and report on
alpha factors without importing FactorMiner internals.

The server itself lives in :mod:`factorminer.mcp.server`. It is imported lazily
so that ``factorminer`` keeps working when the optional ``mcp`` dependency is
not installed; install it with ``pip install 'factorminer[mcp]'``.
"""

__all__ = ["build_server"]


def build_server():
    """Return the FactorMiner :class:`FastMCP` server instance.

    Imported lazily so the optional ``mcp`` dependency is only required when the
    server is actually launched (via ``factorminer mcp-serve``).
    """
    from factorminer.mcp.server import mcp

    return mcp
