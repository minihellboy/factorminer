"""Ingest market data from external MCP connectors.

FactorMiner's own MCP server (``factorminer.mcp.server``) exposes the engine
*outward*. This module is the other direction: an MCP **client** that pulls
OHLCV data *in* from a financial-data MCP connector -- FactSet, Daloopa,
Morningstar, LSEG, or any server that returns tabular price data -- and maps it
onto the canonical FactorMiner schema.

Because each connector defines its own tool names and field names, the adapter
is fully config-driven (see :class:`MCPDataSourceConfig`). Point it at a
connector with a small YAML file; nothing about the connector's schema is
hard-coded here.

The optional ``mcp`` dependency is imported lazily, so importing this module is
safe even when ``mcp`` is not installed.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# Canonical FactorMiner market-data columns. This mirrors the loader contract:
# fetched connector data must be written as a full OHLCV + amount panel so the
# next step (`validate-data` / `mine`) sees the same schema as local files.
CANONICAL_COLUMNS = (
    "datetime",
    "asset_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

KNOWN_MCP_CONNECTORS = (
    {
        "name": "factset",
        "url": "https://mcp.factset.com/mcp",
        "best_for": "market prices, fundamentals, estimates, identifiers",
    },
    {
        "name": "daloopa",
        "url": "https://mcp.daloopa.com/server/mcp",
        "best_for": "company financials, KPI extraction, normalized filings data",
    },
    {
        "name": "morningstar",
        "url": "https://mcp.morningstar.com/mcp",
        "best_for": "funds, holdings, performance, reference data",
    },
    {
        "name": "sp-global",
        "url": "https://kfinance.kensho.com/integrations/mcp",
        "best_for": "Capital IQ-style company, transaction, and market data",
    },
    {
        "name": "moodys",
        "url": "https://api.moodys.com/genai-ready-data/m1/mcp",
        "best_for": "credit, ratings, issuer, and macro risk data",
    },
    {
        "name": "mtnewswire",
        "url": "https://vast-mcp.blueskyapi.com/mtnewswires",
        "best_for": "market news and event context",
    },
    {
        "name": "aiera",
        "url": "https://mcp-pub.aiera.com",
        "best_for": "earnings calls, event transcripts, and corporate events",
    },
    {
        "name": "lseg",
        "url": "https://api.analytics.lseg.com/lfa/mcp",
        "best_for": "cross-asset analytics, rates, FX, options, and market data",
    },
    {
        "name": "pitchbook",
        "url": "https://premium.mcp.pitchbook.com/mcp",
        "best_for": "private markets, deals, funds, and company profiles",
    },
    {
        "name": "chronograph",
        "url": "https://ai.chronograph.pe/mcp",
        "best_for": "private-equity portfolio monitoring data",
    },
    {
        "name": "egnyte",
        "url": "https://mcp-server.egnyte.com/mcp",
        "best_for": "document repositories and diligence files",
    },
    {
        "name": "ccxt",
        "url": "stdio: npx @mcpfun/mcp-server-ccxt (or any ccxt-backed MCP server)",
        "best_for": "crypto spot/futures OHLCV, tickers, and order books across 100+ exchanges via ccxt (Binance, OKX, Bybit, Coinbase, Kraken, ...)",
    },
)


def known_mcp_connectors() -> list[dict[str, str]]:
    """Return the FSI connector endpoints mirrored from the plugin manifest."""
    return [dict(connector) for connector in KNOWN_MCP_CONNECTORS]


@dataclass
class MCPDataSourceConfig:
    """Describes how to reach an MCP connector and shape its response.

    Attributes:
        transport: "http" (streamable HTTP) or "stdio" (local subprocess).
        url: Connector URL -- required for the "http" transport.
        headers: HTTP headers (e.g. auth). ``${ENV}`` placeholders are expanded.
        command: Executable -- required for the "stdio" transport.
        args: Arguments for the stdio command.
        env: Extra environment variables for the stdio subprocess.
        tool: Name of the connector tool that returns price data.
        arguments: Arguments passed to that tool (universe, dates, frequency...).
        records_path: Dotted path into the JSON result that holds the row list,
            e.g. "data.bars". Empty means the result is already a row list.
        field_mapping: Maps canonical column -> connector field name.
        timeout: Per-call timeout in seconds.
        columns_order: Connector field names in positional order, for
            connectors that return array rows (e.g. ccxt's ``[timestamp, open,
            high, low, close, volume]``) instead of named-field rows. When
            set, rows are framed with these names before ``field_mapping`` is
            applied, instead of relying on pandas' default integer columns.
        constant_columns: Canonical column -> literal value applied to every
            fetched row. Use this when a value is a call parameter rather than
            a row field, e.g. injecting the requested symbol as ``asset_id``
            for single-symbol-per-call tools such as ccxt's ``get-ohlcv``.
        derive_amount_from_close_volume: When the connector has no dollar-
            volume field, approximate ``amount`` as ``close * volume`` after
            mapping instead of requiring it in ``field_mapping``.
        datetime_unit: Passed to ``pandas.to_datetime(..., unit=...)`` for
            connectors that return epoch timestamps, e.g. ``"ms"`` or ``"s"``.
            ``None`` parses ``datetime`` as a normal date/timestamp string.
    """

    tool: str
    field_mapping: dict[str, str]
    transport: str = "http"
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    arguments: dict[str, Any] = field(default_factory=dict)
    records_path: str = ""
    timeout: float = 120.0
    columns_order: list[str] | None = None
    constant_columns: dict[str, str] = field(default_factory=dict)
    derive_amount_from_close_volume: bool = False
    datetime_unit: str | None = None

    def __post_init__(self) -> None:
        self.transport = self.transport.lower()
        if self.transport not in {"http", "stdio"}:
            raise ValueError(
                f"Unsupported transport '{self.transport}'. Use 'http' or 'stdio'."
            )
        if self.transport == "http" and not self.url:
            raise ValueError("transport='http' requires a 'url'.")
        if self.transport == "stdio" and not self.command:
            raise ValueError("transport='stdio' requires a 'command'.")
        covered = set(self.field_mapping) | set(self.constant_columns)
        if self.derive_amount_from_close_volume:
            covered.add("amount")
        missing = set(CANONICAL_COLUMNS) - covered
        if missing:
            raise ValueError(
                "field_mapping/constant_columns must cover FactorMiner's required "
                "columns; missing: " + ", ".join(sorted(missing))
            )
        if self.columns_order:
            unknown_targets = set(self.field_mapping.values()) - set(self.columns_order)
            if unknown_targets:
                raise ValueError(
                    "field_mapping targets not present in columns_order: "
                    + ", ".join(sorted(unknown_targets))
                )


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` placeholders in strings within a structure."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_mcp_source_config(path: str | Path) -> MCPDataSourceConfig:
    """Load an :class:`MCPDataSourceConfig` from a YAML file.

    ``${ENV}`` placeholders in any string value are expanded against the
    environment, so credentials never need to live in the file.
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"MCP source config must be a mapping: {path}")
    raw = _expand_env(raw)
    known = {f.name for f in MCPDataSourceConfig.__dataclass_fields__.values()}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown MCP source config keys: {', '.join(sorted(unknown))}")
    return MCPDataSourceConfig(**raw)


def _dig(payload: Any, dotted_path: str) -> Any:
    """Follow a dotted path into nested dicts; empty path returns the payload."""
    if not dotted_path:
        return payload
    cursor = payload
    for key in dotted_path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            raise KeyError(f"records_path '{dotted_path}' not found in connector result")
        cursor = cursor[key]
    return cursor


def _result_to_payload(result: Any) -> Any:
    """Extract a JSON payload from an MCP CallToolResult.

    Prefers ``structuredContent`` (modern connectors); otherwise concatenates
    text content blocks and parses them as JSON.
    """
    structured = getattr(result, "structuredContent", None)
    if structured:
        return structured

    texts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    if not texts:
        raise ValueError("MCP connector returned no text or structured content")

    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "MCP connector response was not JSON; a custom adapter is needed for "
            "this connector's output format."
        ) from exc


def _records_to_frame(records: Any, config: MCPDataSourceConfig) -> pd.DataFrame:
    """Map connector rows onto the canonical FactorMiner OHLCV schema."""
    if isinstance(records, dict):
        # Column-oriented payload: {"close": [...], "open": [...], ...}
        frame = pd.DataFrame(records)
    elif isinstance(records, list):
        if config.columns_order and records and not isinstance(records[0], dict):
            # Positional array rows (e.g. ccxt's [timestamp, o, h, l, c, v]).
            frame = pd.DataFrame(records, columns=config.columns_order)
        else:
            frame = pd.DataFrame(records)
    else:
        raise ValueError(
            f"Expected a list or dict of rows, got {type(records).__name__}"
        )
    if frame.empty:
        raise ValueError("MCP connector returned an empty result set")

    # field_mapping is canonical -> connector; invert to rename connector columns.
    rename = {src: canon for canon, src in config.field_mapping.items()}
    missing_src = [src for src in rename if src not in frame.columns]
    if missing_src:
        raise ValueError(
            "Connector result is missing mapped fields: "
            + ", ".join(missing_src)
            + f". Available columns: {', '.join(map(str, frame.columns))}"
        )
    frame = frame.rename(columns=rename)

    for canonical_col, literal_value in config.constant_columns.items():
        frame[canonical_col] = literal_value

    if config.derive_amount_from_close_volume and "amount" not in frame.columns:
        if "close" not in frame.columns or "volume" not in frame.columns:
            raise ValueError(
                "derive_amount_from_close_volume=True requires 'close' and "
                "'volume' to already be present via field_mapping/constant_columns"
            )
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        frame["amount"] = close * volume

    keep = [c for c in CANONICAL_COLUMNS if c in frame.columns]
    frame = frame[keep].copy()
    if config.datetime_unit:
        frame["datetime"] = pd.to_datetime(
            frame["datetime"], errors="coerce", unit=config.datetime_unit
        )
    else:
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce", utc=False)
    if frame["datetime"].isna().any():
        raise ValueError("Some 'datetime' values failed to parse from the connector")
    frame["asset_id"] = frame["asset_id"].astype(str)
    return frame.sort_values(["asset_id", "datetime"]).reset_index(drop=True)


async def _call_connector(config: MCPDataSourceConfig) -> Any:
    """Open an MCP session, call the configured tool, return its raw result."""
    try:
        from mcp import ClientSession
    except ModuleNotFoundError as exc:  # pragma: no cover - surfaced to operator
        raise ModuleNotFoundError(
            "Fetching data from MCP connectors requires the 'mcp' package. "
            "Install it with:  pip install 'factorminer[mcp]'"
        ) from exc

    if config.transport == "http":
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(
            config.url, headers=config.headers or None
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await session.call_tool(config.tool, arguments=config.arguments)

    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=config.command,
        args=config.args,
        env={**os.environ, **config.env} if config.env else None,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await session.call_tool(config.tool, arguments=config.arguments)


def fetch_market_data(config: MCPDataSourceConfig) -> pd.DataFrame:
    """Fetch market data from an MCP connector as a canonical OHLCV DataFrame.

    The returned frame uses canonical FactorMiner column names, so it can be
    written to CSV/Parquet and passed straight to ``factorminer mine --data``.
    """
    result = asyncio.run(asyncio.wait_for(_call_connector(config), timeout=config.timeout))
    if getattr(result, "isError", False):
        raise RuntimeError(f"MCP connector tool '{config.tool}' reported an error")
    payload = _result_to_payload(result)
    records = _dig(payload, config.records_path)
    return _records_to_frame(records, config)


def fetch_to_file(config: MCPDataSourceConfig, output: str | Path) -> Path:
    """Fetch market data and write it to a CSV/Parquet/HDF5 file.

    Returns the path written. The format is inferred from the extension.
    """
    frame = fetch_market_data(config)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(out, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(out, index=False)
    elif suffix in {".h5", ".hdf5"}:
        frame.to_hdf(out, key="data", index=False)
    else:
        raise ValueError(
            "Cannot infer output format from extension. Use .csv, .parquet, .h5, or .hdf5."
        )
    return out
