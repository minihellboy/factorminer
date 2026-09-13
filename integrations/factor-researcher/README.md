# Factor Researcher integration

Reference agent packages call FactorMiner's CLI/MCP workflows. Formula execution,
evaluation, admission, memory, and benchmark semantics remain in the engine.

## Setup

```bash
uv sync --group dev --extra mcp
uv run factorminer --help
uv run factorminer mcp-serve --transport stdio
```

Model-backed workflows also need the relevant provider extra and credentials.
Credentials stay in the host environment, outside manifests and artifacts.

## Packages

| Path | Purpose |
| --- | --- |
| `plugin/.claude-plugin/plugin.json` | Plugin identity/version |
| `plugin/.mcp.json` | Local server and optional external connector configuration |
| `plugin/agents/factor-researcher.md` | Interactive agent instructions |
| `plugin/commands/` | CLI/MCP workflow entry points |
| `plugin/skills/` | Workflow instructions and reference material |
| `managed-agent/agent.yaml` | Managed-agent configuration |
| `managed-agent/subagents/` | Data, mining, evaluation, and library roles |
| `managed-agent/steering-examples.json` | Example task steering |

The root `.claude-plugin/marketplace.json` points to
`./integrations/factor-researcher/plugin`. Optional remote endpoints in
`plugin/.mcp.json` require their own configured URLs and credentials.

Managed-agent references resolve relative to their YAML files and reuse the
plugin instructions/skills. Handoffs use declared schemas. Only the librarian
leaf has the declared write-capable tools; managed output is confined to `./out/`.
Host-side provisioning and authorization are deployment responsibilities.

## MCP interface

The server exposes data validation/acquisition, Ralph/Helix mining, research
ingestion, evaluation, screening, combination, benchmarking, reporting, export,
and session inspection. Tools delegate through explicit CLI argument arrays and
return structured results or captured error diagnostics.

`factorminer://docs/{topic}` reads a Markdown topic from the repository's flat
`docs/` directory. Documentation paths are part of this interface.

Stdio uses the launching account as its boundary. HTTP is opt-in, defaults to
loopback, and requires `FACTORMINER_MCP_TOKEN`; clients send a bearer token.
Transport commands, endpoint restrictions, and the separate hosted surface are
documented in [security](../../docs/security.md#mcp-transports).

## Data connectors

`MCPDataSourceConfig` describes an outbound connection to an operator-selected
MCP data service. It supports `stdio` and `http` (streamable HTTP) transports.
The schema includes:

| Field | Meaning |
| --- | --- |
| `command`, `args`, `env` | Stdio process configuration |
| `url`, `headers` | Remote endpoint and authentication |
| `tool`, `arguments` | Retrieval call |
| `records_path` | Location of rows in the result |
| `columns_order` | Column order for array-shaped records |
| `field_mapping` | Canonical column to source-field mapping |
| `constant_columns` | Fixed fields such as asset identity |
| `datetime_unit` | Epoch timestamp unit, when needed |
| `derive_amount_from_close_volume` | Explicit amount approximation |

`${ENV}` values expand at runtime. Config validation checks required connection
fields, canonical field coverage, unknown keys, and positional mappings.
Raw payloads are untrusted data; source configuration controls the mapping.

```bash
uv run factorminer mcp-connectors
uv run factorminer fetch-data \
  --mcp-config path/to/source.yaml --output /tmp/universe.parquet
uv run factorminer validate-data /tmp/universe.parquet --strict
```

`factorminer/configs/mcp_sources/ccxt_binance.yaml` shows positional OHLCV mapping.
An opted-in `amount = close * volume` derivation is recorded as an approximation.
Use [reproducibility](../../docs/reproducibility.md) for panel, target, and
resampling contracts. Acquiring data and running research are separate workflows.

## Validation

```bash
uv run python scripts/check.py
```

The checker validates JSON/YAML/frontmatter, managed-agent file references,
required files and output schemas, and the marketplace source. Engine and MCP
regressions are covered by the package test suite.
