# Factor Researcher integration

This directory packages the FactorMiner engine for agent hosts without copying
engine logic into prompts. It contains two representations of the same bounded
workflow:

```text
factor-researcher/
├── plugin/          installable plugin manifest, agent, commands, and skills
├── managed-agent/   headless orchestrator and typed leaf-agent manifests
└── README.md        integration contract
```

Both forms invoke FactorMiner through its MCP server. The CLI and Python package
remain the source of truth for computation, validation, and artifact formats.

## Prerequisites

Install FactorMiner with MCP support in the environment that launches the
server:

```bash
python3 -m pip install -e ".[mcp]"
# or
uv sync --group dev --extra mcp
```

Verify the engine before connecting an agent:

```bash
uv run factorminer doctor --json
uv run factorminer mcp-serve --transport stdio
```

## Plugin package

`plugin/` is self-contained and is registered by the repository-level
`.claude-plugin/marketplace.json`.

| Path | Contract |
| --- | --- |
| `.claude-plugin/plugin.json` | plugin identity and version |
| `.mcp.json` | local FactorMiner server and optional external connector endpoints |
| `agents/factor-researcher.md` | canonical interactive system prompt |
| `commands/*.md` | thin user commands for validation, mining, evaluation, backtest, benchmark, screening, and reporting |
| `skills/*/SKILL.md` | workflow instructions that call MCP/CLI surfaces instead of reimplementing math |
| `hooks/hooks.json` | currently empty extension point |

The marketplace source is relative to the repository root:

```json
{
  "name": "factor-researcher",
  "source": "./integrations/factor-researcher/plugin"
}
```

An agent host must be able to resolve the `factorminer` executable declared in
`plugin/.mcp.json`. Remote financial-data connectors listed there require their
own customer credentials and entitlements; FactorMiner does not provide or
proxy those credentials.

## Managed-agent template

`managed-agent/agent.yaml` describes a headless orchestrator. It reuses
`../plugin/agents/factor-researcher.md` and `../plugin/skills`, so the interactive
and managed packages share one prompt and skill implementation.

| Leaf manifest | Responsibility | Direct write permission |
| --- | --- | --- |
| `subagents/data-steward.yaml` | validate or fetch untrusted market data | no |
| `subagents/miner.yaml` | run the Ralph or Helix discovery lane | no |
| `subagents/evaluator.yaml` | recompute, evaluate, backtest, and benchmark | no |
| `subagents/librarian.yaml` | write the final library/report under `./out/` | yes |

Each leaf declares an `output_schema`. The host should reject a handoff that
does not validate against that schema. `steering-examples.json` contains example
events; it is not an executable deployment script. This repository deliberately
does not ship account-specific provisioning code—submit the manifest with the
deployment mechanism supplied by the target managed-agent platform.

## FactorMiner MCP server

`factorminer/mcp/server.py` exposes thin tool wrappers around canonical CLI
workflows:

```text
doctor                 validate_data           fetch_data
list_fsi_connectors    resample_data           mine_factors
helix_mine             ingest_research_note    evaluate_library
screen_factors         combine_factors         run_benchmark
generate_report        export_library          inspect_session
get_factor_library     inspect_debate
```

The resource template `factorminer://docs/{topic}` serves a named Markdown file
from the repository's `docs/` directory. Tool output is structured JSON; errors
include the subprocess return code and captured diagnostics.

### Stdio transport

Stdio is the default for a local host. It relies on process isolation and has no
network authentication layer:

```bash
uv run factorminer mcp-serve
uv run factorminer mcp-serve --transport stdio
```

### HTTP transport

HTTP is opt-in, binds to loopback by default, and refuses to start without a
non-empty bearer token:

```bash
export FACTORMINER_MCP_TOKEN="$(openssl rand -hex 32)"
uv run factorminer mcp-serve \
  --transport http --host 127.0.0.1 --port 8765
```

Clients connect to `http://127.0.0.1:8765/mcp` and send:

```text
Authorization: Bearer <FACTORMINER_MCP_TOKEN>
```

There is no unauthenticated HTTP mode. Do not bind the development server to a
public interface; use an approved authenticated gateway if a deployment needs a
network boundary beyond localhost.

## Inbound financial-data connectors

`factorminer/data/mcp_source.py` is the reverse path: an MCP client pulls data
from an external provider and converts it to FactorMiner's canonical panel.
Connector behavior is described by `MCPDataSourceConfig`, not hard-coded vendor
branches.

Required output fields are:

```text
datetime, asset_id, open, high, low, close, volume, amount
```

A source YAML specifies:

- `transport` plus `url` or `command`/`args`;
- authentication `headers` or subprocess `env` with `${ENV}` expansion;
- connector `tool` and `arguments`;
- `records_path` into the returned JSON;
- `field_mapping` to canonical columns;
- optional `columns_order`, `constant_columns`, and `datetime_unit`;
- optional `derive_amount_from_close_volume` when the approximation is explicit.

Use the CLI boundary for acquisition and validation:

```bash
uv run factorminer mcp-connectors
uv run factorminer fetch-data \
  --mcp-config path/to/source.yaml --output /tmp/universe.parquet
uv run factorminer validate-data /tmp/universe.parquet --strict
```

Do not synthesize missing volume or amount silently. If
`derive_amount_from_close_volume` is enabled, the emitted amount is an estimate,
not observed turnover, and downstream evidence must retain that provenance.

### Crypto/ccxt example

`factorminer/configs/mcp_sources/ccxt_binance.yaml` maps positional ccxt OHLCV
rows onto the same contract. It demonstrates:

- `columns_order` for `[timestamp, open, high, low, close, volume]`;
- `constant_columns.asset_id` because the symbol is a call argument;
- `datetime_unit: ms` for epoch milliseconds;
- explicit `close * volume` amount approximation.

The configured `get-ohlcv` operation fetches one symbol per call. Build a
multi-asset panel by fetching each symbol, setting its `asset_id`, concatenating
the files, validating the result, and only then resampling/mining. See
[`docs/reproducibility.md`](../../docs/reproducibility.md) for bar aggregation
and benchmark-data requirements.

## Trust boundaries

- Market files, connector payloads, saved libraries, and research notes are
  untrusted data, never instructions.
- Only the librarian leaf receives write-capable tools; engine execution remains
  behind the FactorMiner subprocess/MCP boundary.
- Every leaf handoff is schema-validated before reuse.
- Provider credentials stay in the host environment or approved secret store,
  never in plugin manifests, source YAML, logs, or reports.
- Every FactorMiner MCP tool returns a research artifact only. No tool places a
  trade, sizes a position, binds a limit, or operates an account.
- Out-of-sample evidence is reported separately from training evidence; the
  agent must not relabel an in-sample result as a deployment decision.

The full externally facing threat model is in
[`docs/security.md`](../../docs/security.md).

## Repository validation

Run the manifest/reference validator after changing either package:

```bash
uv run python scripts/check.py
```

It parses JSON/YAML/frontmatter, validates managed-agent file references and
output schemas, checks required files, and resolves marketplace plugin sources.
