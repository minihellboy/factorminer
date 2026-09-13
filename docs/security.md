# Security

FactorMiner processes market data, formulas, saved artifacts, model output, and
agent requests. Credentials, licensed data, filesystem access, and evaluation
integrity are the main trust boundaries.

## Execution boundaries

| Surface | Contract |
| --- | --- |
| Formula DSL | Registered leaves/operators and parser-validated expression trees |
| Custom NumPy operators | Token screening, empty builtins, and NumPy-only globals; compiled with `exec` in process |
| CLI calls from MCP | Explicit argument arrays, without `shell=True`; captured output and return codes |
| Managed-agent output | Declared `./out/` boundary and leaf-specific tools; references checked by `scripts/check.py` |
| Reports | Escaped model-authored names, formulas, rationales, and narrative fields |

Custom-operator restrictions are not an OS security boundary. Treat those
experiments as trusted local code execution. Ordinary factor formulas use the
DSL execution path.

Input loaders use tabular/schema parsers. Connector cache keys use normalized
identifiers. Saved scores are recomputed for analysis; artifact hashes establish
integrity, not source truth or future validity. See [architecture](architecture.md)
and [evidence protocol](evidence-protocol.md).

## Models, prompts, and persistence

Provider keys and bearer tokens belong in environment variables or secret stores.
Committed YAML, manifests, session artifacts, reports, and RFT exports must not
contain them. A configured model endpoint receives the prompt content sent to it.

`OpenAICompatibleProvider` requires an operator-configured `base_url` and does
not fall back to `OPENAI_API_KEY`. Frontier construction strips custom URLs;
draft/frontier requests have independent timeouts. Generated content cannot
select the endpoint.

Research notes enter generation through structured archetypes; memory and
library retrieval uses typed summaries. Sealed-evaluator feedback exposes only
allowed coarse fields. Malformed replies fail to neutral/rejected results, and
rationales remain data rather than instructions. HTML rationales are marked as
unreviewed unless a human attestation is recorded.

Neural-leaf persistence uses `torch.save` and loads weights with
`torch.load(..., map_location="cpu", weights_only=True)`. Retain checkpoint
provenance; restricted weight loading is not validation of model behavior.
RFT export creates a dataset and records `trains_model: false`.

## Data connectors

Acquisition is explicit through workflows such as `fetch-data`, `attach-edgar`,
and `build-futures`. Normal mining does not initiate unattended data acquisition.

- EDGAR fetches use the pinned SEC endpoint, timeouts, response-size controls,
  a descriptive user agent, and request-rate controls. Joins use filing dates.
- Consensus-factor fetches require HTTPS, cap response size, and validate shape.
  Missing or malformed data remains unavailable.
- `MCPDataSourceConfig` validates transport, connection fields, column coverage,
  positional schemas, and unknown keys before connection. `${ENV}` expansion
  keeps credentials outside YAML. Derived `amount = close * volume` is opt-in
  and recorded as an approximation.

Connector configuration and examples are in the
[integration guide](../integrations/factor-researcher/README.md).

## MCP transports

Stdio uses the launching process and host account as its boundary:

```bash
uv run factorminer mcp-serve --transport stdio
```

HTTP is opt-in, defaults to loopback, and requires a non-empty token:

```bash
export FACTORMINER_MCP_TOKEN="$(openssl rand -hex 32)"
uv run factorminer mcp-serve --transport http --host 127.0.0.1 --port 8765
```

Clients send `Authorization: Bearer <token>`. `StaticBearerTokenVerifier` uses
the MCP SDK token-verification interface. The local listener is for one trusted
operator; its token does not supply tenant isolation or per-tool authorization.

The separate [hosted pilot](hosted-pilot.md) provides tenant-bound credentials,
scoped jobs, retention, and consent controls. It is an optional deployment
surface with its own operational boundary. MCP tools return research artifacts
and do not call trading or broker endpoints.

## Verification

Existing tests cover HTTP authentication, connector validation and availability
joins, provider credential separation, sealed feedback, report escaping, and
operator/checkpoint behavior. Run `uv run python scripts/check.py` for integration
manifests and `uv run python scripts/check_architecture.py` for import boundaries.
