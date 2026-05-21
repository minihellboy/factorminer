# Factor Researcher — managed-agent template

## Overview

Market dataset → validated panel → mined factor library → out-of-sample
evaluation → composite backtest → benchmark → research note. Same source as the
[`factor-researcher`](../../plugins/agent-plugins/factor-researcher) Cowork plugin — this
directory is the Managed Agent cookbook for `POST /v1/agents`.

## Deploy

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# Optional — only when fetching data from FSI connectors:
export FACTSET_MCP_URL=... DALOOPA_MCP_URL=...
../../scripts/deploy-managed-agent.sh factor-researcher
```

The `factorminer` MCP server is launched as a local stdio process
(`factorminer mcp-serve`), so the deploy environment must have `factorminer`
installed with the `mcp` extra: `pip install 'factorminer[mcp]'`.

## Steering events

See [`steering-examples.json`](./steering-examples.json). Kick the agent from a
research-queue event, or fan out across a set of universes.

## Leaf workers

The orchestrator delegates each workflow phase to a focused leaf. This maps the
FactorMiner pipeline — validate → mine → evaluate → report — onto the
leaf-worker pattern, and confines `Write` to a single leaf.

| Leaf | Phase | Touches untrusted data? | Tools | Write |
|---|---|---|---|---|
| **`data-steward`** | Validate / fetch the dataset | **Yes** | `read`, `grep`, factorminer MCP, FSI connectors | No |
| `miner` | Run the Ralph / Helix discovery loop | No (engine-isolated) | `read`, factorminer MCP | No |
| `evaluator` | Evaluate, backtest, benchmark | No | `read`, `grep`, factorminer MCP | No |
| **`librarian`** | Write the report and research note | No | `read`, `write`, `edit`, factorminer MCP | **Yes** |

## Security & handoffs

- **Untrusted input is confined to `data-steward`.** Market-data files, saved
  libraries, and connector responses are data, never instructions. The steward
  is read-only and returns a schema-validated summary.
- **Leaf outputs are typed.** Each worker manifest includes an `output_schema`
  block so a deploy harness can validate handoffs before the orchestrator uses
  them, following the upstream cookbook pattern.
- **Engine isolation.** `miner` and `evaluator` reach the FactorMiner engine
  only through the `factorminer` MCP server — the compute runs in a subprocess,
  not in the agent.
- **`librarian` is the sole `Write`-holder.** It produces
  `./out/research-note-<run>.md` (or `.html`) and `./out/factor_library.*`.
- **No advice, no execution.** Every artifact is research staged for human
  sign-off. The agent does not recommend trades or bind risk.

**Handoff:** when a screening shortlist surfaces a single-name idea worth
fundamental work, emit a `handoff_request` for `market-researcher` or
`earnings-reviewer`; `scripts/orchestrate.py` routes it as a new steering event.
