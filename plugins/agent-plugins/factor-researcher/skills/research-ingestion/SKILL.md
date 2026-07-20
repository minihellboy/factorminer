---
name: research-ingestion
description: Absorb external research reports/papers into structured, retrievable hypothesis cues via FactorMiner's Report-to-Memory Absorption (RMA) service — an OHLCV-eligibility gate, a mechanism-family classifier, and research-path cues that feed factor generation prompts. Use to turn a report fragment into a research hypothesis before mining, not to mine factors directly. Triggers on "ingest research", "absorb this report", "is this idea OHLCV-representable", "research archetype", "hypothesis cue", "report-to-memory", "RMA".
---

# Research Ingestion

This skill runs FactorMiner's Report-to-Memory Absorption (RMA) service — a scoped-down implementation of the RMA layer from XAlpha (arXiv:2607.08332): it screens external research fragments for OHLCV-representability, classifies the survivors into a broad mechanism family, and extracts reusable research-path hypothesis cues. It does not mine, generate, or backtest factors; it turns raw research text into structured input for `factor-mining`'s generation prompts.

## Why absorption instead of raw text

Feeding report text directly into a generation prompt lets ungrounded or OHLCV-infeasible claims (analyst EPS revisions, order-book microstructure, fundamentals) leak into hypothesis generation. RMA gates every fragment first, so only price/volume-representable mechanisms reach the mining loop.

## The A/B/C pipeline

| Layer | Question | Output |
|---|---|---|
| A (eligibility) | Can this mechanism be observed, inferred, or proxied from daily OHLCV bars alone? | KEEP/DROP + reason |
| B (mechanism family) | Which broad mechanism bucket does it belong to? | One of `factorminer.architecture.families.MECHANISM_FAMILIES` |
| C (archetype) | What's the reusable research cue? | A `ResearchArchetype` record with `research_paths` |

DROPped fragments (fundamentals, analyst estimates, order-book state, news/sentiment, macro releases) are discarded — they are not representable under the daily OHLCV factor contract.

## Workflow

### 1. Ingest a research note

```bash
factorminer ingest-research path/to/report_fragment.txt
```

Add `--mock` to run offline with the deterministic mock LLM provider (no API calls) — useful for smoke tests, never as a research result.

### 2. Read the classification

The command prints the KEEP/DROP verdict and reason. For a KEPT fragment it also prints the assigned mechanism family, fine-grained family, mechanism role, and research-path cues.

### 3. Hand off to mining

The resulting `ResearchArchetype` records are meant to be threaded into `PromptContextBuilder.build(..., research_archetypes=[...])` so `factor-mining`'s generation prompts carry the research-path text alongside memory and family context. Absorption itself never calls the mining loop — invoke `factor-mining` separately once you have archetypes worth exploring.

## Guardrails

- A-layer eligibility is a feasibility gate, not a quality signal — a KEPT fragment is not yet a validated hypothesis, only one that *could* become an OHLCV factor.
- Never present a `ResearchArchetype`'s `research_paths` as a factor formula; it is a hypothesis cue for `factor-mining`, not executable code.
- `--mock` classification is deterministic keyword heuristics, not real research judgment — never cite mock output as evidence.

## MCP alternative

When the FactorMiner MCP server is connected, `ingest_research_note` exposes the same workflow as a tool, returning the KEEP/DROP verdict and (for KEEP) the archetype record directly.
