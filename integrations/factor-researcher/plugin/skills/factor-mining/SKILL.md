---
name: factor-mining
description: Discover alpha factors by running the FactorMiner research engine — the paper-faithful Ralph loop or the enhanced Helix loop (causal validation, regime conditioning, multi-specialist debate, canonicalization). Use to generate a new factor library from a validated dataset. Triggers on "mine factors", "discover factors", "run mining", "find alpha", "helix loop", "ralph loop", "build a factor library".
---

# Factor Mining

This skill runs FactorMiner's self-evolving discovery loop: it retrieves memory priors, proposes candidate factor formulas with an LLM, evaluates them, and admits the survivors to a factor library.

See `references/loop-architecture.md` for the stage-by-stage loop design and `references/dsl-operators.md` for the factor-formula operator vocabulary.

## Choosing the loop

| Use | When |
|---|---|
| `mine` (Ralph loop) | Default. Paper-faithful Algorithm 1 — retrieve, generate, evaluate, admit, evolve memory. |
| `helix` (Helix loop) | When you want Phase 2 features: do-calculus causal validation, regime-conditional evaluation, multi-specialist debate generation, or SymPy canonicalization. Drop-in superset of Ralph. |

## Workflow

### 1. Confirm prerequisites

The dataset must already pass `factor-data` validation. Confirm the iteration budget — mining cost scales with `iterations × batch-size`.

### 2. Run the Ralph loop

```bash
factorminer -o output/run1 mine \
  --data path/to/market_data.csv \
  --iterations 40 --batch-size 16 --target 30
```

- `--iterations` — maximum mining iterations (the loop also stops early once `--target` factors are admitted).
- `--batch-size` — candidate factors proposed per iteration.
- `--target` — desired library size.
- `--resume path/to/factor_library.json` — continue a previous run.
- `--mock` — synthetic data + mock LLM, no API calls. Use only for smoke tests.

### 3. Or run the Helix loop

```bash
factorminer -o output/run1 helix \
  --data path/to/market_data.csv \
  --iterations 40 --batch-size 16 --target 30 \
  --causal --regime --debate --canonicalize
```

Each `--feature / --no-feature` flag overrides the config; omit a flag to keep the config default. Phase 2 features cost extra compute and LLM calls — enable the ones the research question needs.

### 4. Inspect the result

```bash
factorminer session inspect output/run1 --json
```

Report library size, iteration count, and yield rate. The factor library is written to `output/run1/factor_library.json`; the run log to `session_log.json`.

## Guardrails

- Mining proposes formulas; it does not prove them. Always follow with `factor-evaluation` on the held-out split.
- A low yield rate usually means thresholds are too strict for the dataset, not that the data is bad — tune `ic_threshold` / `correlation_threshold` in config, do not silently relax them in a report.
- `--mock` output is never a research result; never present mock metrics as real.

## MCP alternative

When the FactorMiner MCP server is connected, `mine_factors` and `helix_mine` expose the same workflow as tools, returning a structured session summary directly.
