---
name: factor-evaluation
description: Evaluate a factor library — recompute Information Coefficient (IC), ICIR, win rate, and turnover on held-out data, and surface train→test decay. Use to judge how good a mined library actually is out of sample. Triggers on "evaluate factors", "compute IC", "how good is this library", "factor metrics", "ICIR", "is this factor overfit", "out-of-sample".
---

# Factor Evaluation

Mining proposes factors; evaluation decides whether to believe them. This skill recomputes a library's metrics on a chosen split and exposes overfitting.

See `references/metrics.md` for precise metric definitions (IC vs. paper-IC, ICIR, redundancy correlation).

## Workflow

### 1. Recompute metrics

```bash
factorminer evaluate output/run1/factor_library.json \
  --data path/to/market_data.csv \
  --period test
```

`--period` selects the split: `train`, `test`, or `both`. Always lead with `test` — in-sample IC is not evidence.

### 2. Read the table

The output table reports, per factor: `IC Mean`, `Paper IC`, `Abs IC`, `Paper ICIR`, `Win%`, and `Turnover`. The summary block gives library-level means and the IC range.

### 3. Check decay

```bash
factorminer evaluate output/run1/factor_library.json --data market_data.csv --period both
```

`--period both` adds a **decay table** (train Paper IC → test Paper IC → delta). A large negative delta is the signature of an overfit factor. Report decay honestly; do not quote the train number as the headline.

### 4. Rank the survivors

To shortlist the strongest signals only:

```bash
factorminer evaluate output/run1/factor_library.json --data market_data.csv --period test --top-k 10
```

The top-K-by-IC table is the **signal shortlist** — the natural handoff to a research-idea workflow that wants to know which quantitative signals are currently working. The MCP `screen_factors` tool returns this same shortlist directly.

## Interpreting the numbers

- **IC ≈ 0.03–0.05** out of sample is a respectable single factor on liquid universes.
- **ICIR** matters more than IC: a small but *stable* IC beats a large erratic one.
- **High turnover** quietly erases IC once costs are applied — carry it into `factor-backtest`.

## Guardrails

- Never present `train` metrics as the result. The deliverable is the `test` number.
- If every factor decays to ~0 on test, the library failed — say so. Do not search for a split that flatters it.
