---
name: factor-report
description: Generate static reports, tearsheets, and exports from FactorMiner artifacts — markdown/HTML research notes, plots, and library exports (JSON/CSV/formulas). Use to package a finished research run for review. Triggers on "generate report", "factor report", "research note", "tearsheet", "visualize factors", "export library", "write up the results".
---

# Factor Report

This skill turns a finished run into reviewable artifacts: a static report, plots, and portable exports.

## Workflow

### 1. Static report

```bash
factorminer report output/run1/factor_library.json \
  --session-log output/run1/session_log.json \
  --format markdown --output output/run1/report.md
```

- `--format` — `markdown` or `html`. HTML is self-contained (no JavaScript).
- `--session-log` — includes run metadata (iterations, yield rate).
- `--benchmark path.json` — fold in benchmark results; repeatable.

The report defines every admission metric inline, lists each factor with its formula and IC/ICIR/correlation, and warns on stale artifacts.

### 2. Plots

```bash
factorminer -o output/run1 visualize output/run1/factor_library.json \
  --data market_data.csv --period test \
  --correlation --ic-timeseries --quintile --format png
```

### 3. Export the library

```bash
factorminer export output/run1/factor_library.json --format formulas --output output/run1/formulas.txt
```

`--format` options: `json` (full library), `csv` (tabular metrics), `formulas` (plain formula list for handoff).

## Assembling the research note

A complete note pairs the generated report with narrative the analyst can sign off on:

1. **Objective & data** — universe, horizon, split boundaries, validation status.
2. **Method** — loop used (Ralph/Helix), features enabled, iteration budget.
3. **Library** — admitted factors with formulas and out-of-sample IC/ICIR.
4. **Composite** — backtest result net of costs, with the decay caveat.
5. **Benchmark** — comparison vs. baselines, if run.
6. **Limitations** — overfit risk, capacity, what was *not* tested.

## Guardrails

- The report presents results; it does not certify them. Frame it as research staged for human review.
- Quote out-of-sample metrics as the headline; never the in-sample number.
- Cite the run directory so every figure is reproducible via `factorminer session inspect`.
