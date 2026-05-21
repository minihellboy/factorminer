---
name: factor-benchmark
description: Run FactorMiner benchmark workflows — the Table 1 Top-K freeze benchmark, memory and strategy ablations, transaction-cost pressure tests, and the full suite. Use to compare FactorMiner against baselines or to reproduce paper results. Triggers on "benchmark", "ablation", "compare to baseline", "reproduce table 1", "cost pressure", "benchmark suite".
---

# Factor Benchmark

This skill runs FactorMiner's canonical benchmark surface — the rigorous comparison layer that turns a single run into evidence.

## Modes

| Mode | What it answers |
|---|---|
| `table1` | Top-K freeze benchmark across configured universes vs. baselines — the headline reproduction. |
| `ablation-memory` | How much does experience memory contribute? |
| `ablation-strategy` | Effect of memory policy × dependence metric × backend. |
| `cost-pressure` | How does the library hold up under rising transaction costs? |
| `efficiency` | Operator- and factor-level runtime/compute cost. |
| `suite` | The full benchmark suite in one run. |

## Workflow

### Run a benchmark

```bash
factorminer -o output/bench benchmark table1 --data path/to/market_data.csv
```

```bash
factorminer -o output/bench benchmark suite --data path/to/market_data.csv
```

Pass a pre-mined library to benchmark a specific run rather than mining fresh:

```bash
factorminer -o output/bench benchmark table1 \
  --data market_data.csv \
  --factor-miner-library output/run1/factor_library.json
```

`efficiency` takes no data — it profiles the engine itself.

### Read the result

The CLI prints a per-universe summary (library IC, ICIR, avg |ρ|) and writes JSON payloads into the output directory. Fold those JSON files into the research note with `factor-report --benchmark`.

## Interpreting ablations

- An ablation that *removes* a feature and barely moves the metric means that feature is not earning its compute on this dataset — report that plainly.
- `cost-pressure` is the honesty check: a library that only wins at zero cost is not a result.

## Guardrails

- Benchmark numbers are comparative research evidence, not a performance guarantee.
- Use the same dataset and splits across compared runs, or the comparison is meaningless.
- Reproduction claims must cite the exact config and run directory.
