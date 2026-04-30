# Benchmark Baselines

The benchmark runner records baseline provenance in every manifest. This page
explains what each baseline currently means so benchmark results are not
mistaken for paper-level reproduction when a baseline is still a proxy.

## Provenance Types

| Type | Meaning |
| --- | --- |
| `builtin_paper_catalog` | Uses the repo's built-in 110 FactorMiner paper factors. |
| `saved_library` | Uses factors loaded from a user-provided saved `FactorLibrary`. |
| `runtime_loop` | Runs a mining loop during the benchmark and evaluates the produced library. |
| `catalog_baseline` | Uses deterministic local formulas intended to approximate a named baseline family. |
| `catalog_proxy` | Uses a deterministic stand-in for a baseline whose real implementation is not yet bundled. |
| `synthetic_no_memory_proxy` | Uses a fallback no-memory approximation when no real no-memory run/library is supplied. |

## Current Baselines

| CLI baseline | Current provenance | Paper-fidelity status |
| --- | --- | --- |
| `factor_miner` | `builtin_paper_catalog`, `saved_library`, or `runtime_loop` depending on inputs | strongest current path |
| `factor_miner_no_memory` | `runtime_loop`, `saved_library`, or `synthetic_no_memory_proxy` | faithful only when backed by a real no-memory run/library |
| `alpha101_classic` | `catalog_baseline` | partial subset |
| `alpha101_adapted` | `catalog_baseline` | partial subset with window variants |
| `random_exploration` | `catalog_baseline` | deterministic template generator, not full random-tree search yet |
| `gplearn` | `catalog_proxy` | proxy, not the GPLearn package |
| `alphaforge_style` | `catalog_proxy` | proxy derived from catalog slices |
| `alphaagent_style` | `catalog_proxy` | proxy derived from catalog slices |

## Reading Benchmark Output

Each benchmark manifest includes:

- `metric_version`
- selected baseline name
- baseline kind
- candidate count
- dataset hashes
- runtime contract
- warnings when the run depends on a proxy or fallback

Paper-mode selection uses:

- `ic_paper_mean = abs(mean(IC_t))`
- `ic_paper_icir = abs(mean(IC_t)) / std(IC_t)`

The diagnostic `ic_abs_mean = mean(abs(IC_t))` is still reported, but it is
not the paper-mode quality gate.

## What Counts As Paper-Level Baseline Reproduction

A baseline should only be described as paper-level when all of these are true:

1. The algorithm or formula catalog matches the named paper baseline.
2. Candidate generation, adaptation, and selection rules match the paper.
3. The same train/test protocol is used.
4. The run uses paper-equivalent market data and universe membership.
5. The manifest records the metric version and baseline provenance.

Until then, the result should be described as a repo-local benchmark, smoke
benchmark, catalog baseline, or proxy baseline.

## Replacement Priorities

1. Expand `alpha101_classic` to the full 101 formulas.
2. Expand `alpha101_adapted` to the full adaptation protocol with up to 10
   parameter variants per factor.
3. Replace `random_exploration` with a typed bounded-depth random formula
   sampler.
4. Add an optional real `gplearn` adapter.
5. Keep `alphaforge_style` and `alphaagent_style` explicitly labeled as proxies
   unless real implementations are added.
