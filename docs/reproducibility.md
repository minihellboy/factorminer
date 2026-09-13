# Reproducibility

A reproducible FactorMiner run identifies its data, targets, splits, formulas,
configuration, code/environment, seeds, and baseline provenance. Keep result
JSON, manifests, trial history, and warnings together when moving artifacts.

## Data and targets

The normalized panel requires:

```text
datetime, asset_id, open, high, low, close, volume, amount
```

Identifier aliases include `code`, `ticker`, `symbol`, and `ts_code`; `amt`
aliases `amount`. The loader sorts by asset/time and validates numeric fields.
It can derive `vwap = amount / volume` and per-asset close returns. Derived
fields must remain distinguishable from vendor observations.

```bash
uv run factorminer validate-data path/to/market_data.csv --strict
```

Targets define entry delay, holding period, and price pair. Set
`data.train_period` and `data.test_period` for the supplied panel. Runtime
analysis shares the `EvaluationDataset` split model. Freeze admission, model
fitting, factor weights, hyperparameters, and direction upstream of test data.
The CLI action lane also purges forward-target overlap from the training end.
Array callers own alignment, discovery-only slicing, and feature availability.

The [bundled sample](../data/README.md) has a matching configuration and provenance
manifest. Public-data specifications and checksum locks live in
[`examples/public_evidence/`](../examples/public_evidence/README.md).

Resampling aggregates open/close by first/last source bar, high/low by max/min,
and volume/amount by sum. VWAP and returns are recomputed:

```bash
uv run factorminer resample-data data/binance_crypto_5m.csv \
  /tmp/binance_crypto_10m.csv --rule 10min
```

## Metric contract

The `ic_*` fields use cross-sectional Spearman correlation between signal and
aligned forward return. Explicit `rank_ic_*`, `pearson_ic_*`, and
`ic_definition: spearman_rank` fields distinguish the two correlation measures.

| Field | Definition | Use |
| --- | --- | --- |
| `ic_mean` | `mean(IC_t)` | Direction and signed diagnostics |
| `ic_paper_mean` | `abs(mean(IC_t))` | Default admission, replacement, ranking, Top-K freeze |
| `ic_abs_mean` | `mean(abs(IC_t))` | Magnitude diagnostic |
| `icir` | `mean(IC_t) / std(IC_t)` | Signed ICIR |
| `ic_paper_icir` | `abs(mean(IC_t)) / std(IC_t)` | Default ICIR gate |

For `[0.1, -0.1]`, average magnitude is 0.1 but absolute mean is zero. Admission
uses absolute mean. Current artifacts identify `paper_ic_v2`; older libraries
may be labeled `legacy_abs_ic` and need reevaluation before comparison.
[Evidence protocol](evidence-protocol.md) defines annualization, HAC uncertainty,
risk residualization, turnover, and cost diagnostics.

## Runtime evaluation

Analysis recomputes formulas through the parser and operator registry on the
selected dataset. Missing leaves, invalid windows, unsupported operators, and
incompatible panels produce evaluation failures; stored scores cannot replace
those results.

```bash
uv run factorminer -c path/to/config.yaml evaluate output/run/factor_library.json \
  --data path/to/market_data.csv --period both --top-k 10
```

## Benchmark execution

```bash
uv run factorminer -c path/to/config.yaml -o output/benchmark \
  benchmark table1 --data path/to/market_data.csv --baseline factor_miner
uv run factorminer benchmark ablation-memory --mock
uv run factorminer benchmark ablation-strategy --mock
uv run factorminer benchmark cpcv --mock
uv run python scripts/run_phase2_benchmark.py --mock
```

The CLI and standalone runner share `factorminer.benchmark.runtime`. Manifests
record metric version, candidate count, dataset hashes, runtime contract, and
baseline provenance:

| Provenance | Meaning |
| --- | --- |
| `builtin_paper_catalog` | Built-in catalog of 110 normalized formulas |
| `saved_library` | Supplied factor library |
| `runtime_loop` | Factors mined during the run |
| `catalog_baseline` / `catalog_proxy` | Implemented formula subset or proxy |
| `synthetic_no_memory_proxy` | Fallback when no real no-memory run is supplied |

Mock runs demonstrate execution and artifact shape. Repository-local benchmarks
measure the supplied implementation and panel. [Public evidence releases](public-evidence-release.md)
add checksum-locked data and portable verification; independent reproduction
requires another run on the same declared inputs.

## Experiment-selection comparison

```bash
uv run factorminer -o output/action-comparison benchmark research-actions \
  --seeds 0,1,2,3,4,5,6,7,8,9 --evaluations 32
```

Decision value, generate-only heuristic, random allocation, and contextual
bandit receive the same shuffled proposal tape and evaluation allowance per
seed. The fixed tape isolates allocation from adaptive LLM generation. Finite
allowances apply to this comparison; production mining has no resource quota.

Four fixed diagnostic panels cover persistent signal, independent noise,
transient signal, and an unseen future break. Protocol, formulas, and directions
are frozen before final evaluation. Final results do not reach the planner.
The array API `compare_research_actions` accepts an external frozen catalog and
`purge_bars`; the caller sets purge length to cover target timing.

The primary score sums direction-frozen delayed IC minus the admission threshold
over recommendations. Failed discovery delay checks remove factors from that
advisory set without deleting library entries. Reports also include unstressed
utility, raw-library utility, failed claims, evaluations, and elapsed time.
This diagnostic score is not portfolio return. Bootstrap intervals describe
paired search-seed variation on fixed panels, not variation across markets.

## Research-skill transfer comparison

```bash
uv run factorminer -o output/skill-comparison benchmark research-skills \
  --source-seeds 1000,1001,1002,1003,1004,1005,1006,1007 \
  --target-seeds 2000,2001,2002,2003,2004,2005,2006,2007,2008,2009
```

Three source families cover noisy levels, integrated nuisance, and scale
variation. Five target families change mechanisms and syntax and include
independent noise and a hidden future reversal. Each family/seed pair has a
separate random stream; methods share its panel. Source/target seeds and dataset
identities are disjoint.

Each source episode evaluates a parent, four randomized edits without replacement,
and an available delay challenge. The pack and source cost are frozen in
`transfer_freeze.json` before target execution. Every target method receives the
same parent and one edit choice: two candidate evaluations. Fixed kind/parent
schedules isolate retrieval, not end-to-end planner or live-generation quality.

The score is direction-frozen held-out IC minus 0.04 for the best discovery-admitted
factor, or zero for abstention. A selected factor below that held-out threshold
counts as a false claim under this diagnostic rule. Final returns cannot change
selection. `results.json` includes all runs, recipe choices, false claims, costs,
and paired differences against no transfer, motif, and nearest-trajectory baselines.

Bootstrap intervals resample target panels within each family, condition on one
source pack, and omit multiple-comparison and source-training adjustments.
The existing motif taxonomy groups these wrapper edits together; nearest-trajectory
retrieval is the stronger recipe-specific baseline. Gaussian diagnostic panels
do not establish market performance. Use a new output directory per comparison.
See [research skills](research-skills.md) for retrieval and uncertainty contracts.

## Paper compatibility

`paper_repro.yaml` and `paper_repro_binance.yaml` retain historical reproduction
profiles under `factorminer/configs/`. They use fixed historical windows; the
bundled sample has its own profile. The Binance reproduction profile uses
10-minute bars, so finer source bars need resampling.

The project includes the formula DSL, default admission/target semantics, and
110 normalized catalog formulas. Exact paper datasets, full external baselines,
and A100 timing measurements are not bundled. `alpha101_classic` and
`alpha101_adapted` are partial catalogs; `gplearn`, `alphaforge_style`, and
`alphaagent_style` are proxies. A no-memory comparison is faithful only when it
uses an actual runtime or saved no-memory library. Read manifest labels before
comparing results with a published table.
