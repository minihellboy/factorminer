# Reproducibility and Benchmark Semantics

This document defines what FactorMiner can reproduce, how input panels are
normalized, which metrics control admission and ranking, and how benchmark
provenance must be read. It is a technical contract, not a performance claim.

## Reproduction levels

| Level | Meaning |
| --- | --- |
| Smoke | deterministic local execution on mock or bundled sample data |
| Repository-local benchmark | runtime recomputation with a manifest, possibly using partial/proxy baselines |
| Paper-protocol run | paper target, split, selection, and metric semantics on user-provided compatible data |
| Paper-level reproduction | equivalent data/universe plus faithful external baselines and comparable hardware |

The repository supports the first three levels. It does not bundle the paper's
A-share panel, 64-asset Binance panel, external baseline implementations, or
A100 measurements required for the fourth.

## Deterministic local checks

No API key or private dataset is required for these commands:

```bash
uv run factorminer quickstart
uv run python scripts/run_demo.py
uv run factorminer -o /tmp/factorminer-mock mine --mock -n 2 -b 8 -t 2
uv run factorminer --cpu benchmark table1 --mock --baseline factor_miner
```

`quickstart` runs a health check, mines a small mock library, and emits a static
report under `/tmp/factorminer-quickstart`. Mock results prove execution and
artifact shape only; they are not evidence of market performance.

## Market-data contract

The normalized long-form panel requires:

```text
datetime, asset_id, open, high, low, close, volume, amount
```

Accepted asset identifiers include `asset_id`, `code`, `ticker`, `symbol`, and
`ts_code`; `amt` is accepted as an amount alias. The loader sorts by asset and
time, validates numeric values, and can derive:

- `vwap = amount / volume` where volume is valid;
- per-asset `returns` from close prices.

These derived values are runtime transformations and must not be mistaken for
vendor-observed fields. Strict validation should run before mining:

```bash
uv run factorminer validate-data path/to/market_data.csv --strict
```

MCP connector configs must map or explicitly derive every required field. See
the [Factor Researcher integration](../integrations/factor-researcher/README.md)
for the connector schema and trust boundary.

## Bundled Binance-shaped sample

`data/binance_crypto_5m.csv` is an onboarding and CLI-smoke sample, not the
paper's crypto dataset. Its machine-readable provenance is in
`data/binance_crypto_5m.manifest.json`; descriptive notes are in
[`data/README.md`](../data/README.md).

The sample contains 20,000 five-minute rows: 20 symbols, 1,000 rows each, from
2026-02-14 21:30:00 through 2026-02-18 08:45:00. Use the matching short-window
profile to avoid empty paper-period splits:

```bash
uv run factorminer --config factorminer/configs/binance_sample.yaml \
  validate-data data/binance_crypto_5m.csv
```

The paper-style Binance lane uses ten-minute bars. Binance spot klines do not
provide a native `10m` interval, so resample 1m or 5m source bars:

```bash
uv run factorminer resample-data \
  data/binance_crypto_5m.csv \
  /tmp/binance_crypto_10m.csv \
  --rule 10min
```

Aggregation is deterministic:

| Field | Aggregation |
| --- | --- |
| `open` | first source bar |
| `high` | maximum |
| `low` | minimum |
| `close` | last source bar |
| `volume`, `amount` | sum |
| `vwap` | recomputed as resampled `amount / volume` |
| `returns` | recomputed from resampled close per asset |

For a full user-supplied crypto panel, start from
`factorminer/configs/paper_repro_binance.yaml`. It declares Binance/crypto,
ten-minute frequency, a 2024 training interval, and a held-out 2025 interval.

## Target and split semantics

The paper-facing tensor contract supports the next-period open-to-close target.
Runtime split boundaries come from `data.train_period` and `data.test_period`.
The same `EvaluationDataset` split model is used by `evaluate`, `combine`,
`visualize`, and `benchmark.runtime`.

Do not use test-period evidence for candidate admission, model fitting, factor
weight fitting, or hyperparameter selection. Commands that expose explicit fit
and evaluation periods should retain `train → test` direction.

## Information Coefficient semantics

At each time step, FactorMiner's historical `ic_*` fields compute a
cross-sectional Spearman rank correlation between the factor signal and
forward return. In industry terminology this is **RankIC**, not Pearson IC:

```text
legacy IC_t = RankIC_t = SpearmanRankCorr(signal_t, forward_return_t)
```

The summaries have distinct meanings:

| Field | Definition | Use |
| --- | --- | --- |
| `ic_mean` | `mean(IC_t)` | sign-aware diagnostics and weights |
| `ic_paper_mean` | `abs(mean(IC_t))` | paper-mode admission, replacement, sorting, and Top-K freeze |
| `ic_abs_mean` | `mean(abs(IC_t))` | legacy instability diagnostic only |
| `icir` | `mean(IC_t) / std(IC_t)` | signed diagnostic |
| `ic_paper_icir` | `abs(mean(IC_t)) / std(IC_t)` | paper-mode ICIR gate and reporting |

New evaluation artifacts also expose `rank_ic_*` and `pearson_ic_*` fields,
plus `ic_definition: spearman_rank`. The legacy keys remain unchanged so old
libraries and admission thresholds are not silently reinterpreted. The
industry evidence protocol additionally separates unannualized ICIR,
annualized ICIR, an independence t-stat, and a Newey-West/HAC t-stat. See
[Evidence Protocol](evidence-protocol.md) for the exact formulas and
cross-platform comparison rules.

The difference is material. For `IC_t = [0.1, -0.1]`, `ic_abs_mean` is `0.1`
while `ic_paper_mean` is `0.0`; the series has magnitude but no stable average
direction. Paper-mode gates use the latter.

Artifacts with explicit fields carry `metric_version: paper_ic_v2`. Older
libraries are read with compatibility defaults and identified as
`legacy_abs_ic`; rerun evaluation before comparing them with current output.

## Runtime recomputation

Saved library scores are not authoritative inputs to analysis. Given a formula
and dataset, the runtime parser and operator registry recompute signals, split
statistics, and portfolio diagnostics. This prevents stale metadata from being
reported as evidence on a different panel.

Example:

```bash
uv run factorminer --cpu evaluate output/factor_library.json \
  --data path/to/market_data.csv --period both --top-k 10
```

Recomputation can fail explicitly when a formula uses an unavailable leaf,
unsupported operator, invalid window, or incompatible panel. A failed parse or
evaluation must not be replaced by a stored score.

## Baseline provenance

Every benchmark manifest identifies how its baseline was obtained.

| Provenance | Meaning |
| --- | --- |
| `builtin_paper_catalog` | built-in 110 normalized FactorMiner formulas |
| `saved_library` | user-supplied `FactorLibrary` |
| `runtime_loop` | factors mined during the benchmark run |
| `catalog_baseline` | deterministic local approximation of a named formula family |
| `catalog_proxy` | stand-in for an external algorithm not bundled here |
| `synthetic_no_memory_proxy` | fallback approximation when no real no-memory run is supplied |

Current CLI baseline fidelity:

| Baseline | Implementation status |
| --- | --- |
| `factor_miner` | built-in catalog, saved library, or runtime loop |
| `factor_miner_no_memory` | faithful only with a real runtime/saved no-memory library; otherwise labeled synthetic proxy |
| `alpha101_classic` | deterministic partial catalog, not all 101 formulas |
| `alpha101_adapted` | partial catalog with local window variants |
| `random_exploration` | bounded deterministic templates, not a full paper random-tree implementation |
| `gplearn` | catalog proxy, not the external package |
| `alphaforge_style`, `alphaagent_style` | catalog-derived proxies |

A result is paper-level for a named baseline only when its algorithm/formulas,
generation and selection rules, dataset/universe, train/test protocol, and
metric version all match. Otherwise call it a repository-local benchmark,
catalog baseline, smoke benchmark, or proxy.

## Canonical benchmark execution

`factorminer.benchmark.runtime` coordinates benchmark calculations. Contract
construction, provenance capture, loop execution, datasets, frozen evaluation,
statistics, and reporting live in their named benchmark modules. The CLI and
standalone report builder delegate to it:

```bash
uv run factorminer -c factorminer.local.yaml -o output-benchmark \
  benchmark table1 --data path/to/market_data.csv --baseline factor_miner

uv run factorminer --cpu benchmark ablation-memory --mock
uv run factorminer --cpu benchmark ablation-strategy --mock
uv run factorminer --cpu benchmark cpcv --mock
uv run python scripts/run_phase2_benchmark.py --mock
```

Benchmark artifacts include the metric version, selected baseline and kind,
candidate count, dataset identity/hashes, runtime contract, and warnings for
proxy/fallback behavior. Keep JSON artifacts and their manifests together; a
rendered table without its manifest is not a reproducible benchmark.

## Paper-claim coverage

| Surface | Status |
| --- | --- |
| Typed OHLCV formula DSL and paper-style operator aliases | implemented |
| Ralph generate/evaluate/evolve loop and structured memory | implemented |
| IC, redundancy, replacement, deduplication, and full-validation gates | implemented |
| 110 normalized built-in paper factors | implemented |
| Next-period paper-style target and configurable train/test contract | implemented |
| Exact A-share and 64-asset Binance paper datasets | user-provided; not bundled |
| Full faithful Alpha101/GPLearn/AlphaForge/AlphaAgent baselines | not bundled; partial or proxy as labeled |
| A100 speed table and paper-scale efficiency figures | not bundled |

The NumPy, Bottleneck-backed CPU, and torch/GPU execution surfaces can be
benchmarked locally, but results describe the actual host and code path. Do not
present local timing as the paper's hardware result.

## Reproduction checklist

For a result intended for review, preserve:

1. exact source-data identity, license/entitlement, retrieval timestamp, and
   content hash;
2. normalization/resampling steps and whether any field was derived;
3. config, code commit, Python/package environment, backend, and random seed;
4. train/test boundaries, target construction, universe membership, and costs;
5. formula library plus metric and baseline provenance;
6. machine-readable result/manifest files, not only screenshots or Markdown;
7. warnings, rejected candidates, and failed recomputations;
8. an independent replay on the same inputs before making comparative claims.
