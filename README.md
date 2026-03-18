# FactorMiner

**LLM-driven formulaic alpha mining with experience memory, strict recomputation, and a Phase 2 Helix loop**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

FactorMiner is a research framework for discovering interpretable alpha factors from market data using a typed DSL, an LLM generation loop, structured memory, and a library admission process built around predictive power and orthogonality.

The implementation is based on *FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for Financial Alpha Discovery* (Wang et al., 2026), with an extended Helix-style Phase 2 surface for canonicalization, knowledge-graph retrieval, debate generation, and deeper post-admission validation.

## Architecture

```text
                               FactorMiner
                                   |
          +------------------------+------------------------+
          |                                                 |
          v                                                 v
   Paper Reproduction Lane                           Helix Research Lane
   (strict, benchmark-facing)                        (extended, experimental)
          |                                                 |
          |                                                 +--> KG retrieval
          |                                                 +--> embeddings
          |                                                 +--> debate / critic
          |                                                 +--> canonicalization
          |                                                 +--> causal / regime
          |                                                 +--> capacity / significance
          |
          +--> Data loader -> preprocess -> target -> runtime recomputation
                                   |
                                   v
                          Typed DSL + operator registry
                                   |
                                   v
                      Ralph loop / candidate evaluation core
                                   |
               +-------------------+-------------------+
               |                                       |
               v                                       v
      Experience memory                      Admission / replacement
      (retrieve / distill)                   (IC, ICIR, orthogonality)
               |                                       |
               +-------------------+-------------------+
                                   |
                                   v
                           Factor library / catalogs
                                   |
             +---------------------+----------------------+
             |                                            |
             v                                            v
   Analysis commands                               Benchmark commands
   evaluate / combine /                            table1 / ablation-memory /
   visualize / export                              cost-pressure / efficiency / suite
             |                                            |
             v                                            v
      recomputed metrics,                           manifests, frozen Top-K,
      plots, tearsheets                             reports, machine-readable artifacts
```

## What Is In The Repo

- `RalphLoop` for iterative factor mining with retrieval, generation, evaluation, admission, and memory updates
- `HelixLoop` for enhanced Phase 2 mining with debate, canonicalization, retrieval enrichments, and optional post-admission validation
- 110 paper factors shipped in `library_io.py`
- 60+ operators across arithmetic, statistical, time-series, smoothing, cross-sectional, regression, and logical categories
- A parser + expression tree DSL over the canonical feature set:
  `$open`, `$high`, `$low`, `$close`, `$volume`, `$amt`, `$vwap`, `$returns`
- Analysis commands that now recompute signals on the supplied dataset instead of trusting stored library metadata
- Combination and portfolio evaluation utilities
- Visualization and tear sheet generation
- Mock/demo flows for local end-to-end testing without API keys

## Setup

### Recommended: `uv`

FactorMiner now supports a clean `uv` workflow for local development and reproducible setup.

```bash
git clone https://github.com/minihellboy/factorminer.git
cd factorminer

# Base runtime + dev tools
uv sync --group dev

# Add LLM providers / embedding stack
uv sync --group dev --extra llm

# Full local setup
uv sync --group dev --all-extras
```

Notes:

- The GPU extra is marked Linux-only because `cupy-cuda12x` is not generally installable on macOS.
- `uv sync --group dev --all-extras` is the intended "full environment" path for contributors.
- After syncing, use `uv run ...` for every command shown below.

### `pip` fallback

```bash
python3 -m pip install -e .
python3 -m pip install -e ".[llm]"
python3 -m pip install -e ".[all]"
```

## Quick Start

### Demo without API keys

```bash
uv run python run_demo.py
```

The demo uses synthetic market data and a mock LLM provider. It explicitly uses synthetic fallback for signal failures so local experiments do not get blocked by strict benchmark behavior.

### CLI overview

```bash
uv run factorminer --help
```

Available commands:

- `mine`: run the Ralph mining loop
- `helix`: run the enhanced Phase 2 Helix loop
- `evaluate`: recompute and score a saved factor library on train/test/full splits
- `combine`: fit a factor subset on one split and evaluate composites on another
- `visualize`: generate recomputed correlation, IC, quintile, and tear sheet outputs
- `benchmark`: run strict paper/research benchmark workflows
- `export`: export a library as JSON, CSV, or formulas

## Common Workflows

### 1. Mine with mock data

```bash
uv run factorminer --cpu mine --mock -n 2 -b 8 -t 10
```

### 2. Run Helix with selected Phase 2 features

```bash
uv run factorminer --cpu helix --mock --debate --canonicalize -n 2 -b 8 -t 10
```

### 3. Evaluate a saved library with strict recomputation

```bash
uv run factorminer --cpu evaluate output/factor_library.json --mock --period both --top-k 10
```

Behavior:

- Signals are recomputed from formulas on the supplied dataset.
- `train_period` and `test_period` from config define the authoritative split boundaries.
- `--period both` compares the same factor set across train and test and prints a decay summary.

### 4. Combine factors with explicit fit/eval splits

```bash
uv run factorminer --cpu combine output/factor_library.json \
  --mock \
  --fit-period train \
  --eval-period test \
  --method all \
  --selection lasso \
  --top-k 20
```

Behavior:

- top-k selection is based on recomputed fit-split metrics
- optional selection runs on the fit split
- portfolio evaluation runs on the eval split
- no pseudo-signal fallback is used in benchmark-facing analysis paths

### 5. Visualize recomputed artifacts

```bash
uv run factorminer --cpu visualize output/factor_library.json \
  --mock \
  --period test \
  --correlation \
  --ic-timeseries \
  --quintile \
  --tearsheet
```

Behavior:

- correlation heatmaps are built from recomputed factor-factor correlation
- IC plots use actual IC series from recomputed signals
- quintile plots and tear sheets use actual returns, not library-level summary metadata

### 6. Run the paper benchmark lane

```bash
uv run factorminer --cpu --config factorminer/configs/paper_repro.yaml \
  benchmark table1 --mock --baseline factor_miner
```

Available benchmark commands:

- `benchmark table1`: freeze Top-K on the configured freeze universe and report across universes
- `benchmark ablation-memory`: compare FactorMiner against the relaxed no-memory lane
- `benchmark cost-pressure`: run `1/4/7/10/11` bps stress tests
- `benchmark efficiency`: benchmark operator-level and factor-level compute time
- `benchmark suite`: run the full benchmark bundle

## Configuration

The default config lives at [`factorminer/configs/default.yaml`](factorminer/configs/default.yaml).

Key top-level fields:

- `data_path`: optional source file path when not using `--data`
- `output_dir`: default output directory for libraries, logs, and plots
- `mining`: Ralph/Helix mining thresholds and loop controls
- `evaluation`: backend, worker count, and strictness policy
- `data`: canonical features plus train/test split windows
- `llm`: provider, model, API key, and sampling settings
- `memory`: experience-memory retention settings
- `phase2`: Helix-specific toggles and validation modules

### Signal failure policy

`evaluation.signal_failure_policy` controls what happens when a factor formula cannot be recomputed:

- `reject`: fail the factor or abort the benchmark path
- `synthetic`: use deterministic pseudo-signals for demo/mock flows
- `raise`: propagate the raw exception for debugging

Defaults:

- analysis commands use `reject`
- `mine --mock`, `helix --mock`, and `run_demo.py` use `synthetic`

### Benchmark modes and profiles

The repo now ships explicit benchmark/research profiles:

- `factorminer/configs/paper_repro.yaml`: strict Ralph-only paper lane
- `factorminer/configs/benchmark_full.yaml`: paper lane plus the full benchmark suite
- `factorminer/configs/helix_research.yaml`: research lane with Helix features enabled
- `factorminer/configs/demo_local.yaml`: smaller local/demo settings

`benchmark.mode` controls artifact labeling:

- `paper`: strict paper-reproduction lane
- `research`: Helix-extended lane

### Split semantics

`data.train_period` and `data.test_period` are the source of truth for:

- `evaluate --period train|test|both`
- `combine --fit-period ... --eval-period ...`
- `visualize --period train|test|both`

## Data Format

Input data is expected to contain market bars with at least:

```text
datetime, asset_id, open, high, low, close, volume, amount
```

If `vwap` or `returns` are not present, the runtime layer derives them.

Mock data generation is available through `--mock` and `run_demo.py`.

## Helix / Phase 2

The Helix surface extends the base Ralph loop with additional tooling:

- debate generation via specialist agents
- SymPy canonicalization
- knowledge-graph retrieval
- embedding-assisted retrieval
- auto-inventor hooks
- optional causal, regime, capacity, and significance validation modules

Prompt construction now consumes structured Helix retrieval signals, including:

- complementary patterns
- conflict warnings / saturation warnings
- operator co-occurrence priors
- semantic gaps
- plain-language retrieval summaries

## Analysis Integrity

Recent changes in this repo tightened the benchmark surface:

- `evaluate`, `combine`, and `visualize` now recompute from the supplied dataset
- CLI top-k selection is based on recomputed split metrics
- Helix/Ralph use an explicit signal-failure policy
- mock/demo paths preserve convenience without leaking synthetic fallback into benchmark commands

This means saved library metadata is no longer treated as the final source of truth for analysis.

## Development

### Run tests

```bash
uv run pytest -q factorminer/tests
```

### Build the wheel

```bash
uv run python -m pip wheel --no-deps . -w dist
```

### Lint

```bash
uv run ruff check .
```

## Project Layout

```text
factorminer/
├── agent/         LLM providers, prompt builders, debate, specialists
├── configs/       Default YAML configuration
├── core/          Parser, expression trees, loops, factor library, serialization
├── data/          Loaders, preprocessing, tensor building, mock data
├── evaluation/    Metrics, runtime recomputation, combination, portfolio, validation
├── memory/        Experience memory, KG retrieval, embeddings
├── operators/     Operator registry and implementations
├── tests/         Pytest suite
└── utils/         Config loading, plotting, tear sheets, reporting
```

## Packaging Notes

- The project now uses `setuptools.build_meta` as its build backend.
- `uv` is the recommended local workflow.
- `uv.lock` is generated and should be refreshed when dependency metadata changes.
- The repo metadata points at the current GitHub project: [minihellboy/factorminer](https://github.com/minihellboy/factorminer).

## License

MIT. See [`LICENSE`](LICENSE).
