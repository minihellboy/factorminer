# FactorMiner

**Symbolic factor discovery with LLM-guided search, reproducible evaluation,
and transferable research memory.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/minihellboy/factorminer/actions/workflows/ci.yml/badge.svg)](https://github.com/minihellboy/factorminer/actions/workflows/ci.yml)

FactorMiner generates interpretable formulas, evaluates them on market data,
and records their lineage, metrics, and admission decisions. Shared evaluation
and memory contracts support local mining, controlled experiments, and agent
integrations. Outputs are research artifacts; the project does not execute trades.

## Capabilities

| Component | Function |
| --- | --- |
| Formula engine | Typed DSL with NumPy evaluation; optional compiled operators and Torch kernels |
| Mining | Ralph and Helix loops with generation, evaluation, memory, and checkpoints |
| Experiment selection | Opt-in generation, refinement, delay challenges, and stopping |
| Research skills | Versioned procedures with applicability, uncertainty, and contradiction tracking |
| Evaluation | Runtime recomputation, split-aware metrics, dependence, costs, and statistical diagnostics |
| Evidence | Formula lineage, trial ledgers, content-addressed evidence packs, and verifiable releases |
| Integration | CLI, Python API, MCP server, and reference agent packages |

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/minihellboy/factorminer.git
cd factorminer
uv sync --group dev --extra llm
```

NumPy is the default backend. Optional extras include `mcp`, `research`,
`embeddings`, `visualization`, and `gpu`; the CUDA extra targets Linux.
`uv sync --frozen --extra gpu` installs patched PyTorch with CUDA 12.6 and
CuPy for CUDA 12. The base install needs neither Torch nor a GPU.
See [backend scope](docs/architecture.md#numerical-backends) and
[platform setup and checks](CONTRIBUTING.md#optional-model-and-gpu-environments).

## Quick start

Run a local workflow without credentials:

```bash
uv run factorminer doctor --json
uv run factorminer quickstart
```

`quickstart` writes a sample library and HTML report to
`/tmp/factorminer-quickstart`. For a controlled mock campaign:

```bash
uv run factorminer -o output/mock mine --mock -n 2 -b 8 -t 10
uv run factorminer session inspect output/mock --telemetry
```

To run the action planner with DeepSeek Flash, set `DEEPSEEK_API_KEY` in `.env`:

```bash
uv run --env-file .env factorminer \
  -c factorminer/configs/research_actions_deepseek.yaml \
  -o output/deepseek-research mine --data data/binance_crypto_5m.csv
```

The bundled data is a workflow sample. The action lane uses the configured
training split and purges forward-target overlap. Local mining resumes in the
same output directory and defaults to no iteration ceiling or model-call/wall-time
quota. A positive `-n` sets a total campaign iteration boundary for experiments.
See [research actions](docs/research-actions.md) and
[research skills](docs/research-skills.md) for configuration and recovery.

## Data and evaluation

Input panels require:

```text
datetime, asset_id, open, high, low, close, volume, amount
```

Validate a panel, then recompute a saved library on it:

```bash
uv run factorminer validate-data path/to/market_data.csv --strict
uv run factorminer -c path/to/config.yaml evaluate output/run/factor_library.json \
  --data path/to/market_data.csv --period both --top-k 10
```

Configure targets and train/test periods for the supplied data. Saved scores
remain metadata; analysis recomputes formulas through the evaluation kernel.
See [reproducibility](docs/reproducibility.md) for field aliases, metric semantics,
frozen comparisons, and baseline provenance. Command-level `--help` lists the
available mining, analysis, benchmark, and data workflows.

## Documentation

| Guide | Contents |
| --- | --- |
| [Architecture](docs/architecture.md) | Components, ownership, execution, and persistence |
| [Research actions](docs/research-actions.md) | Experiment selection and campaign recovery |
| [Research skills](docs/research-skills.md) | Compile, transfer, and inspect procedure memory |
| [Reproducibility](docs/reproducibility.md) | Data, metrics, splits, and benchmarks |
| [Evidence protocol](docs/evidence-protocol.md) | Inference, risk residualization, and cost diagnostics |
| [Security](docs/security.md) | Input, model, credential, and MCP boundaries |
| [Agent integration](integrations/factor-researcher/README.md) | Plugin, managed-agent, and MCP setup |

Optional workflows: [public evidence releases](docs/public-evidence-release.md),
[partner review](docs/design-partner-pilot.md), and
[hosted pilot operations](docs/hosted-pilot.md).

## Repository

| Path | Purpose |
| --- | --- |
| `factorminer/` | Engine, CLI, configuration, and tests |
| `docs/` | Technical contracts and workflow guides |
| `data/` | Onboarding sample and provenance manifest |
| `examples/` | Public-data specifications and checksum locks |
| `integrations/` | Reference agent packages |
| `scripts/` | Checks, demos, benchmark entry point, and type baseline |

`output/` contains ignored runtime state. Keep each campaign in its own
directory; archive completed campaigns with their manifests and checkpoints.

## License and reference

FactorMiner is [MIT-licensed](LICENSE). Its original research foundation is
[*FactorMiner: A Self-Evolving Agent with Skills and Experience Memory for
Financial Alpha Discovery*](https://arxiv.org/abs/2602.14670). The project adds
experiment selection, transferable procedures, and evidence/recovery contracts.
Paper-specific profiles and baseline coverage are documented in
[reproducibility](docs/reproducibility.md#paper-compatibility).
