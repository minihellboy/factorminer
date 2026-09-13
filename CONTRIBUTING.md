# Contributing

FactorMiner changes must preserve the research protocol, runtime recomputation,
artifact provenance, and package boundaries. This file defines the repository
workflow; technical behavior belongs in the owning documentation.

## Setup

Python 3.12 or newer is required. CI and the repository tooling run on Python
3.12, which is the compatibility baseline for contributors and releases.

```bash
git clone https://github.com/minihellboy/factorminer.git
cd factorminer
uv sync --group dev --all-extras
```

Smaller environments can use `uv sync --group dev` and add `--extra llm` or
`--extra mcp` only when needed. Run repository commands through `uv run`.

### Optional model and GPU environments

The lock selects official `torch==2.13.0+cu126` wheels on Linux and
`cupy-cuda12x==14.2.0`. Both use CUDA 12; runtime libraries and NVRTC come from
the environment, so a system CUDA toolkit is not required. Validation uses
Python 3.12, Linux x86_64, and an NVIDIA RTX 4090 with driver 580.173.02.
The Linux Torch wheels require glibc 2.28 or newer. The locked macOS Torch
wheels require Apple Silicon and macOS 14+; Intel Macs can use the base CPU
install without model extras. Other GPU/platform combinations are unverified.
The CUDA index selection is uv-specific; pip users must select the official
PyTorch CUDA 12.6 distribution separately before installing the GPU extra.

```bash
uv sync --frozen --group dev --extra gpu --extra embeddings
uv run --no-sync factorminer --gpu doctor
FACTORMINER_REQUIRE_CUDA=1 FACTORMINER_TEST_EMBEDDINGS=1 \
  uv run --no-sync pytest -q factorminer/tests/test_gpu_compatibility.py
```

The required-CUDA flag fails if Torch or CUDA is missing. Tests compare real
CUDA tensors against NumPy and CPU Torch, check gradients against NumPy finite
differences, run a CuPy compiled kernel with DLPack interchange, train/reload a
neural leaf, and compare pipeline admission decisions. Embedding validation
downloads `all-MiniLM-L6-v2` on first use; normal CI skips this network test.

macOS uses the native Torch wheel without CuPy. When Torch and Homebrew-linked
XGBoost load different OpenMP runtimes, the process can crash. Launch mixed
model workflows with Torch's library directory selected before Python starts
([upstream workaround](https://github.com/pytorch/pytorch/issues/191933)):

```bash
FACTOR_TORCH_LIB=$(uv run --no-sync python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')
DYLD_LIBRARY_PATH="$FACTOR_TORCH_LIB" uv run --no-sync pytest -q factorminer/tests
```

This preserves parallel execution and does not alter system libraries. The
base CPU environment (`uv sync --group dev`) does not install Torch or CuPy.

## Branches and pull requests

Branch from an up-to-date `main`:

```bash
git switch main
git pull --ff-only
git switch -c <type>/<descriptive-scope>
```

Use descriptive names such as `fix/ic-split-leakage`,
`refactor/cli-command-registration`, or `docs/repository-structure`. Do not add
tool, editor, or author suffixes to branch names.

Keep each PR independently reviewable:

- one behavioral contract or one mechanical restructuring;
- focused tests proving the change;
- updates to the document that owns the changed behavior;
- no unrelated formatting or generated runtime output;
- green lint, manifest, test, and package checks.

Prefer a sequence of small PRs over a branch that mixes moves, semantic changes,
new features, and cleanup. Pure compatibility removal must identify the migrated
consumers and the test that proves the old path is no longer required.

## Package ownership

| Path | Owns |
| --- | --- |
| `factorminer/domain/` | dependency-free numerical contracts shared by application workflows |
| `factorminer/application/` | typed execution context and cross-workflow application contracts |
| `factorminer/architecture/` | protocol contracts, policies, stages, reusable research services |
| `factorminer/core/` | loop orchestration, DSL/parser, expressions, factor library, session I/O |
| `factorminer/agent/` | model providers, prompt construction, generation, debate |
| `factorminer/data/` | ingestion, normalization, connectors, tensor construction |
| `factorminer/evaluation/` | runtime recomputation, metrics, validation, reports |
| `factorminer/benchmark/` | comparative contracts, datasets, runners, statistics, and reports |
| `factorminer/memory/` | stores, retrieval primitives, knowledge graph, embeddings |
| `factorminer/operators/` | typed operator definitions and execution backends |
| `factorminer/mcp/` | external MCP tools/resources over stable engine workflows |
| `integrations/` | deployable agent packaging; no duplicated engine logic |
| `scripts/` | repository validation, demos, and standalone entry points |

See [Architecture](docs/architecture.md) for the dependency direction and
runtime contracts.

## Engineering rules

1. Put benchmark-facing semantics on architecture contracts or
   `factorminer.benchmark.runtime`, not in CLI handlers or a single loop.
2. Recompute formula signals on the supplied dataset. Stored summary scores are
   metadata, not authoritative analysis input.
3. Put retrieval, evolution, and persistence behavior behind `MemoryPolicy`.
4. Prefer a reusable service or stage implementation over adding branches to
   `RalphLoop` or `HelixLoop`.
5. Mutate factor libraries through `FactorAdmissionService` so admission and
   replacement invariants stay centralized.
6. Keep optional experiments opt-in and admission-neutral unless a contract
   change is explicit and tested.
7. Keep `output/`, credentials, private data, and local configuration out of
   source control.
8. Add regression coverage for every new public contract, policy, stage,
   benchmark path, manifest field, or public export.
9. All repository-authored source, documentation, and assets are MIT-licensed.
   By contributing, you agree that your contribution is licensed under MIT.
   Do not add code or assets whose terms prevent distribution under MIT.

## Documentation ownership

Technical documentation describes implemented behavior only:

- `README.md`: product boundary, supported surfaces, setup, common commands,
  and repository map;
- `docs/architecture.md`: contracts, flows, invariants, and package ownership;
- `docs/reproducibility.md`: data, splits, metrics, baseline provenance, and
  benchmark interpretation;
- `docs/security.md`: trust boundaries, controls, and verification;
- `integrations/<name>/README.md`: integration-specific manifests, interfaces,
  deployment requirements, and permissions.

Do not put roadmaps, audit diaries, speculative features, market forecasts, or
completed work logs into technical docs. Git history, issues, and PRs are the
record for planning and completed work.

## Validation

Run fast, focused checks while editing:

```bash
uv run ruff check <changed-python-files>
uv run pytest -q path/to/relevant_test.py
```

Before opening a PR, run:

```bash
uv run ruff check .
uv run python scripts/check_architecture.py
uv run python scripts/check.py
uv run pytest -q factorminer/tests
uv build
```

Useful focused suites:

| Change | Minimum focused coverage |
| --- | --- |
| architecture contracts | `factorminer/tests/test_architecture.py` |
| Ralph/Helix generation or stages | `test_ralph_loop.py`, `test_helix_loop.py`, generator tests |
| benchmark runtime or standalone script | `factorminer/tests/test_benchmark.py` |
| MCP/integration | MCP tests plus `uv run python scripts/check.py` |
| package exports/dependencies | `factorminer/tests/test_import_boundaries.py` |

Use `git diff --check` before committing. Build artifacts must contain the
`factorminer` package/configs and must not contain tests, local output, docs
archives, integration templates, or private data unless packaging metadata
explicitly says otherwise.

## PR description

A reviewer should be able to answer:

- What contract or path changed?
- Which execution surfaces are affected?
- Which public imports or artifacts remain stable?
- Which tests demonstrate the result?
- Which technical document was updated?
- Are there data, security, migration, or reproducibility implications?
