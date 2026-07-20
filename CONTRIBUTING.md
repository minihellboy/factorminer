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
| `factorminer/architecture/` | protocol contracts, policies, stages, reusable research services |
| `factorminer/core/` | loop orchestration, DSL/parser, expressions, factor library, session I/O |
| `factorminer/agent/` | model providers, prompt construction, generation, debate |
| `factorminer/data/` | ingestion, normalization, connectors, tensor construction |
| `factorminer/evaluation/` | runtime recomputation, metrics, validation, reports |
| `factorminer/benchmark/` | canonical comparative runtime and compatibility exports |
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
   benchmark path, manifest field, or compatibility export.

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
- What remains intentionally compatible?
- Which tests demonstrate the result?
- Which technical document was updated?
- Are there data, security, migration, or reproducibility implications?
