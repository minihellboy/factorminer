# Contributing

This repository is no longer a small prototype. Changes should preserve the current architecture and reproducibility guarantees.

## Setup

Recommended local setup:

```bash
uv sync --group dev
uv sync --group dev --extra llm
```

For the broadest local environment:

```bash
uv sync --group dev --all-extras
```

Use `uv run ...` for commands in this repo.

## Core Rules

1. Keep benchmark-facing semantics in the architecture layer.
   Put protocol, dataset, memory, dependence, and benchmark rules under `factorminer/architecture` or `factorminer/benchmark/runtime`, not ad hoc in loop code.

2. Treat runtime recomputation as authoritative.
   Analysis and benchmark surfaces should recompute from formulas on the supplied dataset rather than trust stored library summaries.

3. Prefer services and policies over growing the loops.
   `RalphLoop` and especially `HelixLoop` should not keep absorbing new feature logic if a service, policy, or stage implementation is cleaner.

4. Keep `output/` out of source control.
   It is mutable runtime state and should stay untracked.

5. Add tests for every new architecture surface.
   If you add a new policy, stage, benchmark path, or contract, add regression coverage in `factorminer/tests`.

## Repo Structure

High-level ownership:

- `factorminer/architecture`: canonical contracts, stages, policies, services
- `factorminer/core`: loops, parser, factor library, expression execution
- `factorminer/benchmark`: canonical benchmark runtime and legacy helpers
- `factorminer/evaluation`: metrics, runtime recomputation, validation, portfolio analysis
- `factorminer/memory`: raw memory store, retrieval logic, KG, embeddings
- `factorminer/operators`: operator specs and execution backends

## Development Workflow

Typical local loop:

```bash
uv run pytest -q factorminer/tests
uv run ruff check <changed files>
```

Repo-wide Ruff is not yet clean. Do not block useful changes on unrelated style debt, but avoid introducing new issues in files you touch.

## What To Put Where

### New mining rule or benchmark rule

Prefer:

- `factorminer/architecture/paper_protocol.py`
- `factorminer/architecture/evaluation_kernel.py`
- `factorminer/architecture/geometry.py`
- `factorminer/benchmark/runtime.py`

### New retrieval or memory behavior

Prefer:

- `factorminer/architecture/memory_policy.py`
- `factorminer/architecture/prompt_context.py`
- `factorminer/architecture/families.py`
- `factorminer/memory/`

### New Helix-only capability

Prefer:

- a service in `factorminer/architecture/phase2_services.py`
- or a new stage implementation pattern

Avoid putting all new logic directly into `core/helix_loop.py`.

## Testing Expectations

Before opening a PR, run:

```bash
uv run pytest -q factorminer/tests
```

If your change touches:

- architecture contracts: run `factorminer/tests/test_architecture.py`
- Ralph/Helix loops: run `factorminer/tests/test_ralph_loop.py` and `factorminer/tests/test_helix_loop.py`
- runtime benchmarks: run `factorminer/tests/test_benchmark.py`

## Documentation Expectations

If your change alters repo architecture or public workflow, update:

- `README.md`
- `docs/architecture.md`
- `docs/repo-audit.md` when the repo inventory or debt picture materially changes

## Current Known Debt

These are known and non-blocking unless your change touches them directly:

- repo-wide Ruff debt
- `core/helix_loop.py` remains too large
- legacy benchmark surfaces still coexist with the runtime benchmark surface
- NaN-window warnings in `core/expression_tree.py`

## Pull Request Standard

A good PR in this repo should make it easy to answer:

- what contract changed
- which execution lane changed: paper, Helix, analysis, benchmark, or all
- what tests cover it
- what docs were updated
