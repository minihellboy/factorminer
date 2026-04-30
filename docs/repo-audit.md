# FactorMiner Repo Audit

This document is a technical audit of the current repository state after the architecture and runtime benchmark refactors.

## Snapshot

This audit intentionally avoids hardcoded fast-moving repository counts. For a
fresh local snapshot, run:

```bash
uv run pytest --collect-only -q factorminer/tests
uv run python - <<'PY'
from pathlib import Path
files = sorted(Path("factorminer").rglob("*.py"))
lines = sum(p.read_text(errors="ignore").count("\n") + 1 for p in files)
print(f"Python files: {len(files)}")
print(f"Python lines: {lines}")
PY
```

Stable paper-facing inventory:

- `110` built-in paper factors
- canonical paper-style and research mining lanes
- runtime recomputation for analysis and benchmark reporting

Package inventory:

| Package | Python files | Role |
| --- | ---: | --- |
| `agent` | 8 | providers, prompting, debate |
| `architecture` | 14 | canonical contracts, policies, stages, services |
| `benchmark` | 5 | runtime benchmark suite and legacy benchmark helpers |
| `configs` | 1 | packaged YAML profile resources |
| `core` | 13 | loops, factor library, parser, expression tree, I/O |
| `data` | 6 | loading, preprocessing, tensorization, mock generation |
| `evaluation` | 17 | metrics, recomputation, validation, portfolio analysis |
| `memory` | 10 | memory store, retrieval, KG, embeddings |
| `operators` | 15 | operator implementations, backends, and sandboxing |
| `tests` | 33 | regression coverage |
| `utils` | 6 | config, reporting, plotting |

## What Is Structurally Strong

### Canonical architecture layer exists now

The largest structural improvement is the existence of a real `factorminer.architecture` package. The codebase now has explicit surfaces for:

- protocol
- dataset contract
- dependence metrics
- evaluation kernel
- memory policy
- family discovery
- prompt context
- stage model
- admission service
- lifecycle logging
- Phase 2 helper services

That is a substantial improvement over the earlier state where semantics were spread across Ralph, Helix, benchmark code, and config projections.

### Runtime benchmark surface is the correct productized path

`factorminer.benchmark.runtime` is now the right center of gravity. It owns:

- dataset loading
- runtime mining loop execution
- benchmark library build
- frozen Top-K selection
- cross-universe evaluation
- manifest and provenance capture
- runtime ablations

This is the correct direction. The old benchmark layer still exists, but it is no longer the best architectural path.

### The test surface is strong

The repo has broad regression coverage. This matters because the codebase is no longer a small research prototype. It now behaves like a maintainable framework with multiple contracts and execution lanes.

The CI surface now checks Ruff, tests, package artifact contents, import boundaries, and a CPU-safe CLI smoke that fails when `factor_library.json` is empty.

## Canonical Execution Paths

### Mining path

```mermaid
flowchart LR
    A["Config"] --> B["PaperProtocol"]
    A --> C["DatasetContract"]
    A --> D["MemoryPolicy"]
    A --> E["EvaluationKernel"]
    B --> F["RalphLoop / HelixLoop"]
    C --> F
    D --> F
    E --> F
    F --> G["FactorLibrary"]
    F --> H["Session + Manifests + Checkpoints"]
```

### Analysis path

```mermaid
flowchart LR
    A["Saved Library"] --> B["Runtime recomputation"]
    C["Dataset"] --> B
    B --> D["evaluate"]
    B --> E["combine"]
    B --> F["visualize"]
```

### Benchmark path

```mermaid
flowchart LR
    A["Runtime manifest / baseline catalog"] --> B["run_table1_benchmark"]
    B --> C["Evaluate factors on freeze universe"]
    C --> D["Build benchmark library"]
    D --> E["Freeze Top-K"]
    E --> F["Evaluate on report universes"]
    F --> G["Artifacts + provenance manifests"]
```

## Findings By Area

### 1. `architecture/`

Status: strong.

This package now contains the right kinds of abstractions. It meaningfully reduces conceptual duplication and gives the rest of the repo a place to grow without bloating the loops further.

Most valuable modules:

- `paper_protocol.py`
- `dataset_contract.py`
- `evaluation_kernel.py`
- `memory_policy.py`
- `families.py`
- `stages.py`

Main remaining gap:

- the architecture layer exists, but not every older concern has been moved into it yet

### 2. `core/`

Status: improved but still the largest debt surface.

`RalphLoop` is better than before because it now composes stages and delegates more work. `HelixLoop` is still structurally heavy. It still owns many optional features and phase-specific concerns in one file.

Main hotspots:

- `core/helix_loop.py`
- `core/ralph_loop.py`
- `core/expression_tree.py`

What improved:

- library mutation logic moved toward a service
- memory persistence is policy-aware
- family/category inference is no longer purely ad hoc

What still needs work:

- more Helix feature logic should move into policies/services
- expression-tree warning behavior should be made explicit and quieter

### 3. `benchmark/`

Status: good direction, not fully cleaned.

`benchmark/runtime.py` is clearly the canonical path now. It supports:

- real runtime loop execution
- memory ablation
- strategy-grid ablation
- cost-pressure analysis
- efficiency benchmarking

Remaining issue:

- `benchmark/helix_benchmark.py` and `run_phase2_benchmark.py` still exist as legacy-facing surfaces

Recommendation:

- either fully document them as legacy analysis/reporting tools or continue collapsing their useful pieces into `benchmark/runtime.py`

### 4. `memory/`

Status: solid base, promising extension surface.

The repo now has a real policy boundary over the raw memory components. That is the right abstraction. The next gains will come from richer concrete policies rather than more loop-local heuristics.

Current state:

- flat memory retrieval exists
- KG retrieval exists
- regime-aware retrieval exists
- family-aware retrieval exists
- `SuccessPattern.confidence` is backward-compatible for older memory JSON
- online regime forgetting now applies deterministic confidence decay and uses `RegimeState.label()` semantics

Most valuable next steps:

- learned family discovery instead of heuristic family inference
- tighter integration between policy-level persistence and any legacy memory-manager path

### 5. `evaluation/`

Status: broad and important.

This package is doing a lot:

- runtime recomputation
- metrics
- portfolio combination
- regime/capacity/causal/significance validation

This is both a strength and a future refactor target. The benchmark/runtime layer currently relies heavily on this package, which is correct, but there is still conceptual spread between evaluation kernel logic and the wider evaluation package.

### 6. `operators/`

Status: materially better after the backend and sandbox work.

The repo now has a real `c` backend contract via Bottleneck-backed compiled CPU operations. That closes one of the bigger credibility gaps between paper claims and implementation reality.

Remaining concern:

- backend availability and numerical equivalence should continue to be treated as test-critical surfaces
- custom operator code paths now share one NumPy-only sandbox, but this should continue to be treated as a security-sensitive surface

### 7. Packaging and first-run UX

Status: materially improved.

Fixed in this stabilization pass:

- packaged distributions include `factorminer/configs/default.yaml`
- packaged distributions exclude `factorminer.tests*`
- default backend is CPU-safe `numpy`
- `--gpu/--cpu` are explicit overrides instead of forcing GPU by default
- `factorminer doctor` checks install/config/dependency/key/path readiness
- `factorminer init-config` writes a mock-friendly starter YAML
- `factorminer quickstart` runs doctor, mines a tiny mock library, and writes a static report
- `factorminer session inspect` summarizes run artifacts and warns on library-size mismatches
- paper-mode quality gates now use `ic_paper_mean = abs(mean(IC_t))`; legacy
  `ic_abs_mean = mean(abs(IC_t))` remains diagnostic only

Deferred:

- full-repo mypy remains non-blocking; scoped type-health checks are documented instead

## Documentation Findings

Before this audit pass, the markdown surface was behind the code in several ways:

- missing the new `architecture/` package from the repo story
- under-documenting the benchmark runtime surface
- missing the strategy-grid ablation lane
- not describing policy-based memory or family discovery adequately
- not showing the new loop/stage model clearly

This audit pass updates:

- `README.md`
- `docs/architecture.md`
- `docs/repo-audit.md`
- `docs/metrics.md`
- `docs/faq.md`
- `docs/reproducibility.md`

## Repository Health Risks

### Medium-risk structural issues

- `HelixLoop` still concentrates many optional concerns in one file
- legacy benchmark/reporting surfaces still coexist with the canonical path
- some config projection logic still exists in multiple places
- full-repo mypy debt is visible but not yet suitable as a blocking CI gate

### Medium-risk product issues

- documentation can drift again if architecture changes are not reflected in the docs layer
- heuristic family inference is useful but not yet a robust research surface

### Low-risk but noisy issues

- NaN-window runtime warnings in expression-tree execution
- style debt in some older loop code

## Recommended Next Implementation Order

If continuing from the current state, the best next order is:

1. Land the trust/onboarding release and GitHub issue replies.
2. Continue shrinking `core/helix_loop.py` into policies and services.
3. Resolve the legacy benchmark split by moving more reporting concerns to the runtime suite.
4. Add learned or cluster-based factor-family discovery.
5. Make policy-level experiment manifests richer and easier to compare across runs.
6. Address the expression-tree warning surface explicitly.
7. Expand scoped mypy cleanup before considering a blocking type gate.

## Recommended GitHub Docs Surface

The best public-facing docs structure for the repo now is:

- `README.md`: project entry point, architecture summary, quick start, benchmark surface
- `docs/metrics.md`: IC semantics and compatibility details
- `docs/faq.md`: answers to setup, data, model, and API-key confusion
- `docs/reproducibility.md`: mock versus private-data reproduction paths
- `docs/architecture.md`: technical architecture and runtime contract
- `docs/repo-audit.md`: implementation inventory, strengths, debt, roadmap

That is enough structure for a serious GitHub repo without overbuilding a docs system too early.

## Bottom Line

The repo is no longer just a paper prototype. It now has:

- a canonical architecture layer
- a canonical benchmark-runtime surface
- stage-composed loops
- policy-based memory
- pluggable dependence metrics
- stronger reproducibility and artifact semantics

The next challenge is not invention of more surfaces. It is disciplined consolidation of the remaining heavy modules around the architecture layer that now exists.
