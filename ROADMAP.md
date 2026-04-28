# Roadmap

This roadmap reflects the current post-refactor state of the repository.

## Current State

Recently completed:

- paper-faithful IC semantics with explicit `ic_mean`, `ic_paper_mean`,
  `ic_abs_mean`, `icir`, and `ic_paper_icir`
- first-run `factorminer quickstart` and richer `validate-data` next steps
- static reports that define admission metrics and warn on legacy artifacts
- explicit `PaperProtocol` and `DatasetContract`
- stage-composed Ralph and Helix loops
- policy-based memory surface
- pluggable dependence metrics
- Bottleneck-backed `c` backend
- canonical runtime benchmark suite
- strategy-grid benchmark ablations over `memory policy × dependence metric × backend`
- family-discovery prompt context
- packaging checks for wheel/sdist config inclusion and test exclusion
- CPU-safe default config plus `doctor`, `init-config`, and `session inspect`
- deterministic synthetic fallback and shared custom-operator sandboxing
- improved GitHub-facing documentation

## Immediate Priorities

### 1. Trust and onboarding release hygiene

Goal:

- keep `v0.2-trust-and-onboarding` focused on correctness, docs, issue
  templates, reproducible quickstart, and static reports

Why:

- public adoption depends on users being able to verify metrics, data
  requirements, and first-run behavior before deeper research features

### 2. Continue shrinking `HelixLoop`

Goal:

- move more optional feature logic into services, policies, and reusable validation surfaces

Why:

- `factorminer/core/helix_loop.py` is still the largest concentration of architectural debt

### 3. Finish benchmark-surface consolidation

Goal:

- keep `factorminer.benchmark.runtime` as the single productized benchmark surface

Why:

- legacy benchmark/reporting paths still exist beside it

### 4. Richer experiment manifests

Goal:

- improve comparison across runs by carrying more policy, family, dependence, and benchmark metadata into artifacts

Why:

- the architecture is now explicit enough that cross-run comparability should be first-class

## Research Priorities

### 1. Learned factor-family discovery

Current family discovery is heuristic. The next step is learned or clustered family structure over admitted and rejected formulas.

### 2. Regime-conditioned memory beyond prompt text

Current regime-aware retrieval changes ranking/context. The next step is deeper regime-conditioned evolution and persistence.

### 3. Library-utility optimization

Move beyond single-factor admission toward marginal contribution to composite or portfolio utility.

### 4. Dependence-metric expansion

Potential extensions:

- partial correlation
- mutual information
- HSIC-style nonlinear dependence

## Engineering Priorities

### 1. Continue type-health cleanup

Ruff is clean in the standard workflow. Full-repo mypy still exposes legacy type debt, so the near-term target is scoped typing on config loading, runtime fallback, custom operators, online regime memory, and CLI helper surfaces.

### 2. Eliminate remaining persistence split

Policy-based memory persistence exists, but some older manager-style paths still remain in the codebase.

### 3. Quiet the expression-tree warning surface

The current NaN-window warnings are known and should be handled more explicitly.

### 4. Better CI depth

The repo now has tests, linting, package artifact checks, and a non-empty CLI smoke. Later improvements should include:

- scoped linting on changed files
- optional benchmark smoke tests
- issue-template and label hygiene for setup, data, metrics, paper fidelity,
  docs, and good-first-issue triage
- non-blocking full-repo mypy reporting

## Suggested Next Build Order

1. Land the trust/onboarding release notes and GitHub issue replies
2. Continue extracting Helix services
3. Collapse more legacy benchmark/reporting paths into runtime
4. Build learned family discovery
5. Expand benchmark ablations and cross-run manifests
6. Address expression-tree warnings
7. Expand scoped mypy coverage before making any type gate blocking

## Not A Priority Right Now

These are intentionally lower priority than the structural items above:

- frontend/demo polish
- broad style cleanup in untouched modules
- large new benchmark families before the runtime surface is fully consolidated

## Definition Of “Healthy Main Branch”

The repo should be considered healthy when:

- tests pass in CI
- benchmark runtime remains canonical
- architecture docs stay in sync with code
- `output/` remains untracked
- new feature work lands through architecture-layer boundaries instead of loop bloat
