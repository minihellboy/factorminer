# Roadmap

This roadmap reflects the current post-refactor state of the repository.

## Current State

Recently completed:

- explicit `PaperProtocol` and `DatasetContract`
- stage-composed Ralph and Helix loops
- policy-based memory surface
- pluggable dependence metrics
- Bottleneck-backed `c` backend
- canonical runtime benchmark suite
- strategy-grid benchmark ablations over `memory policy × dependence metric × backend`
- family-discovery prompt context
- improved GitHub-facing documentation

## Immediate Priorities

### 1. Continue shrinking `HelixLoop`

Goal:

- move more optional feature logic into services, policies, and reusable validation surfaces

Why:

- `factorminer/core/helix_loop.py` is still the largest concentration of architectural debt

### 2. Finish benchmark-surface consolidation

Goal:

- keep `factorminer.benchmark.runtime` as the single productized benchmark surface

Why:

- legacy benchmark/reporting paths still exist beside it

### 3. Richer experiment manifests

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

### 1. Resolve repo-wide lint debt

The repo is not Ruff-clean yet. This is important, but it should be done as a deliberate cleanup effort rather than mixed into unrelated feature work.

### 2. Eliminate remaining persistence split

Policy-based memory persistence exists, but some older manager-style paths still remain in the codebase.

### 3. Quiet the expression-tree warning surface

The current NaN-window warnings are known and should be handled more explicitly.

### 4. Better CI depth

The repo now has a basic test workflow. Later improvements should include:

- scoped linting on changed files
- optional benchmark smoke tests
- packaging/build checks

## Suggested Next Build Order

1. Continue extracting Helix services
2. Collapse more legacy benchmark/reporting paths into runtime
3. Build learned family discovery
4. Expand benchmark ablations and cross-run manifests
5. Address expression-tree warnings
6. Run a dedicated lint cleanup pass

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
