# Roadmap

This roadmap reflects the current state of the repository after three
completed work passes: the architecture/protocol refactor (this file's
original baseline), a 17-item landscape-extension round (data connectors,
crowding/risk/portfolio evaluation, MCP hardening, prompt caching/cascade
routing, sealed multi-evaluator search), and a deep math/logic audit +
primary-source research pass that fixed 12 real bugs and closed 4
previously-deferred methodology gaps. See
[Landscape Review & Extension Roadmap](docs/landscape-and-extensions.md) for
the full extension inventory,
[July 2026 Research & Business Outlook](docs/july-2026-research-and-business-outlook.md)
for the newer paper/agent/business scan, and [Repo Audit](docs/repo-audit.md)
for the technical audit this file summarizes into action items.

## Current State

Consolidation roadmap completed on `codex/financial-services-integration`:

- the 107-file architecture/research/audit working tree was verified,
  committed as `5fd351d`, and pushed; package builds include the expanded
  `architecture`, `evaluation`, `data`, `mcp`, and config surfaces while
  excluding tests
- `RalphLoop` and `HelixLoop` now share
  `agent.factor_generator.FactorGenerator`, validated output parsing,
  repair/cascade/cache behavior, and policy-based memory persistence
  (`2f509aa`); `ExperienceMemoryManager` is a frozen compatibility facade,
  not a production loop dependency
- Phase-2 comparison and ablation execution now runs through
  `benchmark.runtime`; `helix_benchmark.py` fell from 2,203 lines to a
  46-line import shim and the combined benchmark implementation shrank by
  1,349 lines (`673a024`)
- optional Phase-2 component construction moved into
  `Phase2ComponentFactory`; `HelixLoop` fell from 1,490 to 1,270 lines
  (`af5f22d`)
- rolling and cross-sectional all-NaN reductions now return NaN explicitly
  without warning suppression; focused expression/Ralph/island coverage is
  free of project RuntimeWarnings (`751c027`)
- final repository validation passes: 855 tests, Ruff, all 26 repository
  checks, and `git diff --check`; the remaining 10 pytest warnings are outside
  the completed expression-tree scope (three third-party SWIG deprecations,
  six portfolio-fixture warnings, and the intentional legacy-shim warning)

Completed in the preceding audit/research pass:

- 12 confirmed math/logic bugs found and fixed with regression tests proving
  both the bug and the fix (capacity net-of-cost was a complete no-op;
  Deflated Sharpe used the wrong kurtosis convention; a BPR ranking-loss
  gradient had the wrong sign; a span-basis centering bug caused false
  "worth an LLM jump" recommendations; and 8 more — see chat history for the
  full list)
- real Lou–Polk factor-crowding residualization (`crowding.py`
  `residual_mode="factor_regression"`), reusing the existing Ken French
  connector, opt-in and fail-closed to the previous cross-sectional proxy
- `temporal_reweight`'s first-rebalance-block look-ahead removed; now
  genuinely walk-forward end to end, matching AlphaForge's actual algorithm
- MCP server hardening: authenticated HTTP transport (`sse` was previously
  unauthenticated), install-path redaction on `doctor`
- prompt caching, cheap-first cascade routing, `parent_formula` lineage,
  crowding/risk/portfolio/capacity evaluation modules, EDGAR/futures data
  connectors, sealed multi-evaluator research mode, RFT dataset export —
  see [Landscape Review](docs/landscape-and-extensions.md) §10 for all 17
  items

Earlier (pre-existing baseline this roadmap already reflected):

- paper-faithful IC semantics, stage-composed Ralph/Helix loops,
  policy-based memory surface, pluggable dependence metrics, canonical
  runtime benchmark suite, CPU-safe default config, `doctor`/`init-config`/
  `session inspect`

## Completed Immediate Priorities

| # | Item | Outcome | Evidence |
| ---: | --- | --- | --- |
| 0 | Commit and push the 107-file working tree | Complete on the tracked feature branch; distribution contents were inspected after `uv build` | `5fd351d`; 839-test baseline, Ruff, `scripts/check.py`, wheel and sdist checks |
| 1 | Unify Ralph/Helix generation and memory | Complete; one generator/parser path and one policy persistence path serve both loops | `2f509aa`; 90 focused loop/cascade/cache tests |
| 2 | Collapse the benchmark split | Complete; the old module is import-only and Phase-2 execution delegates to `runtime.py` | `673a024`; 46-line shim, 33 focused tests |
| 3 | Continue shrinking `HelixLoop` | Completed for this pass; optional dependency/component construction is now a reusable service | `af5f22d`; 1,490 to 1,270 lines, 79 focused tests |
| 4 | Quiet expression-tree NaN-window warnings | Complete; explicit validity/degree-of-freedom guards, no blanket filters | `751c027`; 154 focused tests with no project RuntimeWarnings |

The completed sequence was then exercised as one integrated change set:
`855 passed, 10 warnings` in the full test suite, with Ruff and all 26
repository checks clean.

## Immediate Follow-ups

1. Merge the pushed feature branch, observe the repository CI result, then
   close issue #5 with a link to
   [Financial Services Integration](docs/financial-services-integration.md).
   Merge/issue closure remains coordination work; the implementation and
   branch push are complete.
2. Continue extracting Helix validation, auto-invention, and checkpoint
   persistence in similarly bounded services. At 1,270 lines the loop is
   materially smaller, but it remains the largest debt concentration in
   `core/`.
3. Remove the `helix_benchmark.py` and `ExperienceMemoryManager`
   compatibility shims after a documented deprecation window and a call-site
   audit of downstream users.
4. Address the separate `evaluation/portfolio.py` empty-quintile warning
   surface now visible after expression-tree noise was removed.

## Research Priorities

The explicitly requested Round 3 scan is complete. It did not justify a broad
new feature round. The strongest evidence instead moves validation,
reproducibility, and measured harness selection ahead of more generation
surface; see the [July 2026 outlook](docs/july-2026-research-and-business-outlook.md)
for sources, commercial positioning, proof gates, and rejected directions.

### 0. One credible, reproducible evidence baseline

Complete a paper-comparable Qlib/FactorMiner reproduction with a versioned
dataset contract, typed random and canonical-factor controls, explicit costs,
and an immutable result bundle. This remains the prerequisite for interpreting
any new search, memory, or agent result.

### 1. Asset-pricing evidence and novelty gate

Make a reproducible evidence pack—not another generator—the next major research
surface. It should combine point-in-time EDGAR/accounting provenance,
multivariate horse races, factor/library spanning, multiple-testing controls,
novelty against a versioned anomaly catalog, liquidity/capacity filters, held-out
evaluation, and explicit falsification criteria. The highest-value new source
screened 280 hypotheses down to only a small final set after these filters.

### 2. Trajectory and harness laboratory

Replay the existing prompts, six memory policies, retrieval modes, providers,
cascade choices, and bounded agent topologies on identical historical tasks and
budgets. `architecture/lifecycle.py`, policy schemas, provenance, sealed
evaluation, and RFT export already provide the substrate. Add no seventh fixed
memory policy until a held-out comparison proves a gap.

### 3. General-agent gateway, conditional on customer pull

Keep FactorMiner a governed laboratory that general agents call. With three to
five design partners, extend the existing MCP/API boundary with durable jobs,
an artifact registry, RBAC/SSO, data entitlements, signed manifests, provider
failover, budgets, approval gates, and audit exports. Do not build this control
plane before the evidence artifact itself is valuable.

### 4. Typed trajectory evolution

Experiment with QuantaAlpha-style weak-step mutation and compatible trajectory
crossover while preserving the typed AST, complexity bounds, and provenance.
Separate structural LLM edits from cheaper numeric-window optimization. Promote
the approach only if it improves accepted-candidate yield per dollar and OOS
survival on the common baseline.

### 5. Later research bets

Learned factor families, portfolio/library marginal utility, a falsification
critic, and a separately sandboxed code/program-factor lane remain justified
experiments. They stay behind the evidence baseline and trajectory laboratory;
arbitrary Python must not become the default factor language.

## Engineering Priorities

### 1. Continue type-health cleanup

Ruff is clean repo-wide (verified after every change in the last two
passes). Full-repo mypy still exposes legacy type debt. Round 2 added many
new `@dataclass(frozen=True)` contracts (`CoMetricResult`, `SealedFeedback`,
`ModelZooConfig`, and others) that are naturally well-typed already — these
are a good anchor for the first *blocking* scoped mypy target, rather than
starting from the oldest, loosest-typed modules.

### 2. Retire persistence compatibility after the deprecation window

Production loops now persist through `MemoryPolicy`; the old
`ExperienceMemoryManager` has no production loop dependency and is explicitly
frozen. Remove its public compatibility export after downstream call sites
have had a deprecation window.

### 3. Better CI depth

- scoped linting on changed files
- optional benchmark smoke tests
- issue-template and label hygiene
- non-blocking full-repo mypy reporting
- confirm the pushed branch and eventual merge run successfully under hosted CI

## Suggested Next Build Order

1. Merge the pushed consolidation branch and close issue #5 after hosted CI
2. Complete one paper-comparable, immutable evidence baseline
3. Build the asset-pricing evidence and novelty gate
4. Build offline trajectory/harness replay and compare existing memory policies
5. Continue extracting Helix validation/auto-invention/checkpoint services
6. Remove compatibility shims after their deprecation window
7. Quiet `evaluation/portfolio.py`'s empty-quintile warnings
8. Add the general-agent enterprise control plane only with design-partner pull
9. Expand scoped mypy coverage, anchored on round-2's already-typed
   dataclasses, before making any type gate blocking

## Not A Priority Right Now

These are intentionally lower priority than the structural items above:

- general finance chat/search, Excel/pitchbook automation, or terminal replacement
- live trading/execution or an alpha marketplace
- a multi-agent rewrite or another hand-designed memory policy without replay evidence
- arbitrary Python as the default factor representation
- proprietary data acquisition before design-partner traction
- frontend/demo polish
- broad style cleanup in untouched modules
- large new benchmark families before the runtime surface is fully
  consolidated

## Definition Of "Healthy Main Branch"

The repo should be considered healthy when:

- tests pass locally and in hosted CI (local verification is complete; hosted
  status depends on the pushed branch/merge workflow)
- benchmark runtime remains the single canonical execution surface (now true;
  the old module is import-only)
- architecture docs stay in sync with code
- `output/` remains untracked
- new feature work lands through architecture-layer boundaries instead of
  loop bloat (the generator/memory fork is closed and Phase-2 construction is
  now service-owned)
