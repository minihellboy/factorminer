# Roadmap

This roadmap reflects the current state of the repository after three
completed work passes: the architecture/protocol refactor (this file's
original baseline), a 17-item landscape-extension round (data connectors,
crowding/risk/portfolio evaluation, MCP hardening, prompt caching/cascade
routing, sealed multi-evaluator search), and a deep math/logic audit +
primary-source research pass that fixed 12 real bugs and closed 4
previously-deferred methodology gaps. See
[Landscape Review & Extension Roadmap](docs/landscape-and-extensions.md) for
the full extension inventory and [Repo Audit](docs/repo-audit.md) for the
technical audit this file summarizes into action items.

## Current State

Recently completed (this pass):

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

## Immediate Priorities

### 0. Get three passes of work out of the working tree

**This is the actual top blocker, not a backlog item.** `git log` still ends
at `fea3553 add financial services integration`; every item above —
architecture refactor follow-ups, all 17 landscape-extension items, all 12
bug fixes — is sitting as ~106 uncommitted files. None of it has run through
CI, none of it is reviewable on GitHub, and issue #5 (financial-services
integration) can't be closed against a merged PR because nothing has been
pushed since it was implemented.

Goal:

- split into reviewable commits/PRs along the natural seams that already
  exist (the 9 landscape-extension epics; the bug-fix pass as its own PR
  with each fix's regression test alongside it; docs sync separately)
- run `scripts/check.py` and the packaging checks against the new
  `architecture/`, `evaluation/`, `data/`, and `mcp/` files specifically —
  these packages grew the most and packaging config (`pyproject.toml`
  `include`/`exclude`) has not been re-verified against them
- close issue #5 once merged, linking to `docs/financial-services-integration.md`

Why:

- none of the correctness work in this roadmap matters to anyone until it's
  reachable outside this working tree

### 1. Unify `RalphLoop` and `HelixLoop` onto shared generation/memory infrastructure

**New finding from this pass, and now the single highest-value structural
item** — higher priority than the pre-existing `HelixLoop`-shrinking goal
below, because it's actively getting worse: every recent improvement
(prompt caching, cascade routing, retry/repair, policy-based memory) landed
*only* on the `HelixLoop` + `agent.factor_generator` + debate path.
`RalphLoop` — the default, non-debate mining path — still uses:

- its own `class FactorGenerator` (`core/ralph_loop.py:180`, ~54 lines): no
  cascade support, no `cacheable_prefix`, no repair/retry, and a hand-rolled
  regex parser instead of the shared, validated `agent/output_parser.py`
- `ExperienceMemoryManager` (`memory/experience_memory.py`, 594 lines)
  instead of the policy-based memory surface (`architecture/memory_policy.py`)
  that `HelixLoop` uses

Goal:

- point `RalphLoop` at `agent.factor_generator.FactorGenerator` and a
  `PaperMemoryPolicy`-equivalent default instead of its private shadow
  classes, so cost savings and quality-repair benefits apply to every
  mining run, not only debate-mode Helix runs
- retire `ExperienceMemoryManager` once nothing depends on it, or
  explicitly document it as a frozen legacy shim if full retirement isn't
  safe in one pass

Why:

- shipping infrastructure improvements that only reach one of two loops is
  a widening architectural fork, not a net improvement to the whole system

### 2. Collapse the benchmark surface split

`benchmark/helix_benchmark.py` (2,203 lines) and `benchmark/runtime.py`
(2,289 lines) are now nearly equal in size — this is two full parallel
implementations, not "one canonical path plus a thin legacy shim." This has
been flagged as a priority for at least two prior passes without action;
it is now large enough that consolidation is a multi-session project in its
own right, not a quick win.

Goal:

- either finish collapsing `helix_benchmark.py`'s useful, non-duplicated
  pieces into `runtime.py` and delete it, or make an explicit, documented
  decision to freeze it as legacy-only with a clear deprecation note at the
  top of the file (stop letting it silently keep pace in size)

Why:

- `factorminer.benchmark.runtime` is documented repo-wide as *the*
  canonical benchmark surface; a same-size shadow implementation undermines
  that claim every time someone has to figure out which one a given call
  site should use

### 3. Continue shrinking `HelixLoop`

Goal:

- move more optional feature logic into services, policies, and reusable
  validation surfaces (`core/helix_loop.py` is still 1,489 lines / 45
  class+def members — essentially unchanged in size despite three rounds of
  feature work landing around it, which is a good sign the architecture
  layer held for *new* work, but the pre-existing bulk was never reduced)

Why:

- still the largest single concentration of architectural debt in `core/`

### 4. Quiet the expression-tree NaN-window warning surface

Goal:

- the `RuntimeWarning: Mean of empty slice` / `Degrees of freedom <= 0`
  warnings from `core/expression_tree.py`'s rolling-statistic helpers
  (`nanmean`/`nanvar` on all-NaN windows) fired in every full test run
  across all three of this repo's work passes and have been on the roadmap
  each time without being addressed — either guard the all-NaN case
  explicitly (return NaN without invoking the warning-triggering reduction)
  or suppress with a documented, narrowly-scoped `warnings.catch_warnings`
  at the exact call sites, not a blanket filter

Why:

- cheap to fix, has been deferred three times, and noisy warnings make real
  regressions harder to spot in CI output

## Research Priorities

### 1. Let the round-2 research surfaces mature before scanning for round 3

Round 2 added substantial new research surface: factor crowding (now with
two residualization modes), risk/portfolio construction, capacity/impact
modeling, sealed multi-evaluator search, GraphSAGE model co-optimization,
RFT dataset export. **Recommendation: do not run another landscape-scan
round yet.** Each of these is genuinely new surface area with its own
failure modes (the deep audit found 12 real bugs in round 2's own work
within one pass); let them accumulate real mining-run usage and test
coverage before adding a fourth layer on top. This mirrors this repo's own
stated principle (`ROADMAP.md`, `docs/repo-audit.md`): "the next challenge
is not invention of more surfaces, it is disciplined consolidation."

### 2. Learned factor-family discovery

Still heuristic (unchanged across all three passes). The next step is
learned or clustered family structure over admitted and rejected formulas.

### 3. Library-utility optimization

Move beyond single-factor admission toward marginal contribution to
composite or portfolio utility. `evaluation/model_zoo.py`'s
`EnsembleMarginalUtilityService` (landed in round 2) is a starting point for
this, not yet the finished feature.

### 4. Regime-conditioned memory beyond prompt text

Current regime-aware retrieval changes ranking/context. The next step is
deeper regime-conditioned evolution and persistence.

## Engineering Priorities

### 1. Continue type-health cleanup

Ruff is clean repo-wide (verified after every change in the last two
passes). Full-repo mypy still exposes legacy type debt. Round 2 added many
new `@dataclass(frozen=True)` contracts (`CoMetricResult`, `SealedFeedback`,
`ModelZooConfig`, and others) that are naturally well-typed already — these
are a good anchor for the first *blocking* scoped mypy target, rather than
starting from the oldest, loosest-typed modules.

### 2. Eliminate the remaining persistence split

Covered concretely under Immediate Priorities #1 above (`ExperienceMemoryManager`
vs. policy-based memory) — this item is no longer abstract; it has a name
and a call-site list.

### 3. Better CI depth

- scoped linting on changed files
- optional benchmark smoke tests
- issue-template and label hygiene
- non-blocking full-repo mypy reporting
- a CI job that actually runs (none of this has been exercised in CI since
  nothing has been pushed — see Immediate Priorities #0)

## Suggested Next Build Order

1. Push three passes of uncommitted work as reviewable PRs; verify
   packaging config against the new packages; close issue #5
2. Unify `RalphLoop` onto `agent.factor_generator.FactorGenerator` and
   policy-based memory
3. Quiet the expression-tree NaN-window warnings (small, deferred 3x)
4. Collapse or formally freeze `benchmark/helix_benchmark.py`
5. Continue extracting `HelixLoop` services
6. Build learned family discovery
7. Expand scoped mypy coverage, anchored on round-2's already-typed
   dataclasses, before making any type gate blocking

## Not A Priority Right Now

These are intentionally lower priority than the structural items above:

- a round-3 landscape/feature scan (explicitly deferred — see Research
  Priorities #1)
- frontend/demo polish
- broad style cleanup in untouched modules
- large new benchmark families before the runtime surface is fully
  consolidated

## Definition Of "Healthy Main Branch"

The repo should be considered healthy when:

- tests pass in CI (currently: passes locally, has not run in CI since
  nothing has been pushed)
- benchmark runtime remains the single canonical surface (currently: not
  true — see Immediate Priorities #2)
- architecture docs stay in sync with code
- `output/` remains untracked
- new feature work lands through architecture-layer boundaries instead of
  loop bloat (round 2's own new work largely held this line; the
  pre-existing `RalphLoop`/`HelixLoop` fork did not)
