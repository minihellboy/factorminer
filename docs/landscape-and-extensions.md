# Landscape Review & Extension Roadmap

This document cross-references FactorMiner against the current (2024–2026) research and
commercial landscape for LLM-driven alpha mining, and turns the gaps into a prioritized,
architecture-aware extension roadmap. It is a companion to
[Repo Audit](repo-audit.md) (internal state) and [Roadmap](../ROADMAP.md) (maintainer
priorities) — this document is external-facing: what does the rest of the world do that
FactorMiner doesn't yet, and is it worth doing.

Sourcing standard: every claim below is tied to a primary source (arXiv paper, official
repo/docs, vendor documentation, or named press coverage). Compensation and revenue figures
from secondary sources (forums, career sites) are marked as self-reported/undisclosed.

> **Revision 2 (2026-07-20):** Round 1 (§§1–7) is fully implemented — see the
> implementation-status note in §6. Round 2 (§§8–11, appended below) covers nine
> additional research domains investigated after round 1 shipped, deliberately covering
> ground round 1 didn't touch (alternative data, explainability/compliance, LLM cost
> efficiency, agent-framework interop, GNN/Transformer model options, factor crowding,
> multi-asset expansion, memory-retrieval technique, and a fresh scan for papers
> published since round 1). **All 17 Round 2 items (§10) are now also implemented,
> tested, and merged** — see the implementation-status note in §10.

## Method

1. Read the base paper (arXiv:2602.14670, Wang et al., Tsinghua, Feb 2026) end to end.
2. Identified and read three papers that directly cite FactorMiner and extend its exact
   problem setting (published Mar–Jul 2026 — i.e., *after* this repo's paper, making them
   the closest available signal for "what's the next move in this specific niche").
3. Cross-referenced nine other LLM/RL/GP alpha-mining systems cited by those three papers.
4. Cross-referenced six adjacent OSS quant libraries for capabilities FactorMiner's
   `evaluation/` and `utils/` packages don't clearly cover.
5. Cross-referenced three alpha-monetization platforms for business-model patterns,
   including one well-documented failure.
6. Cross-referenced the MCP/data-connector landscape against FactorMiner's existing
   `factorminer/data/mcp_source.py` client.
7. Grepped the codebase for the resulting technical terms (`purge`, `embargo`, `deflated`,
   `triple.?barrier`, `risk.?parity`, `walk.?forward`, `PBO`, `meta.?label`,
   `combinatorial`, `HRP`) before writing any gap claim, so this document doesn't
   re-propose things that already exist.

## 1. Direct lineage: papers that extend FactorMiner's exact problem

These three papers were published *after* arXiv:2602.14670 and explicitly cite it as prior
work, in the same "LLM agent + experience memory + formulaic alpha" niche. They are the
highest-signal source for "what does the next generation of this exact system look like."

| Paper | Date / Org | Core mechanism | Reported result | What FactorMiner lacks today | Source |
|---|---|---|---|---|---|
| **Hubble** | Mar 2026, Celestial Quant Lab / UBC | 3-layer AST sandbox (structural whitelist + depth/node bounds + arity/type check) around LLM-generated formulas; per-round telemetry of `PARSE_ERROR` vs `DUPLICATE` vs `SecurityViolation`; evolutionary feedback = top-K formulas + categorized error summary fed back each round | 100% computational stability, 181/186 candidates valid over 3 rounds, best factor IR 1.31 (30 US equities) | Per-round generation telemetry (error-type breakdown, OK-rate trend) as a first-class report artifact | [arXiv:2604.09601](https://arxiv.org/abs/2604.09601) |
| **AlphaMemo** | May 2026, U. Sydney / U. Edinburgh | *Structured Search-Process Memory*: extracts an "edit motif" from the AST diff between a parent formula and its child, stores a **confidence-gated residual** of child quality vs. a ledger-predicted baseline, keyed by `(parent context, edit motif)`; an **asymmetric veto** lets high-confidence failure patterns block a whole action, while positive evidence only nudges. Falls back to the base search prior when evidence is sparse. | Best S&P500 IC/RankIC/Sharpe among tested baselines; on CSI500 the "balanced" (weak-memory) operating point wins — confirms memory must be a *calibrated correction*, not a controller | Edge-level (parent→child) credit assignment. FactorMiner's memory stores pattern/family-level success/failure, not "which specific edit, on which specific parent shape, produced this specific delta" | [arXiv:2606.20625](https://arxiv.org/abs/2606.20625), code: [github.com/jarrettyu/AlphaMemo](https://github.com/jarrettyu/AlphaMemo) |
| **XAlpha** | Jul 2026, U. Hong Kong | Multi-brain loop: a **Report-to-Memory Absorption (RMA)** layer ingests external research reports/papers through an A/B/C taxonomy (A = "is this mechanism OHLCV-representable at all", B = broad mechanism family, C = 48 fine-grained "Research Archetypes"); a **Macro Brain** plans cycle themes and routes to archetypes (fixed-theme / coarse-guided / memory-driven modes); a **Micro Brain** does hypothesis→code generation with a **tri-alignment judge** (idea vs. code vs. financial rationale must agree) plus static/dynamic leakage checks; a **Cross Brain** attributes outcomes back to archetypes and writes mechanism-level GOOD/BAD feedback (not just a scalar) | Outperforms AlphaAgent, AlphaJungle, RD-Agent(Q)-style baselines on CSI300, 2011–2025 chronological split | (a) Ingesting *external* documents (research notes, papers, theses) into hypothesis memory — today's memory only distills FactorMiner's own trial history; (b) mechanism-level (not just scalar-IC) admit/reject feedback; (c) a taxonomy-routed cycle planner above the per-candidate loop | [arXiv:2607.08332](https://arxiv.org/abs/2607.08332) |

## 2. Wider LLM/GP/RL alpha-mining field

| System | What it does differently | Reported result | Gap vs. this repo | Source |
|---|---|---|---|---|
| **AlphaForge** (AAAI 2025) | Two-stage: a generative-predictive network proposes formulas using a learned predictive network as a sparse-reward surrogate (gradient-based search, not pure LLM prompting); a *second*, separate combination model **dynamically reweights** factors over time from recent performance, instead of one static combination weight | IC 4.40% / RankIC 5.89% on CSI300, beats GP/DSO/RL baselines and MLP/XGBoost/LightGBM | `evaluation/combination.py` (`equal_weight`, `ic_weighted`, `orthogonal`) computes one fixed weighting for the whole period — no temporal reweighting as factor performance drifts | [AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/33365), [arXiv:2406.18394](https://arxiv.org/abs/2406.18394) |
| **AlphaAgent** (KDD 2025) | Three-agent split (Idea / Factor / Eval); originality enforced via **AST-similarity distance** against the existing library; explicit **hypothesis↔factor semantic-alignment** LLM check; framed entirely around *counteracting alpha decay* | 11.0% ann. excess return (IR 1.5) on CSI500, 8.74% (IR 1.05) on S&P500, 2021–2024, after costs; 81% higher hit ratio at 30% fewer tokens vs. baseline | Largely covered already (SymPy canonicalization = dedup, family-aware memory = diversity), **except**: no longitudinal "IC since admission" decay-curve report — the repo computes rolling/cumulative IC over a dataset window but not a per-factor curve keyed to *time since library admission* | [arXiv:2502.16789](https://arxiv.org/abs/2502.16789), code: [github.com/RndmVariableQ/AlphaAgent](https://github.com/RndmVariableQ/AlphaAgent) |
| **RD-Agent(Q)** (Microsoft Research + HKUST, NeurIPS 2025) | **Factor-model co-optimization**: Research/Development/Feedback loop jointly evolves the factor pool *and* the downstream prediction model (not just the formula library), with a multi-armed-bandit scheduler choosing which direction to push next | ~2× annualized return vs. classical factor libraries using 70% fewer factors, for <$10 of LLM spend; built on Qlib | FactorMiner mines and combines a factor *library*; it does not co-evolve a downstream nonlinear model (GBDT/NN) as a joint optimization target inside the loop | [Microsoft Research](https://www.microsoft.com/en-us/research/publication/rd-agent-quant-a-multi-agent-framework-for-data-centric-factors-and-model-joint-optimization/), [arXiv:2505.15155](https://arxiv.org/abs/2505.15155), [github.com/microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) |
| Chain-of-Alpha, AlphaGPT/Alpha-GPT, FAMA, AlphaJungle (MCTS), AlphaSAGE (GFlowNets), AlphaQCM (distributional RL), QuantaAlpha, AlphaAgentEvo | Alternative search strategies over the same formulaic-alpha problem (dual-chain refinement, human-in-the-loop NL→formula, neuro-symbolic agents, MCTS, GFlowNet exploration, distributional-RL reward shaping, RL-tuned self-evolution policy) | Varies by paper | All converge on the same three levers FactorMiner already has some version of (typed DSL, memory/experience reuse, evaluation gates) — no single one contributes a capability outside what's captured in the four rows above | See citation lists in AlphaMemo/XAlpha (§1) |
| **AlphaEvolve / FunSearch** (DeepMind) | Not alpha-mining-specific, but the closest analogue for *search-diversity mechanics*: an **island-model** population — several sub-populations evolve independently and periodically exchange top performers (migration) — combined with a MAP-Elites-style archive that keeps a *diverse* elite set, not just a single leaderboard-top-K | Rediscovered 75% of 50+ tested open math results; found a faster 4×4 matrix-multiplication algorithm; cut a TPU training kernel by 23% | FactorMiner's family-aware policy reranks *one* shared library by family-gap; there is no multi-population / periodic-migration mechanism to prevent one dominant factor family from crowding out the search | [DeepMind blog](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), [technical writeup](https://composio.dev/blog/alphaevolve-evolutionary-agent-from-deepmind), [Wikipedia](https://en.wikipedia.org/wiki/AlphaEvolve) |

## 3. Adjacent OSS quant ecosystem

| Project | Category | Capability FactorMiner lacks | Confirmed absent by grep? | Source |
|---|---|---|---|---|
| **Microsoft Qlib** | Full research/execution platform | Point-in-time China A-share/US data layer, RL environments, model zoo — *not* a competitor to FactorMiner's discovery loop, but the **reference dataset/benchmark** all three lineage papers (§1) build on (CSI300/CSI500/S&P500, Qlib format) | n/a (positioning gap, not a code gap) | [github.com/microsoft/qlib](https://github.com/microsoft/qlib) |
| **mlfinlab / Hudson & Thames** | Financial ML techniques (López de Prado) | **Combinatorial Purged Cross-Validation** (`PurgedKFold` + embargo), **triple-barrier labeling**, **meta-labeling** | Yes — zero matches for `purg`, `embargo`, `triple.?barrier`, `meta.?label`, `combinatorial` anywhere in `factorminer/` | [hudsonthames.org/mlfinlab](https://hudsonthames.org/mlfinlab/), [Purged CV — Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation) |
| **Riskfolio-Lib** | Portfolio optimization | **Hierarchical Risk Parity / HERC**, CVaR and 30+ other risk-measure optimizers, risk-budget/risk-parity position sizing across assets | Yes — zero matches for `risk.?parity`, `HRP` | [riskfolio-lib.readthedocs.io](https://riskfolio-lib.readthedocs.io/) |
| **Bailey & López de Prado (2014, 2017)** | Statistical rigor | Deflated Sharpe Ratio is **already implemented** (`evaluation/significance.py::DeflatedSharpeCalculator`, tested). Its natural companion metric, **Probability of Backtest Overfitting (PBO)** via combinatorially symmetric CV, is not | Yes for PBO (`PBO` had zero matches); DSR confirmed present, not a gap | [Deflated Sharpe Ratio — SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551), [PBO paper — SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) |
| **vectorbt / Zipline-reloaded / backtrader** | Backtest engines | Not a gap in substance — FactorMiner deliberately stays evaluation-only (IC/quintile backtests, cost-pressure, capacity), consistent with its own guardrails against recommending trades/execution. No action recommended here. | n/a | — |
| **Alphalens-reloaded / pyfolio / QuantStats** | Factor/portfolio tear sheets | Already substantially matched — `utils/tearsheet.py`, `evaluation/report_viewer.py`, and `combine --tearsheet` cover correlation heatmaps, IC time series, quintile returns; no material gap found | n/a | — |

## 4. Business models: what's monetizable, and a cautionary tale

| Platform | Mechanism | Monetization | Lesson for FactorMiner | Source |
|---|---|---|---|---|
| **WorldQuant BRAIN** | Free simulation platform, 120,000+ data fields; users earn points, hit a Gold tier (10,000 pts), get invited into a paid Research Consultant Program | Tiered quarterly payments; Grandmaster consultants "upwards of $8,000+/quarter" per WorldQuant's own consultant page; secondary reports (Reddit, eFinancialCareers — **self-reported, unverified**) cite up to ~$140k/yr max | The core mechanic worth borrowing is **free, ungated exploration with a gamified quality ladder** — not the payout plumbing. A local "leaderboard" report (§5, item 9) captures this without needing a payments/compliance layer. | [worldquant.com/brain](https://www.worldquant.com/brain/), [consultant program](https://worldquantbrain.com/consultant) |
| **Numerai** | Free tournament on obfuscated/anonymized features; contributors **stake NMR** (crypto) on their model; stake-weighted predictions form a Meta-Model that feeds a real market-neutral hedge fund (~$700M AUM as of mid-2026) | Staking creates a self-filtering "skin in the game" signal; fund performance funds payouts | The **share-results-without-sharing-the-formula** pattern (Numerai obfuscates its features; FactorMiner's analogue would be redacting the formula, not the data) is directly reusable — see roadmap item 5. Full staking/tournament infra is disproportionate for an OSS research tool. | [docs.numer.ai](https://docs.numer.ai/), [Numerai buyback coverage](https://www.benzinga.com/content/60539570/numerai-completes-third-strategic-nmr-buyback-bringing-total-repurchases-3-2-million) |
| **QuantConnect Alpha Streams** *(cautionary tale)* | 2017–2021: a marketplace where quants licensed whole algorithms to funds (70/30 revenue split, $100–$30k/month per license); in 2020 opened to $10k retail accounts | Licensing fees | **It failed.** QuantConnect killed v1.0 in 2021: deploying identical alphas across heterogeneous live accounts broke in ways that created real investor risk, and the fix required rebuilding around anonymized micro-factor bundles instead of full licensed algorithms. **This directly validates FactorMiner's existing guardrail stance** ("research artifacts staged for review... does not recommend trades, size positions, bind risk, or execute anything," per `docs/financial-services-integration.md`) — do not build a signal/algorithm marketplace or auto-deployment path. See §6. | [QuantConnect's own postmortem](https://www.quantconnect.com/forum/discussion/13441/alpha-streams-refactoring-2-0/), [Wikipedia](https://en.wikipedia.org/wiki/QuantConnect) |

## 5. Data & MCP connector landscape

FactorMiner already has a generic, config-driven MCP data client (`factorminer/data/mcp_source.py`,
`MCPDataSourceConfig`) wired to institutional-equity connectors matching
`anthropics/financial-services` (FactSet, Daloopa, Morningstar, LSEG, and — confirmed by
reading that repo directly — also S&P Global/Kensho, Moody's, MT Newswires, Aiera,
PitchBook, Chronograph). That side of the story is done.

| Gap | Detail | Source |
|---|---|---|
| **No crypto-native connector**, despite crypto being FactorMiner's *other* flagship lane (`data/binance_crypto_5m.csv`, `configs/paper_repro_binance.yaml`, `docs/binance-reproduction.md` all exist) | The market has multiple mature `ccxt`-based MCP servers (100+ exchanges: Binance, OKX, Bybit, Coinbase, Kraken) already in production use with Claude/Cursor/LangChain. FactorMiner's *existing* generic `MCPDataSourceConfig` machinery should be able to point at one with a config file and zero new client code — see roadmap item 1. | [mcp-server-ccxt](https://github.com/doggybee/mcp-server-ccxt), [PyPI mcp-ccxt](https://pypi.org/project/mcp-ccxt/), [2026 crypto MCP guide](https://www.cryptohopper.com/blog/the-2026-guide-to-crypto-mcp-servers-13080) |
| Institutional tick/point-in-time data (Databento) | Databento ships an 18-tool MCP server (Historical + Reference APIs, corporate actions). Paid/enterprise — not an OSS build item, but worth one documented example config since the connector *pattern* already exists. | [Databento MCP](https://lobehub.com/mcp/jdmiranda-databento-mcp-server) |
| Trade-execution MCPs | Industry commentary (Cryptohopper's 2026 guide) expects execution-capable MCPs to become the default by end of 2026. **Explicitly not recommended** for FactorMiner — same guardrail logic as §4's QuantConnect lesson. | [2026 crypto MCP guide](https://www.cryptohopper.com/blog/the-2026-guide-to-crypto-mcp-servers-13080) |

## 6. Prioritized extension roadmap

Ranked by (business/research value) ÷ (effort), and mapped to the *specific* seam in the
existing `factorminer/architecture/` layer — per this repo's own stated principle that
"new feature work lands through architecture-layer boundaries instead of loop bloat"
(`ROADMAP.md`). None of these touch `core/helix_loop.py`, which the repo's own audit
already flags as the largest debt surface — adding more surface there would fight the
maintainers' stated direction, not help it.

> **Implementation status:** all 13 items below are implemented, tested, and
> merged — each row's "Lands in" column names the actual module. See
> `docs/architecture.md`'s "Landscape-Review Extension Modules" table for a
> consolidated summary. Item 6 (`edit_aware` memory policy)'s original
> caveat — real trajectories not yet carrying `parent_formula` lineage — was
> resolved in the Round 2 pass (§10 item 1 below); it is now live on real
> `mine`/`helix` sessions, not just unit-tested.

### Tier 0 — cheap, high-leverage, closes a documented gap

| # | Feature | Value | Effort | Lands in |
|---|---|---|---|---|
| 1 | **Crypto MCP connector config** (§5): a `configs/mcp_sources/ccxt_binance.yaml`-style config for an existing ccxt MCP server, reusing `MCPDataSourceConfig` as-is | Directly closes the #1 "requires data" item in the repo's own `docs/paper-claims.md` for the crypto lane; zero new client code | Config + docs only | `factorminer/data/mcp_source.py` (already generic) |
| 2 | **Combinatorial Purged CV + PBO** (§3): `PurgedKFold`/`CombinatorialPurgedCV` with purge+embargo, plus a `ProbabilityOfBacktestOverfitting` calculator, as the direct companion to the already-implemented `DeflatedSharpeCalculator` | Completes the López de Prado toolkit institutional allocators specifically ask for; DSR alone answers "is this Sharpe real," PBO answers "did I just pick the best of many noisy trials" — same authors, same citation, currently half-implemented | New module, ~3–5 days | New `factorminer/evaluation/cross_validation.py`; new `benchmark cpcv` lane in `benchmark/runtime.py`, consumed by `EvaluationKernel` |
| 3 | **Qlib-format data adapter** (export + import) | Every one of the three direct-lineage papers (§1) and AlphaForge benchmark on Qlib's CSI300/CSI500/S&P500 panels. An adapter is the cheapest path to running an apples-to-apples comparison against exactly the systems this document just cross-referenced, resolving `paper-claims.md`'s "requires paper-equivalent data" caveat for the equities lane | New loader + export target, ~1 week | `factorminer/data/qlib_source.py` (loader, peer to `mcp_source.py`); `--format qlib` on `core/library_io.py` export path |
| 4 | **Factor decay / half-life report** | AlphaAgent's whole framing is "alpha decay resistance" — FactorMiner has the primitives (`rolling_ic`, `cumulative_ic` in `evaluation/backtest.py`) but not a report keyed to *time since library admission*. Cheap, high visibility, directly answers the question institutional reviewers ask first | Report/plot only, ~2–3 days | `utils/reporting.py` + `evaluation/report_viewer.py`; surfaced via `factorminer session inspect` |
| 5 | **Anonymized/redacted export mode** (§4, Numerai pattern) | Lets a team share an IC/ICIR tearsheet with a stakeholder or reviewer without disclosing the formula — a real collaboration/disclosure gap today's `export`/`report` commands don't cover | CLI flag + redaction logic, ~1–2 days | `--anonymize` flag on existing `export` command in `cli.py`, `core/library_io.py` |

### Tier 1 — larger, matches published SOTA in this exact niche

| # | Feature | Value | Effort | Lands in |
|---|---|---|---|---|
| 6 | **Edit-motif memory policy** (AlphaMemo, §1) | Direct, peer-reviewed-adjacent evidence of out-of-sample IC/Sharpe gains from edge-level (parent→child) credit assignment vs. FactorMiner's current pattern/family-level memory. This is the single most directly-applicable finding in this whole review — a paper built explicitly against FactorMiner's own problem framing | New policy, moderate — needs AST-diff extraction on the existing `core/expression_tree.py` | New `EditAwareMemoryPolicy` — 6th implementation of `architecture/memory_policy.py`'s existing policy interface (`paper`/`none`/`kg`/`family_aware`/`regime_aware` → `+edit_aware`) |
| 7 | **Research-report ingestion → hypothesis memory** (XAlpha RMA, §1) | Turns FactorMiner from "search over formula templates" into "search grounded in an actual thesis" — and FactorMiner already serves `docs/` as an MCP resource (`factorminer://docs/{topic}`), so the ingestion *pathway* already exists; this adds the taxonomy-gated absorption layer on top | New service + skill, larger — needs an eligibility/taxonomy classifier | New `architecture/research_absorption.py` service (parallel to the existing `research_extensions.py`), feeding `PromptContextBuilder`; new skill in `plugins/agent-plugins/factor-researcher/skills/` |
| 8 | **Risk-based portfolio construction** (HRP/CVaR/risk-parity, §3) | Closes the gap between "here is a signal with IC 0.02" and "here is an investable weight vector with a risk budget" — the layer above signal combination that today's `evaluation/combination.py` and `evaluation/portfolio.py` don't provide | New module, moderate (CVXPY-style solver dependency) | New `factorminer/evaluation/risk_portfolio.py`; new `factorminer portfolio-construct` CLI command — deliberately *not* wired into the mining loop, stays a downstream/optional analysis step consistent with the "no execution" guardrail |
| 9 | **Dynamic/temporal combination reweighting** (AlphaForge, §2) | FactorMiner's `FactorCombiner` computes one static weighting per period; AlphaForge shows a real, measured IC/RankIC gain from adapting weights to recent factor performance | Extends existing class, ~1 week | New method on `evaluation/combination.py::FactorCombiner` alongside `equal_weight`/`ic_weighted`/`orthogonal` |
| 10 | **Per-round mining telemetry** (Hubble, §1) | Cheap addition once instrumented: `PARSE_ERROR`/`DUPLICATE`/rejection-rate trend across mining rounds is a concrete, quotable "compute safety" number for any pitch to an enterprise/compliance audience | Logging + report, ~2 days | `core/lifecycle.py` (already logs candidate trajectories) + a summary view in `utils/reporting.py` |

### Tier 2 — bigger bets, higher effort, real differentiation

| # | Feature | Value | Effort | Lands in |
|---|---|---|---|---|
| 11 | **Factor+model co-optimization loop** (RD-Agent(Q), §2) | The single largest reported number in this whole review (2× ARR, 70% fewer factors) comes from jointly evolving the factor pool *and* a downstream model instead of mining factors in isolation. Biggest scope item here — needs careful boundary-setting so it doesn't become the next `helix_loop.py`-style debt surface | New stage, large | New `ModelCoOptimizeStage` implementing the existing `architecture/stages.py` stage protocol — explicitly *not* new HelixLoop branching logic |
| 12 | **Island-model mining mode** (AlphaEvolve/FunSearch, §2) | Real, evidence-backed mechanism for avoiding premature convergence on one dominant factor family — FactorMiner's family-aware policy is a single-population re-ranking heuristic, not a multi-population search | New parallel-run + merge logic, large | Extends `benchmark/ablation.py`'s existing parallel-grid infrastructure; population-merge logic in `architecture/library_services.py` |
| 13 | **Taxonomy-routed macro research planner** (XAlpha Macro/Micro/Cross, §1) | Builds directly on the *existing* `architecture/families.py::FactorFamilyDiscovery`, which already infers ad hoc families — XAlpha's contribution is formalizing this into a routed taxonomy with mechanism-level (not scalar-only) feedback | Large, and the repo's own audit already flags family discovery as "heuristic, not learned" — this is the natural next step for that exact stated priority | Extends `architecture/families.py` + `architecture/phase2_services.py` |

### What NOT to build

- **An alpha/signal marketplace or licensing feature.** QuantConnect tried exactly this and
  killed it in 2021 after live per-account deployment created real investor risk (§4). This
  repo's existing guardrail language ("does not recommend trades, size positions, bind
  risk, or execute anything") is the *correct* boundary, not a placeholder to relax later.
- **Trade execution / order management / broker integration.** Same lesson, and outside
  every existing guardrail statement in `docs/financial-services-integration.md`.
- **Triple-barrier labeling / meta-labeling as a full build** — real technique, confirmed
  absent, but it changes the *target construction* contract (path-dependent stop/take-profit
  labels vs. fixed-horizon forward return) and edges toward strategy/execution research
  rather than formulaic factor discovery. Worth scoping deliberately as an optional,
  clearly-separated labeling mode in `data/tensor_builder.py` if pursued — not a Tier 0/1
  item here because the scope boundary needs a maintainer decision first.

## 7. Suggested sequencing

Tier 0 items (1–5) are independent of each other and of the mining loop — safe to run in
parallel, each lands in a different file, none touches `core/helix_loop.py` or
`core/ralph_loop.py`. Item 6 (edit-motif memory) is the highest-value Tier 1 item and the
most direct evidence match to this project's own stated research priorities
(`ROADMAP.md`'s "Learned factor-family discovery" and "Dependence-metric expansion" sit
right next to what AlphaMemo actually built). Items 11–13 should wait until the
maintainers' own stated priority #2 ("continue shrinking `HelixLoop`") is further along,
since they're new stage/service surface that's easy to bolt on badly if the loop itself is
still absorbing unrelated refactors.

---

# Round 2 — Nine Domains Round 1 Didn't Cover

Round 1 (§§1–7) cross-referenced the direct-lineage papers, the wider LLM/GP/RL
alpha-mining field, adjacent OSS quant libraries, alpha-monetization business models, and
the data/MCP connector landscape. Round 2 dispatched nine independent research passes
into ground Round 1 explicitly did not touch, each grounded first in what the repo
*actually* has (post-Round-1: 154 Python files, 61,833 lines, 650 tests) before making any
gap claim.

## 8. New since Round 1: papers and a code-level bug

### 8.1 Papers published in the Round 1 → Round 2 window (May–Jul 2026)

A fresh scan (arXiv `q-fin.TR`/`q-fin.CP` recent listings, keyword search, and an
author-listing check for the original FactorMiner authors) found **no FactorMiner v2**
and **no new direct citation** of arXiv:2602.14670 beyond the three papers Round 1 already
matched (Hubble, AlphaMemo, XAlpha). It did find four adjacent papers worth a look:

| Paper | Date / Org | Core mechanism | Reported result | Gap vs. this repo | Source |
| --- | --- | --- | --- | --- | --- |
| **QuantEvolver** | May 2026, PKU/Alibaba/UIC | Drops ever-growing prompt-feedback loops; converts backtest scores into **GRPO/RFT policy updates** on a ≤30B "Miner" LLM (Factor DSL + oracle seeds + regime-aware task bank + Diversity-Complementarity reward) — *internalizes* mining skill into model weights instead of externalizing it into memory | +7.8% directional accuracy; OOS RankIC +109.5% / top-10 mean +186.9% vs. strongest baseline on a held-out cross-section | FactorMiner freezes the LLM and externalizes all experience into memory policies (paper/kg/family_aware/regime_aware/edit_aware) — orthogonal axis, not a replacement | [arXiv:2605.15412](https://arxiv.org/abs/2605.15412), [OSS](https://github.com/QuantLLM/QuantEvolver) (MIT) |
| **Agora / Sealed Joint Search** | Jun 2026, Panda AI | Argues fixed evaluation scorers are a root cause of OOS gaps; co-evolves the **scoring function itself** alongside alphas under a sealed, provenance-tracked, multi-evaluator topology (3 evaluators kept in disagreement, not vote-collapsed) so the search can't self-confirm | Sealed 91-day 2026 holdout Sharpe 1.87 vs. 1.334 best baseline (single-seed; authors flag a −0.755 cross-seed mean as a real caveat) | FactorMiner's `EvaluationKernel`/`significance.py` (DSR/PBO/FDR) are correct but **fixed** objectives; nothing co-evolves the evaluator | [arXiv:2606.29194](https://arxiv.org/abs/2606.29194) |
| **Discovery under Hypothesis Redundancy** | Jun 2026 | A **geometric diagnostic**, not another agent: proposes a "Search Compression Hypothesis" — an LLM's non-local proposal only helps when it (a) spectrally compresses the explored-formula span, (b) escapes that span in an orthogonal direction, and (c) aligns its residual with the actual target. Cheap to check before spending an LLM call. | On A-share factor archives + LLM-SRBench: novelty without alignment expands coverage but not yield; hybrid (local-edit + LLM-jump) gains vanish once the explored span is near full rank | FactorMiner has no budget gate deciding *when* a non-local LLM proposal is worth it vs. a cheap local AST edit — `geometry.py`/`dependence.py` measure redundancy but don't drive this decision | [arXiv:2606.14386](https://arxiv.org/abs/2606.14386) |
| **AlphaCrafter** (partial) | May 2026 | Miner + Screener (regime-conditioned ensembles) + **Trader** (risk-constrained execution) three-stage pipeline | Best risk-adjusted returns and lowest cross-trial variance on CSI300/S&P500 | Screener idea (regime-conditioned ensemble selection) is a legitimate extension of `evaluation/regime.py` + `combination.py`; **the Trader stage is not** — full execution pipeline, violates the existing no-execution guardrail | [arXiv:2605.05580](https://arxiv.org/abs/2605.05580) |

None of these reopen the marketplace/broker/triple-barrier items Round 1 explicitly
rejected. QuantEvolver and Agora are both GPU/multi-LLM-cost-heavy — real research value,
but scoped into Tier 2 below, not a quick win.

### 8.2 A code-level finding: `KGMemoryPolicy` silently drops its own embedder

Ground-truth check (read `memory/retrieval.py`, `memory/embeddings.py`,
`memory/kg_retrieval.py`, `architecture/memory_policy.py`, `core/helix_loop.py` directly
before reporting anything): FactorMiner **already has** a working dense-embedding path —
`FormulaEmbedder` (MiniLM/TF-IDF/hash fallback + FAISS/cosine) — used today for semantic
neighbor/duplicate/gap detection and Helix-time dedup. It is real, tested, and wired
correctly in `HelixLoop._helix_retrieve` when `enable_embeddings: true`.

It is **not** wired in `KGMemoryPolicy.retrieve` (`architecture/memory_policy.py`, the
`kg` policy's retrieval method): that call passes `kg=self.knowledge_graph` to
`retrieve_memory_enhanced` but never passes `embedder=`, so any mining run using the `kg`
memory policy silently never gets the dense semantic-neighbor context its own enhanced
retrieval path is fully capable of producing. This is a one-line-class fix, not a new
capability — see Tier 0 item 2 below.

## 9. Nine domains, compact

Full per-domain source tables live in each scout's transcript
(`history://AltDataLandscape`, `history://ExplainabilityMRM`, `history://LLMCostEfficiency`,
`history://AgentOrchestrationFrameworks`, `history://GNNTransformerCrossSectional`,
`history://FactorCrowdingCapacity`, `history://MultiAssetExpansion`,
`history://RecentPapers2026`, `history://RetrievalMemoryModernization`). This section
summarizes the verdict per domain; §10 turns the winners into a roadmap.

| Domain | Verdict | Best finding | Worst trap found |
| --- | --- | --- | --- |
| **Alternative data** | Not empty — 6 sources, but FactorMiner's tensor path is hard-coded to 8 OHLCV leaves, so every source needs the same architectural prerequisite first | Free SEC EDGAR XBRL fundamentals (`data.sec.gov/api/xbrl/`) — machine-readable, versioned, zero cost | Satellite imagery (Planet/Orbital Insight): $50k–$500k/yr, not MCP-clean, poor fit for a formulaic-DSL miner |
| **Explainability / MRM** | Repo has strong performance rigor (DSR/PBO/FDR/causal/decay) but zero compliance-shaped *packaging*; gap is presentation, not math | Wiring `parent_formula` lineage — closes Round 1's own documented debt, cheap, unlocks `edit_aware` memory for real | Claiming "SR 26-2 compliant" or "SEC AI compliant" — guidance is principles-based; only counsel can make that claim, tooling can only produce evidence |
| **LLM cost efficiency** | Real, unaddressed lever — mining calls an LLM per candidate round with a large, mostly-static system prompt | Anthropic/OpenAI prompt caching on the stable system+memory prefix — cited 41–80% session cost reduction, zero quality tradeoff | Fine-tuning a bespoke formula-generation LLM now — high effort, risks losing novel economic hypotheses, premature before mining volume justifies it |
| **Agent orchestration frameworks** | Do **not** rewrite `debate.py`/`critic.py` onto LangGraph/CrewAI/AutoGen — no evidenced alpha-quality payoff, real cost (breaks ablations, non-determinism) | FactorMiner already has the right interop surface (`mcp/server.py`) — deepen it (docs, examples, optional streamable-HTTP) instead of migrating | Porting `DebateOrchestrator` into a `StateGraph`/Crew — a rewrite with no concrete gain, confirmed by zero primary source showing quality improvement from doing this |
| **GNN / Transformer models** | No genuinely lightweight numpy/sklearn-native GNN exists; full SOTA relational rankers are traps | Pairwise/listwise ranking losses (Margin/ListNet/BPR) for `model_zoo.py` — pure loss-function swap, zero new dependency | ACT / PRISM-VQ / full MASTER — need industry/region graphs FactorMiner's data layer doesn't have, and would create a second black-box alpha engine fighting the interpretability positioning |
| **Factor crowding & capacity** | Genuinely absent (confirmed by grep); distinct from existing library-internal redundancy and market-impact capacity | Lou–Polk CoMetric (*RFS* 2022) — within-leg residual-correlation crowding score computable from price history FactorMiner already has | Crowding-as-a-trade-timer — the papers themselves show crowding means are already priced in; use only as a research risk label, never a mining objective |
| **Multi-asset expansion** | Not empty, but only one asset class is a clean bolt-on | Continuous-contract futures (basis/premium/OI as new leaves) — one 2025 primary precedent (arXiv:2509.23609), same panel geometry FactorMiner already has | Options surfaces / FX carry / fixed-income curves — geometry-breaking (need tenor×strike rank, not a leaf list), unprecedented in this tooling class, real scope creep |
| **Recent papers (May–Jul 2026)** | Not empty — denser than the three already-known lineage papers | Hypothesis-Redundancy geometric gate — cheap, pure diagnostic, zero execution surface | AlphaCrafter's **Trader** stage — live risk-constrained execution, a direct guardrail violation if adopted whole |
| **Retrieval / memory technique** | Partially a false gap — dense embeddings already exist — but real gaps remain in fusion, reranking, and one live wiring bug | `KGMemoryPolicy` never passes its embedder to enhanced retrieval (§8.2) — a bug, not a research gap, essentially free to fix | Treating this as "add a vector DB" — wrong shape; FactorMiner's memory is structured patterns and typed formulas, not documents |

## 10. Round 2 prioritized extension roadmap

Same ranking discipline as §6: (value ÷ effort), mapped to a real seam, architecture-layer
first. ~45 raw findings across 9 domains collapse to 17 roadmap items below — findings
that land in the same file are bundled into one item; findings the scouts themselves
flagged as traps are excluded here and listed under "What NOT to build" instead.

> **Implementation status:** all 17 items below are implemented, tested (812/812
> `factorminer/tests` passing, `ruff check` clean), and merged. Real, live evidence
> gathered during implementation review (not just unit tests): a real HTTPS call to
> `data.sec.gov` returned genuine Apple Inc. XBRL fundamentals correctly joined onto a
> synthetic OHLCV panel by point-in-time filed date (item 8); a real MCP client executed
> a live stdio tool call end-to-end and confirmed the research-artifact guardrail text on
> every tool (item 7); `mcp-serve --transport http` was confirmed to refuse startup
> without a bearer token (item 7); a rendered HTML MRM report was grepped for zero raw
> `<script>` tags with a dedicated XSS-payload test (item 9/10/11 compliance pack); the
> GraphSAGE ranker was verified end-to-end with real torch installed, not just the
> import-guard path (item 15). One real bug was caught and fixed during this review:
> `ingest-research --eligibility-mode alt_enabled` (item 8) initially never actually kept
> anything, because it gated on the process-local feature registry rather than the known
> alt-data leaf catalog — fixed in `architecture/research_absorption.py`, with a
> regression test. See [Security Considerations](security.md) for the full threat model
> behind every new externally-facing surface introduced here.

### Tier 0 — cheap, ship-it, no new dependency

| # | Feature | Value | Effort | Lands in |
| --- | --- | --- | --- | --- |
| 1 | **Wire `parent_formula` lineage into `RalphLoop`/`HelixLoop` trajectories** | Closes Round 1's own documented gap (§ Current Technical Debt in `docs/architecture.md`); makes `EditAwareMemoryPolicy` (AlphaMemo) actually live instead of degrading to `paper` behavior; doubles as the SR-26-2 "developmental history" audit trail §9 asks for | Cheap — plumbing only, `edit_aware` tests already use synthetic `parent_formula` | `core/ralph_loop.py`, `core/helix_loop.py`, `core/provenance.py` |
| 2 | **Fix `KGMemoryPolicy` embedder wiring** (§8.2) | Bug fix: dense semantic-neighbor retrieval already works in Helix, silently absent from the `kg` memory policy | Cheap — one call-site | `architecture/memory_policy.py::KGMemoryPolicy.retrieve` |
| 3 | **Prompt caching on the stable system+memory prefix** | 41–80% session LLM cost reduction, 13–31% TTFT reduction per cited benchmark (arXiv:2601.06007); zero quality tradeoff if the prefix is exact; `debate.py`'s 4 parallel specialists multiply the win | Cheap — provider SDK flag + prompt-builder ordering (stable content first, variable memory tail last) | `agent/llm_interface.py` (Anthropic `cache_control`, OpenAI auto-prefix), `agent/prompt_builder.py` |
| 4 | **Factor crowding module**: consensus-factor novelty screen (Ken French/AQR free data) + Lou–Polk CoMetric + hyperbolic alpha-decay crowding taxonomy (Lee 2025) | Distinguishes genuinely novel formulas from rediscovered Fama-French/AQR styles; only pure-price-history crowding score with a named academic pedigree; explains *why* mechanical (Trend/Momentum) families decay faster than judgment ones | Cheap → moderate — numpy correlation + one cached CSV loader per source, no paid API required for v1 | New `evaluation/crowding.py`; optional soft gate in `architecture/geometry.py`; extends `evaluation/decay.py` with hyperbolic fit |
| 5 | **Hypothesis-Redundancy geometric gate** | Cheap research diagnostic: decides whether the explored-formula span justifies an expensive non-local LLM jump vs. a cheap local AST edit, before spending the LLM call | Cheap → moderate — numpy/scipy spectral check | `architecture/geometry.py`, `architecture/dependence.py`; consumed by `island_model.py`'s exploration schedule |
| 6 | **Ranking-loss training objectives for `model_zoo.py`** (Margin / ListNet / BPR alongside MSE) | Cited result: Margin-loss ranker SR 0.75 / AR 16.2% vs. MSE SR 0.66 / AR 14.8% on the same S&P500 panel — portfolio utility moved more than IC did; directly optimizes the cross-sectional order FactorMiner's own IC/quintile evaluation already cares about | Cheap — new loss function on existing sklearn/optional-torch predictions, no new dependency | `evaluation/model_zoo.py` (new `train_objective` alongside `ridge`/`lasso`/`xgboost`) |
| 7 | **MCP interop hardening**: richer tool docstrings, consumer-recipe docs for LangGraph/OpenAI Agents SDK/CrewAI calling FactorMiner as a node, optional streamable-HTTP transport | FactorMiner already ships the right interop surface (`mcp/server.py`); every major agent framework now speaks MCP natively — the win is documentation + one transport option, not a rewrite | Cheap — docs + examples, optional transport flag | `factorminer/mcp/server.py`, `docs/financial-services-integration.md` |

### Tier 1 — moderate, real new capability

| # | Feature | Value | Effort | Lands in |
| --- | --- | --- | --- | --- |
| 8 | **Extensible feature-leaf architecture** + **SEC EDGAR XBRL fundamentals connector** as its first consumer | Prerequisite for *every* alt-data source found in §9 (fundamentals, on-chain, options-flow, 13F all blocked on this today); EDGAR itself is free, official, versioned, and unlocks `$eps`/`$revenue`/`$book_equity`-style leaves composable with existing `Ts*`/`Cs*` operators | Moderate — generalizes `FEATURES`, `data/loader.py`, `data/tensor_builder.py`, `agent/prompt_builder.py` beyond the hardcoded 8 OHLCV names, plus a new EDGAR loader | `core/types.py`, `data/loader.py`, `data/tensor_builder.py`, new `data/edgar_source.py`, `architecture/research_absorption.py` (dual-track the A-layer so fundamentals notes aren't hard-dropped) |
| 9 | **MRM validation pack + model inventory report** | Composes existing DSR/PBO/FDR/causal/decay/provenance into one examiner-shaped artifact (conceptual soundness, outcomes analysis, ongoing monitoring, inventory row) mapped to Fed/OCC SR 26-2 (2026-04-17, supersedes SR 11-7) — the math already exists, nobody can find it in one place today | Moderate — mostly glue + schema over existing modules, no new math | New `evaluation/mrm_pack.py`; rendered in `evaluation/report_viewer.py`; checklist in `docs/financial-services-integration.md` |
| 10 | **Formula AST sensitivity / ablation module** | Proves interpretability rather than asserting it — leaf-feature ablation, operator-subtree leave-one-out ΔIC, and window/parameter local sensitivity over `core/expression_tree.py`; answers "which subexpression actually drives this factor's IC" without resorting to SHAP-on-a-black-box (which would undercut the formulaic-alpha story) | Moderate — pure tree walks over existing IC metrics, no new dependency | New `evaluation/formula_sensitivity.py`; panels in `utils/tearsheet.py` and `evaluation/report_viewer.py` |
| 11 | **Economic-rationale field with mandatory human attestation** | Persists a structured (math structure → financial semantics → market logic) triple per factor, addressing SR 26-2's "conceptual soundness" pillar; cheapest conceptual-soundness win available | Cheap → moderate — schema + LLM-drafted field + report render; **must** ship with a human-attestation checkbox, never auto-presented as validated theory (SEC anti-AI-washing exam focus, per §9) | Extends `core/provenance.py::FactorProvenance`; rendered in `evaluation/report_viewer.py` |
| 12 | **Hybrid BM25+dense retrieval fusion for memory** + swap the embedder for a code-specialized model | Formula DSL is short and lexical (`TsRank`, `$close`) where BM25 wins, plus semantic paraphrase cases where dense wins — cited NDCG@10 57.7 (BM25) vs. 63.7 (best dense) on code-adjacent benchmarks, i.e. genuinely complementary, not redundant; today's primary pattern selector (`memory/retrieval.py`) uses neither, only a heuristic score | Moderate — RRF fusion is rank-based/score-free math; embedder swap is a checkpoint change | `memory/retrieval.py`, `memory/kg_retrieval.py`; optional local `CodeRankEmbed`-class checkpoint in `memory/embeddings.py::FormulaEmbedder` |
| 13 | **OpenAI-compatible local LLM provider + cascade routing** (local/small draft → deterministic DSL-parse gate → frontier repair/critic only on failures) | FrugalGPT-style cascades report up to 98% cost reduction in matched-quality setups; FactorMiner already has a free, deterministic confidence signal (`output_parser.py`'s parse success/failure) that generic cascade papers don't have access to | Moderate — new provider class + router; GPU only if self-hosting a useful 7B+ local model | `agent/llm_interface.py` (new `OpenAICompatibleProvider`), `agent/debate.py`, `agent/factor_generator.py`'s repair path |

### Tier 2 — bigger bets, real differentiation, GPU/data-heavy

| # | Feature | Value | Effort | Lands in |
| --- | --- | --- | --- | --- |
| 14 | **Continuous-contract futures asset-class lane** (basis/premium/spot/OI leaves, roll-adjusted continuous series) | Only asset class with a real 2025 LLM-factor-mining precedent (arXiv:2509.23609); fits FactorMiner's existing panel geometry *if* scoped to front-month continuous contracts only (multi-tenor curves are a different, invasive geometry — see traps) | Substantial — new data vendor, roll-calendar handling, futures-aware capacity/cost model | New `data/futures_source.py`; extends `core/types.py`, `data/tensor_builder.py`, `evaluation/capacity.py` |
| 15 | **Hand-rolled optional-torch GraphSAGE/GAT on a rolling-correlation asset graph** for `model_zoo.py` | Real cited results (RankIC 0.438, min-var Sharpe 0.984 vs. 0.635 HAR baseline) using only a rolling-correlation graph FactorMiner can build from data it already has — no torch-geometric/DGL, mirrors the existing optional-torch pattern in `operators/neuro_symbolic.py` | Substantial — breaks `model_zoo`'s current row-flatten contract, needs a panel-aware (N×K + adjacency) path | `evaluation/model_zoo.py` (new `model_kind`), reusing `operators/neuro_symbolic.py`'s optional-torch pattern |
| 16 | **Offline RFT/policy-update mining mode** (QuantEvolver-style, opt-in) | Orthogonal axis to all 6 existing memory policies: internalizes mining skill into model weights instead of externalizing into memory; cited OOS RankIC +109.5% vs. strongest baseline | Substantial — GPU + RL training infra (Verl/vLLM), offline-only, must not become a live-serving product concern | New `training/rft_miner.py`; feeds `evaluation_kernel.py` rewards |
| 17 | **Sealed multi-evaluator scorer co-evolution** (Agora-style, research-mode only) | Co-evolves the evaluation objective itself under an anti-self-confirmation sealed topology, rather than trusting one fixed `EvaluationKernel` — real conceptual advance over "search harder against the same score" | Substantial — multi-LLM cost, and the paper's own single-seed caveat (mean Sharpe drops to −0.755 cross-seed) means this needs careful scoping as a research mode, not a default | New `architecture/sealed_joint_search.py`; extends `evaluation/significance.py` |

### What NOT to build (Round 2 additions)

- **Full SOTA relational GNNs** (ACT, PRISM-VQ, full MASTER, THGNN). Each needs data
  FactorMiner's loaders don't have (industry/region/supply-chain adjacency, external prior
  factor streams) and/or torch-geometric/DGL-class dependencies. Building one would create
  a second, black-box alpha engine sitting next to the formulaic one — directly undercuts
  the "interpretable factors, not opaque ML" positioning that is this project's actual
  differentiator.
- **AlphaCrafter's Trader stage, or anything execution-shaped.** Same guardrail as Round 1
  §4's QuantConnect lesson. The Screener idea is fine (Tier 2 material if ever pursued);
  the Trader is not, full stop.
- **Options surfaces, FX carry, fixed-income curves as asset classes.** All three need a
  fundamentally different panel geometry (tenor×strike rank, not a flat leaf list) that no
  comparable open project has solved for typed-DSL LLM mining. Scope creep away from
  FactorMiner's proven equity/crypto/near-term-futures strength, not a bolt-on.
- **Rewriting `agent/debate.py`/`agent/critic.py` onto LangGraph, CrewAI, or AutoGen.** No
  primary source shows an alpha-quality gain from doing this; it would break the existing
  `no_debate` ablation and introduce a second, less deterministic orchestration stack for
  no evidenced payoff. MCP is the correct interop surface and it already exists.
- **Claiming formal regulatory compliance** ("SR 26-2 compliant," "SEC AI-compliant," "EU
  AI Act cleared"). SR 26-2 and the EU AI Act are principles-based / risk-classification
  frameworks a firm's own counsel and MRM function must apply — tooling can only produce
  *evidence* (the validation pack, the attestation export, the documentation kit), never
  the compliance determination itself. Ship the evidence, not the claim.
- **Fine-tuning/distilling a bespoke formula-generation LLM now** (LLM cost item 6 from the
  scout pass). Real technique, but premature: needs a training-data volume from continuous
  mining that the project doesn't have logged yet, and risks losing novel economic
  hypotheses in exchange for cheaper syntax-matching. Revisit once Tier 0/1 cost levers
  (caching, cascading) are in and mining volume is high enough to justify curating a
  distillation dataset.
- **Paid alt-data sources before the free ones are proven** (Daloopa fundamentals,
  Glassnode/CoinMetrics on-chain, Unusual Whales options-flow, RavenPack sentiment,
  satellite imagery). All are legitimate *later* additions once the extensible feature-leaf
  architecture (Tier 1 item 8) and its free EDGAR consumer are shipped and working — adding
  a paid vendor dependency before the architecture prerequisite exists, or before proving
  the pattern on free data, is the wrong sequencing.

## 11. Round 2 suggested sequencing

Tier 0 items (1–7) are independent of each other and of the mining loop core — safe to run
in parallel, same file-ownership discipline as Round 1. Item 1 (`parent_formula` lineage)
is the single highest-leverage item in this whole pass: it is already-documented debt, it
is cheap, and it makes an already-shipped Round 1 feature (`edit_aware` memory) live for
the first time. Item 8 (extensible feature-leaf architecture) is the biggest lever in Tier
1 — it is a prerequisite for six of the nine research domains' findings (everything in
§9's "Alternative data" row, plus the futures lane in Tier 2), so it should be sequenced
before any specific alt-data connector is built, not after. Tier 2 items are real but
GPU/data/cost-heavy — good candidates for a dedicated follow-up pass once Tier 0/1 land,
not a same-week parallel batch with everything else.
