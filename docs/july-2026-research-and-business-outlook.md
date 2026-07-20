# FactorMiner: July 2026 Research and Business Outlook

**Evidence cutoff:** 20 July 2026

**Scope:** alpha-discovery papers, general-agent research, finance-agent
benchmarks, commercial landscape, regulation, business models, and valuation
scenarios.

This is Round 3 of the repository's landscape work. It extends the earlier
[Landscape Review](landscape-and-extensions.md), but uses a different rule:
new work is recommended only where current primary evidence, a concrete
FactorMiner seam, and a measurable proof gate all exist.

## Executive answer

The market does not need FactorMiner to become another general finance agent.
Foundation-model vendors now ship finance-specific agents, office integrations,
connectors, permissions, and audit logs. Market-intelligence vendors already own
large document corpora and established analyst workflows. Competing with either
layer would put this project against capital, distribution, and proprietary data
it does not have.

The defensible direction is narrower and more valuable:

> **FactorMiner should be the governed, model-agnostic alpha-research laboratory
> that general agents call: a safe symbolic search space plus reproducible
> evaluation, novelty testing, provenance, and reviewable evidence.**

That thesis follows from four findings:

1. New alpha-mining papers keep expanding search with MCTS, code evolution,
   report absorption, trajectory mutation, and structured memory. Their reported
   results are promising, but their datasets, periods, objectives, and execution
   assumptions are not comparable.
2. The strongest asset-pricing evidence found in this pass shows that generation
   is easy and survival is rare: 280 proposed signals became 159 conventionally
   significant results, then 38 multivariate survivors, and only a small final
   set after multiple-testing, factor-spanning, and novelty checks against 209
   published anomalies.
3. General-agent research finds no universal memory system and large performance
   swings from the harness alone. FactorMiner already has the raw material to
   measure and optimize those choices: lifecycle traces, policy-based memory,
   provenance, and reward-annotated trajectory export.
4. Commercial and regulatory evidence points to the same moat: trusted data,
   source links, deterministic recomputation, permissions, auditability, and
   human approval—not a larger prompt or a larger agent swarm.

## How to read the evidence

- Paper results below are **authors' reported results**, not independent
  replications. No numerical result is used as a FactorMiner performance claim.
- arXiv and SSRN work is identified as preprint evidence; venue versions are used
  where an official proceeding is available.
- Benchmark versions are not compared across releases. In particular, Vals
  Finance Agent v1.1 and v2 are different tests.
- Commercial funding, revenue, and usage figures come from company announcements
  unless stated otherwise. They are signals of buyer demand, not audited market
  sizing.
- The valuation section is a scenario model, not an appraisal of this repository
  or a promise of funding outcomes.

## 1. The closest research lineage

### 1.1 Directly comparable alpha-discovery systems

| Work | Date/status at cutoff | Main mechanism | What it changes for FactorMiner |
| --- | --- | --- | --- |
| [FactorMiner](https://arxiv.org/abs/2602.14670) | Feb 2026 preprint | Skills, structured experience memory, and a Ralph-style generate/evaluate/evolve loop | Baseline lineage; faithful evaluation remains the reference obligation |
| [AlphaJungle](https://arxiv.org/abs/2505.11122), [AAAI-26 version](https://doi.org/10.1609/aaai.v40i2.37069) | AAAI 2026 | LLM-guided MCTS, multidimensional backtest feedback, frequent-subtree avoidance | Supports search-allocation experiments, but only after one comparable evaluation protocol exists |
| [Chain-of-Alpha](https://arxiv.org/abs/2508.06312) | Aug 2025 preprint | Separate generation and optimization chains with backtest feedback | Reinforces stage specialization; does not justify a second independent loop implementation |
| [CogAlpha](https://arxiv.org/abs/2511.18850), [ACL 2026 version](https://aclanthology.org/2026.acl-long.538/) | ACL 2026; arXiv revised 11 Jul | Code-level alpha representation with LLM mutation and recombination across five datasets and three markets | Shows the expressiveness ceiling of a formula DSL; arbitrary code should remain an isolated research lane, not the safe default |
| [QuantaAlpha](https://arxiv.org/abs/2602.07085) | Feb 2026; revised May | Treats an end-to-end run as a trajectory, then mutates weak steps and crosses over useful segments; reports cross-market transfer | Best match to FactorMiner's existing lifecycle/provenance records; test typed-AST trajectory evolution before adding another generator |
| [FactorEngine](https://arxiv.org/abs/2603.16365) | Mar 2026 preprint | Turing-complete program factors, report knowledge, separate LLM logic revision and Bayesian parameter optimization | Supports separating structural and parameter search; its arbitrary-program surface conflicts with the current safety and auditability advantage |
| [Hubble](https://arxiv.org/abs/2604.09601) | Mar 2026; revised Apr | Dual retrieval, AST sandboxing, family selection, and validated generation | Much of the architecture is already proportionately represented; priority is comparative proof, not reimplementation |
| [AlphaMemo](https://arxiv.org/abs/2606.20625) | May 2026 preprint | AST-diff edit motifs, confidence residuals, and veto memory | FactorMiner now records parent formulas and edit motifs; the missing step is an offline policy comparison on held-out runs |
| [QuantEvolver](https://arxiv.org/abs/2605.15412) | May 2026 preprint | Evolutionary search over agent trajectories | Supports the trajectory-laboratory direction, subject to strict cost and leakage controls |
| [XALPHA](https://arxiv.org/abs/2607.08332) | 9 Jul 2026; revised 13 Jul | Research-report absorption plus Macro, Micro, and Cross brains; hypothesis/code/rationale alignment and leakage checks | FactorMiner has a scoped report-absorption substrate, but not XALPHA's full 48-archetype taxonomy or three-brain loop; validate document-grounded hypotheses before expanding it |
| [Can AI Do Financial Research?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6569258) | 71-page SSRN paper posted 13 Apr 2026 | Symbolic accounting laboratory, fixed empirical pipeline, multivariate horse races, multiple-testing corrections, multi-model spanning, and novelty tests against 209 anomalies | **Highest-priority transfer:** build an evidence and novelty gate around fundamental hypotheses, not another unconstrained generator |

Two boundaries matter more than the papers' headline metrics:

- **Search representation:** CogAlpha and FactorEngine gain expressiveness with
  code or programs; FactorMiner gains safety, comparability, and auditability
  from a typed DSL. A future code lane should run in a separate sandbox and
  should never weaken the formula lane's guarantees.
- **Generation versus scientific evidence:** reported IC, return, Sharpe, and
  drawdown figures use different markets, periods, data treatments, costs, and
  portfolio constructions. A reproducible common harness is more valuable than
  copying whichever search method reports the largest number.

### 1.2 The overlooked lesson: discovery is a funnel

The asset-pricing paper above is the clearest future-direction signal. Its
candidate funnel is:

```text
280 generated signals
        ↓ conventional screen
159 directionally significant
        ↓ multivariate horse races
38 independently predictive candidates
        ↓ multiple testing + factor spanning + novelty vs 209 anomalies
a small final set
```

FactorMiner has strong candidate-generation and validation components, including
CPCV/PBO, DSR, FDR controls, dependence checks, risk/capacity/crowding analysis,
and an EDGAR feature lane. It does not yet provide a single asset-pricing
evidence object that answers:

- Is the signal incremental to canonical factors and the current library?
- Does it survive competing specifications and multiple-testing correction?
- Is it novel relative to a versioned anomaly catalog?
- Does it survive a point-in-time, microcap/liquidity-aware, held-out protocol?
- Can a reviewer reproduce every transformation from raw filing to final table?

That object should become the product's center of gravity.

## 2. What newer general-agent research transfers

| Evidence | Finding | FactorMiner implication |
| --- | --- | --- |
| [EvoMemBench](https://arxiv.org/abs/2605.18421) | Fifteen memory methods under a standardized protocol produced no universal winner; long context remained competitive, retrieval led knowledge-heavy tasks, and procedural memory helped when past experience matched task structure | Do not add a seventh fixed memory policy. Replay the existing policies and learn or select by task, regime, and cost |
| [SelfMem](https://arxiv.org/abs/2607.03726) | A self-optimizing memory strategy reported 40.8–48.7% gains over the strongest baseline across 100K–1M-token BEAM settings | Treat memory behavior as an optimizable policy with held-out evaluation, not a static config label |
| [Retrospective Harness Optimization](https://arxiv.org/abs/2606.05922) | Re-solves difficult past trajectories and proposes harness changes using self-consistency and self-preference; reports SWE-Bench Pro moving from 59% to 78% without external labels | Build offline prompt/tool/memory/harness replay from lifecycle logs, but retain market-data ground truth and sealed holdouts instead of relying only on self-preference |
| [MAFBench](https://arxiv.org/abs/2602.03128) | Framework choices alone reportedly changed latency by more than 100×, planning by up to 30 points, and coordination from above 90% to below 30% | Do not assume more agents are better. Benchmark single-agent, staged, debate, and sealed-panel configurations under the same budget |
| [Beyond the Leaderboard](https://arxiv.org/abs/2607.05775) | Review of agent tool-use, planning, and reasoning failures finds failures compound with task length and scaffolding is not a consistent cure | Favor bounded research jobs with explicit artifacts, stop conditions, and validators over an open-ended "general researcher" loop |
| [AHOIS](https://arxiv.org/abs/2606.26722) | Uses a Socratic critic, counterexamples, and explicit falsification criteria in closed-loop scientific work | Add a falsification plan and disconfirming-test artifact; avoid debate that produces prose but no testable rejection condition |

The repository is unusually well positioned for this transfer. It already has
`architecture/lifecycle.py`, `core/provenance.py`, policy-based memory,
`architecture/rft_export.py`, and sealed evaluation. The next missing component
is not another runtime agent. It is a **trajectory laboratory** that can replay
the same historical tasks across policies, prompts, providers, and budgets
without contaminating the final holdout.

## 3. Finance-agent reality in July 2026

General finance agents are improving, but still fail the kind of dynamic,
auditable correctness FactorMiner needs.

- The original [Finance Agent Benchmark](https://arxiv.org/abs/2508.00828)
  contains 537 expert-authored research questions across nine categories, with
  search and SEC tools. It is useful evidence that finance needs tool use, but
  its original scores are a May 2025 snapshot.
- [BlueFin](https://arxiv.org/abs/2605.30907) has 131 professional spreadsheet
  tasks and 3,225 rubric criteria. Its judge was validated against experts, yet
  the strongest tested models averaged below 50%, with particular weakness in
  dynamic correctness.
- [MBABench](https://arxiv.org/abs/2605.22664) evaluates end-to-end financial
  workbooks for accuracy, formulas, and professional formatting. Its results
  show that the agent harness materially affects output even with the same
  underlying model, and difficult chained work remains weak.
- [Meta-Benchmarks for Financial-Services LLM Evaluation](https://arxiv.org/abs/2607.01740)
  maps 452 public benchmarks to 41 O*NET work activities and 38 BIAN domains
  across 288 models through June 2026. Its core lesson is that "best model" is
  use-case-specific.
- The live [Vals benchmarks](https://www.vals.ai/benchmarks) had already moved
  from Finance Agent v1.1 to v2 by this cutoff. The public v2 repository defines
  the [current research-and-SEC task harness](https://github.com/vals-ai/finance-agent-v2).
  Scores across versions should not be read as a time series.
- [FinHarness](https://arxiv.org/abs/2605.27333) reports that routing requests
  through an inline safety harness reduced attack success while using advanced
  models far less often than an all-frontier configuration. The transferable
  idea is policy enforcement and cost routing around the model, not dependence
  on a single model.

This evidence rules out "replace the quant researcher" as a credible near-term
claim. A better claim is measurable and narrower: **reduce the time and cost from
a hypothesis to a reproducible, reviewer-approved evidence pack.**

## 4. Commercial landscape and where not to compete

### 4.1 The layers are separating

| Layer | Current evidence | Strategic consequence |
| --- | --- | --- |
| Foundation/general finance agents | [Anthropic launched ten finance-agent templates](https://www.anthropic.com/news/finance-agents) in May 2026 with skills, connectors, subagents, per-tool permissions, credentials, and audit logs. [OpenAI and PwC](https://openai.com/index/openai-pwc-finance-collaboration/) announced governed enterprise finance agents and human oversight. | Do not build a general chat/search/office agent. Make FactorMiner callable by these systems through stable MCP/API contracts |
| Market intelligence and workflow | [AlphaSense announced](https://www.alpha-sense.com/press/alphasense-raises-350m-at-7-5b-valuation-and-surpasses-600m-in-annual-recurring-revenue/) more than $600M ARR, a $7.5B valuation, 7,000 enterprise customers, and a corpus of more than 500M business documents | Do not compete on broad document search or terminal replacement. Integrate, or accept research briefs as inputs |
| Finance-specific agent applications | [Rogo raised a $75M Series C](https://rogo.ai/news/scaling-rogo-to-build-the-future-of-investment-banking-our-75m-series-c-and-european-expansion) to expand end-to-end financial workflows | Pitchbooks, diligence, and general banking workflows are well-funded categories; alpha verification is a narrower, less crowded wedge |
| Trusted financial data | [Daloopa raised $47M](https://daloopa.com/blog/press-release/47-million-series-c) around source-linked, auditable finance data and AI workflows | Provenance and data lineage are commercial features. FactorMiner should integrate licensed data and preserve entitlements, not try to own every dataset |
| Quant research platforms | Qlib/RD-Agent, WorldQuant BRAIN, Numerai, and commercial quant stacks cover data, modeling, tournaments, or execution | FactorMiner's opening is a model-agnostic symbolic research and evidence layer, not a fund, broker, or alpha marketplace |

### 4.2 Recommended position

**Category:** governed alpha R&D / research verification infrastructure.

**Buyer:** systematic asset managers, quant pods, research-engineering leads,
model-risk teams, and internal AI-platform teams.

**User:** quantitative researchers who need more throughput without giving up
review, reproducibility, or control.

**Integration:** general agents, data vendors, notebooks, CI, and internal model
registries call FactorMiner; FactorMiner returns immutable research artifacts.

The product boundary should remain explicit:

- It proposes and evaluates research artifacts.
- It does not recommend a live trade, size a position, bind a risk limit, route
  an order, or operate an autonomous account.
- A human or separately governed downstream system decides what becomes a
  production signal.

That boundary is technically coherent and commercially useful. It also avoids
the higher-risk autonomous-execution direction while regulators are focusing on
agent governance, third-party concentration, resilience, and supervision.

## 5. Regulation and adoption outlook

The newest official evidence is directionally consistent:

- The FCA's 6 July [Mills Review](https://www.fca.org.uk/news/press-releases/fca-publishes-landmark-review-impact-ai-retail-financial-services)
  examines how AI may transform retail finance through 2030 and explicitly
  treats autonomous agents, trust, control, competition, fraud, and cyber risk
  as strategic questions.
- The UK's 14 July [Financial Services AI Adoption Plan](https://www.gov.uk/government/publications/ai-adoption-plan-financial-services/financial-services-ai-adoption-plan)
  calls for scaling beyond pilots while addressing regulatory clarity,
  resilience, skills, and agentic-payments readiness. It reports 21% AI adoption
  in finance and real estate in an early-2025 survey versus 16% economy-wide,
  while noting much higher adoption in larger firms.
- The FSB's 10 June [consultation report](https://www.fsb.org/2026/06/sound-practices-for-responsible-adoption-of-artificial-intelligence-ai-consultation-report/)
  proposes 12 practices spanning organization-wide governance and the AI
  lifecycle, including risks from third-party dependence and changing models.
- [FINRA's 2026 oversight guidance](https://www.finra.org/rules-guidance/guidance/reports/2026-finra-annual-regulatory-oversight-report/gen-ai)
  reiterates that existing technology-neutral securities rules continue to
  apply to GenAI use. Its broader report emphasizes model risk, records, data
  integrity, security, and third-party risk.

For FactorMiner, the practical requirements are:

1. provider-agnostic model interfaces and failover;
2. versioned model, prompt, tool, data, and policy manifests;
3. point-in-time data lineage and entitlement-aware access;
4. per-tool permissions, budgets, and human approval gates;
5. immutable job records, reproducible replay, and exportable evidence;
6. no hidden transition from research output to execution.

These are not merely compliance costs. They are the proposed product moat.

## 6. Ranked future directions

### Priority 0 — Establish one credible evidence baseline

**Build:** complete a paper-comparable Qlib/FactorMiner reproduction with a
versioned dataset contract, typed random baseline, canonical factor controls,
cost assumptions, and an immutable result bundle.

**Why first:** no later search or memory improvement is interpretable if it is
measured on a moving or proxy baseline. The current paper-claims matrix still
records partial/proxy fidelity in data and some baselines.

**Proof gate:** an independent user can reproduce the primary tables from one
command and obtain the same hashes and metrics within documented tolerances.

### Priority 1 — Build the asset-pricing evidence and novelty gate

**Build:** a first-class `ResearchEvidencePack` that combines:

- point-in-time EDGAR/accounting feature provenance;
- univariate and multivariate horse races;
- canonical factor and current-library spanning tests;
- multiple-testing correction and existing CPCV/PBO/DSR diagnostics;
- novelty matching against a versioned published-anomaly catalog;
- liquidity/microcap/capacity screens and truly held-out evaluation;
- a human-readable falsification plan and rejection reasons.

**Existing seams:** `data/edgar_source.py`, `evaluation/significance.py`,
dependence and crowding services, the Qlib adapter, factor provenance, and
research absorption.

**Proof gate:** reproduce the published funnel logic on a documented public
subset, then demonstrate that the gate rejects apparently strong but redundant
or multiple-testing-sensitive candidates.

**Business value:** this is the artifact a CIO, research head, or model-risk
reviewer can approve. It converts the system from an idea generator into a
research-control product.

### Priority 2 — Build a trajectory and harness laboratory

**Build:** offline replay that compares generator prompts, memory policies,
retrieval, cascade routing, providers, and agent topologies under identical
tasks, data, seeds, and token/dollar budgets.

Start with selection among the six existing memory policies. Do not add a new
policy until replay shows a repeatable gap. Retain a sealed final holdout so
self-optimization cannot grade itself into overfitting.

**Existing seams:** lifecycle trajectories, parent/secondary-parent lineage,
edit motifs, policy schemas, prompt caching/cascade routing, sealed evaluation,
and RFT JSONL export.

**Proof gate:** pre-register a policy-selection rule on historical runs and beat
the best fixed policy on held-out mining tasks at equal cost, with confidence
intervals and no degradation in false-discovery controls.

### Priority 3 — Make FactorMiner an enterprise agent gateway

**Build only with design-partner pull:** durable job APIs, an artifact registry,
RBAC/SSO, tenant and data-entitlement boundaries, signed manifests, approval
gates, per-tool budgets, provider failover, and audit export around the existing
MCP surface.

**Proof gate:** three to five design partners can run the same governed research
workflow from their preferred general agent while keeping data private and
reproducing the resulting evidence pack.

**Why third:** this is valuable only after the evidence artifact is worth
governing. Building enterprise plumbing around a weak scientific result would
create infrastructure without a wedge.

### Priority 4 — Typed trajectory evolution, not arbitrary-code default

**Build as a research experiment:** identify low-reward trajectory steps, mutate
them, and cross over compatible parent segments while preserving the typed AST,
complexity bounds, and provenance.

Separate structural edits from numeric-window optimization. A Bayesian or other
cheap local optimizer can tune windows after the hypothesis structure is fixed,
reducing LLM calls and making the source of improvement auditable.

**Proof gate:** higher accepted-candidate yield per dollar on the common baseline,
with equal or better OOS survival, diversity, and expression complexity.

### Priority 5 — Falsification critic and isolated program lane

Two later research bets are justified, but should not enter the default loop:

- A falsification critic that proposes counterexamples, alternative
  explanations, and tests that could reject a candidate.
- A separately sandboxed code/program-factor lane to measure whether CogAlpha or
  FactorEngine-style expressiveness adds durable value beyond the DSL.

The program lane must have stricter resource limits, static and dynamic leakage
checks, dependency isolation, and a conversion path back to a reviewable
artifact. If it cannot preserve those guarantees, it should remain out of scope.

## 7. What not to build

| Direction | Decision | Reason |
| --- | --- | --- |
| General finance chat/search agent | Do not build | Foundation vendors and AlphaSense already bundle models, content, connectors, office tools, and distribution |
| Excel/pitchbook automation | Do not build | Crowded, well-funded, and not connected to FactorMiner's scientific advantage |
| Live autonomous trading or execution | Do not build | Changes the risk, regulatory, and operational category; contradicts the current research-artifact boundary |
| Alpha marketplace | Do not build | Incentives, leakage, heterogeneous deployment, and live-account risk overwhelm the current product |
| Multi-agent rewrite | Do not build | MAFBench shows topology itself can degrade cost, latency, and coordination; compare bounded variants first |
| Seventh hand-designed memory policy | Do not build yet | EvoMemBench shows no universal winner; use measured selection over policies already present |
| Arbitrary Python as the default factor language | Do not build | It discards the typed DSL's safety, deterministic evaluation, and auditability moat |
| Proprietary data acquisition before traction | Do not build | Integrate licensed/customer data and preserve entitlements; prove workflow value before buying a corpus |

## 8. Business model

Keep the MIT research core open. Monetize the control plane and the work needed
to make it safe inside an institution.

### Product packages

| Package | Customer outcome | Indicative annual price hypothesis |
| --- | --- | ---: |
| Open-source core | Local research, reproducible examples, community adoption | Free |
| Team research registry | Shared jobs, immutable evidence packs, comparisons, private model/data connections | $15k–$30k |
| Enterprise/private deployment | SSO/RBAC, entitlements, audit export, provider controls, SLAs, private networking | $75k–$250k |
| Validation and data onboarding | Reproduction, custom benchmark, point-in-time data mapping, policy setup | $25k–$100k project, designed to convert to recurring software |

These are pricing hypotheses to test, not observed willingness to pay. The first
commercial experiment should be a paid design partnership around one evidence
workflow, not a broad self-serve launch.

### Initial market scenarios

Broad comparables report thousands of institutional customers—AlphaSense says
7,000 enterprises, while [FactSet reported](https://investor.factset.com/news-releases/news-release-details/factset-reports-results-fourth-quarter-and-fiscal-2025)
more than 9,000 client firms in fiscal 2025. Only a small subset is relevant to
systematic alpha research, so the useful exercise is bottom-up:

| Scenario | Assumption | Annual recurring opportunity |
| --- | --- | ---: |
| Beachhead | 500 relevant teams × $25k–$75k ACV | $12.5M–$37.5M |
| Expanded control plane | 2,000 research/model-risk teams × $50k–$150k ACV | $100M–$300M |

These are addressable-revenue scenarios, not a sourced TAM forecast. They should
be replaced with interview and pipeline data as soon as it exists.

### Metrics that matter

- time from hypothesis to reproducible reviewed evidence;
- accepted-candidate yield per model dollar and compute hour;
- OOS survival and false-discovery rate, not in-sample Sharpe alone;
- percentage of evidence packs reproduced by a second user;
- audit completeness and reviewer time;
- weekly active research teams and recurring workflows;
- expansion from one team/data source to multiple governed teams.

## 9. Valuation landscape and scenario model

There is no defensible present valuation for FactorMiner from repository code
alone. There is no verified revenue, growth, customer retention, proprietary
dataset, or paid deployment evidence in scope. Any precise current number would
be invented.

The observable comparables illustrate why the business model matters:

| Comparable | 20 Jul 2026 observable input | Simple equity-value/revenue signal | Interpretation |
| --- | --- | ---: | --- |
| AlphaSense | $7.5B private valuation and >$600M ARR in its [June announcement](https://www.alpha-sense.com/press/alphasense-raises-350m-at-7-5b-valuation-and-surpasses-600m-in-annual-recurring-revenue/) | <12.5× | High growth plus proprietary content/workflow distribution |
| FactSet | ~$9.25B market-cap snapshot; [$2.322B FY2025 revenue](https://investor.factset.com/news-releases/news-release-details/factset-reports-results-fourth-quarter-and-fiscal-2025) | ~4.0× | Mature public financial-data/workflow company |
| MSCI | ~$45.93B market-cap snapshot; [$3.302B 2025 run-rate](https://ir.msci.com/news-releases/news-release-details/msci-reports-financial-results-fourth-quarter-and-full-year-2025) | ~13.9× | High-value indexes, data, analytics, and embedded recurring workflows |

The public-company figures are point-in-time market-cap—not enterprise-value—
comparisons and are deliberately approximate. They are not directly comparable
to a venture financing or to one another.

For planning only, use three explicit revenue-multiple cases:

| ARR | Tool/services case, 2–4× | Vertical workflow case, 5–8× | Data/workflow moat case, 10–13× |
| ---: | ---: | ---: | ---: |
| $1M | $2M–$4M | $5M–$8M | $10M–$13M |
| $5M | $10M–$20M | $25M–$40M | $50M–$65M |
| $10M | $20M–$40M | $50M–$80M | $100M–$130M |

The upper case is not earned by adding AI labels. It would require proprietary
or hard-to-recreate evidence data, strong net retention, embedded workflows,
credible growth, and low concentration risk. FactorMiner currently belongs in
none of these revenue cases because it has no verified ARR. The purpose of the
table is to show which business qualities could eventually change valuation.

## 10. 12/24/36-month outlook

### 0–12 months: prove the scientific wedge

- Finish one exact, public, reproducible baseline.
- Ship the evidence/novelty pack for accounting and price-based hypotheses.
- Publish two end-to-end case studies, including candidates that fail.
- Recruit three to five design partners and measure reviewer time and OOS
  survival.
- Keep general-agent support at the MCP/API boundary.

**Kill/rethink trigger:** users value generated ideas but will not pay for the
evidence/review workflow, or independent reproduction remains too expensive or
data-dependent.

### 12–24 months: productize governance

- Add private deployment, job/artifact registry, SSO/RBAC, entitlements, budgets,
  approvals, and audit exports.
- Support two or three customer-selected data providers instead of building a
  data business prematurely.
- Benchmark providers and harnesses continuously; allow the customer's general
  agent to invoke the same governed job.
- Convert design partners into recurring contracts.

**Kill/rethink trigger:** each deployment remains bespoke services work and no
common evidence schema or integration repeats across customers.

### 24–36 months: earn a data and learning moat

- Optimize prompts, memory, and search allocation from privacy-preserving,
  permissioned trajectory evidence.
- Build a versioned anomaly/mechanism/evaluation corpus whose value increases
  with use.
- Introduce typed trajectory evolution only where held-out replay proves it.
- Consider an isolated program-factor lane only if the DSL's measured ceiling is
  commercially material.

**Upside case:** FactorMiner becomes the research-evidence control plane shared
by internal agents, models, and data vendors.

**Base case:** it is a strong open-source research framework with a focused
private-deployment business.

**Downside case:** frontier finance agents absorb most orchestration; the durable
value remains the evaluation protocol, evidence schema, and reproducibility
tooling.

## Bottom line

The July 2026 landscape rewards disciplined narrowing. Search systems are
proliferating; trustworthy scientific evidence is not. General finance agents
are becoming distribution channels, while trusted data and governed workflows
capture enterprise value.

FactorMiner should therefore optimize for one outcome:

> **Turn a machine-generated alpha hypothesis into a falsifiable, novel,
> point-in-time, reproducible, and reviewable evidence pack—independent of which
> general agent or model proposed it.**

That direction uses the repository's real strengths, has a clear buyer, avoids
the most dangerous scope expansions, and creates the conditions for a credible
commercial control plane later.
