# Industry Benchmarking and Evidence Protocol

Research snapshot: 2026-07-21.

This document answers two different questions that are often collapsed into
one leaderboard:

1. can a system generate, translate, rank, or search formulaic factors; and
2. does a frozen factor retain economically meaningful, risk-adjusted value
   out of sample after turnover, costs, capacity, and research selection?

The first is an AI/research-system capability. The second is investment
evidence. A system can score well on the first and fail the second. FactorMiner
therefore treats benchmark names as components of an evidence chain, not as
interchangeable scores.

## Executive conclusion

There is no single industry benchmark for formulaic alpha mining. The most
credible practical protocol is a stack:

```text
point-in-time data and a frozen target/universe
  -> explicit Pearson IC and Spearman RankIC
  -> serial-dependence-aware uncertainty and multiple-testing control
  -> purged/walk-forward selection evidence
  -> style/industry-neutral residual evidence
  -> exact portfolio turnover, costs, impact, and capacity
  -> independent event-driven replay
  -> paper/live forward evidence
```

Qlib, Alpha101/158/360, and AlphaBench provide useful common inputs and
research tasks. LEAN provides a useful independent execution surface. FinBen,
FinQA, TAT-QA, and FinanceBench evaluate financial language or document
reasoning, not tradable alpha. None of them replaces the stack above.

FactorMiner already had strong pieces of this stack: temporal dataset splits,
rolling research evaluation, CPCV/PBO, block-bootstrap tests, Benjamini-Hochberg
FDR, a Deflated Sharpe implementation, cost and capacity models, and frozen
benchmark manifests. Its most consequential gaps were fragmented reporting,
no separately labeled Pearson IC, and no general style/industry signal
neutralizer. `industry_evidence_v1` addresses those gaps while keeping the
historical `paper_ic_v2` admission contract compatible.

## The trust hierarchy

| Layer | What it answers | Minimum credible evidence | FactorMiner state |
| --- | --- | --- | --- |
| Data integrity | Was information available at decision time? | point-in-time fields, delistings, corporate actions, source hashes, target lag | caller contract and manifests; vendor truth remains external |
| Predictive association | Does the signal order or linearly predict forward returns? | Pearson IC and Spearman RankIC by period, coverage, sign | both explicit; legacy `ic_*` remains RankIC |
| Sampling uncertainty | Is the mean stable under serial dependence? | IC series, Newey-West/HAC t-stat, block-bootstrap CI | implemented |
| Selection hygiene | Was the winner chosen from many trials? | complete trial count, family p-values, FDR, DSR, CPCV-path matrix/PBO | implemented when the complete inputs are supplied |
| Temporal generalization | Was all fitting upstream of held-out evaluation? | train/validation/test freeze, purge label overlap, embargo, rolling regimes | fixed splits, rolling splits, CPCV and PBO exist |
| Risk attribution | Is the signal a repackaged style/industry exposure? | point-in-time style and industry exposures, raw and residual metrics | open OLS/WLS residualization implemented; proprietary exposure estimation is not |
| Economic implementation | Does it survive churn and trading frictions? | target-weight turnover, cost curve, spread/fees/impact, capacity | linear cost stress plus richer cost/capacity modules |
| Independent replay | Does another execution engine reproduce orders and P&L? | frozen weights/orders replayed with calendars, fills, actions and constraints | not bundled; LEAN/another engine is the recommended external check |
| Forward truth | Does the behavior survive after research ends? | paper/live track record with unchanged rules | external |

The report deliberately labels each gate `measured`, `partial`,
`not_supplied`, or `external_required`. A missing point-in-time or risk-model
input is not converted into a reassuring default.

## Metric contract: the names are not enough

For asset `i` and decision period `t`, let `s[i,t]` be a factor value and
`r[i,t+1]` the already aligned forward return.

### IC and RankIC

```text
Pearson IC_t = PearsonCorr_i(s[i,t], r[i,t+1])
RankIC_t     = SpearmanCorr_i(s[i,t], r[i,t+1])
```

Pearson IC tests a linear cross-sectional relation and is sensitive to signal
magnitude and outliers. RankIC tests monotonic ordering and is invariant to
monotone transforms. Formula-mining systems often optimize ranks, so RankIC
can look good while the signal has unstable magnitudes; the reverse can also
happen. Report both.

FactorMiner historically called its Spearman series `ic_series`. That contract
is preserved for saved artifacts and admission. New artifacts add
`ic_definition: spearman_rank`, `rank_ic_*`, and `pearson_ic_*` fields.

### ICIR, annualization, and t-statistics

FactorMiner now reports four distinct quantities:

```text
ICIR                 = mean(IC_t) / sample_std(IC_t)
annualized ICIR      = ICIR * sqrt(periods_per_year)
independence t-stat  = ICIR * sqrt(number_of_periods)
HAC t-stat           = mean(IC_t) / NeweyWestSE(mean(IC_t))
```

These must not be compared by label alone. At the research snapshot, the
[AlphaBench metric code](https://github.com/CityU-MLO/AlphaBench/blob/main/backtest/factor_metrics/metrics.py)
defines `FL_Ir` as `mean/std` and `FL_Icir` as `sqrt(n)*mean/std`. The second is
an independence t-stat, not FactorMiner's unannualized ICIR. Positive serial
correlation can make that independence statistic materially too optimistic;
the HAC statistic is the primary asymptotic inference in the new report.

### Sign and magnitude

`mean(abs(IC_t))` is not evidence of a stable direction. A series alternating
`+0.1, -0.1` has high average magnitude but zero usable signed mean.
FactorMiner's paper gate remains `abs(mean(IC_t))`, with the signed mean kept
for orientation. Never choose a factor's sign on the held-out test panel; set
the orientation on training data and freeze it.

### Turnover and costs

The evidence report constructs equal-weight top and bottom legs, each with
unit notional, then computes:

```text
one_way_turnover_t = 0.5 * sum_i(abs(w[i,t] - w[i,t-1]))
net_return_t       = gross_return_t - one_way_cost * sum_i(abs(delta_weight[i,t]))
```

This uses actual long and short target weights, not only top-bucket membership
churn. It excludes the initial entry and says so in the artifact. The linear
basis-point curve is a reproducible stress test, not a fill simulator. When
dollar volume is supplied, the existing square-root impact/capacity estimator
adds participation and capital scenarios. Spread, fee, borrow, limit-up/down,
latency, queue, and market-specific tax assumptions still need a production
execution model. The AQR study of [trading costs of asset-pricing
anomalies](https://www.aqr.com/insights/research/working-paper/trading-costs-of-asset-pricing-anomalies),
based on a large live institutional trade sample, is a useful reminder that
cost and break-even capacity are strategy-, market-, size-, and execution-
specific; a generic basis-point haircut is only a first screen.

FactorMiner's current capacity estimator uses the selected long leg's average
liquidity to parameterize a shared impact haircut for both legs. That is a
useful diagnostic scenario only when long and short liquidity are similar. A
production study must estimate each trade on both legs, including
borrow availability and asymmetric market constraints.

### Multiple testing

The number of trials includes rejected formulas, prompt variants, model
variants, seeds, hyperparameters, universes inspected, and manual restarts—not
only saved factors. The [factor-zoo analysis by Harvey, Liu and
Zhu](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2249314) explains why a
plain `t > 2` is not a credible discovery threshold after extensive search.

FactorMiner uses complementary controls:

- family-wide Benjamini-Hochberg FDR on two-sided HAC mean tests;
- block-bootstrap IC confidence intervals;
- Deflated Sharpe with the declared trial count and non-normal return moments;
- PBO from a complete trial-by-CPCV-path performance matrix.

FDR asks how many discoveries in a family are expected to be false. DSR asks
whether a selected return series clears a multiple-search/non-normality bar.
PBO asks whether the selection rule repeatedly picks an in-sample winner that
falls below the out-of-sample median. They are not substitutes. See the
[original DSR paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
and [PBO paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253).
Benjamini-Hochberg is most defensible for independent or positively dependent
tests. A highly correlated factor family should also consider the more
conservative Benjamini-Yekutieli procedure or a resampling-based family test;
the current implementation reports this limitation rather than claiming
arbitrary-dependence control.

### Risk-model residualization

A strong factor can be a disguised size, beta, momentum, volatility, country,
or industry position. The open residualization primitive fits, independently
at each date:

```text
signal_t = intercept + exposure_t * beta_t + residual_signal_t
```

It accepts static or point-in-time numeric exposures, one-hot classifications,
and optional positive WLS weights. Metrics and turnover/cost stress are then
recomputed on the residual signal. This implements the attribution question,
not MSCI's proprietary model construction. Barra models also estimate and
standardize exposures, factor returns, factor covariance, and specific risk;
the [MSCI USE4 methodology](https://www.msci.com/documents/10199/242721/Barra_US_Equity_Model_USE4.pdf/d7625289-cade-4e88-96ae-696219af5b67)
shows why a complete commercial risk model is a much larger system.

## Platform engines: what should and should not be integrated

### Microsoft Qlib

[Qlib](https://github.com/microsoft/qlib) is the closest open common substrate
for the listed formula-mining research. It spans data handlers, model training,
records, backtesting, portfolio analysis, and execution research. FactorMiner
already supports Qlib-compatible I/O and formula concepts, so the highest-value
integration is artifact parity: same point-in-time panel, same target, same
universe, same expressions, and reconciled per-date metrics.

Two frequent naming mistakes matter:

- `Alpha360` is 360 normalized raw lag features: 60 lags each of close, open,
  high, low, VWAP, and volume. It is not 360 independently handcrafted alpha
  theses.
- Qlib's default `Alpha158` handler combines nine candlestick features, current
  normalized price fields, and rolling operators over windows 5/10/20/30/60.
  The exact count depends on configuration. Treat the handler config and Qlib
  commit as part of the baseline identity.

The official implementations are in Qlib's
[`loader.py`](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)
and [`handler.py`](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py).
“Beating Alpha158” is meaningless if labels, preprocessing, dates, universe,
fees, or model class differ.

### QuantConnect LEAN

[LEAN](https://github.com/QuantConnect/Lean) is an active event-driven engine
with calendars, securities, orders, brokerage models, corporate actions,
multi-asset support, and backtest/live symmetry. FactorMiner should not embed a
second execution engine. The credible boundary is to export frozen target
weights/orders and compare dates, fills, holdings, fees, and P&L in LEAN. A
disagreement becomes a testable reconciliation item instead of an in-process
assumption shared by both results.

### WorldQuant BRAIN

BRAIN is important as an industry alpha-expression and competition culture,
but it is not a fully open, locally reproducible benchmark contract. Operator
semantics, data entitlements, delay/neutralization settings, and platform
scoring are part of the result. Treat platform scores as external evidence
with a recorded configuration, not as a baseline FactorMiner can honestly
claim to reproduce.

### Zipline/Pyfolio lineage

[Zipline Reloaded](https://github.com/stefan-jansen/zipline-reloaded) keeps the
Quantopian event-driven research API alive and remains useful for notebooks and
cross-checks. Its lineage is influential, but LEAN is the stronger default for
the proposed independent, multi-asset execution replay. Pyfolio-style tearsheets
are reporting surfaces; they do not repair data leakage or execution semantics.

## Formula libraries

| Library | What it really provides | Use here | Fidelity requirement |
| --- | --- | --- | --- |
| Alpha101 | 101 explicit short-horizon formulas from Kakushadze | classic formula/regression baseline | exact formulas, fields, operator semantics, universe and delay |
| GTJA/Alpha191 | a widely circulated Chinese brokerage-style formula family | A-share stress baseline | pin a specific source/version; public variants disagree |
| Qlib Alpha158 | configurable engineered price/volume feature handler | open ML feature baseline | Qlib commit plus handler config and processors |
| Qlib Alpha360 | normalized 60-lag OHLCV/VWAP tensor | raw-history model baseline | Qlib commit plus normalization/label config |
| “Alpha20” | small experimental subsets used in some agent studies | smoke or ablation set | publish the exact 20 expressions; the name is not canonical |

The [Alpha101 paper](https://arxiv.org/abs/1601.00991) reports 101 formulas,
short holding periods, and low average pairwise correlation. It is a useful
reference library, not proof that a new market implementation should retain
the original performance years later.

FactorMiner currently bundles only 12 Alpha101-inspired entries and derived
window variants. Manifests correctly mark them partial/local. The
`alphaforge_style` and `alphaagent_style` catalogs are also proxies, not the
external algorithms. Comparative claims must use faithful imported catalogs
or retain the proxy label.

Alpha191 needs extra caution: unlike the Alpha101 paper or Qlib handlers, no
single authoritative public executable specification dominates the open
implementations. Operator definitions, missing factors, benchmark series, and
price-adjustment choices vary. A checksum-addressed formula manifest is part of
the benchmark—not an optional implementation detail.

## Agent and search systems: current research frontier

The research frontier is moving toward an executor-grounded loop: generate an
interpretable program, compile it against a fixed DSL, evaluate it with a
point-in-time oracle, preserve complete search history, control duplication,
and retest the frozen output under the Tier-0 protocol.

| System | Main contribution | Public implementation signal | What it does not prove |
| --- | --- | --- | --- |
| [AlphaBench](https://alphabench.cc/) | standardized LLM generation, zero-shot evaluation, search, and newer atomic tasks over executable factors | official repository includes FFO, Qlib/Assay adapters, EA/CoT/ToT search and Alpha158 factors | net, capacity-constrained, independently replayed alpha |
| [RD-Agent](https://github.com/microsoft/RD-Agent) / RD-Agent(Q) | multi-agent research and development loop, factor/model co-optimization on Qlib | large active codebase and Qlib scenario | a frozen universal benchmark or buy-side production result |
| [RD2Bench](https://arxiv.org/abs/2404.11276) | evaluates end-to-end data-centric R&D operations | general agent R&D benchmark | portfolio truth |
| [AlphaAgent](https://arxiv.org/abs/2502.16789) | idea/factor/evaluation agents with AST originality, alignment, and complexity regularization against decay | official code is available | common-protocol dominance after costs across markets |
| [AlphaForge](https://ojs.aaai.org/index.php/AAAI/article/view/33365) | generate a factor zoo, then dynamically select/combine factors | official code uses Qlib and includes GP/RL/DSO/ML comparisons | fixed-formula interpretability of the dynamic portfolio |
| [Alpha-GPT](https://arxiv.org/abs/2308.00016) | human–AI translation of trading ideas into factor search | system/paper line, later human-in-the-loop expansion | autonomous selection hygiene |
| [RiskMiner](https://arxiv.org/abs/2402.07080) | risk-seeking MCTS and collection-aware reward | paper-level method; public reproductions are less canonical | protocol-independent SOTA |
| [Alpha Jungle](https://doi.org/10.1609/aaai.v40i2.37069) | LLM-guided MCTS with backtest feedback and frequent-subtree avoidance | AAAI 2026 paper | immunity to oracle overfitting |
| [Chain-of-Alpha](https://arxiv.org/abs/2508.06312) | generation and optimization chains with iterative feedback | young preprint/code line | mature, independently reproduced standard |
| [CogAlpha](https://aclanthology.org/2026.acl-long.538/) | code-level representation plus LLM reasoning and evolutionary mutation/recombination | ACL 2026 research frontier | a shared live-trading benchmark |
| [qfinzero](https://github.com/CityU-MLO/qfinzero) | adjacent CityU-MLO quant tooling | young repository | a recognized alpha benchmark |

There is therefore no defensible universal “SOTA winner.” The papers use
different markets, periods, formula grammars, labels, budgets, baselines,
costs, and selection rules. The state-of-the-art design pattern is more useful
than a cross-paper rank:

1. typed/executable formulas rather than prose-only judging;
2. deterministic compiler and operator tests;
3. search diversity and duplicate/complexity control;
4. complete trial accounting and train-only feedback;
5. dynamic combination only after individual-factor evidence;
6. evaluator separation from the generating model;
7. frozen, costed, risk-neutral, cross-market and independent replay.

AlphaBench is the best fit for measuring FactorMiner's LLM formula workflow.
Qlib plus the Tier-0 report is the better fit for measuring economic evidence.
These should be two lanes in one evaluation, not one blended score.

## Financial LLM benchmarks

| Benchmark family | Measures | Relevance to FactorMiner |
| --- | --- | --- |
| [FinBen](https://proceedings.neurips.cc/paper_files/paper/2024/hash/adb1d9fa8be4576d28703b396b82ba1b-Abstract-Datasets_and_Benchmarks_Track.html), PIXIU/FLARE, FLUE | financial NLP breadth: extraction, text, QA, forecasting, decisions | model-selection context for narrative agents; not factor evidence |
| [FinQA](https://aclanthology.org/2021.emnlp-main.300/), ConvFinQA, [TAT-QA](https://aclanthology.org/2021.acl-long.254/) | table/text retrieval and multi-step numerical reasoning | useful if agents read filings or derive fundamentals |
| [FinanceBench](https://github.com/patronus-ai/financebench) | open-book analyst QA over filings with evidence | useful for a future filing-research lane, not OHLCV alpha mining |
| FiQA and Chinese financial-language suites | sentiment, retrieval, domain knowledge | only relevant when those inputs enter the factor hypothesis/data contract |
| [FinRL/FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL) | RL environments, train/test/trade workflows | execution/agent research, not a formula-library baseline |
| MMLU, GPQA, SWE-bench, LiveCodeBench | general knowledge/reasoning/coding | choose capable components; never cite as trading validity |

FinBen's final NeurIPS 2024 version reports 42 datasets across 24 tasks; the
earlier arXiv abstract reported 36. Pin the released benchmark version when
stating coverage. FinanceBench contains 10,231 questions, but its public
repository exposes a 150-case annotated evaluation sample; do not imply the
entire corpus is openly bundled.

FLUE, BizBench, CFBenchmark/DISC-FinLLM/Fin-Eva and similar suites fit the
same language-capability lane: sentiment, structure, domain knowledge, or
business arithmetic. [BizBench](https://aclanthology.org/2024.acl-long.452/)
is particularly relevant to quantitative business reasoning, but still does
not execute a factor portfolio. BloombergGPT was evaluated on financial NLP,
general NLP, and internal tasks, as described in the
[BloombergGPT paper](https://arxiv.org/abs/2303.17564), not on Alpha101-style
search with net trading evidence.

The supplied list also names “DiligenceBench (2026).” No stable primary paper,
official repository, versioned dataset, or evaluation contract was found in
this research snapshot. It should remain an unverified watch item, not a
trusted dependency or claim. A social-media announcement alone is not enough
to establish benchmark provenance.

## FactorMiner implementation

The public API is additive:

```python
from factorminer.evaluation import (
    IndustryEvidenceConfig,
    evaluate_industry_evidence,
)

report = evaluate_industry_evidence(
    "candidate_042",
    signals,                 # (assets, periods)
    forward_returns,         # already aligned (assets, periods)
    config=IndustryEvidenceConfig(
        periods_per_year=252,
        cost_bps=(0, 5, 10, 20),
        primary_cost_bps=10,
    ),
    risk_exposures=style_and_industry_exposures,  # (M,K) or (M,T,K)
    risk_weights=float_market_cap,                # optional (M,) or (M,T)
    exposure_names=["size", "beta", "momentum", "industry_software"],
    family_ic_series=every_tried_factor_rankic,
    n_trials=total_search_trials,
    pbo_performance_matrix=trial_by_cpcv_path_scores,
    volume=point_in_time_dollar_volume,
)

payload = report.to_dict()  # strict JSON-compatible; NaN/inf become null
```

The result includes:

- exact Pearson IC and RankIC series and summaries;
- unannualized/annualized ICIR, independence t-stat, and HAC inference;
- block-bootstrap RankIC confidence interval;
- target-weight turnover, gross return series, cost curve, and absolute-spread
  break-even cost (no test-set sign flip is applied);
- raw versus style/industry-residual metrics and cost stress;
- optional family FDR, DSR, PBO, and square-root capacity outputs;
- validation coverage and warnings for evidence the arrays cannot establish.

The standard runtime evaluator also emits explicit Pearson and RankIC fields.
Existing `ic_*` consumers retain the Spearman RankIC behavior and
`paper_ic_v2` version, avoiding a silent historical-score rewrite.

## Recommended benchmark ladder

### Gate A: executable correctness

- parser and operator conformance fixtures;
- exact temporal-window and cross-sectional semantics;
- NaN, ties, constants, suspensions, and universe-entry tests;
- Qlib parity for every shared operator and formula.

### Gate B: frozen formula baselines

- import checksum-addressed Alpha101 and selected Qlib Alpha158/360 configs;
- keep GTJA191 and Alpha20 versioned rather than name-only;
- report compile coverage and failures before performance;
- separate faithful, adapted, and proxy baselines in tables.

### Gate C: research-system capability

- run AlphaBench generation, evaluation, and search tasks;
- freeze model, prompt, budget, operator grammar, evaluator, and seed set;
- report validity, diversity, wall time, tokens/cost, and full trial count;
- never reuse test/evaluation-oracle feedback for final selection.

### Gate D: economic evidence

- train/validation/test and rolling walk-forward panels;
- purge the full label horizon and embargo adjacent samples;
- raw and residual Pearson IC/RankIC with HAC and block-bootstrap inference;
- family FDR, DSR, PBO, turnover, cost, impact, capacity, and regime slices;
- at least one cross-universe or cross-market holdout.

### Gate E: independent execution and forward validation

- export frozen weights/orders to LEAN or another independent engine;
- reconcile calendar, actions, fills, fees, borrow, holdings and P&L;
- paper trade without changing rules;
- promote only after a predeclared observation period and failure policy.

## Highest-value next work

1. Add faithful, checksum-addressed Alpha101 and Qlib handler imports instead
   of expanding proxy catalogs. Formula/operator parity tests come first.
2. Export a frozen-weight replay bundle for LEAN: instrument, decision time,
   target weight, price convention, currency, fee model, and dataset hash.
3. Put the complete research trial ledger into benchmark manifests so DSR,
   FDR, and PBO cannot be run on only the surviving factors.
4. Add point-in-time universe/corporate-action adapters and survivorship tests.
5. Run AlphaBench as a separate agent-capability lane once its data and engine
   commit are frozen; do not merge its NLP/search score into economic alpha.
6. Add risk-model data adapters (open style definitions plus user/vendor
   exposures) while keeping the residualizer vendor-neutral.

The promotion rule should remain simple: a factor is not “industry validated”
because it beat one formula set or one LLM benchmark. It advances only when
each required gate has evidence, and missing gates remain visible.
