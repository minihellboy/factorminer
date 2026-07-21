# Evidence Protocol

FactorMiner's evidence report composes predictive, statistical, risk, and
implementation diagnostics without treating a single score as proof of an
alpha. The protocol is:

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

FactorMiner provides temporal dataset splits,
rolling research evaluation, CPCV/PBO, block-bootstrap tests, Benjamini-Hochberg
FDR, a Deflated Sharpe implementation, cost and capacity models, and frozen
benchmark manifests. `industry_evidence_v1` composes those primitives, adds
explicit Pearson IC and a style/industry signal neutralizer, and keeps the
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

These must not be compared by label alone. The
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

## Public API

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
