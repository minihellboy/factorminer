# Evidence protocol

`industry_evidence_v1` combines predictive, statistical, risk, and implementation
diagnostics. Reports label each gate `measured`, `partial`, `not_supplied`, or
`external_required`. Missing inputs remain explicit. The default admission
contract remains `paper_ic_v2`; see [metric definitions](reproducibility.md#metric-contract).

## Metrics and inference

Reports include cross-sectional Pearson IC and Spearman RankIC. For the evidence
report's IC series:

```text
ICIR                = mean(IC_t) / sample_std(IC_t)
annualized ICIR     = ICIR * sqrt(periods_per_year)
independence t-stat = ICIR * sqrt(number_of_periods)
HAC t-stat          = mean(IC_t) / NeweyWestSE(mean(IC_t))
```

HAC inference accounts for serial dependence; annualized ICIR and a t-statistic
answer different questions. Freeze factor direction before held-out evaluation.

| Diagnostic | Required input / interpretation |
| --- | --- |
| Block-bootstrap confidence interval | IC series; uncertainty under the resampling assumptions |
| Benjamini–Hochberg FDR | Complete family of two-sided HAC tests; assumes independent or suitably positively dependent tests |
| Deflated Sharpe | Declared search-trial count and return moments |
| PBO | Complete trial-by-CPCV-path performance matrix |

Trial accounting includes rejected candidates and inspected alternatives, not
just the final library. These diagnostics require complete inputs and do not
repair leaked selection data or establish arbitrary-dependence error control.

## Turnover and cost stress

The report constructs equal-weight top and bottom legs, each with unit notional:

```text
one_way_turnover_t = 0.5 * sum_i(abs(w[i,t] - w[i,t-1]))
net_return_t = gross_return_t - one_way_cost * sum_i(abs(delta_weight[i,t]))
```

Turnover uses actual target weights and excludes initial entry. The basis-point
cost curve is a diagnostic stress, not an order-fill simulation. With dollar
volume, the square-root impact/capacity estimator adds participation and capital
scenarios. It uses the selected long leg's average liquidity for a shared haircut;
asymmetric liquidity, borrow, fees, and execution constraints need separate inputs.

## Risk residualization

At each date, OLS or positive-weight WLS fits:

```text
signal_t = intercept + exposures_t * beta_t + residual_signal_t
```

Inputs can include static or point-in-time numeric exposures, one-hot industry
classifications, and positive weights. Metrics and cost stress are recomputed
on residual signals. This implements exposure attribution, not estimation of a
complete commercial risk model or verification of exposure availability.

## Python API

```python
from factorminer.evaluation import IndustryEvidenceConfig, evaluate_industry_evidence

report = evaluate_industry_evidence(
    "candidate_042",
    signals,                    # (assets, periods)
    forward_returns,            # aligned to signal decisions
    config=IndustryEvidenceConfig(
        periods_per_year=252,
        cost_bps=(0, 5, 10, 20),
        primary_cost_bps=10,
    ),
    risk_exposures=exposures,    # optional (M,K) or (M,T,K)
    risk_weights=weights,        # optional (M,) or (M,T)
    exposure_names=names,
    family_ic_series=all_trial_rankic,
    n_trials=total_search_trials,
    pbo_performance_matrix=trial_by_path_scores,
    volume=dollar_volume,
)
payload = report.to_dict()       # nonfinite values become JSON null
```

Outputs include correlation series, ICIR/HAC statistics, confidence intervals,
turnover, gross returns, cost curves, break-even costs, residual metrics, and
available family/capacity diagnostics. Coverage and warnings identify evidence
the arrays cannot establish: source truth, point-in-time availability, independent
execution replay, and prospective performance.
