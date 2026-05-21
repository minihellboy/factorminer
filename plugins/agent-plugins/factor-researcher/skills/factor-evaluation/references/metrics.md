# Metric definitions

FactorMiner reports several related metrics. Knowing exactly which one is being
quoted prevents the most common reporting error — passing an in-sample or
absolute-value number off as the real result.

## Information Coefficient (IC)

The IC for one period is the **Spearman rank correlation** between a factor's
cross-sectional signal and the forward returns of the same assets. The factor's
IC is the average of those per-period correlations.

| Field | Meaning | Use |
|---|---|---|
| `ic_mean` | Signed mean IC | The honest directional number — a negative IC means the signal is inverted. |
| `ic_paper_mean` | Paper-convention IC | Sign-resolved per the paper protocol, so an inverted signal is credited at its usable strength. The headline metric for paper comparisons. |
| `ic_abs_mean` | Mean of \|IC\| | Magnitude only — useful as a screen, **never** as a result; it cannot be negative and so always flatters. |

## ICIR — Information Coefficient Information Ratio

`ICIR = mean(IC) / std(IC)` across periods.

| Field | Meaning |
|---|---|
| `icir` | ICIR from signed IC |
| `ic_paper_icir` | ICIR under the paper convention |

ICIR is usually **more decision-relevant than IC**: a small but stable IC
(high ICIR) is a more trustworthy signal than a large erratic one.

## Supporting metrics

- **`ic_win_rate`** — fraction of periods with IC in the favorable direction.
  A coin-flip win rate (~50%) with a non-zero IC mean signals a few outlier
  periods carrying the factor.
- **`turnover`** — how much the signal's cross-sectional ranking churns
  period to period. High turnover quietly erases IC once transaction costs
  apply; carry it into `factor-backtest`.

## Redundancy correlation

When a factor is *admitted*, FactorMiner also checks its correlation against
every factor already in the library. A candidate whose signal correlates above
`correlation_threshold` with an existing factor is rejected as redundant — this
is what keeps the library diversified rather than 30 copies of momentum.

## Reporting rule

The deliverable is the **out-of-sample (`test`) `ic_paper_mean` and
`ic_paper_icir`**. Quote `train` numbers only inside an explicit train→test
decay comparison, and never quote `ic_abs_mean` as a headline result.
