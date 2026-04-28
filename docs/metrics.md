# FactorMiner Metric Semantics

FactorMiner evaluates a factor by computing a cross-sectional Spearman
Information Coefficient for each time step:

```text
IC_t = SpearmanRankCorr(signal_t, forward_return_t)
```

The important detail is how that time series is summarized.

## Current Metrics

| Field | Definition | Default use |
| --- | --- | --- |
| `ic_mean` | signed `mean(IC_t)` | diagnostics, sign-aware weighting |
| `ic_paper_mean` | `abs(mean(IC_t))` | paper-mode admission, replacement, sorting, Top-K freeze, benchmark selection |
| `ic_abs_mean` | `mean(abs(IC_t))` | legacy diagnostic and backward compatibility |
| `icir` | signed `mean(IC_t) / std(IC_t)` | diagnostics |
| `ic_paper_icir` | `abs(mean(IC_t)) / std(IC_t)` | paper-mode ICIR gate and reports |

`metric_version: paper_ic_v2` marks artifacts written with these explicit fields.
Older libraries without the new fields load with compatibility defaults and are
reported as `legacy_abs_ic`.

## Why This Changed

Issue #3 was valid. The previous implementation used `mean(abs(IC_t))` in
places where the paper-style scalar should be `abs(mean(IC_t))`.

Those are not equivalent:

```text
IC_t = [0.1, -0.1]
mean(abs(IC_t)) = 0.1
abs(mean(IC_t)) = 0.0
```

The first value says the factor was often strong in either direction. The second
value says the factor had no stable average direction. Paper-mode admission
should reject the alternating example at any positive IC threshold.

## Practical Reading

- Use `ic_paper_mean` when comparing FactorMiner results to paper-style tables.
- Use `ic_mean` when you need the sign of the relationship.
- Use `ic_abs_mean` only as a diagnostic for unstable or alternating factors.
- If a report warns about legacy metric fields, rerun mining or evaluation to
  regenerate current artifacts.
