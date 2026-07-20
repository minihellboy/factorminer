---
name: factor-backtest
description: Combine a factor library into a composite signal and quintile-backtest it under transaction costs — long-short return, monotonicity, turnover, and tearsheets. Use for the portfolio-level view that single-factor IC does not give. Triggers on "backtest", "composite signal", "combine factors", "long-short return", "portfolio", "quintile", "tearsheet", "transaction costs".
---

# Factor Backtest

A library of individually-decent factors is not a strategy. This skill combines them into one composite signal and backtests the portfolio that signal implies — the level at which transaction costs and capacity actually bite.

## Workflow

### 1. Combine and backtest

```bash
factorminer combine output/run1/factor_library.json \
  --data path/to/market_data.csv \
  --method all --fit-period train --eval-period test
```

- `--method` — `equal-weight`, `ic-weighted`, `orthogonal`, or `all` to compare every method.
- `--fit-period` — split used to fit weights / run selection (use `train`).
- `--eval-period` — split used to score the composite (use `test`).
- `--selection` — optional pre-filter: `lasso`, `stepwise`, `xgboost`, or `none`.
- `--top-k` — keep only the top-K factors before combining.

The report gives composite `IC Mean`, `ICIR`, `Long-Short` return, `Monotonicity`, and `Avg Turnover`.

### 2. Generate tearsheets

For the visual portfolio view — quintile returns, IC time series, correlation heatmap:

```bash
factorminer -o output/run1 visualize output/run1/factor_library.json \
  --data market_data.csv --period test --tearsheet --quintile --correlation
```

## What to look for

- **Monotonicity** — quintile returns should step up Q1→Q5. A non-monotone composite is fragile regardless of headline IC.
- **Long-short return net of turnover** — high `Avg Turnover` means the gross return is optimistic; FactorMiner's transaction-cost model is what makes the net number honest.
- **Method spread** — if `orthogonal` and `equal-weight` disagree sharply, the library has redundant or unstable factors; revisit `factor-evaluation`.

## Guardrails

- Fit weights on `train`, score on `test` — never fit and score on the same split.
- The backtest estimates historical behavior; it is not a forward return promise. Present it as a research artifact for review.
- Report net-of-cost numbers as the headline; gross numbers only as context.
