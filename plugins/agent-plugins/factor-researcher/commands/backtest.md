---
description: Combine factors into a composite signal and quintile-backtest it
argument-hint: "[library path and combination method, e.g. 'factor_library.json ic-weighted']"
---

Load the `factor-backtest` skill. Combine the library into a composite signal
and quintile-backtest the implied portfolio under transaction costs. Fit weights
on `train`, score on `test`. Report long-short return net of turnover costs as
the headline.
