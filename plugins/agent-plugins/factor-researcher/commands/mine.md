---
description: Mine a new alpha-factor library from a market dataset
argument-hint: "[dataset path and objective, e.g. 'data/universe.csv, 30 factors']"
---

Load the `factor-mining` skill and run a FactorMiner discovery loop.

First confirm the dataset has passed `factor-data` validation. Then run `mine`
(Ralph loop) by default, or `helix` when causal, regime, debate, or
canonicalization features are wanted. If no dataset is given, ask for one —
do not silently mine on synthetic data.
