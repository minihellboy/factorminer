---
description: Validate or fetch a market dataset before mining
argument-hint: "[dataset path, or an MCP-source config to fetch from]"
---

Load the `factor-data` skill. Schema-check the OHLCV file and confirm the
train/test split has coverage. If an MCP-source config is given instead, run
`fetch-data` to pull the dataset from the connector first, then validate it.
Never proceed to mining on a dataset that failed validation.
