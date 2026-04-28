# Reproducibility Guide

This repository separates smoke reproducibility from paper-scale data
reproduction.

## Reproducible Without Private Data

Quickstart:

```bash
uv run factorminer quickstart
```

Validate the bundled sample:

```bash
uv run factorminer validate-data examples/quickstart/sample_market_data.csv
```

Mine against mock data:

```bash
uv run factorminer -o /tmp/factorminer-mock mine --mock -n 2 -b 8 -t 2
```

Generate a static report:

```bash
uv run factorminer report /tmp/factorminer-mock/factor_library.json \
  --session-log /tmp/factorminer-mock/session_log.json \
  --format html \
  --output /tmp/factorminer-mock/report.html
```

## Requires User-Provided Market Data

Paper-style A-share and Binance evaluations require OHLCV plus amount data over
the configured train and test windows. Start by validating the file:

```bash
uv run factorminer validate-data path/to/market_data.csv
```

Then mine:

```bash
uv run factorminer -c factorminer.local.yaml -o output-real \
  mine --data path/to/market_data.csv
```

## Table 1-Style Runtime Evaluation

Use the canonical benchmark runner:

```bash
uv run factorminer -c factorminer.local.yaml -o output-benchmark \
  benchmark table1 --data path/to/market_data.csv
```

To restrict the run while checking setup:

```bash
uv run factorminer -c factorminer.local.yaml -o output-benchmark \
  benchmark table1 --data path/to/market_data.csv --baseline factor_miner
```

Every benchmark result and manifest includes `metric_version`. Paper-mode
selection uses `ic_paper_mean = abs(mean(IC_t))` and
`ic_paper_icir = abs(mean(IC_t)) / std(IC_t)`.
