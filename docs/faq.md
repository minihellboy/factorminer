# FactorMiner FAQ

## Is this a trained model?

No. FactorMiner is a mining and evaluation framework. It uses an LLM provider to
propose formulaic factors, then recomputes and filters those formulas against
market data. The saved output is a factor library, not a trained neural model.

## Is the code fully open source?

The repository code is open source under the project license. External LLM APIs,
market data feeds, and optional third-party services are separate from the code
in this repository.

## What data columns are required?

The loader requires:

```text
datetime, asset_id, open, high, low, close, volume, amount
```

Common aliases are accepted, for example `timestamp` for `datetime`, `ticker`
or `code` for `asset_id`, and `amt` or `turnover` for `amount`. Run:

```bash
uv run factorminer validate-data path/to/market_data.csv
```

## Do I need an API key?

No for local smoke tests and quickstart. Use `--mock` or:

```bash
uv run factorminer quickstart
```

Real LLM-guided mining needs credentials for the configured provider, such as
OpenAI, Anthropic, or Google.

## Why does IC differ from paper IC?

`ic_mean` is signed `mean(IC_t)`. The paper-style metric used for admission and
benchmark selection is `ic_paper_mean = abs(mean(IC_t))`. The legacy diagnostic
`ic_abs_mean = mean(abs(IC_t))` remains visible because it is useful for finding
unstable factors. See [Metric Semantics](metrics.md).

## What can I reproduce without private data?

You can reproduce CLI behavior, quickstart reports, mock-data mining, and small
benchmark smoke tests. Paper-scale A-share or Binance tables require
user-provided data covering the configured train/test periods and universes.

## Is `data/binance_crypto_5m.csv` the paper crypto dataset?

No. It is a small 20-symbol, 5-minute sample for validation and CLI workflows.
The paper-style Binance benchmark uses 10-minute bars, 64 major Binance assets,
and 2024 train / 2025 held-out test periods. See
[`data/README.md`](../data/README.md) and
[`docs/binance-reproduction.md`](binance-reproduction.md).

## Why does the bundled Binance sample use 2026 dates?

The bundled file is not intended for paper Table 1 reproduction. Use
`factorminer/configs/binance_sample.yaml` when working with the sample because
its train/test split matches the short 2026 sample window.
