# Bundled Data Notes

The repository includes a small Binance-shaped OHLCV sample:

- file: `data/binance_crypto_5m.csv`
- manifest: `data/binance_crypto_5m.manifest.json`
- matching config: `factorminer/configs/binance_sample.yaml`

This file is for schema validation, CLI smoke tests, examples, and local
workflow checks. It is not the full paper benchmark dataset.

## Current Sample

- frequency: 5 minutes
- rows: 20,000
- assets: 20
- rows per asset: 1,000
- date range: `2026-02-14 21:30:00` through `2026-02-18 08:45:00`
- columns: `datetime`, `asset_id`, `open`, `high`, `low`, `close`, `volume`, `amount`
- sha256: `f6664790ead548fc2d440a5cf8d065492d8e5635f2d0d7fd114b6d09b2be2d33`

Symbols:

```text
ADA, APT, ARB, ATOM, AVAX, BNB, BTC, DOGE, DOT, ETC,
ETH, FIL, LINK, LTC, NEAR, OP, SHIB, SOL, UNI, XRP
```

## What This Does Not Claim

The paper describes a 10-minute Binance benchmark over 64 major assets and a
2024 train / 2025 held-out test protocol. This bundled file is smaller, uses
5-minute bars, and covers a short 2026 window. It should not be used to claim
paper Table 1 reproduction.

Use it to verify that your installation, data columns, validation, resampling,
and report generation work.

## Validate The Sample

```bash
uv run factorminer --config factorminer/configs/binance_sample.yaml \
  validate-data data/binance_crypto_5m.csv
```

## Mine Against The Sample

```bash
uv run factorminer --config factorminer/configs/binance_sample.yaml \
  -o /tmp/factorminer-binance-sample \
  mine --data data/binance_crypto_5m.csv
```

The sample config uses the mock LLM provider so it does not require an API key.

## Convert To 10-Minute Bars

For paper-style Binance workflows, convert 1-minute or 5-minute source data to
10-minute bars first:

```bash
uv run factorminer resample-data \
  data/binance_crypto_5m.csv \
  /tmp/binance_crypto_10m.csv \
  --rule 10min
```

See `docs/binance-reproduction.md` for the paper-style Binance workflow.
