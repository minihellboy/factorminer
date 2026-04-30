# Binance Reproduction Notes

The paper-style Binance lane expects 10-minute OHLCV bars. Binance public spot
klines do not include a native `10m` interval in the
[documented kline interval set](https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data),
so use 1-minute or 5-minute source data and resample it to 10 minutes before
mining or benchmarking.

## Bundled Sample Data

`data/binance_crypto_5m.csv` is a small public sample for data validation and
CLI smoke testing. It is not the full paper benchmark dataset.

Current sample summary:

- frequency: 5 minutes
- rows: 20,000
- symbols: 20
- rows per symbol: 1,000
- date range: `2026-02-14 21:30:00` to `2026-02-18 08:45:00`
- symbols: `ADA`, `APT`, `ARB`, `ATOM`, `AVAX`, `BNB`, `BTC`, `DOGE`, `DOT`,
  `ETC`, `ETH`, `FIL`, `LINK`, `LTC`, `NEAR`, `OP`, `SHIB`, `SOL`, `UNI`, `XRP`

## Resample 5m To 10min

```bash
uv run factorminer resample-data \
  data/binance_crypto_5m.csv \
  /tmp/binance_crypto_10m.csv \
  --rule 10min
```

Aggregation uses standard candle semantics:

- `open`: first source bar
- `high`: max
- `low`: min
- `close`: last source bar
- `volume`: sum
- `amount`: sum
- `vwap`: recomputed as `amount / volume`
- `returns`: recomputed from resampled close prices per symbol

## Paper-Style Config

Use `factorminer/configs/paper_repro_binance.yaml` as the starting point. It
pins:

- market: `crypto`
- universe: `Binance`
- frequency: `10min`
- train period: `2024-01-01` through `2024-12-31`
- held-out test period: `2025-01-01` through `2025-12-31`

Example:

```bash
uv run factorminer \
  --config factorminer/configs/paper_repro_binance.yaml \
  -o output-binance-paper \
  benchmark table1 --data /tmp/binance_crypto_10m.csv --baseline factor_miner
```
