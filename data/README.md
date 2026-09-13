# Onboarding data

`binance_crypto_5m.csv` is a Binance-shaped OHLCV workflow sample, distributed
under the project MIT License. Its provenance and checksum are recorded in
[`binance_crypto_5m.manifest.json`](binance_crypto_5m.manifest.json).

| Property | Value |
| --- | --- |
| Frequency | 5 minutes |
| Rows | 20,000 |
| Assets | 20, with 1,000 rows each |
| Dates | 2026-02-14 21:30:00 through 2026-02-18 08:45:00 |
| Fields | datetime, asset_id, open, high, low, close, volume, amount |
| Configuration | `factorminer/configs/binance_sample.yaml` |

The matching configuration uses a mock provider and needs no API key:

```bash
uv run factorminer -c factorminer/configs/binance_sample.yaml \
  validate-data data/binance_crypto_5m.csv
uv run factorminer -c factorminer/configs/binance_sample.yaml \
  -o output/sample mine --data data/binance_crypto_5m.csv
```

Use this panel for installation, schema, and workflow checks. It is not the
full research-paper dataset. Larger comparisons need a separately sourced
panel with matching targets and splits; see [reproducibility](../docs/reproducibility.md).
