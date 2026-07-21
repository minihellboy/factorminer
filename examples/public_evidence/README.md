# Public evidence datasets

The YAML files in this directory are editable source descriptions. They are not
accepted directly by a benchmark. First resolve every provider checksum into a
lock, then prepare the canonical panel:

```bash
uv run factorminer public-data lock \
  examples/public_evidence/binance_spot_daily_2023_2024.yaml \
  examples/public_evidence/binance_spot_daily_2023_2024.lock.json

uv run factorminer public-data prepare \
  examples/public_evidence/binance_spot_daily_2023_2024.lock.json \
  output/public-evidence-data

uv run factorminer public-data verify output/public-evidence-data
```

The lock freezes every archive URL and upstream SHA-256 before evaluation. The
prepared directory contains a deterministic `market_data.csv`, the lock, and a
manifest binding their hashes and limitations.

The configured source is publicly retrievable, but this repository does not
assert permission to redistribute its raw market data. Reproducers fetch the
same checksum-pinned archives from the provider. Do not use
`--bundle-public-data` unless an independently confirmed redistribution grant
applies to the exact input.

`ecb_fx_daily_2023_2024.yaml` is the redistribution-safe release dataset. The
ECB permits reuse subject to accurate reproduction, source citation, and clear
disclosure of modifications. Its lock hashes the exact API responses because
the API does not publish sidecar checksums. The prepared panel may be bundled
with `--bundle-public-data`; its manifest records the required attribution and
the derived OHLC/volume schema fields. The Binance panel remains the richer
price/volume technical comparison, but must be fetched rather than
redistributed because this project does not assert raw-data redistribution
rights for it.
