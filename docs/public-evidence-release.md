# Public Evidence Release Runbook

This runbook turns a public-data experiment into an independently verifiable,
content-addressed release. It separates three claims that must not be blurred:

1. the provider archives were fetched at the URLs and SHA-256 values in the
   committed lock;
2. FactorMiner normalized and evaluated those exact bytes under the frozen
   configuration and seeds;
3. the resulting receipt and every declared artifact still match their
   recorded hashes.

It does not claim that historical results will persist, that the selected
universe was point-in-time, or that raw provider data may be redistributed.

## Reference dataset

The release-safe reference covers 12 ECB daily euro foreign-exchange reference
rate series from 2023-01-01 through 2024-12-31. The ECB permits free reuse of
information obtained from its site when it is reproduced accurately, the ECB
is cited, and modifications are stated. The editable YAML freezes the universe,
transformations, and limitations; the JSON lock freezes the exact SHA-256 of
each API response.

ECB says these information-only reference observations are generally published
at about 16:00 CET on working days. The normalized midnight timestamp is only an
observation-date label: the pipeline treats the value as available after that
publication and evaluates only the subsequent daily observation.

```bash
uv sync --extra dev --extra visualization

uv run factorminer public-data prepare \
  examples/public_evidence/ecb_fx_daily_2023_2024.lock.json \
  output/public-evidence-data

uv run factorminer public-data verify output/public-evidence-data
```

The expected canonical panel has 6,132 rows, 12 assets, 511 daily timestamps,
and SHA-256
`d9a17748808f29892901fae6c1ef37bfba3bdaca06df703c943d954afe5d1096`.
The committed lock SHA-256 is
`9c5c4630ee789bf7e78ce192cd8e05bf349756247ef55aaa312227ca7bf9a0b6`.
A mismatch is a new dataset version, not the same experiment.

To audit the provider checksum resolution without changing the committed lock:

```bash
uv run factorminer public-data lock \
  examples/public_evidence/ecb_fx_daily_2023_2024.yaml \
  output/recreated-public-data.lock.json
cmp examples/public_evidence/ecb_fx_daily_2023_2024.lock.json \
  output/recreated-public-data.lock.json
```

## Full evidence run

Run the three credential-free formula baselines, ten derived seeds, and the
Tier-0 evidence report. Ralph/Helix generation requires an external model
provider and is excluded from the fresh-clone claim. The normalized ECB panel
is bundled under its stated reuse conditions and attribution.

The runner deterministically divides unique timestamps into 60% train, 20%
validation, and 20% test windows. One timestamp is purged at each boundary so
the one-bar forward target at the end of a fit/selection window cannot enter the
next window; the recorded embargo is zero bars. Admission is fit on train,
Top-K selection and combination weights use validation, and the Tier-0 report
uses test only.

```bash
uv run python scripts/run_phase2_benchmark.py \
  --data output/public-evidence-data/market_data.csv \
  --dataset-manifest output/public-evidence-data/dataset_manifest.json \
  --data-license-class redistributable_with_attribution \
  --evidence-tier public_reproducible \
  --portable-release \
  --bundle-public-data \
  --runs 10 \
  --seed 42 \
  --n-factors 40 \
  --skip-ablation \
  --output output/public-evidence-run \
  --methods random_exploration alpha101_classic alpha101_adapted
```

The output includes per-seed runtime manifests, baseline comparison tables,
Pearson IC, RankIC, HAC uncertainty, turnover/cost-pressure paths, DSR and
explicit missing-gate states, statistical tests, the effective configuration,
reports, and a content-addressed directory below
`output/public-evidence-run/releases/`.

## Fresh-clone verification

On another machine, check out the receipt's `code_sha`, rebuild the data from
the committed lock, then run:

```bash
uv sync --frozen --extra dev --extra visualization
uv run factorminer verify-receipt \
  output/public-evidence-run/releases/RELEASE_ID
```

Verification fails on a changed receipt, manifest, declared artifact,
environment lock, or dataset commitment. Relocatable artifact paths mean the
release directory can be copied without retaining the producer's absolute
filesystem layout.

## Publication checklist

- Attach the portable release directory as an archive, retaining the ECB
  attribution and transformation disclosures in its dataset descriptor.
- Record the repository commit and release ID in the GitHub Release notes.
- Report every method and seed, not only the best run.
- Retain the manifest's baseline provenance. The repository's
  `alpha101_classic` and `alpha101_adapted` catalogs are versioned implemented
  subsets/proxies and must not be described as a complete Alpha101, Alpha158,
  Alpha191, or Alpha360 implementation.
- State that reference observations are not executable closes, OHLC fields are
  derived from one observation, volume is a schema placeholder, liquidity and
  capacity are unavailable, the universe is ex-post selected, and no result is
  live evidence.
- Link the exact lock and this runbook.
- Have an independent person follow the fresh-clone procedure before calling
  the artifact independently reproduced.

The source series are published through the
[ECB Data Portal](https://data.ecb.europa.eu/data/datasets/EXR), and reuse is
governed by the ECB's
[disclaimer and copyright conditions](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html).
Publication timing and the information-only/transaction warning come from the
[ECB reference-rate page](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html).
The panel explicitly states that FactorMiner inverted the rates and derived
schema fields. The separate Binance example retains real price/volume fields
for technical comparison but is not the redistribution-safe release input.
