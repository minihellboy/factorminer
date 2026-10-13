# Qlib CN benchmark

This benchmark scores FactorMiner factor sets and Qlib's reference workflows on
the same China A-share rows, label, splits, and backtest. It answers two
questions: how good are the factors on their own, and do they add to a strong
handcrafted baseline (Alpha158 + LightGBM)?

## Data

The panel comes from the community Qlib CN provider
([chenditc/investment_data](https://github.com/chenditc/investment_data)),
which publishes a daily `qlib_bin.tar.gz` with a manifest and SHA-256. It has
split/dividend-adjusted daily bars and point-in-time CSI300/CSI500 membership.

`scripts/qlib_cn/export_panel.py` runs in a separate interpreter with `pyqlib`
installed. It exports one long panel per universe:

- every instrument that was a member at any date in the window, over the whole
  window, so time-series operators see full histories;
- `in_universe`, set on the dates the instrument was a member.

FactorMiner masks targets outside membership (`in_universe = 0`). Metrics
therefore score each date over that day's members, which matches Qlib's dynamic
instrument filter. Prices stay visible to formulas.

## Protocols

| Protocol | Train | Validation | Test |
| --- | --- | --- | --- |
| `classic` (Qlib `examples/benchmarks`) | 2008-01-01 – 2014-12-31 | 2015-01-01 – 2016-12-31 | 2017-01-01 – 2020-08-01 |
| `recent` | 2013-01-01 – 2019-12-31 | 2020-01-01 – 2021-12-31 | 2022-01-01 – 2026-09-30 |

The label is Qlib's `Ref($close, -2) / Ref($close, -1) - 1`, which is a
`close_to_close` target with `entry_delay_bars: 1` and `holding_bars: 1`.
`factorminer/configs/qlib_cn_csi300.yaml` declares the classic profile. Its
`preprocess` section turns off cross-sectional z-scoring, winsorising, and
peer-median filling, so formulas read adjusted price levels as Qlib's handlers
do.

Mining sees only the train split, minus the target horizon. Factor directions
are fixed on train, and validation and test report signed RankIC under that
direction.

## Conformance

On CSI300 2008–2020, FactorMiner's target and Qlib's label agree to within 6e-8
on all 878,599 rows where both are defined. FactorMiner scores no row outside
Qlib's membership. One difference is accepted: 5,208 rows (0.6%) where the stock
has no bar at the signal date (suspended) still carry a Qlib label, but have no
FactorMiner target.

Reproducing Qlib's published Alpha158 + LightGBM CSI300 result on this provider
gives IC 0.0447 and RankIC 0.0484 for seed 0 (published: 0.0448 and 0.0469).

## Running

```bash
# 1. Provider and isolated Qlib environment
gh release download <tag> -R chenditc/investment_data -p 'qlib_bin*'
uv venv --python 3.12 qlib-venv && VIRTUAL_ENV=qlib-venv uv pip install pyqlib==0.9.7 lightgbm==4.6.0 pyarrow
QPY=qlib-venv/bin/python

# 2. Panels
$QPY -I scripts/qlib_cn/export_panel.py --provider qlib_cn --universe csi300 \
  --start 2007-01-01 --end 2026-10-12 --out panels/csi300.parquet \
  --source-manifest qlib_bin.manifest.json
# FactorMiner reads CSV unless pyarrow is installed in its environment
$QPY -I -c "import pandas as pd; pd.read_parquet('panels/csi300.parquet').to_csv('panels/csi300.csv', index=False)"
# Raw Qlib labels for paired comparisons
$QPY -I -c "import qlib; from qlib.data import D; qlib.init(provider_uri='qlib_cn', region='cn'); \
  df = D.features(D.instruments('csi300'), ['Ref(\$close, -2)/Ref(\$close, -1) - 1'], start_time='2008-01-01'); \
  df.columns = ['label']; df.reset_index().to_parquet('labels/csi300.parquet')"

# 3. Qlib reference matrix (Alpha158/Alpha360 x LightGBM/linear, three seeds)
scripts/qlib_cn/run_reference_matrix.sh $QPY qlib_cn output/qlib_cn

# 4. A FactorMiner factor set: direction-frozen scores plus exported values
uv run python scripts/qlib_cn/evaluate_library.py -c factorminer/configs/qlib_cn_csi300.yaml \
  --data panels/csi300.csv --library output/run/factor_library.json \
  --out output/qlib_cn/csi300/classic/mined

# 5. The same LightGBM on those factors, alone and appended to Alpha158
$QPY -I scripts/qlib_cn/run_qlib_baselines.py --provider qlib_cn --universe csi300 \
  --protocol classic --handler factors --factors output/qlib_cn/csi300/classic/mined/factor_values.csv \
  --out output/qlib_cn/csi300/classic/mined_factors_lgb_s0
$QPY -I scripts/qlib_cn/run_qlib_baselines.py ... --handler alpha158_plus ...

# 6. Paired HAC test of daily RankIC differences (seeds averaged per side)
$QPY -I scripts/qlib_cn/compare_predictions.py --label labels/csi300.parquet \
  --a output/qlib_cn/csi300/classic/mined_alpha158_plus_lgb_s0/predictions.parquet \
  --b output/qlib_cn/csi300/classic/alpha158_lgb_s0/predictions.parquet

# 7. One results table
python scripts/qlib_cn/aggregate.py output/qlib_cn > output/qlib_cn/results.json
```

Supplied `vwap` columns are kept: in this provider `amount / volume` is 0.1x
the true VWAP (unit mismatch), so deriving VWAP would misscale every formula
that mixes `$vwap` with prices.

Market data and factor values stay local. Run outputs record the provider
release, panel SHA-256, splits, model parameters, and seeds.

## Limitations

- Cross-sectional operators (`CsRank`, ...) rank over every exported
  instrument, including non-members on that date. Member ordering is preserved,
  but nonlinear combinations of ranks differ slightly from member-only ranking.
- LLM-driven mining is not seeded end to end. Mining comparisons use one
  campaign per arm, and differences should be read alongside the seed spread of
  the downstream models.
- The provider is a community rebuild, not Qlib's original Yahoo-sourced
  dataset. Absolute numbers differ slightly from Qlib's published tables.
