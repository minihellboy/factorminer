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

## Missing data

Almost every CSI300 instrument has gaps: listing after the panel start,
suspensions, delistings. Since operator semantics `numpy-expression-v2`, a gap
is never turned into a number. NaN denominators give NaN (not 0), empty
windows and missing latest values give NaN, and EMA/DEMA/KAMA skip gaps
instead of turning NaN for the rest of the series. Under v1, `EMA($close, 10)`
was NaN on 98.7% of member cells.

`scripts/qlib_cn/alpha158_conformance.py` compares FactorMiner's Alpha158
transcription (`core.reference_catalogs.ALPHA158`) with Qlib's raw feature
values. On CSI300 2008–2014 the median per-feature daily Spearman agreement is
0.9999998, and 131 of 158 features agree above 0.99. The rest are
IMAX/IMIN/IMXD. Qlib's `IdxMax` takes `argmax` without skipping NaN, and every
disagreeing cell has a gap in its window.

## Evaluation backend

`evaluation.backend: gpu` evaluates compiled formula plans on CUDA
(`operators.torch_executor`). Window sums follow NumPy's pairwise summation
order, and scalar divisions are true IEEE divisions. Arithmetic, rolling
moments, regressions, order statistics and ranks are therefore bit-identical
to NumPy. Only `exp`/`log1p`/`tanh`/`pow` (hence Skew, Kurt) can differ in the
last ulp. On the CSI300 panel, 178 of 180 catalog and mined formulas are
bit-identical, and the GPU is 29x faster than the NumPy plan. A step that
finds the device full runs on the host instead.

## Marginal-value admission

`admission.policy: marginal` admits a candidate only for what it adds beyond a
basis made of the 16 leading principal components of Alpha158 (77% of its
variance) plus the admitted library. On each day, members are
rank-transformed, and the candidate and the target are residualized by ridge
regression on the basis. The admission rules are:

- Newey-West t of the mean partial rank IC ≥ 3, and |partial IC| ≥ 0.005;
- residual variance share ≥ 0.2;
- the residual's long-only TopkDropout book (top 1/6 of members, at most 10%
  of the book replaced per day) has positive excess return.
  `min_net_bps` additionally charges 20 bps round trip on its turnover.

Candidates in a batch are admitted greedily, each judged against the earlier
picks. `recent_periods` also requires the partial IC to hold over the end of
the discovery window.

Calibration on 175 catalog and mined factors (train to test, CSI300 classic):
standalone train IC is mostly restated Alpha158 (median residual share 0.29).
Factors with train partial-IC t ≥ 3 kept their sign out of sample in 97% of
cases, and had 4x the test partial IC of the others.

## Campaigns, walk-forward and multiplicity

LLM mining is stochastic, so each arm runs three independent campaigns.
`scripts/qlib_cn/arm_lift.py` scores each campaign's seed-ensembled
`alpha158_plus` model against the seed-ensembled Alpha158 model, using the
daily RankIC difference. An arm's lift is the mean campaign lift series
(Newey-West t). `campaign_sd` shows campaign-to-campaign luck. Arms are compared
on their lift difference.

The `wf2018`, `wf2020`, `wf2022` and `wf2024` protocols have disjoint two-year
test windows covering 2018–2025, each after seven training and two
validation years.

`scripts/qlib_cn/audit_campaigns.py` corrects admitted factors for the whole
search recorded in each campaign's tamper-evident trial ledger. It reports BH
q-values over all trials, and the Deflated Sharpe Ratio with the effective
trial count and the cross-trial Sharpe variance (Bailey & López de Prado,
2014, on the per-period scale).

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
- LLM-driven mining is not seeded end to end. Arms run three campaigns each,
  and `campaign_sd` should be read next to any arm difference.
- The provider is a community rebuild, not Qlib's original Yahoo-sourced
  dataset. Absolute numbers differ slightly from Qlib's published tables.
