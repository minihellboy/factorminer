# Qlib CN benchmark

This benchmark scores FactorMiner factor sets and Qlib's reference workflows on
the same China A-share rows, label, splits, and backtest. It asks one question:
do LLM-mined factors add predictive value to a strong handcrafted baseline
(Alpha158 + LightGBM), out of sample, after accounting for the whole search?

## Findings

CSI300, 2008–2025, LLM mining with DeepSeek. 30 campaigns, 35,711 evaluated
formulas, and 471 LightGBM fits.

- **Marginal admission is the only rule that adds ranking value, and only
  over a tuned baseline.** Against Alpha158 + LightGBM with robust feature
  scaling and a ranked label, marginal admission raises daily RankIC by
  +0.0032 over 2018–2025 (pooled walk-forward, t 2.03). It is positive in
  three of four windows and in all three classic campaigns (+0.0032, sd
  0.0006). It beats the paper rule in walk-forward (+0.0034, t 2.59) and
  every paper arm on classic (t 1.5–2.0). The paper rule adds nothing
  (−0.0002 to +0.0011). Against Qlib's default baseline, no arm is
  significant (best +0.0018, t 1.21) or distinguishable from a negative
  control that feeds Alpha158 in twice (+0.0011, t 1.41).
- **No ranking gain reaches the portfolio.** No mined arm changes the daily
  top-50 spread that a TopkDropout book trades (|t| ≤ 1.43 against either
  baseline). After-cost IR changes stay within seed-to-seed noise (single
  campaigns range over ±0.4). The extra signal lives in the body of the
  cross-section, which RankIC weights and a top-50 book ignores.
- **Preprocessing beats mining on RankIC and loses on the portfolio.** The
  tuned processors lift the baseline's RankIC from 0.0523 to 0.0583 (classic)
  and from 0.0281 to 0.0398 (walk-forward), more than any mined set. But they
  cut the top-50 spread (13.5 → 11.1 and 6.2 → 5.1 bps/day) and the IR
  (1.21 → 0.94 and 1.00 → 0.72). RankIC is not the portfolio's objective,
  and lift measured on it does not transfer.
- **Marginal value decays out of sample.** A marginal-arm factor's partial IC
  beyond Alpha158 keeps 27% of its train value in walk-forward and 42% on
  classic. In walk-forward, its sign holds for 61% of factors. The t ≥ 3 gate
  is well placed: above it, factors keep their sign 75–85% of the time
  (58% below), with three to five times the test partial IC. But even at
  t ≥ 6 only 28% of the value survives.
- **The handcrafted catalog's classic win is era-specific.** The 110-formula
  FactorMiner catalog adds +0.0061 RankIC (t 2.31) on 2017–2020 but −0.0005
  across 2018–2025 against the default baseline. Against the tuned baseline it
  matches marginal admission (+0.0032, t 1.68; difference 0.0000).
- **Significance is not novelty.** The paper rule's admitted factors all pass
  BH-FDR over the full ledger and a noise-only Deflated Sharpe hurdle, and on
  classic all keep their standalone sign out of sample. Yet their average
  partial IC beyond Alpha158 is about 40% below marginal admission's (train
  0.0050 against 0.0080, test 0.0021 against 0.0028), and they add no lift.
- **Memory, the research planner and the cost gate change nothing
  measurable.** All their pairwise differences are within |t| 1.5, and
  campaign-to-campaign spread (sd 0.0006–0.0019) is the size of the effects.

The plan's next step was new data types (fundamentals, flows, intraday). The
evidence argues for fixing the objective first: admit and train on what the
portfolio trades (top-of-book marginal spread, a top-weighted loss), not on
RankIC. Until a mined set moves the top-50 spread, more data feeds a target
that does not pay.

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

Supplied `vwap` columns are kept. In this provider `amount / volume` is 0.1x
the true VWAP (a unit mismatch), so deriving VWAP would misscale every formula
that mixes `$vwap` with prices.

## Protocols

| Protocol | Train | Validation | Test |
| --- | --- | --- | --- |
| `classic` (Qlib `examples/benchmarks`) | 2008 – 2014 | 2015 – 2016 | 2017-01 – 2020-07 |
| `recent` | 2013 – 2019 | 2020 – 2021 | 2022-01 – 2026-09 |
| `wf2018` | 2009 – 2015 | 2016 – 2017 | 2018 – 2019 |
| `wf2020` | 2011 – 2017 | 2018 – 2019 | 2020 – 2021 |
| `wf2022` | 2013 – 2019 | 2020 – 2021 | 2022 – 2023 |
| `wf2024` | 2015 – 2021 | 2022 – 2023 | 2024 – 2025 |

The label is Qlib's `Ref($close, -2) / Ref($close, -1) - 1`. In FactorMiner
terms that is a `close_to_close` target with `entry_delay_bars: 1` and
`holding_bars: 1`. `factorminer/configs/qlib_cn_csi300.yaml` declares the
classic profile. Its `preprocess` section turns off cross-sectional
z-scoring, winsorising, and peer-median filling, so formulas read adjusted
price levels as Qlib's handlers do. The walk-forward configs differ from it
only in their splits.

Mining sees only the train split, minus the target horizon. Factor directions
are fixed on train, and validation and test report signed RankIC under that
direction. LightGBM fits on train, early-stops on validation, and predicts
test.

## Conformance

On CSI300 2008–2020, FactorMiner's target and Qlib's label agree to within 6e-8
on all 878,599 rows where both are defined. FactorMiner scores no row outside
Qlib's membership. One difference is accepted: 5,208 rows (0.6%) where the stock
has no bar at the signal date (suspended) still carry a Qlib label, but have no
FactorMiner target.

Reproducing Qlib's published Alpha158 + LightGBM CSI300 result on this provider
gives IC 0.0447 and RankIC 0.0484 for seed 0 (published: 0.0448 and 0.0469).

`scripts/qlib_cn/alpha158_conformance.py` compares FactorMiner's Alpha158
transcription (`core.reference_catalogs.ALPHA158`) with Qlib's raw feature
values. On CSI300 2008–2014, the median per-feature daily Spearman agreement is
0.9999998, and 131 of 158 features agree above 0.99. The rest are
IMAX/IMIN/IMXD: Qlib's `IdxMax` takes `argmax` without skipping NaN, and every
disagreeing cell has a gap in its window.

## Missing data

Almost every CSI300 instrument has gaps: listing after the panel start,
suspensions, delistings. Since operator semantics `numpy-expression-v2`, a gap
is never turned into a number:

- NaN denominators give NaN, not 0;
- empty windows and missing latest values give NaN;
- EMA/DEMA/KAMA skip gaps instead of turning NaN for the rest of the series.

Under v1, `EMA($close, 10)` was NaN on 98.7% of member cells. All results
below use v2.

## Evaluation backend

`evaluation.backend: gpu` evaluates compiled formula plans on CUDA
(`operators.torch_executor`). Window sums follow NumPy's pairwise summation
order, and scalar divisions are true IEEE divisions. Arithmetic, rolling
moments, regressions, order statistics and ranks are therefore bit-identical
to NumPy. Only `exp`/`log1p`/`tanh`/`pow` (hence Skew, Kurt) can differ in the
last ulp. On the CSI300 panel (904 assets × 4,802 days), 178 of 180 catalog and
mined formulas are bit-identical, and the GPU is 29x faster than the NumPy
plan.

A step that finds the device full runs on the host instead, so campaigns can
share a card. Mock mining went from 16.7 s to about 5 s per iteration with the
GPU backend and the dependence-index fix.

## Marginal-value admission

`admission.policy: marginal` admits a candidate only for what it adds beyond a
basis: the 16 leading principal components of Alpha158 (77% of its variance)
plus the admitted library. On each day, members are rank-transformed, and the
candidate and the target are residualized by ridge regression on the basis. A
candidate is admitted when all of these hold:

- Newey-West t of the mean partial rank IC ≥ 3, and |partial IC| ≥ 0.005;
- residual variance share ≥ 0.2;
- the residual's long-only TopkDropout book has positive excess return. The
  book holds the top 1/6 of members and sells held names only from the bottom
  10% of held plus new candidates, as Qlib does. `min_net_bps` additionally
  charges 20 bps round trip on its turnover.

Candidates in a batch are admitted greedily, each judged against the earlier
picks. `recent_periods` also requires the partial IC to hold over the end of
the discovery window (t ≥ `min_recent_t`). Every trial's marginal statistics
are recorded in the trial ledger.

## Experimental design

**Arms.** Every campaign runs 60 iterations of 40 LLM candidates
(`deepseek-flash`) on the train split. Planner arms take up to 240 research
actions and stopped after 73–176.

| Arm | Admission | Other settings |
| --- | --- | --- |
| `memory` / `control` | paper rule (IC ≥ 0.02, ICIR ≥ 0.2, max corr 0.5) | experience memory on |
| `nomemory` | paper rule | memory off |
| `planner` | paper rule | research planner (`decision_value`) |
| `marginal` | marginal | |
| `marginalcost` | marginal, `min_net_bps: 0` | |
| `plannermarginal` | marginal | research planner |
| `marginalrecent` | marginal, `recent_periods: 488` (two years) | |
| `catalog` | none: the 110-formula FactorMiner catalog | |
| `alpha158fm` | none: FactorMiner's Alpha158 (negative control) | |

On `classic`, each mined arm runs three independent campaigns. Each
walk-forward window runs one campaign per arm (`control`, `marginal`,
`marginalrecent`), so the windows are the replicates. Every factor set is
scored by `evaluate_library.py` and fed to LightGBM twice, three seeds each:
alone (`factors` handler) and appended to Alpha158 (`alpha158_plus`).

**Lift.** `scripts/qlib_cn/arm_lift.py` rank-averages each campaign's seeds
and compares them with the seed-ensembled Alpha158 model, date by date. It
uses two baselines: Qlib's default Alpha158 handler and a tuned variant (see
[Tuned baseline](#tuned-baseline)), each against Alpha158+factors under the
same processors.

- the *lift* is the daily RankIC difference;
- the *top lift* is the difference in the daily top-50 spread (mean label of
  the 50 best-scored names minus the day's mean, in bps), the part of the
  cross-section a TopkDropout book trades.

An arm's lift is the mean over its campaigns, with a Newey-West t over
dates. `campaign_sd` is the campaign-to-campaign spread. Walk-forward windows
are pooled by concatenating their disjoint lift series, each against its own
baseline. A campaign that admitted nothing (`wf2024_control`) counts as zero
lift.

**Persistence.** `evaluate_library.py --marginal` rebuilds the marginal basis
on each split and reports each factor's partial IC beyond Alpha158, oriented
by its train sign. `scripts/qlib_cn/marginal_persistence.py` aggregates the
results.

**Multiplicity.** `scripts/qlib_cn/audit_campaigns.py` re-tests the admitted
factors against the full search in each campaign's tamper-evident trial ledger.
It reports:

- BH q-values over all trials;
- the effective trial count (Bailey & López de Prado average-correlation
  estimate);
- two Deflated Sharpe hurdles on each factor's daily IC series. `dsr_null`
  uses the noise-only variance 1/(T−1). `dsr` uses the observed cross-trial
  variance of signed Sharpe ratios.

## Results

### Baselines

| Protocol | Alpha158 LGB RankIC (3-seed ensemble) | Top-50 spread (bps/day) | IR after cost (seed mean) |
| --- | --- | --- | --- |
| classic | 0.0523 | 13.5 | 1.21 |
| wf2018 | 0.0434 | 10.1 | 1.52 |
| wf2020 | 0.0353 | 9.8 | 1.80 |
| wf2022 | 0.0126 | 1.2 | −0.08 |
| wf2024 | 0.0208 | 3.8 | 0.77 |
| wf pooled | 0.0281 | 6.2 | 1.00 |

The baseline itself decays: by 2022–23 the top-50 spread is about a tenth of
its 2017–20 level.

### Classic: six mining arms, three campaigns each

Against Qlib's default baseline:

| Arm | Library | RankIC lift | t | campaign sd | Top-50 lift (bps) | t | IR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| memory | 15 / 11 / 13 | +0.0023 | 1.42 | 0.0016 | +0.21 | 0.34 | 1.28 |
| nomemory | 11 / 8 / 10 | +0.0019 | 0.92 | 0.0019 | −0.19 | −0.26 | 1.15 |
| planner | 8 / 14 / 6 | +0.0015 | 1.31 | 0.0016 | +0.19 | 0.36 | 1.35 |
| marginal | 32 / 31 / 24 | +0.0028 | 1.41 | 0.0008 | −0.82 | −1.10 | 1.11 |
| marginalcost | 9 / 7 / 5 | +0.0009 | 0.64 | 0.0006 | −0.80 | −1.39 | 1.07 |
| plannermarginal | 19 / 15 / 15 | +0.0020 | 1.28 | 0.0014 | −0.45 | −0.78 | 1.29 |
| catalog | 110 | +0.0061 | 2.31 | – | +0.50 | 0.49 | 0.98 |
| alpha158fm | 158 | −0.0019 | −2.22 | – | −0.20 | −0.35 | 1.29 |

Between mined arms, no RankIC difference is significant (|t| ≤ 1.50), for
example memory − nomemory +0.0003 (t 0.29) and marginal − marginalcost
+0.0019 (t 1.50). On the top-50 spread, the paper arms lean ahead of the
marginal ones (memory − marginal t 1.75, planner − marginalcost t 1.94): the
residual signal marginal admission selects sits in the body of the
cross-section, not in its top. The catalog beats every mined arm on RankIC
(t −1.9 to −2.7). All arms beat the duplicated-Alpha158 control (t 1.9–3.1),
which is itself significantly negative: 158 redundant columns cost LightGBM
about 0.002 RankIC.

Per-seed IRs vary by ±0.2 within an arm. IR is a single number per run, so it
cannot be paired across arms; the top-50 lift is the paired test of the same
question.

### Walk-forward, 2018–2025

Against Qlib's default baseline:

| Arm | 2018–19 | 2020–21 | 2022–23 | 2024–25 | Pooled lift (t) | Pooled top-50 (bps, t) |
| --- | --- | --- | --- | --- | --- | --- |
| control | +0.0048 | −0.0007 | +0.0005 | 0 (empty) | +0.0012 (1.06) | −0.33 (−0.62) |
| marginal | +0.0036 | +0.0025 | −0.0006 | +0.0017 | +0.0018 (1.21) | −0.00 (−0.00) |
| marginalrecent | −0.0023 | −0.0005 | +0.0014 | +0.0066 | +0.0013 (0.87) | −0.99 (−1.43) |
| catalog | +0.0052 | −0.0015 | −0.0030 | −0.0028 | −0.0005 (−0.29) | +0.23 (0.29) |
| alpha158fm | +0.0001 | +0.0003 | +0.0008 | +0.0033 | +0.0011 (1.41) | −0.27 (−0.49) |

Library sizes, in window order: control 29/13/6/0, marginal 39/36/33/13,
marginalrecent 31/36/22/10. Every pairwise difference is insignificant
(|t| ≤ 1.50 on both metrics). The duplicated-Alpha158 control shows the noise
floor of the design: about ±0.0015 RankIC over 8 years.

The paper rule finds less and less as windows move forward, and admitted
nothing for 2024–25. Marginal admission keeps finding 13–39 factors. The
recent-decay gate helps in the latest window (+0.0066, t 1.88) and hurts in
the first (−0.0023). Pooled, it is no better than the plain marginal rule.

### Factors alone

LightGBM on each set alone (`factors` handler), mean RankIC over seeds:

| | classic | wf2018 | wf2020 | wf2022 | wf2024 |
| --- | --- | --- | --- | --- | --- |
| Alpha158 (Qlib handler) | 0.0498 | 0.0413 | 0.0337 | 0.0112 | 0.0178 |
| alpha158fm (factor handler) | 0.0560 | 0.0495 | 0.0372 | 0.0286 | 0.0356 |
| catalog | 0.0511 | 0.0407 | 0.0359 | 0.0288 | 0.0292 |
| best mined arm | 0.0372 (marginal) | 0.0365 (marginal) | 0.0311 (marginal) | 0.0365 (marginalrecent) | 0.0293 (marginal) |

The same Alpha158 features score higher through the factor handler than
through Qlib's Alpha158 handler. That is a preprocessing effect, measured
directly below. On their own, the best mined sets of 13–39 formulas reach
75–92% of the Qlib-handler Alpha158 in 2017–2021, and beat it in 2022–2025,
when Alpha158 is weak. But Alpha158 already carries most of what they
contain.

### Tuned baseline

`run_qlib_baselines.py --robust` gives the Alpha158 and Alpha158+factors
handlers the factor handler's processors: a robust z-score of features fit on
train, zero fill, and a cross-sectionally ranked label. Lift is then measured
against Alpha158 under the same processors (`arm_lift.py --baseline
alpha158r_lgb --handler alpha158r_plus`).

| Baseline | classic RankIC | top-50 (bps/day) | IR | wf pooled RankIC | top-50 | IR |
| --- | --- | --- | --- | --- | --- | --- |
| Qlib default (raw features, z-scored label) | 0.0523 | 13.5 | 1.21 | 0.0281 | 6.2 | 1.00 |
| tuned (robust features, ranked label) | 0.0583 | 11.1 | 0.94 | 0.0398 | 5.1 | 0.72 |

Lift over the tuned baseline:

| Arm | classic lift (t) | campaign sd | classic top-50 (bps, t) | wf pooled lift (t) | wf top-50 (bps, t) |
| --- | --- | --- | --- | --- | --- |
| memory / control | +0.0011 (0.54) | 0.0006 | +0.11 (0.15) | −0.0002 (−0.16) | −0.45 (−0.80) |
| nomemory | +0.0010 (0.42) | 0.0010 | −0.09 (−0.12) | | |
| planner | +0.0003 (0.22) | 0.0016 | +0.56 (0.90) | | |
| marginal | +0.0032 (1.41) | 0.0006 | −0.11 (−0.14) | **+0.0032 (2.03)** | −0.08 (−0.12) |
| marginalcost | +0.0014 (0.89) | 0.0013 | −0.26 (−0.41) | | |
| plannermarginal | +0.0021 (1.20) | 0.0006 | −0.03 (−0.05) | | |
| marginalrecent | | | | +0.0023 (1.66) | −0.46 (−0.70) |
| catalog | +0.0065 (2.28) | – | +2.01 (1.95) | +0.0032 (1.68) | −0.21 (−0.28) |
| alpha158fm | −0.0014 (−1.92) | – | −0.26 (−0.54) | −0.0004 (−0.78) | −0.31 (−0.69) |

Walk-forward marginal lift by window: +0.0038, +0.0027, +0.0071 (t 2.29),
−0.0009. The decay-gated arm is strongest in 2022–23 (+0.0094, t 3.50) and
weakest in 2024–25 (−0.0030), the reverse of its pattern against the default
baseline. Paired differences in walk-forward:

- marginal − control: +0.0034 (t 2.59);
- marginal − alpha158fm: +0.0036 (t 2.34);
- marginalrecent − control: +0.0025 (t 1.97);
- marginal − catalog: +0.0000 (t 0.01).

On classic, marginal − memory/nomemory/planner is +0.0021/+0.0022/+0.0029
(t 1.72/1.52/1.97).

Three readings:

1. Over the stronger baseline, the arm ranking is stable across protocols:
   marginal admission adds about +0.003 RankIC, and the paper rule adds
   nothing. That matches what each rule selects for (partial IC beyond
   Alpha158 versus standalone IC).
2. The baseline's processing changes which rule looks useful. Under Qlib's
   default, the paper rule's libraries add +0.0012 and marginal's +0.0018.
   Under the tuned baseline, the paper rule falls to −0.0002 and marginal
   rises to +0.0032. A plausible reading is that, against raw Alpha158
   features, well-scaled restatements of Alpha158 information pass for lift.
   Once the baseline is scaled well, only information Alpha158 lacks still
   counts, and that is what marginal admission selects for.
3. These are a few comparisons among the 30+ that this study ran
   (two baselines × arms × protocols). Marginal admission's walk-forward t of
   2.03 alone would not survive a Bonferroni correction. Its consistency does
   carry weight: the same sign in both protocols, in 3 of 4 windows and 3 of 3
   campaigns, and significant paired wins over the paper rule and the
   negative control. None of it reaches the top-50 spread.

### Does marginal value persist?

Partial IC beyond Alpha158's 16 principal components, train vs test, oriented
by the train sign:

| Set | Factors | Train partial IC | Test partial IC | Retention | Sign held | Test t > 2 | Composite test partial IC (mean t) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| classic marginal | 87 | 0.0077 | 0.0033 | 42% | 82% | 31% | 0.0024 (0.86) |
| classic plannermarginal | 49 | 0.0086 | 0.0038 | 45% | 84% | 37% | 0.0062 (1.99) |
| classic memory (paper) | 39 | 0.0044 | 0.0018 | 42% | 67% | 18% | 0.0056 (2.18) |
| classic nomemory (paper) | 29 | 0.0090 | 0.0043 | 47% | 86% | 28% | 0.0107 (3.69) |
| classic planner (paper) | 28 | 0.0039 | 0.0017 | 43% | 64% | 14% | 0.0040 (1.45) |
| classic marginalcost | 21 | 0.0099 | 0.0028 | 28% | 71% | 33% | 0.0032 (1.13) |
| classic catalog | 110 | 0.0063 | 0.0027 | 43% | 73% | 22% | 0.0034 (1.35) |
| wf control (paper) | 48 | 0.0038 | 0.0011 | 29% | 67% | 4% | 0.0022 (0.68) |
| wf marginal | 121 | 0.0077 | 0.0021 | 27% | 61% | 17% | 0.0043 (1.17) |
| wf marginalrecent | 99 | 0.0079 | 0.0026 | 33% | 68% | 16% | 0.0060 (1.58) |
| wf catalog | 440 | 0.0051 | 0.0010 | 20% | 57% | 11% | 0.0004 (0.11) |
| wf alpha101 | 48 | 0.0056 | 0.0022 | 39% | 65% | 13% | 0.0007 (0.22) |

Pooled over all 521 mined factors, binned by train partial-IC |t|:

| Train \|t\| | Factors | Train partial IC | Test partial IC | Retention | Sign held | Test t > 2 |
| --- | --- | --- | --- | --- | --- | --- |
| < 3 | 189 | 0.0032 | 0.0008 | 24% | 58% | 8% |
| 3 – 4 | 141 | 0.0071 | 0.0025 | 35% | 75% | 20% |
| 4 – 6 | 151 | 0.0101 | 0.0044 | 43% | 79% | 35% |
| ≥ 6 | 40 | 0.0154 | 0.0043 | 28% | 85% | 35% |

Composite partial IC is the equal-weight rank composite of a set's oriented
factors, with the mean over campaigns or windows of its per-split t. One
classic composite, nomemory, carries clearly significant out-of-sample value
beyond Alpha158 (t 3.69, averaged over three campaigns), yet its LightGBM lift
is +0.0019 (t 0.92). Partial IC against 16 principal components is a
necessary condition for model lift, not a sufficient one: LightGBM also uses
the 142 components that the basis drops, and nonlinear interactions.

The admission gate at t ≥ 3 is well placed: below it, marginal value is mostly
noise. Above it, value persists but shrinks by more than half, and the
strongest train results shrink the most. That pattern is consistent with
selection on noise plus genuine decay. A partial IC of 0.002–0.004 spread over
a library is what the LightGBM lift measures: too small to clear the noise
floor of 8 test years.

### Multiplicity

| Arm (campaigns) | Trials per campaign | Effective trials | BH q < 0.05 | Null-DSR p < 0.05 | Marginal BH q < 0.05 | Test sign held |
| --- | --- | --- | --- | --- | --- | --- |
| classic paper (9) | 410 – 770 | 220 – 660 | 100% | 100% | – | 100% |
| classic marginal (9) | 360 – 1,530 | 260 – 1,370 | 56 – 74% | 14 – 47% | 92 – 100% | 60 – 100% |
| wf control (4) | 1,600 – 1,980 | 1,410 – 1,760 | 100% | 100% | – | 67 – 100% |
| wf marginal (8) | 1,550 – 1,870 | 1,410 – 1,700 | 69 – 83% | 23 – 47% | 94 – 100% | 40 – 83% |

Test sign held refers to standalone test RankIC. The strict DSR, using the
observed cross-trial Sharpe variance, passes no factor in any campaign. Trial
Sharpe ratios spread about 4x wider than sampling noise (sd 0.09–0.10 against
0.024 for T = 1,681). So most trials carry real but similar signal, and the
expected best of about 1,400 draws from that spread (0.30–0.33) is as high as
the best trial itself. That version of the DSR cannot be passed by
construction here. The noise-only version is reported as the informative one.

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

# 3. Qlib reference matrix (Alpha158/Alpha360 x LightGBM/linear, three seeds).
#    run_qlib_baselines.py also takes --protocol wf2018..wf2024 and --robust (tuned baseline).
scripts/qlib_cn/run_reference_matrix.sh $QPY qlib_cn output/qlib_cn

# 4. Mine one campaign (replicates: same config, another output directory)
uv run factorminer -c campaigns/classic_marginal_c1.yaml -o campaigns/classic_marginal_c1 \
  mine --data panels/csi300.csv -n 60

# 5. Score a factor set: direction-frozen metrics, marginal persistence, exported values
uv run python scripts/qlib_cn/evaluate_library.py -c factorminer/configs/qlib_cn_csi300.yaml \
  --data panels/csi300.csv --library campaigns/classic_marginal_c1/factor_library.json \
  --out output/qlib_cn/csi300/classic/classic_marginal_c1 --backend gpu --marginal

# 6. The same LightGBM on those factors, alone and appended to Alpha158 (seeds 0-2)
$QPY -I scripts/qlib_cn/run_qlib_baselines.py --provider qlib_cn --universe csi300 \
  --protocol classic --handler alpha158_plus \
  --factors output/qlib_cn/csi300/classic/classic_marginal_c1/factor_values.csv --seed 0 \
  --out output/qlib_cn/csi300/classic/classic_marginal_c1_alpha158_plus_lgb_s0

# 7. Lift per arm (classic) and pooled over walk-forward windows
$QPY -I scripts/qlib_cn/arm_lift.py --label labels/csi300.parquet --root output/qlib_cn/csi300/classic \
  --arm marginal=classic_marginal_c1,classic_marginal_c2,classic_marginal_c3 --arm catalog=fm_catalog
$QPY -I scripts/qlib_cn/arm_lift.py --label labels/csi300.parquet --campaigns campaigns \
  --window wf2018=output/qlib_cn/csi300/wf2018 --window wf2020=output/qlib_cn/csi300/wf2020 \
  --window wf2022=output/qlib_cn/csi300/wf2022 --window wf2024=output/qlib_cn/csi300/wf2024 \
  --arm marginal={w}_marginal --arm control={w}_control

# 8. Persistence and multiplicity
uv run python scripts/qlib_cn/marginal_persistence.py output/qlib_cn/marginal_oos
uv run python scripts/qlib_cn/audit_campaigns.py --campaigns campaigns/classic_* \
  --scores output/qlib_cn/csi300/classic
```

Market data and factor values stay local. Run outputs record the provider
release, panel SHA-256, splits, model parameters, and seeds.

## Limitations

- Cross-sectional operators (`CsRank`, ...) rank over every exported
  instrument, including non-members on that date. Member ordering is preserved,
  but nonlinear combinations of ranks differ slightly from member-only ranking.
- LLM-driven mining is not seeded end to end. Classic arms run three campaigns
  each. Walk-forward windows run one campaign per arm, so a window's arm
  differences mix arm effects with campaign luck. The classic campaign spread
  (sd 0.0006–0.0019) suggests that luck is of the same size as the effects.
- One LLM (`deepseek-flash`), one universe (CSI300), and daily price-volume
  data. A stronger model, or other data, may behave differently.
- The provider is a community rebuild, not Qlib's original Yahoo-sourced
  dataset. Absolute numbers differ slightly from Qlib's published tables.
