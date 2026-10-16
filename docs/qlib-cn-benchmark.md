# Qlib CN benchmark

FactorMiner factor sets and Qlib reference models scored on the same China
A-share rows, label, splits and TopkDropout backtest. Question: do LLM-mined
factors (or new data) add out-of-sample value to Alpha158 + LightGBM, after
accounting for the search, and does it reach the portfolio?

## Findings

CSI300, 2008–2025, DeepSeek (`deepseek-flash`) mining: 30+ campaigns, ~36k
evaluated formulas, 500+ model fits.

- **Marginal admission adds RankIC, only over a tuned baseline.** Against
  Alpha158 + LightGBM with robust features and a ranked label, the pooled
  walk-forward lift is +0.0032 (t 2.03). It is positive in 3/4 windows and
  3/3 classic campaigns, and beats the paper rule (+0.0034, t 2.59). Against
  Qlib's default baseline no arm is significant or distinguishable from a
  duplicated-Alpha158 negative control.
- **RankIC gains do not reach the book.** No mined arm, admission gate or
  appended data feature changes the daily top-50 spread (|t| ≤ 1.4) or
  raises pooled after-cost IR. Gates on the top of book (zone partial IC,
  top-lift t) still raise RankIC (+0.001 to +0.005) and still lower the book
  (wf2018+2020 IR 0.81–1.18 vs 1.45; excess t −0.5 to −1.7).
- **Ranking objectives and book tweaks do not pay.** Listwise softmax CE,
  rank_xendcg, top-sextile logloss and top-weighted MSE all lose to
  regression in walk-forward. Score smoothing lowers every book. Lowering
  `n_drop` saves costs one-for-one with the gross alpha it loses.
- **The best book is Qlib's default processing (raw features, z-scored
  label):** pooled IR 1.29 vs 1.00 for the tuned top-stopped model
  (+3.2%/yr, t 1.24). It has the lowest RankIC (0.028 vs 0.042). The z-scored
  label keeps return tails that a rank label flattens.
- **Marginal value decays.** Partial IC beyond Alpha158 keeps 27%
  (walk-forward) to 42% (classic) of its train value. The t ≥ 3 admission
  gate is where persistence starts (sign held 75–85% above it, 58% below).
- **Significance ≠ novelty.** Paper-rule factors pass BH-FDR and a noise-only
  Deflated Sharpe hurdle but carry ~40% less partial IC than marginal ones
  and add no lift.
- **Memory, research planner and cost gate: no measurable effect.** All
  pairwise |t| < 1.5; campaign-to-campaign sd (0.0006–0.0019) matches the
  effect sizes.

- **Training on a 5-day label reaches the book.** With everything else
  Qlib-default and seven seeds per book, `--horizon 5` raises the pooled
  after-cost IR from 0.95 to 1.47 on CSI300, 0.43 to 1.06 on CSI500 and
  −0.16 to 0.79 on CSI1000; pooled over the three universes the paired t is
  3.8 and the IR gain 0.91 (90% CI 0.51–1.32). The 10-day label and a
  validation-chosen horizon are weaker. This is the adopted model; the
  prospective holdout from 2026-10-16 scores it against the 1-day incumbent.
- **Round 3 (2026-10-16): faster loop, mechanism tests.** `fast_fit.py`
  (bit-identical cached fits) and parallel `book_eval.py` make a
  three-universe judgement 2–5 minutes. h5 survives every robustness check
  (size exposure, realistic execution of suspended names, 32 seeds) and
  replicates in direction on never-used 2012–2017 windows, with a smaller
  relative gain. The survival-weighted label, GLS weights, seed counts above
  7, no-trade bands, smoothing, fundamentals and mined libraries (re-exported
  without a look-ahead) do not raise the book. More training data raises
  RankIC everywhere but not the book: the model is far better at losers than
  winners and a top-50 long-only book sees only the winners. Wider books in
  the larger universes are the open lead.
- **Round 4: the signal's shape changed in 2017.** Short-term reversal, the
  main source of the score's top, lost its long side (CSI500 reversal RankIC
  4–8% a year before 2017, ~1% after; recent losers no longer rebound, recent
  winners still lag), consistent with buyers arbitraging the underpriced side
  while short-sale limits protect the overpriced side. A broad exclusion book
  that harvests the loser tail wins in 2018–2025 (size-adjusted ΔIR +0.90
  pooled) and loses decisively in its registered 2012–2017 test (−2.19
  [−2.80, −1.46]); the trailing shape does not predict the next window, so
  the top-50 book stays.
- **Seed sets matter as much as most ideas.** The three-seed default book
  quoted at 1.26–1.29 is the best of the 35 three-seed subsets of seven
  seeds (mean 1.05, sd 0.10). Books are now judged with seven seeds.

Dead ends this round: industry-neutral scores, more trees or lower learning
rates, bagging, refitting on train+validation, fundamentals features, a
holding bonus for incumbents.

## Data

**Provider.** [chenditc/investment_data](https://github.com/chenditc/investment_data)
`qlib_bin.tar.gz` (manifest + SHA-256): split/dividend-adjusted daily bars,
point-in-time CSI300/CSI500 membership. `export_panel.py` exports every
instrument that was ever a member, over the whole window, with `in_universe`
marking membership. Targets are masked outside membership, so metrics score
each day's members (Qlib's dynamic filter) while formulas see full histories.
Supplied `vwap` is kept: in this provider `amount / volume` is 0.1× the true
VWAP.

**Extra leaves** (FactorMiner `data.features` + prompt
`feature_descriptions`; `make_campaigns.py` shorthands):

| Shorthand | Source | Leaves |
| --- | --- | --- |
| `+fund` | BaoStock daily (`fetch_baostock.py`, `merge_panel_features.py`) | `turnover`, `pe_ttm`, `pb`, `ps_ttm`, `pcf_ttm`, `mcap`, `st`, `index_close` |
| `+fq` | `+fund` + Eastmoney quarterly reports (`fetch_em_fundamentals.py`, `data_factors.py --panel-out`) | `roe`, `gp_margin`, `np_margin`, `yoy_ni`, `yoy_rev`, `cfo_to_book`, `ind_return` |

Quarterly figures come from `stock_yjbb_em` plus each report's actual
disclosure date from `stock_yysj_em` (53% of reports). Without an actual date,
the earlier of the latest notice and the statutory deadline is used. A report
is effective from the first trading day after its disclosure and is
forward-filled. `ind_return` is the equal-weight daily return of the stock's
Eastmoney industry over all panel stocks (groups < 5 → NaN; 78% member
coverage). Labels before a stock's first labelled report use its earliest
label, a small classification look-ahead. BaoStock is fetched serially; it
blocks clients that open parallel sessions.

**Fixed data features.** `data_factors.py --factors-out` writes cross-sectional
ranks in [−1, 1] of value (ep, bp, sp, cfp), size, turnover (level, 20-day
mean, jump), 60-day beta, 20-day idiosyncratic volatility, ST, industry
momentum (20d), the industry-relative 5-day return and the quarterly fields.
These are appended to Alpha158 without mining. They set the bar that a mined
use of the data must clear.

## Protocols

| Protocol | Train | Validation | Test |
| --- | --- | --- | --- |
| `classic` (Qlib `examples/benchmarks`) | 2008–2014 | 2015–2016 | 2017-01 – 2020-07 |
| `recent` | 2013–2019 | 2020–2021 | 2022-01 – 2026-09 |
| `wf2018` / `wf2020` / `wf2022` / `wf2024` | 7 years | next 2 | next 2 (2018–19 … 2024–25) |

Label `Ref($close, -2) / Ref($close, -1) - 1` (FactorMiner: `close_to_close`,
`entry_delay_bars: 1`, `holding_bars: 1`). `configs/qlib_cn_csi300.yaml`
disables cross-sectional z-scoring, winsorising and peer-median fill, so
formulas read adjusted price levels as Qlib's handlers do. Mining sees train
only (minus the target horizon). Factor directions are frozen on train.
LightGBM fits on train, early-stops on validation and predicts test.

## Conformance

- Target vs Qlib label (CSI300 2008–2020): agreement within 6e-8 on all
  878,599 shared rows. 5,208 rows (0.6%, suspended at signal date) have a
  Qlib label but no FactorMiner target.
- Alpha158 + LightGBM seed 0: IC 0.0447 / RankIC 0.0484 (published 0.0448 /
  0.0469).
- FactorMiner's Alpha158 transcription vs Qlib (`alpha158_conformance.py`):
  median daily Spearman 0.9999998, 131/158 features > 0.99. The rest are
  IMAX/IMIN/IMXD, where Qlib's `IdxMax` does not skip NaN.

## Missing data

Operator semantics `numpy-expression-v2`: a gap never becomes a number. NaN
denominators give NaN (not 0), as do empty windows and a missing latest
value. EMA/DEMA/KAMA skip gaps rather than going NaN for the rest of the
series. Under v1, `EMA($close, 10)` was NaN on 98.7% of member cells.

## Evaluation backend

`evaluation.backend: gpu` runs compiled formula plans on CUDA
(`operators.torch_executor`) with NumPy's pairwise summation order and true
IEEE division. Results are bit-identical to NumPy except `exp`/`log1p`/`tanh`/
`pow` (last ulp). 178 of 180 formulas are bit-identical on 904 × 4,802, and
the GPU is 29× faster. A step that finds the device full falls back to the
host.

## Marginal-value admission

`admission.policy: marginal` scores a candidate against a basis: Alpha158's 16
leading principal components (77% of variance) plus the admitted library.
Daily member ranks of the candidate and the target are ridge-residualized on
the basis. Gates:

- Newey-West t of mean partial rank IC ≥ 3 and |partial IC| ≥ 0.005;
- residual variance share ≥ 0.2;
- the residual's long-only TopkDropout book (top 1/6, Qlib dropout) has
  positive excess return; `min_net_bps` charges 20 bps round trip;
- optional top-of-book gates:
  - `min_top_t`: partial IC t within the contention zone, the best
    `zone_fraction` (1/3) of members by the basis's pooled score;
  - `min_top_lift_bps` / `min_top_lift_t`: change in the daily top-spread
    when the candidate joins that score;
- optional `recent_periods` / `min_recent_t`: the partial IC must hold over
  the end of the discovery window.

A batch is admitted greedily, each candidate judged against earlier picks.
All statistics go to the trial ledger. `replay_admission.py` re-runs a
campaign's ledger under other gate settings without the LLM.

## Experimental design

Campaigns: 60 iterations × 40 candidates on train. Planner arms take up to 240
research actions.

| Arm | Admission | Other |
| --- | --- | --- |
| `memory` / `control` | paper (IC ≥ 0.02, ICIR ≥ 0.2, corr ≤ 0.5) | memory on |
| `nomemory` | paper | memory off |
| `planner` | paper | research planner |
| `marginal` | marginal | |
| `marginalcost` | marginal, `min_net_bps: 0` | |
| `plannermarginal` | marginal | research planner |
| `marginalrecent` | marginal, `recent_periods: 488` | |
| `fund` / `fq` | marginal | `+fund` / `+fq` leaves |
| `catalog` | none: 110-formula FactorMiner catalog | |
| `alpha158fm` | none: FactorMiner's Alpha158 (negative control) | |

Classic arms run three campaigns each. Walk-forward windows run one per arm,
so the windows are the replicates. Each set is fed to LightGBM alone
(`factors` handler) and appended to Alpha158 (`alpha158_plus`), seeds 0–2.

- **Lift** (`arm_lift.py`): seed-ensembled Alpha158+factors vs seed-ensembled
  Alpha158 under the same processors, per date. RankIC lift and top lift
  (daily top-50 spread: mean label of the 50 best-scored names minus the
  day's mean, bps). An arm is averaged over its campaigns with a Newey-West t
  over dates. Walk-forward lift series are concatenated across windows.
- **Book** (`book_eval.py`): a book is a rank-average of saved predictions.
  It is backtested per window with TopkDropout (50, `n_drop` 5) and Qlib's
  costs, and the daily after-cost excess returns are pooled over 2018–2025
  (1,942 days). Comparison is the paired daily excess difference vs a
  baseline book, with a Newey-West t.
- **Persistence** (`evaluate_library.py --marginal`,
  `marginal_persistence.py`): partial IC beyond Alpha158 on each split,
  oriented by the train sign.
- **Multiplicity** (`audit_campaigns.py`): BH q over every ledger trial,
  effective trial count (average-correlation estimate), and Deflated Sharpe
  with noise-only (`dsr_null`) and observed cross-trial variance (`dsr`).

## Model options

`run_qlib_baselines.py` runs Qlib's benchmark-config models on any handler:

- `--model lgb | linear | dens | xgb | cat | mlp | gru | alstm`: LightGBM,
  OLS, DoubleEnsemble, XGBoost, CatBoost, and Qlib's PyTorch MLP/GRU/ALSTM.
  The sequence models use 20-day windows of Qlib's 20 kept Alpha158 columns
  plus any appended factors.
- `--robust`: train-fit robust z-score with clipping, zero fill, ranked
  label (the tuned baseline).
- `--objective mse | topmse | lambdarank | xendcg | listnet | topclass`
  (LightGBM). Rank objectives use per-day sextile grades and no L1/L2.
  `listnet` is a custom per-day softmax CE against `softmax(label / T)`.
- `--early-stop top`: stop on validation top-50 spread instead of loss.
- `--horizon H`: train on `Ref($close, -H-1) / Ref($close, -1) - 1`. Train
  and validation segments end H−1 trading days early so labels stay
  in-segment. Metrics and backtest remain 1-day.

## Results

### Baselines

| Protocol | Alpha158 LGB RankIC (3-seed ens.) | Top-50 (bps/day) | IR after cost (seed mean) |
| --- | --- | --- | --- |
| classic | 0.0523 | 13.5 | 1.21 |
| wf2018 | 0.0434 | 10.1 | 1.52 |
| wf2020 | 0.0353 | 9.8 | 1.80 |
| wf2022 | 0.0126 | 1.2 | −0.08 |
| wf2024 | 0.0208 | 3.8 | 0.77 |
| wf pooled | 0.0281 | 6.2 | 1.00 |

By 2022–23 the top-50 spread is about a tenth of its 2017–20 level.

### Classic arms vs Qlib's default baseline

| Arm | Library (c1/c2/c3) | RankIC lift (t) | campaign sd | Top-50 lift bps (t) | IR |
| --- | --- | --- | --- | --- | --- |
| memory | 15/11/13 | +0.0023 (1.42) | 0.0016 | +0.21 (0.34) | 1.28 |
| nomemory | 11/8/10 | +0.0019 (0.92) | 0.0019 | −0.19 (−0.26) | 1.15 |
| planner | 8/14/6 | +0.0015 (1.31) | 0.0016 | +0.19 (0.36) | 1.35 |
| marginal | 32/31/24 | +0.0028 (1.41) | 0.0008 | −0.82 (−1.10) | 1.11 |
| marginalcost | 9/7/5 | +0.0009 (0.64) | 0.0006 | −0.80 (−1.39) | 1.07 |
| plannermarginal | 19/15/15 | +0.0020 (1.28) | 0.0014 | −0.45 (−0.78) | 1.29 |
| catalog | 110 | +0.0061 (2.31) | – | +0.50 (0.49) | 0.98 |
| alpha158fm | 158 | −0.0019 (−2.22) | – | −0.20 (−0.35) | 1.29 |

No mined-arm RankIC difference is significant (|t| ≤ 1.50). On top-50 the
paper arms lean ahead of marginal ones (memory − marginal t 1.75): the residual
signal marginal admission selects lives in the body of the cross-section. All
arms beat the duplicated-Alpha158 control (t 1.9–3.1). That control is itself
negative: 158 redundant columns cost LightGBM ~0.002 RankIC.

### Walk-forward vs Qlib's default baseline

| Arm | 2018–19 | 2020–21 | 2022–23 | 2024–25 | Pooled lift (t) | Pooled top-50 bps (t) |
| --- | --- | --- | --- | --- | --- | --- |
| control | +0.0048 | −0.0007 | +0.0005 | 0 (empty) | +0.0012 (1.06) | −0.33 (−0.62) |
| marginal | +0.0036 | +0.0025 | −0.0006 | +0.0017 | +0.0018 (1.21) | −0.00 (−0.00) |
| marginalrecent | −0.0023 | −0.0005 | +0.0014 | +0.0066 | +0.0013 (0.87) | −0.99 (−1.43) |
| catalog | +0.0052 | −0.0015 | −0.0030 | −0.0028 | −0.0005 (−0.29) | +0.23 (0.29) |
| alpha158fm | +0.0001 | +0.0003 | +0.0008 | +0.0033 | +0.0011 (1.41) | −0.27 (−0.49) |

Library sizes by window: control 29/13/6/0, marginal 39/36/33/13,
marginalrecent 31/36/22/10. The paper rule dries up over time; marginal
keeps finding factors. The duplicated-Alpha158 control sets the noise floor at
about ±0.0015 RankIC.

### Factors alone

LightGBM on each set alone, mean RankIC over seeds:

| | classic | wf2018 | wf2020 | wf2022 | wf2024 |
| --- | --- | --- | --- | --- | --- |
| Alpha158 (Qlib handler) | 0.0498 | 0.0413 | 0.0337 | 0.0112 | 0.0178 |
| alpha158fm (factor handler) | 0.0560 | 0.0495 | 0.0372 | 0.0286 | 0.0356 |
| catalog | 0.0511 | 0.0407 | 0.0359 | 0.0288 | 0.0292 |
| best mined arm | 0.0372 | 0.0365 | 0.0311 | 0.0365 | 0.0293 |

The same features score higher through the factor handler: a preprocessing
effect, measured below. The best mined sets (13–39 formulas) reach 75–92% of
Alpha158 in 2017–21 and beat it in 2022–25.

### Tuned baseline

`--robust` gives the Alpha158 handlers the factor handler's processors.

| Baseline | classic RankIC | top-50 | IR | wf RankIC | top-50 | IR |
| --- | --- | --- | --- | --- | --- | --- |
| Qlib default (raw features, z-scored label) | 0.0523 | 13.5 | 1.21 | 0.0281 | 6.2 | 1.00 |
| tuned (robust features, ranked label) | 0.0583 | 11.1 | 0.94 | 0.0398 | 5.1 | 0.72 |

Lift over the tuned baseline:

| Arm | classic lift (t) | campaign sd | classic top-50 (t) | wf lift (t) | wf top-50 (t) |
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

Marginal by window: +0.0038, +0.0027, +0.0071 (t 2.29), −0.0009. Paired
walk-forward differences:

| Pair | Difference (t) |
| --- | --- |
| marginal − control | +0.0034 (2.59) |
| marginal − alpha158fm | +0.0036 (2.34) |
| marginalrecent − control | +0.0025 (1.97) |
| marginal − catalog | +0.0000 (0.01) |

The arm ranking is stable over the tuned baseline and matches what each rule
selects for: partial IC beyond Alpha158 versus standalone IC. Against raw
Alpha158 features, well-scaled restatements of Alpha158 pass for lift. Over
30+ comparisons, a single t of 2.03 would not survive Bonferroni correction.
The evidence is its consistency across protocols, windows and campaigns. None
of it reaches the top-50 spread.

### Book evaluation

**Tied scores.** Qlib's default LightGBM (learning rate 0.2, L1 206,
loss-stopped at 8–64 trees) emits few distinct scores: about 92% of a day's
names share their score with another. `TopkDropoutStrategy` sorts them in
whatever order the index came in, which followed Python's per-process string
hash, so the same predictions backtested to a wf2024 IR anywhere between
0.64 and 0.96 across reruns. `_backtest` now breaks ties with a stable
pseudo-random key (`_untie`), and `book_eval.py --tie-draws K` averages a
book over K keys and reports the IR spread across draws. Every IR in the
table below is a single draw from before the fix; the 16-draw re-evaluation
is in `book_wf_ties.json`. A "holding bonus" for incumbents within ties or
small percentile margins was tried and removed: TopkDropout only ever sells
the `n_drop` lowest-ranked holdings against the five best new names, which
sit far above any incumbent, so a bonus below about 0.5 of the daily range
changes no decision and a larger one is `n_drop` by another name.

Pooled 2018–2025 walk-forward. Baseline: tuned top-stopped LightGBM
(`alpha158r_mtop_lgb`), 3 seeds. Single tie draw (see above).

| Book (3 seeds unless noted) | After-cost IR | Excess/yr | Top-50 bps | RankIC | Excess diff t |
| --- | --- | --- | --- | --- | --- |
| tuned, top-stopped (baseline) | 1.00 | 7.7% | 6.40 | 0.042 | – |
| tuned, loss-stopped | 0.98 | 7.8% | 5.07 | 0.040 | +0.05 |
| Qlib default processors | **1.29** | **10.9%** | 6.24 | 0.028 | +1.24 |
| baseline + default, 50/50 | 1.06 | 8.6% | 6.66 | 0.038 | +0.62 |
| baseline, score EMA (half-life 3) | 0.68 | 5.6% | 4.55 | 0.039 | −1.24 |
| default, score EMA (half-life 3) | 0.72 | 6.1% | 5.38 | 0.021 | −0.69 |
| baseline, 1 seed | 1.05 | 7.9% | 6.12 | 0.040 | +0.16 |
| listnet T=1, 1 seed | 0.57 | 6.2% | 3.83 | −0.003 | −0.46 |
| rank_xendcg, 1 seed | 0.71 | 7.3% | 6.17 | 0.002 | −0.15 |
| top-sextile logloss, 1 seed | 0.53 | 6.2% | 3.85 | −0.013 | −0.37 |
| + fixed data features (valuation/size/liquidity), tuned | 0.98 | 7.6% | 5.15 | 0.046 | −0.10 |
| + fixed data features, default | 1.05 | 9.7% | 6.07 | 0.030 | +0.66 |

- **Seed ensembling** adds nothing: 1 seed IR 1.05, 3 seeds 1.00.
- **`n_drop`** 1/2/3 cuts turnover from 0.20 to 0.04/0.08/0.12. Baseline IR
  is 1.12/0.97/0.99; no paired |t| > 0.6.
- **Top-of-book admission replays** (wf2018 + wf2020, appended to the
  baseline): every gate (`min_top_t` 2/3, `min_top_lift_t` 0/1, both) raises
  RankIC and lowers IR from 1.45 to 0.81–1.18, excess t −0.5 to −1.7. On
  classic the gates lift RankIC (t 1.1–2.5) with top lift |t| < 1.
- **Top-weighted MSE** (`topmse`), best on classic, did not replicate in
  walk-forward: top lift +0.13 bps (t 0.15).
- **16 tie draws** (`book_wf_ties.json`): default processing 1.27 (draws
  1.21–1.36) vs tuned top-stopped 0.96 (0.93–1.00), paired t 1.29, ΔIR 90%
  CI [−0.27, 0.85], reality-check p 0.20. The pre-fix headline stands but is
  not significant. Single default seeds score 0.92/1.13/1.08 with 22–27% of
  names tied; the three-seed rank average (21% tied) reaches 1.27, so for
  this model seed averaging does help, partly by resolving ties with
  information rather than at random. The tuned model's single seed has no
  ties and the same IR as its ensemble (1.02 vs 0.96).
- **Seed count** (`book_wf_seedcount.json`, default 1-day model): seeds
  0–2 give 1.26, seeds 0–4 1.01 (t −2.00 vs the 3-seed book), seeds 0–6
  0.95 (t −2.27), seeds 3–6 alone 1.12. The three-seed incumbent is a
  favourable seed draw. All 35 three-seed subsets of seven seeds
  (`book_wf_seedsets.json`): mean 1.05, sd 0.10, range 0.79–1.27, and
  seeds 0–2 are the best of the 35; single seeds average 0.98 (sd 0.08).
  The expected default book is therefore about 1.05, not 1.27, and every
  three-seed comparison in this table carries a seed-set sd of 0.10 on each
  side.
- **Seven-seed re-judgement of the horizon** (`book_csi300_h7b.json`,
  `book_csi500_h7.json`, `book_csi1000_h7.json`; 4 tie draws). 1-day →
  5-day label: CSI300 0.95 → 1.47 (paired t 1.93, ΔIR 90% CI [0.08, 0.96],
  reality-check p 0.044), CSI500 0.43 → 1.06 (t 1.94, p 0.044), CSI1000
  −0.16 → 0.79 (t 3.87). Day-averaged across the three universes: t 3.8 at
  bandwidths 7, 20 and 60, ΔIR +0.91 (CI [0.51, 1.32]); the two seed halves
  (0–2 and 3–6) agree (+0.92 and +0.80). The adoption survives the seed
  correction; the 5-day book gains about 0.5 IR over a fairly seeded
  incumbent on every universe.
- **Tree count is not a lever under Qlib's parameters.** Fitted to 512
  rounds with no early stopping, the default model (learning rate 0.2, L1
  206) stops by itself when no split clears the penalty: 5 trees on wf2022,
  under 64 on every window, so books from the first 64, 128, 256 or 512
  trees are identical to the early-stopped one and 8 or 16 trees lose
  slightly (`book_wf_trees.json`). Capacity comes from the preset
  (`--lgb-config slow31`/`slow63`: lower rate, no or light L1, 165–300
  trees), which is tested below.
- **Label horizon** (`--horizon`, train on the 3/5/10-day forward return,
  trade daily; `book_wf_horizon.json`, 4 tie draws): pooled IR h1 1.26, h3
  1.07, h5 1.52 (paired t +0.72, reality-check p 0.41), h10 1.29, h5+h10
  blend 1.41. Choosing the horizon on validation IR gives 1.04: validation
  picked h10 for wf2024 (test −0.14) and h3 for wf2018. Not adopted; h5 is
  up in three windows of four and is being replicated on CSI500.
- **Replication of the horizon family on CSI500 and CSI1000** (36 fits
  each, the hypothesis "h5 beats h1" fixed from the CSI300 result before
  fitting). CSI500 (`book_csi500_horizon.json`): h1 0.45, h5 0.90 (t 1.29),
  h10 0.90, h5+h10 1.06 (t 1.66, reality-check p 0.084). CSI1000
  (`book_csi1000_horizon.json`): h1 −0.13, h5 0.93 (t 4.12, ΔIR 90% CI
  [0.64, 1.54], reality-check p < 0.001), h10 0.47 (t 2.10). Pooled over
  the three universes (`cross_universe.py`, day-averaged paired
  differences): h5 vs h1 t 3.0 at bandwidths 7, 20 and 60, ΔIR +0.78 (CI
  [0.35, 1.24]), positive in every universe; h10 t 1.5; validation-selected
  horizon t 1.0. **The 5-day training label is the first change that
  reaches the book**, and it is adopted as a fixed choice (`--horizon 5`,
  default processing, 3 seeds), not as a validation-selected procedure,
  because validation IR picked the wrong horizon in most windows. The gain
  is not a capacity effect in disguise: the 5-day model does grow more trees
  (tie share 0.21 → 0.10), but the `slow31`/`slow63` presets with 165–300
  trees and no ties on the 1-day label gain nothing. Caveats:
  on CSI300 alone t 0.72 and wf2024 loses 0.58 IR; the gain is in 2018–19
  and 2022–23 on CSI300 and across all windows on CSI1000.
- **Review-driven variants, 3 seeds** (`book_wf_review3.json`, vs default
  1.26): refit on train+valid 0.73 (t −2.45, CI excludes 0), raw features
  with rank label 0.77 (t −1.85), robust features with z label 1.28
  (+0.19), top-stopped 0.96, top-stopped refit 0.77, with fundamentals
  0.81 (t −1.93, CI excludes 0), fundamentals top-stopped 1.14. Nothing from
  the review helps the book; refitting and the fundamentals features hurt.
- **LightGBM presets with real capacity** (`book_wf_lgbcfg.json`, 3 seeds,
  4 tie draws, vs default 1.26): `slow31` (rate 0.03, 31 leaves, no L1, ~300
  trees) 1.21, `slow63` 1.20, their blend 1.20, validation-selected preset
  1.21 (reality-check p 0.85); turning Qlib's row bagging on (`qlib_bag`,
  seed 0) 0.86 (t −2.15, CI excludes 0). The five-to-forty-tree default is
  not beaten by a model with ten times the trees.
- **Robust processing under the 5-day label** (`book_wf_h5_robust.json`,
  7 seeds, 4 draws): robust features with a rank label 1.20 vs default
  processing 1.47 (t −1.38); their blend 1.35. Default processing stays:
  the z-scored label's tails matter at 5 days as at 1.
- **XGBoost, 7 seeds** (`book_wf_xgb7.json`, 4 draws): 1-day label 1.09
  (LightGBM 0.95), 5-day label 1.52 (LightGBM 1.47, paired excess t −0.37),
  blend of both 5-day learners 1.48. XGBoost adds nothing over LightGBM, but
  the horizon effect replicates in a second learner (+0.43 IR), which is
  further evidence that the gain belongs to the label, not the model.
- **Reference models, seed 0, 1-day label** (`book_wf_reference_s0.json`,
  4 tie draws, vs LightGBM seed 0 at 0.89): XGBoost 1.22 (t 0.94), CatBoost
  0.94, MLP 0.85 (RankIC 0.041, the highest, yet no better book), ALSTM 0.26
  (t −1.67). Single seeds carry a seed sd near 0.1, so XGBoost is the only
  one worth a seven-seed follow-up; none approaches the 5-day LightGBM book
  (1.47).
- **Industry-neutral scores** (`--neutral`, Eastmoney industry of the latest
  report, groups of five or more names demeaned): pooled IR 1.25 → 0.88 at
  half weight and 0.60 at full weight. The book's sector tilts are part of
  its return, not noise to remove.

Univariate signal of the data features (2017–2025, 1-day label):
- ep, bp, size, cfo_to_book and roe_yoy have RankIC t 4–5 but top-50 spread
  < 1.6 bps.
- The industry-relative 5-day return (reversal) gives −2.5 bps (t −2.4) and
  20-day industry momentum +2.7 bps: these are the more book-relevant
  features.

### Round 3: fast loop and mechanism tests

**Loop.** `fast_fit.py` caches Alpha158 and the training labels once per
universe and protocol start (one cache per start: Qlib's rolling std/corr
round differently by load start, enough to move a LightGBM bin edge) and
fits every window, label and seed in one process with Qlib's own processors.
Refits are bit-identical to `run_qlib_baselines.py` (checked on h1/h5,
raw/robust, three windows; `fast_fit.py fit --check RUN`). A 7-seed, 4-window
family takes ~100 s instead of ~10 min; `book_eval.py --jobs` runs every
(book, tie draw) backtest in forked workers sharing one Exchange (156 s →
35 s); `family.sh` judges books in CSI300/500/1000 and pools them with
`cross_universe.py`. A full three-universe judgement takes 2–5 minutes.

Every test below is the adopted h5 book (7 seeds, default processing) against
a single pre-specified challenger or grid, pooled over the three universes,
4 tie draws (`output/qlib_cn/family/<tag>_*.json`).

- **Survival-weighted label** (`ew90`). A name ranked on day t earns
  Σ_k 1[held at t+k] r_{t+k}; the measured TopkDropout exit hazard is ~0.10
  a day (mean holding 9.9–10.0 days), so the matched target is
  Σ_{k≤20} 0.9^{k−1} r_{t+k}. Pooled t −0.76, ΔIR −0.05 [−0.32, 0.22]: up
  in CSI300 (1.59 vs 1.47), down in CSI1000 (0.62 vs 0.80). The alpha-decay
  profile (`book_diagnostics.py decay`) explains it: the CSI300 book's top-50
  spread is 27 bps over days 1–5 and 25 bps over days 6–20, while CSI1000's
  late lags are negative. The unbiased target adds more return noise than the
  signal it recovers; h5 is the better bias–variance point.
- **Size exposure.** Regressing each book's excess on equal-weight universe
  minus index (`book_diagnostics.py size`): the EW-minus-index return is only
  1.5–2.8%/yr, and h5 − h1 net of size is +4.7%/yr (t 2.16), +4.0%/yr
  (t 1.50) and +8.9%/yr (t 3.29). The label gain is selection, not a small-cap
  tilt.
- **Seed count.** Nested books of 1/3/7/15/32 seeds: pooled IR 1.16, 1.39,
  1.38, 1.35, 1.20 (k32 vs k7 t −1.7). Three disjoint 7-seed sets give
  1.31–1.42 (seed noise ≈ 0.05 pooled). Averaging per-model percentile ranks
  of heavily tied models is the cause: with z-score combining, k32 is 1.31
  vs k7 1.36 (t −0.5). The ensemble saturates by ~7 seeds; rank averaging is
  harmless at 7 but should not be used for large ensembles.
- **Turnover.** `n_drop` 3/4/5/7: pooled 1.31/1.41/1.38/1.18; gross IR rises
  with turnover and costs take it back. n_drop 5 is on the frontier.
- **No-trade band** (`--band`, sell a holding once its rank falls below
  B × 50, refill from the top): pooled 0.19–0.99 for B = 1.5–4, turnover not
  even monotone in B. The scores re-rank sharply day to day; TopkDropout's
  fixed daily count acts as the noise filter a band lacks.
- **Score smoothing** under h5 (EMA half-life 1/2/4 days): pooled ΔIR −0.13,
  −0.25, −0.31; even gross IR falls. The slow decay belongs to the current
  score; older scores carry less.
- **Suspended names.** Qlib's TopkDropout tries to buy top-ranked names that
  are suspended on the trade day and leaves their slots in cash: 2.2% of the
  CSI300 h5 book's daily top 50 and 8.5% of CSI1000's are suspended on the
  signal day (the model ranks NaN-feature rows high), so the books hold 49.2
  and 46.5 names with 2.5% and 8.7% cash. `book_eval.py --tradable`
  (`only_tradable`: buy the next-ranked name, as a trader at the close would)
  lowers IR slightly (CSI300 1.48 → 1.42, CSI1000 0.79 → 0.76); h5 vs h1
  under it: pooled t 3.1, ΔIR +0.83 [0.37, 1.32]. New tests use `--tradable`.
- **Fundamentals and valuation under h5** (CSI300, `csi300_fq3`, 20 PIT
  features appended): 1.41 → 0.96 (t −2.25, ΔIR −0.45 [−0.76, −0.15]),
  lower in every window, the same verdict as under h1.
- **Inverse-variance (GLS) weights** (`fast_fit.py --weight invvar`,
  1/σ²_20d): squared loss on z-scored returns is dominated by volatile names,
  and GLS is the efficient estimator of the same mean. Pooled t −1.0, ΔIR
  −0.10 [−0.48, 0.26]; RankIC lower in every universe. Rejected.

- **Mined libraries under h5, without the export look-ahead.** Every
  library export before 2026-10-15 17:17 had its rows masked by the future
  return (fixed in `evaluate_library.py`, not re-run until now). Re-exported
  (`<window>_{marginal,fq}_v2`; wf2024's control campaign never finished,
  so the control arm is out) and appended to Alpha158 under h5, CSI300:
  marginal 1.33, fq 1.09 vs 1.41 (t −0.33 and −1.68, reality-check p 0.77),
  while RankIC rises from 0.0344 to 0.0375 for both. The old verdict was not
  an artefact of the leak: RankIC lift still does not reach the book.
- **Power.** One universe's 8 years give a 90% interval on ΔIR of about
  ±0.35; pooled over three universes about ±0.25. A RankIC lift of 0.003 is
  worth perhaps +0.05 IR: invisible here. More hill-climbing on these days is
  mostly noise-chasing; more independent days are the lever.
- **Earlier windows: out-of-sample replication of h5** (registered before any
  fit, `docs/registrations/spec_h5_vs_h1_early.json`). CSI300 membership
  starts 2005 and CSI500 2007, so walk-forward windows wf2012/2014/2016
  (tests 2012–2017) were added; no h5-vs-h1 decision had used them. Result
  (`output/qlib_cn/family/early_*`): CSI300 h1 2.83 → h5 3.06 (t 0.65),
  CSI500 5.74 → 6.07 (t 0.81), pooled t 0.91, ΔIR +0.20 [−0.24, 0.65]:
  positive in both universes, so it replicates by the registered rule, but
  the relative gain (+6–8%) is far smaller than in 2018–2025 (+50% and more).
  The early books are extraordinary in every year (CSI500 IR 4.4–7.6, ~40%/yr
  excess, frictionless and without impact costs); every member has rows
  (300/500 a day, delisted names included), so this is not survivorship but
  a less efficient market whose 1-day reversal alpha the 1-day label already
  captured. As that alpha was arbitraged away, what remained sits at longer
  horizons, where the 5-day label is matched.

- **Pooled-universe training** (`fast_fit.py --train-universes
  csi300,csi500,csi1000`: one model on the union of the three indices' rows,
  ~5.5M instead of ~1.1M, label z-scored over the union, scored in each
  universe). Early stopping runs much longer (wf2018 51–81 trees, wf2020
  22–41, vs 2–15 per universe), so the per-universe fits are
  variance-limited. RankIC rises in every universe (0.034 → 0.038,
  0.030 → 0.036, 0.041 → 0.047), yet the book does not: 1.54/0.52/0.98 vs
  1.41/0.91/0.76, pooled t −0.02, ΔIR −0.08 [−0.34, 0.17].
- **Why RankIC lift misses the book: asymmetric predictability.** Mean
  5-day excess return by score bucket (h5, 2018–2025): top 50 +27/+17/+19
  bps (CSI300/500/1000), bottom 20% −33/−29/−40 bps. The model is 1.5–2×
  better at finding losers than winners, and RankIC counts both tails. Pooled
  training's gain in CSI500 sits in the bottom 20% (−3.0 bps) with +0.5 bps in
  the top 50; in CSI1000 it reaches the top (+7.9 bps), the one universe where
  the pooled book gains. In CSI500 and CSI1000 the top 50 is no better than
  the rest of the top 20% (17 vs 18, 19 vs 20 bps): the extreme top of the
  score is saturated.
- **Book size** (`book_eval.py --size f`: topk = f × N, n_drop scaled with
  it). A wider book trades the saturated top for breadth and lower tracking
  error: excess return falls a little and IR rises, so the paired mean t
  stays near zero while the IR interval moves. 2018–2025, IR for top 50 /
  20% / 50% / 80% of N: CSI300 1.41/1.50/1.28/0.49, CSI500
  0.91/1.08/1.26/0.68, CSI1000 0.76/1.34/1.50/1.21; pooled ΔIR of 20% vs 50
  names +0.36 [0.16, 0.56]; net of size exposure CSI1000 20% is IR 1.09
  (t 2.92) vs 0.45. Registered check on the early windows
  (`docs/registrations/spec_book_size_early.json`): CSI500 (50 → 100 names)
  6.07 → 7.47, ΔIR [0.91, 1.98]; CSI300 (50 → 60) 3.06 → 2.85,
  [−0.35, −0.05]; pooled −0.10 [−0.29, 0.10]: inconclusive by the registered
  rule. What both periods agree on: in the larger universes a fixed 50 names
  is too narrow. Not adopted; a 100-name CSI500 book is the candidate for the
  next prospective registration.

### Round 4: the shape of the signal

Book construction is a function of one object, the conditional mean of
forward excess return given the score's cross-sectional rank. `book_diagnostics.py
curve` measures it (5-day excess by score ventile) together with
frictionless daily-rebalanced books and their break-even costs;
`reversal` and `size` (now with a paired IR-difference interval of the
size residuals, per universe and pooled) explain and adjust it.

- **2018–2025: the top saturated, the bottom carries the signal.** 5-day
  excess by ventile, top three (18, 19, 20) / bottom: CSI300 +25.5/+23.5/+34.0
  / −44; CSI500 +21.8/+21.6/+12.7 / −50; CSI1000 +20.2/+23.4/+18.2 / −87.
  Frictionless, a book that holds the universe minus its bottom 10–20% has IR
  2.4–3.6 at 5–10% daily turnover (break-even 11–19 bps one-way), against
  1.2–2.1 for the top 50 at 37–47% (break-even 4.5–10 bps).
- **Exclusion book, in sample.** `--size 0.8 --band 1.1` (hold 80% of N,
  sell a name only when it falls below rank 0.88 N, refill from the top)
  raises the size-adjusted IR over the top-50 book by +0.19/+0.55/+1.49,
  pooled +0.90 [+0.19, +1.63], with lower return (3–5%/yr vs 6–12%) at much
  lower tracking error. Bands at small books (band 1.1–1.2 at 50 names)
  multiply turnover and lose 1–1.6 IR.
- **Registered early-window test: refuted**
  (`docs/registrations/spec_exclusion_book_early.json`, committed before
  running). 2012–2017, `--tradable`: size-adjusted IR difference CSI300
  −1.96 [−2.58, −1.21], CSI500 −1.82 [−2.90, −0.68], pooled −2.19
  [−2.80, −1.46]. The top-50 book earned 28%/yr (CSI300) and 39%/yr
  (CSI500) there; concentration wins. A 50/50 split of capital between the
  two books does not hedge the regime either: it loses 0.25–0.7 pooled IR in
  2012–2017 for +0.12–0.27 in 2018–2025.
- **Why: the shape changed.** In 2012–2017 the curve is convex at the top
  (top ventile +77 bps CSI300, +93 bps CSI500, the largest of all; frictionless
  top-50 IR 5.2/8.0). Across 17 window × universe cells the top's convexity
  (top ventile minus ventiles 17–19) sets which construction wins in the same
  window (Spearman −0.85 with the exclusion book's IR gain), but the previous
  window's convexity barely predicts the next (−0.42, n = 14): construction
  cannot be timed from the trailing shape.
- **Mechanism: short-term reversal lost its long side in 2017.** The top
  ventile is mostly recent losers (past-5-day return percentile 0.32–0.44).
  Reversal RankIC in CSI500 was 3.8–8.5 (×100) every year 2010–2016 and
  −1.3 to 2.0 in 2017–2024; the decile of biggest 5-day losers went from
  +12 bps over the next 5 days (mean of 2010–2016) to −10 (2017–2024),
  while the biggest winners still lag (−7 to −33 bps in six of those eight
  years). CSI300 breaks the same way (4–6 → about 1); CSI1000, whose
  history starts in 2014, falls from 4–8 to 2–3. What remains of reversal is the overpriced side, which
  needs shorting, and A-shares restrict it: the pattern Miller (1977) predicts
  when short sales are constrained and buyers compete away the underpriced
  side. It is why the model now finds losers 1.5–2× better than winners, why
  RankIC gains (which count both tails) miss a long-only book, and why
  2012–2017 books are so much stronger.
- **Volatility is not the reason.** Within the top 10% by score, the
  higher-volatility half earns as much or more (CSI500 20.5 vs 14.9 bps,
  CSI1000 25.0 vs 20.3); a volatility filter would not lift the top.

- **Recency-weighted training does not change the shape.** `fast_fit.py
  --weight decay2` / `decay4` (sample weight 0.5^(age / 2 or 4 years)):
  pooled ΔIR +0.01 [−0.33, 0.33] and +0.02 [−0.18, 0.22] (`--tradable`), and
  the CSI500 top still saturates (ventiles 16–20: +18/+16/+16/+18/+15 bps).
  The saturated top is in the post-2017 data, not in a model fitted to the old
  regime.

- **Price-volume anomalies are short-side only.** `book_diagnostics.py
  screen` over 13 classic features (overnight and intraday 20-day returns,
  MAX, skewness, abnormal amount and volume, 12-1 momentum, distance to the
  240-day high, close/VWAP, Amihud illiquidity, 5- and 20-day returns,
  volatility), 2018–2025. The significant deciles are all losers: in CSI1000
  the top deciles of intraday run-up, MAX, abnormal amount, volume surge,
  close/VWAP, 5- and 20-day return and volatility lag by 32–52 bps over five
  days (t −4.0 to −5.5); in CSI500 intraday run-up (−24, t −2.3). Only 12-1
  momentum and the 240-day-high ratio lean positive on the long side
  (CSI300 +15 and +20 bps, t 1.2 and 1.8). Within the score's top 20%,
  where the book buys, no feature splits returns significantly (largest
  |t| 2.19 of 39 tests, CSI500 5-day return, in the momentum direction).

Consequence for the search: improvements must find *winners* in a market
where the cheapest winner signal has been arbitraged. Tail-symmetric metrics
(RankIC, IC) and losses (MSE on z-scored labels) spend capacity on the short
side. Not adopted: the exclusion book, sleeve blends, adaptive construction,
recency weights. Price and volume alone have little left to say about
winners; the open routes are information from outside them (earnings and
analyst revisions, holdings and flows) or a benchmark-relative book that can
underweight index names, which needs index weights (free-float shares) the
provider does not have.

### Does marginal value persist?

Partial IC beyond Alpha158's 16 PCs, oriented by the train sign:

| Set | Factors | Train | Test | Retention | Sign held | Test t > 2 | Composite test (mean t) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| classic marginal | 87 | 0.0077 | 0.0033 | 42% | 82% | 31% | 0.0024 (0.86) |
| classic plannermarginal | 49 | 0.0086 | 0.0038 | 45% | 84% | 37% | 0.0062 (1.99) |
| classic memory | 39 | 0.0044 | 0.0018 | 42% | 67% | 18% | 0.0056 (2.18) |
| classic nomemory | 29 | 0.0090 | 0.0043 | 47% | 86% | 28% | 0.0107 (3.69) |
| classic planner | 28 | 0.0039 | 0.0017 | 43% | 64% | 14% | 0.0040 (1.45) |
| classic marginalcost | 21 | 0.0099 | 0.0028 | 28% | 71% | 33% | 0.0032 (1.13) |
| classic catalog | 110 | 0.0063 | 0.0027 | 43% | 73% | 22% | 0.0034 (1.35) |
| wf control | 48 | 0.0038 | 0.0011 | 29% | 67% | 4% | 0.0022 (0.68) |
| wf marginal | 121 | 0.0077 | 0.0021 | 27% | 61% | 17% | 0.0043 (1.17) |
| wf marginalrecent | 99 | 0.0079 | 0.0026 | 33% | 68% | 16% | 0.0060 (1.58) |
| wf catalog | 440 | 0.0051 | 0.0010 | 20% | 57% | 11% | 0.0004 (0.11) |
| wf alpha101 | 48 | 0.0056 | 0.0022 | 39% | 65% | 13% | 0.0007 (0.22) |

Pooled over 521 mined factors, by train |t|:

| Train \|t\| | Factors | Train | Test | Retention | Sign held | Test t > 2 |
| --- | --- | --- | --- | --- | --- | --- |
| < 3 | 189 | 0.0032 | 0.0008 | 24% | 58% | 8% |
| 3–4 | 141 | 0.0071 | 0.0025 | 35% | 75% | 20% |
| 4–6 | 151 | 0.0101 | 0.0044 | 43% | 79% | 35% |
| ≥ 6 | 40 | 0.0154 | 0.0043 | 28% | 85% | 35% |

Composite = equal-weight rank composite of a set's oriented factors. The
classic nomemory composite has significant test value beyond Alpha158 (t
3.69) yet LightGBM lift of +0.0019 (t 0.92). Partial IC against 16 PCs is
necessary for model lift, not sufficient. The pattern (persistence above
t 3, strongest train values shrinking most) is consistent with selection on
noise plus genuine decay.

### Multiplicity

| Arm (campaigns) | Trials / campaign | Effective trials | BH q < 0.05 | Null-DSR p < 0.05 | Marginal BH q < 0.05 | Test sign held |
| --- | --- | --- | --- | --- | --- | --- |
| classic paper (9) | 410–770 | 220–660 | 100% | 100% | – | 100% |
| classic marginal (9) | 360–1,530 | 260–1,370 | 56–74% | 14–47% | 92–100% | 60–100% |
| wf control (4) | 1,600–1,980 | 1,410–1,760 | 100% | 100% | – | 67–100% |
| wf marginal (8) | 1,550–1,870 | 1,410–1,700 | 69–83% | 23–47% | 94–100% | 40–83% |

The strict DSR (observed cross-trial variance) passes nothing. Trial Sharpe
ratios spread ~4× wider than sampling noise, so the expected maximum of
~1,400 draws (0.30–0.33) equals the best trial. The noise-only DSR is the
informative one.

## Running

```bash
# Provider and isolated Qlib environment
gh release download <tag> -R chenditc/investment_data -p 'qlib_bin*'
uv venv --python 3.12 qlib-venv && VIRTUAL_ENV=qlib-venv uv pip install pyqlib==0.9.7 lightgbm==4.6.0 pyarrow
QPY=qlib-venv/bin/python

# Panel and raw labels
$QPY -I scripts/qlib_cn/export_panel.py --provider qlib_cn --universe csi300 \
  --start 2007-01-01 --end 2026-10-12 --out panels/csi300.parquet --source-manifest qlib_bin.manifest.json
$QPY -I -c "import pandas as pd; pd.read_parquet('panels/csi300.parquet').to_csv('panels/csi300.csv', index=False)"
$QPY -I -c "import qlib; from qlib.data import D; qlib.init(provider_uri='qlib_cn', region='cn'); \
  df = D.features(D.instruments('csi300'), ['Ref(\$close, -2)/Ref(\$close, -1) - 1'], start_time='2008-01-01'); \
  df.columns = ['label']; df.reset_index().to_parquet('labels/csi300.parquet')"

# Extra data (separate env with baostock/akshare)
python -I scripts/qlib_cn/fetch_baostock.py --panel panels/csi300.parquet --out baostock --workers 1
python -I scripts/qlib_cn/merge_panel_features.py --panel panels/csi300.parquet --baostock baostock \
  --out panels/csi300_fund
python -I scripts/qlib_cn/fetch_em_fundamentals.py --out eastmoney --start 2006 --end 2026
$QPY -I scripts/qlib_cn/data_factors.py --panel panels/csi300_fund.parquet --fundamentals eastmoney \
  --panel-out panels/csi300_fq --factors-out data_factors/csi300_fq.parquet

# Reference matrix; any handler/model/objective/horizon via run_qlib_baselines.py
QPY=$QPY PROVIDER=qlib_cn UNIVERSE=csi300 scripts/qlib_cn/fit_queue.sh scripts/qlib_cn/jobs/reference.txt

# Campaign configs (arm = JSON override of the marginal template) and mining
uv run python scripts/qlib_cn/make_campaigns.py --campaigns campaigns \
  --arm fq='{"data": {"features": "+fq"}}' --protocol classic:2 --protocol wf2018 --protocol wf2024
uv run factorminer -c campaigns/wf2018_fq.yaml -o campaigns/wf2018_fq mine --data panels/csi300_fq.csv -n 60

# Score a library, then LightGBM on Alpha158 + factors
uv run python scripts/qlib_cn/evaluate_library.py -c factorminer/configs/qlib_cn_csi300.yaml \
  --data panels/csi300.csv --library campaigns/classic_marginal_c1/factor_library.json \
  --out output/qlib_cn/csi300/classic/classic_marginal_c1 --backend gpu --marginal
$QPY -I scripts/qlib_cn/run_qlib_baselines.py --provider qlib_cn --universe csi300 --protocol classic \
  --handler alpha158_plus --factors output/qlib_cn/csi300/classic/classic_marginal_c1/factor_values.csv \
  --seed 0 --out output/qlib_cn/csi300/classic/classic_marginal_c1_alpha158_plus_lgb_s0

# Admission replay under other gates
uv run python scripts/qlib_cn/replay_admission.py -c campaigns/classic_marginal_c1.yaml --data panels/csi300.csv \
  --campaign campaigns/classic_marginal_c1 --rule 'zone2={"min_top_t": 2.0}' --out replay/classic_marginal_c1

# Lift, book, persistence, multiplicity
$QPY -I scripts/qlib_cn/arm_lift.py --label labels/csi300.parquet --root output/qlib_cn/csi300/classic \
  --arm marginal=classic_marginal_c1,classic_marginal_c2,classic_marginal_c3
$QPY -I scripts/qlib_cn/book_eval.py --provider qlib_cn --universe csi300 --label labels/csi300.parquet \
  --window wf2018=output/qlib_cn/csi300/wf2018 --window wf2020=output/qlib_cn/csi300/wf2020 \
  --book base='alpha158r_mtop_lgb_s?' --book q='alpha158_lgb_s?' --baseline base --out book.json
uv run python scripts/qlib_cn/marginal_persistence.py output/qlib_cn/marginal_oos
uv run python scripts/qlib_cn/audit_campaigns.py --campaigns campaigns/classic_* --scores output/qlib_cn/csi300/classic
```

Market data and factor values stay local. Outputs record the provider
release, panel SHA-256, splits, labels, model parameters and seeds.

## Limitations

- `CsRank` and the other cross-sectional operators rank over all exported
  instruments, non-members included. Member order is preserved; nonlinear rank
  combinations differ slightly.
  `CsNeutralize` demeans only; there is no industry neutralization.
- LLM mining is not seeded end to end. Walk-forward runs one campaign per arm
  per window, and campaign luck (sd 0.0006–0.0019) is the size of the
  effects.
- One LLM, one universe (CSI300), daily data. Fundamentals have actual
  disclosure dates for 53% of reports; the rest use a conservative deadline
  fallback.
- The provider is a community rebuild of Qlib's dataset; absolute numbers
  differ slightly from Qlib's published tables.
