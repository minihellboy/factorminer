# Qlib CN selection and confirmation protocol

How a change to the CSI300 top-50 book is chosen, judged and confirmed. Results
are in [qlib-cn-benchmark.md](qlib-cn-benchmark.md).

## Status of the data

- **2008–2025 is development data.** The walk-forward tests (wf2018–wf2024,
  test 2018–2025) have been used to choose objectives, processors, features,
  admission rules and book variants. A test result there is evidence for
  ranking ideas, not an unbiased estimate of live performance.
- **2026-10-16 onward is the prospective holdout** (protocol `live`). No
  decision may use those days until a procedure is registered for them.

## Power

Paired daily after-cost excess differences between two of our books have a
standard deviation of about 46 bps (long-run 52 bps), close to a single
book's 55 bps: different top-50 books share little noise. At 80% power and
5% two-sided:

| Test days | Detectable annual excess difference | Detectable IR (book vs zero) |
|---:|---:|---:|
| 240 (1 year) | 22 pp | 2.8 |
| 480 | 16 pp | 2.0 |
| 1,942 (wf 2018–2025) | 8 pp | 1.0 |

Most variants differ by 1–4 pp a year, so no single comparison is decisive.
Prospective data can detect a broken procedure, not rank close variants.

## Selecting on development data

1. **Register the family before running it.** A family is a jobs file under
   `scripts/qlib_cn/jobs/` and a `book_eval.py` command. Every evaluated book
   is appended to `output/qlib_cn/registry.jsonl`, so the number of books
   tried is recorded.
2. **Choose on validation, report test.** `book_eval.py --select
   proc=a,b,c` picks per window the book with the best validation IR, using
   each run's `valid_predictions.parquet` from the model fitted on train
   only. The procedure book's test result is the estimate for "choose among
   a, b, c". The best fixed book's test result is not.
3. **Simulate continuously.** `--sim continuous` (default) carries
   holdings across windows. Quote this, not per-window resets.
4. **Correct for the search.** `reality_check.p_value` is the max-statistic
   block-bootstrap p of the best book over the baseline across all books of
   the run. Quote it next to the best book's t.

## Adoption rule (development data)

A challenger replaces the incumbent only if all hold:

- its procedure book beats the incumbent's on pooled continuous test after
  costs, with paired NW t ≥ 2 at bandwidths 7, 20 and 60 days, and the
  ΔIR 90% block-bootstrap interval excludes 0;
- the reality-check p over the run's books is < 0.10;
- it does not lose more than 0.3 IR in more than one of the four windows,
  and its turnover is no more than 20% above the incumbent's;
- the gain does not depend on fewer than 10 days (drop the 10 best paired
  days; the mean difference stays positive).

A change that fails stays out. The response is never a search over more
thresholds on the same windows.

**Replication universes.** CSI500 and CSI1000 (labels, protocols and books
as for CSI300) are confirmation data for a hypothesis fixed on CSI300
before their fits are made. `cross_universe.py` averages the paired daily
differences across universes day by day and reports the pooled NW t, so a
change that is too small to show on one universe can be judged on three.
The adoption rule then applies to the pooled series, and a validation-
selected procedure is only required when validation actually picks the
right book (for the label horizon it did not, so the fixed choice was
registered).

## Registered procedures

| Registration | Frozen | Books | Decision |
| --- | --- | --- | --- |
| `docs/registrations/csi300_h5_vs_h1.json` | 2026-10-15, provider to 2026-10-12, 6 boosters | `h5` (5-day training label) vs `h1` (incumbent), Alpha158 default processing, 3 seeds (its globs `s?` now also match seeds 3–6; score the registered runs only, `s[012]`) | 2027-10-15: paired NW(20) t > 0 confirms, < −2 refutes |
| `docs/registrations/csi300_h5_vs_h1_7seeds.json` | 2026-10-15, provider to 2026-10-12, 14 boosters | same books with seeds 0–6 (primary prospective test; the 3-seed pair is reported alongside) | same rule |

## Prospective holdout

1. **Fit.** Fit the procedure's runs under `--protocol live`: train
   2017-10-01 to 2024-09-30, validate 2024-10-01 to 2026-09-30, refit if
   the procedure refits. Fits go to `output/qlib_cn/csi300/live/`. Each
   saves `model.txt`; the empty test segment saves no predictions.
2. **Freeze.** Run `register.py freeze --spec <spec.json> --runs
   output/qlib_cn/csi300/live --out docs/registrations/<name>.json`, then
   commit and push it. The spec names the books, the baseline and the
   rule. The record keeps the git commit, the provider's last date and the
   SHA-256 of every booster. A registration is never edited or replaced.
3. **Score.** `scripts/qlib_cn/live_score.sh docs/registrations/<name>.json`
   (optionally `TAG=<release>`) does all of the following against a separate
   copy of the release under `~/fm-data/live/<tag>/`, leaving the research
   provider untouched, and writes `output/qlib_cn/live/<name>_<tag>.json`
   with the release tag, last date and calendar hash. By hand, after each
   provider update:
   - `register.py verify`;
   - `run_qlib_baselines.py --protocol live --load-model <run>/model.txt
     --out output/qlib_cn/csi300/live_pred/<run>` for every frozen run;
   - `book_eval.py --window live=output/qlib_cn/csi300/live_pred`.
4. **Decide at 2027-10-15 (about 240 sessions).**
   - **Pass:** the realised IR is above the development IR minus two
     sampling SEs (about 2.4 IR at 240 days), and paired turnover and cost
     match the development distribution.
   - **Fail:** an implementation or data fault is assumed and investigated
     before anything else.
   - Comparisons between registered books are reported but not acted on
     before about 1,000 sessions.

Provider rebuilds can revise history. The provider's last date and a hash of
`calendars/day.txt` at each scoring are logged with the results.
