# Research actions

FactorMiner can select what experiment to run next, record why it selected it,
and learn from the result. This first implementation is opt-in and supports
both Ralph and Helix. It provides an inspectable experiment-selection system;
the measured evidence does not establish superior general search performance.

## Running a campaign

The ready-to-run profile is `factorminer/configs/research_actions_deepseek.yaml`.
It uses the official DeepSeek endpoint and the `deepseek-flash` model. Credentials
come from `DEEPSEEK_API_KEY`; `uv run --env-file .env` can load a local file without
putting its contents in the configuration or research artifacts.

```bash
uv run --env-file .env factorminer \
  -c factorminer/configs/research_actions_deepseek.yaml \
  -o output/deepseek-research mine --data data/binance_crypto_5m.csv
```

The bundled panel is a short workflow sample, not the full paper dataset. Keep
the admission thresholds when testing; zero admissions is a meaningful outcome.
The CLI action lane slices the configured training interval and removes its
last `entry_delay_bars + holding_bars` rows (the maximum over targets). Analysis
of the test interval happens separately. Array callers must supply discovery
data themselves. The planner does not establish point-in-time provenance for
external features or the model's training data.

Local runs default to `max_iterations: 0`: continue until the library target,
a planner stop, or interruption. Existing LLM-call and wall-time exhaustion
flags no longer stop local mining. An explicit positive `-n` is still supported
as a total iteration boundary for an experiment. Same-directory runs resume
automatically; a fresh campaign needs a fresh output directory. `--resume FILE`
imports a library instead of automatically restoring another checkpoint.

## Available experiments

| Action | Execution | Recorded result |
| --- | --- | --- |
| Generate | Existing LLM generator with family and memory guidance | All candidate evaluations, admission and rejection outcomes |
| Refine | `Mean(parent, refinement_window)` for an observed parent | Exact parent/child lineage and centrally evaluated library gain |
| Challenge | Recompute parent and `Delay(parent, delay_bars)` | Paired-support IC, retained direction, retention ratio, pass/fail/inconclusive |
| Stop | No generation or evaluation | Estimates and reason for stopping |

Refinement is currently one deterministic smoothing edit. Its parent can be a
previously rejected candidate. A challenge is an advisory stress check on the
supplied discovery panel: it neither independently confirms a factor nor alters
the factor library. The original direction is fixed for the delayed comparison,
and both series use the same finite observations. Fewer than three usable IC
periods or a parent that fails the paired-support quality gate is inconclusive.

## Decision model

`research.planner.policy` accepts `decision_value`, `heuristic`, `random`, or
`contextual_bandit`. The default configuration leaves the entire planner disabled.

Generation and refinement use beta admission estimates and observed positive
library gain. Library utility is the sum of admitted factors' absolute mean IC
above the admission threshold; it is not a portfolio objective. Estimates are
conditioned on a coarse empty/growing/stalled context. Empty generations supply
failed proposals; failed and interrupted actions supply no success evidence.

Challenge benefit estimates the reducible binary decision risk for the measured
discovery-panel delay-retention predicate. It assumes that this particular
measurement resolves that predicate. It is not a model of future profit or a
general estimate of the value of independent validation.

Decision value selects the largest positive estimated benefit minus declared
utility costs, with randomized exploration. If all active actions have
nonpositive value, it selects stop. The costs are utility proxies, not dollar
prices or quotas. The initial priors and costs are configurable assumptions;
they have not been calibrated across independent markets.

The heuristic always generates and retains the existing family router. Random
selects uniformly among active actions. The contextual bandit samples beta
posteriors 512 times to construct a finite mixture, then samples from that
mixture. Its logged probability is the probability actually used for dispatch.
The ledger includes all offers, estimates, context, configuration, randomness
seed, chosen probability, exact formulas, outcome, and elapsed time.

## Persistence and recovery

`research_actions.sqlite3` is authoritative; `research_actions.jsonl` is an atomic
readable export. Each decision is committed before dispatch. A terminal outcome
and the latest canonical library, signal, memory, and loop-state snapshot commit
together. Resume can reconstruct missing or torn regular checkpoint files from
this snapshot, preserving completed work without making model calls again.
Helix snapshots also include its state and knowledge graph when present.

An action interrupted before that commit has an unknown outcome and is marked
interrupted. External provider requests already in flight cannot be guaranteed
exactly once after a process crash. The trial ledger retains data-contact
records, but evaluation accounting for uncommitted work is a lower bound.
Auxiliary stores such as research knowledge and custom operators retain their
existing persistence contracts; the action snapshot does not make every output
artifact one distributed transaction. Use one process per campaign directory.

Dataset, protocol, and planner changes require a separate campaign directory.
Regular and interrupted exits retain a final checkpoint even when periodic
checkpointing is disabled. SQLite recovery snapshots apply to the opt-in action
lane; ordinary mining retains its existing file checkpoint format.

## Initial comparison, 12 September 2026

Ten paired search seeds, four policies, and 32 candidate evaluations per episode
were run in each of four fixed diagnostic worlds. All policies used the same
candidate catalog and data. Controlled admission used IC 0.04, ICIR 0.01, and
correlation 0.9; the live sample used IC 0.04, ICIR 0.05, and correlation 0.9. These are diagnostic panels, not tradable prices.

| World | Decision value | Heuristic | Random | Contextual bandit |
| --- | ---: | ---: | ---: | ---: |
| Persistent signal | 2.5452 | 2.8191 | 2.3013 | 2.5246 |
| Independent noise | -0.0226 | 0.0000 | -0.0065 | -0.0098 |
| Transient signal | -0.1208 | -0.2548 | -0.1194 | -0.1010 |
| Unannounced future break | -0.3219 | -0.3543 | -0.2927 | -0.3196 |

Values are mean held-out delay utility; higher is better for this declared
objective. On persistent signal, decision value lost to the heuristic by 0.2739
(paired bootstrap interval: -0.4292 to -0.1186). On transient signal it improved
delay utility by 0.1340 (0.0938 to 0.1750), while reducing unstressed utility from
2.6006 to 0.9857. Under independent noise it admitted false positives and lost
to the heuristic. Differences against random and contextual bandit were not
resolved by these intervals. Improvements on the future-break panel reflected
fewer claims exposed to failure, not anticipation of that break.

The live Flash integration generated 30 valid formulas in five calls during
its first 16 actions (five generate, eleven refine; 41 evaluations), using
37,613 reported tokens. No formula passed the configured admission gates on
the bundled crypto sample. The frozen catalog comparison over ten further
paired search seeds produced zero admissions for every policy: it establishes
execution parity, not market advantage. Production ceilings were disabled;
the initial 8- and 16-action boundaries specifically exercised automatic resume.

An uncapped continuation used the shipping DeepSeek provider and stopped by its
own decision at action 46: nine generate, 36 refine, and one stop. Across the
whole campaign it recorded 90 evaluations and nine successful Flash calls
(64,661 reported tokens). Both earlier history prefixes remained unchanged;
all formulas still failed admission. This validates quota-free continuation
and an evidence-driven stop on this sample, without establishing alpha quality.

The implementation therefore keeps decision value opt-in. The next policy
improvement needs independent calibration of null false admissions and the
value of refinement, a broader edit/challenge menu, and testing across distinct
datasets. Reusing the current final panels to tune parameters would require a
new untouched evaluation set before making a stronger claim.
