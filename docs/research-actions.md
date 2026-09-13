# Research actions

The opt-in action planner selects experiments in Ralph and Helix, records each
decision before dispatch, and updates estimates from completed outcomes.

## Run a campaign

The DeepSeek profile uses `deepseek-flash` and reads `DEEPSEEK_API_KEY` from the
environment. A local `.env` can supply it:

```bash
uv run --env-file .env factorminer \
  -c factorminer/configs/research_actions_deepseek.yaml \
  -o output/deepseek-research mine --data data/binance_crypto_5m.csv
```

For another dataset, configure targets and training periods in a local YAML.
The CLI action lane takes the training interval and purges its final
`entry_delay_bars + holding_bars` rows, using the maximum across targets.
Array callers supply their own discovery-only panels and feature provenance.

`max_iterations: 0` runs until the library target, planner stop, or interruption.
Local model-call and wall-time quotas do not stop mining. A positive `-n` is a
total campaign iteration boundary. Repeating a command in the same output
directory resumes its checkpoint; a new campaign needs a new directory.
`--resume FILE` imports a library instead of selecting another checkpoint.

## Experiments

| Action | Execution | Evidence |
| --- | --- | --- |
| Generate | LLM proposals with family and memory guidance | Candidate evaluations and admission/rejection outcomes |
| Refine | Smoothing by default; a selected recipe when research skills are enabled | Exact parent/child lineage and evaluated library gain |
| Challenge | Recompute parent and `Delay(parent, delay_bars)` | Paired-support IC, retention, pass/fail/inconclusive |
| Stop | End experiment dispatch | Value estimates and stopping reason |

The default refinement is `Mean(parent, refinement_window)`. An observed parent
can be a previously rejected candidate. [Research skills](research-skills.md)
add smoothing, differencing, and rank/smoothing recipes through `MemoryPolicy`.
All candidate evaluation and admission use the shared engine services.

A delay challenge fixes the parent's direction and compares both signals on
the same finite observations. Fewer than three usable IC periods, or a parent
below the paired-support quality gate, is inconclusive. Challenges are advisory
discovery evidence and do not mutate library admission.

## Decision model

Enable `research.planner.enabled` and choose `research.planner.policy`:

| Policy | Selection |
| --- | --- |
| `decision_value` | Largest positive estimated benefit minus utility costs, with exploration; otherwise stop |
| `heuristic` | Generate using the existing family router |
| `random` | Uniform among active actions |
| `contextual_bandit` | Sample from a mixture constructed from 512 beta-posterior draws |

Generation/refinement estimate admission probability and positive library gain
within empty/growing/stalled contexts. Library utility sums admitted factors'
absolute mean IC above the admission threshold. Empty generations count as
failed proposals; failed or interrupted actions supply no success evidence.

Challenge value estimates the reducible binary decision risk for the specified
discovery delay test. Costs and priors are configurable utility assumptions,
not monetary charges or calibrated estimates of future profitability. The
planner remains opt-in. See [benchmark semantics](reproducibility.md#experiment-selection-comparison).

The ledger records offers, estimates, context, configuration, seed, actual
selection probability, exact formulas, outcome, and elapsed time. Skill-enabled
refinement additionally records the conditional recipe distribution and joint
selection probability.

## Persistence and recovery

`research_actions.sqlite3` is authoritative; `research_actions.jsonl` is an atomic
readable export. Decisions commit before dispatch. Each terminal outcome and
the latest library, signal, memory, and loop snapshot commit together. Helix
includes its additional state and knowledge graph when present.

Resume can reconstruct missing or torn checkpoint files from that snapshot.
Uncommitted in-flight actions become interrupted with an unknown outcome.
Provider calls already in flight cannot be guaranteed exactly once after a
crash; evaluation counts for uncommitted work are a lower bound.

Use one process per campaign directory. Dataset, protocol, planner, or frozen
skill-setting changes require a new campaign. Regular and interrupted exits
retain a final checkpoint even when periodic checkpointing is disabled.
Auxiliary knowledge/custom-operator stores retain their own persistence
contracts. Ordinary mining retains its file-based checkpoint format.
