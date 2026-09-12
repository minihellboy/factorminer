# Transferable research skills

FactorMiner can compile completed research actions into a versioned collection
of empirical edit and testing procedures, then use that frozen evidence in a
new campaign. This is an opt-in extension of the [action planner](research-actions.md).
It operates through `MemoryPolicy` in both Ralph and Helix; evaluation and
admission still use the central services.

## Collect and freeze source experience

Copy `factorminer/configs/research_actions_deepseek.yaml` to a local config.
Retain its data, split, evaluation, and DeepSeek settings, and replace its
`research` section with:

```yaml
research:
  planner:
    enabled: true
    policy: decision_value
    seed: 42
  skills:
    enabled: true
    mode: none
    source: ""
```

Run each source campaign with its own data configuration and output directory:

```bash
uv run --env-file .env factorminer -c factorminer.local.yaml \
  -o output/source-a mine --data path/to/source-a.csv
```

Here `mode: none` chooses uniformly among available edit recipes and records
their outcomes. The configured ordinary memory policy continues to operate.
The skill extension has no iteration, model-call, or wall-time quota. Existing
automatic resume and planner stopping behavior apply.

Once source actions have finished, compile a snapshot:

```bash
uv run factorminer research-skills compile \
  output/source-a output/source-b output/source-c \
  --destination output/skills/v1.json
uv run factorminer research-skills inspect output/skills/v1.json
```

Inputs can be campaign directories or `research_actions.sqlite3` files. The
compiler opens read transactions, refuses snapshots with an in-flight action,
deduplicates repeated campaigns, and rejects conflicting snapshots of the same
campaign. A source may continue later; compiling that history produces a new
version. An existing destination may only be reused for identical content.

The pack contains executable recipe identifiers, exact parent/child formulas,
action and dataset identities, record hashes, logged selection probabilities,
positive and negative observations, inconclusive outcomes, applicability
conditions, and descriptive effect intervals. Legacy histories without measured
applicability are counted as skipped. Failed or interrupted actions do not
supply a measured effect. Content hashes detect modification; they do not
authenticate a source or establish independence between its datasets.

## Transfer into a new campaign

Use a new output directory and set:

```yaml
research:
  planner:
    enabled: true
    policy: decision_value
    seed: 42
  skills:
    enabled: true
    mode: structured
    source: output/skills/v1.json
    minimum_datasets: 3
    minimum_effect: 0.002
    confidence_level: 0.90
    exploration_probability: 0.10
    temperature: 0.02
    recipes: [smooth_3, smooth_8, difference_1, rank_smooth_3]
```

`research.skills.enabled` requires the action planner. The source pack is loaded
once, copied to `research_skill_pack.json` in the campaign directory, and pinned
by content hash and retrieval settings in the campaign identity. The run
manifest includes that identity and artifact path. Resume retains the copy if
the original source moves; replacing the original with a different version is
rejected. Change the source version or retrieval settings in a new campaign.

## Executable procedures and applicability

| Recipe | Exact child for parent `P` |
| --- | --- |
| `smooth_3` | `Mean(P, 3)` |
| `smooth_8` | `Mean(P, 8)` |
| `difference_1` | `Delta(P, 1)` |
| `rank_smooth_3` | `Mean(CsRank(P), 3)` |
| `delay_1` | `Delay(P, 1)`; advisory challenge only |

These are fixed, parser-checked, single-step recipes. They are not generated
Python or learned multistep programs. Refinement chooses among unseen valid
children of the planner's selected parent. The delay recipe applies only when
the planner's configured delay is one bar.

Applicability requires matching market/frequency, target definitions and timing,
default target, signal failure policy, and delay-test thresholds. It also uses
the parent's discovery-quality band, lag-one signal persistence, and variation
in cross-sectional signal scale. The latter two descriptors use already
evaluated signals without accessing targets. Quality bands do use discovery
targets. Formula family is deliberately omitted so behavior can transfer across
different expressions. Missing signal support supplies no structured guidance.
Imported library statistics are unverified until measured in this campaign;
they cannot establish a parent/child improvement effect.

Refinement effects are child minus parent absolute mean discovery IC, requiring
full-panel evaluation for both. Fast-screen-only results are inconclusive.
Different formulas can have different warm-up support, so this is a descriptive
IC difference, not a paired causal effect. A delay challenge recomputes both
signals on paired support with a frozen direction. Its reward is the recomputed
parent's excess IC when the challenge reveals fragility, and zero when it passes.
These observations do not change admission or confirm future validity.

## Retrieval, uncertainty, and local disagreement

| Mode | Procedure choice |
| --- | --- |
| `none` | Uniform choice among available recipes; no transferred score |
| `motif` | Existing edit-aware residual model reconstructed from source and local edit trajectories |
| `trajectory` | Mean effect of up to five nearest same-scope observations, using signal dynamics and parent quality |
| `structured` | Context-matched evidence with dataset-level uncertainty and local contradiction review |

Imported observations from the current dataset identity are excluded. Structured
retrieval averages effects within each dataset and constructs a Student interval
across those dataset means. At least `minimum_datasets` are required. An interval
entirely above `minimum_effect` supports a recipe; one entirely below its negative
discourages it. Otherwise the evidence remains exploratory. The closest interval
bound to zero supplies the source score. Intervals assume independent dataset
means and are unadjusted for adaptive selection or multiple comparisons. More
trials on the same data do not increase the dataset count.

A local observation contradicting a supported direction marks that skill
`under_review`. Its source influence becomes `1 / (1 + contradictions)` and the
remaining weight goes to the mean local effect. Review records preserve the
contradictory event IDs; the imported pack is unchanged. This is an explicit
adaptation heuristic, not a calibrated posterior or a learned validity period.
Matched delay outcomes can also supply dataset-weighted pass/fail pseudocounts
to the planner's discovery-test prior, discounted by local disagreement.

Scores produce a softmax distribution with a positive uniform exploration
component. The ledger retains every available recipe, evidence summary, score,
and conditional probability. Root action probability remains
`selection_probability`; `joint_selection_probability` is its product with the
conditional recipe probability. The seed and action sequence reproduce the
choice. These logs support auditing; the implementation reports no off-policy
causal estimate.

Local procedure memory is rebuilt from committed action history, including after
resume. Uncommitted or interrupted outcomes cannot become successful transfer
evidence. Ordinary memory formation and persistence remain owned by the wrapped
`MemoryPolicy`.

## What the comparison establishes

The [transfer benchmark](reproducibility.md#research-skill-transfer-comparison)
isolates procedure choice with frozen source evidence, independent target seeds,
different target syntax, and equal candidate-evaluation allowances. It includes
nearest-trajectory retrieval, null targets, and an unseen future reversal.

The existing motif taxonomy maps all four wrapper edits to the same
`structural_grow` motif; it cannot distinguish these recipes in that comparison.
Nearest-trajectory retrieval is therefore the stronger procedure-specific
baseline. Neither skill retrieval nor descriptive confidence establishes market
profitability, handles an unobservable future break, or makes the planner optimal.
The extension remains opt-in.
