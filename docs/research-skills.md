# Research skills

The opt-in skill extension compiles completed action histories into versioned
edit/testing procedures and retrieves that evidence in a later campaign.
`MemoryPolicy` owns retrieval and local feedback in both Ralph and Helix.
Evaluation and admission use the normal engine services.

## Collect source experience

Copy the [action-planner profile](../factorminer/configs/research_actions_deepseek.yaml)
to a local config, retain its model/data settings, and set:

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

Run separate source campaigns with their own data and output directories:

```bash
uv run --env-file .env factorminer -c path/to/source-config.yaml \
  -o output/source-a mine --data path/to/source-a.csv
```

`none` chooses uniformly among available recipes while retaining the configured
ordinary memory policy. The extension adds no resource quota; normal planner
stopping and automatic resume apply.

## Freeze and transfer

```bash
uv run factorminer research-skills compile \
  output/source-a output/source-b output/source-c \
  --destination output/skills/v1.json
uv run factorminer research-skills inspect output/skills/v1.json
```

Inputs are campaign directories or `research_actions.sqlite3` files. Compilation
uses read transactions, rejects in-flight actions and conflicting campaign
snapshots, and deduplicates repeated sources. A destination can be reused only
for identical content. Further source experience requires a new pack version.

For a target campaign, keep the planner enabled and replace the `skills` section:

```yaml
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

This section belongs under `research`. Use a new output directory for the target.
The campaign retains `research_skill_pack.json` and pins its hash and retrieval
settings in its identity/manifest. Resume uses that copy if the original moves;
a different source version or retrieval configuration requires a new campaign.

## Procedures and evidence

| Recipe | Child for parent `P` |
| --- | --- |
| `smooth_3` | `Mean(P, 3)` |
| `smooth_8` | `Mean(P, 8)` |
| `difference_1` | `Delta(P, 1)` |
| `rank_smooth_3` | `Mean(CsRank(P), 3)` |
| `delay_1` | `Delay(P, 1)`; advisory challenge when delay is one bar |

Recipes are fixed, parser-checked, single-step edits. Refinement considers unseen
valid children of the selected parent. Packs retain exact formulas, action and
dataset IDs, record hashes, propensities, outcomes, and applicability conditions.
Legacy histories without measured applicability are skipped explicitly.

Applicability matches market/frequency, targets/timing, default target, signal
failure policy, delay-test thresholds, parent-quality band, signal persistence,
and scale variation. Signal descriptors use already evaluated signals; quality
bands use discovery targets. Formula family is omitted to permit transfer
across expressions. Missing support supplies no structured guidance.

Refinement reward is child minus parent absolute mean discovery IC, requiring
full-panel evaluation for both. Fast-screen-only or unverified imported parent
statistics cannot establish improvement. Different warm-up support makes this
a descriptive IC difference, not a paired causal effect. Delay reward is the
recomputed parent's excess IC when a paired-support challenge reveals fragility,
and zero when it passes. Failed/interrupted work supplies no measured effect.

## Retrieval and local review

| Mode | Score |
| --- | --- |
| `none` | Uniform choice; no transferred score |
| `motif` | Existing edit-aware residual model over source/local trajectories |
| `trajectory` | Mean effect of up to five nearest same-scope observations |
| `structured` | Context-matched effect intervals and local contradiction review |

Imported observations with the target dataset identity are excluded. Structured
retrieval averages effects within each dataset and constructs a Student interval
across dataset means. With at least `minimum_datasets`, an interval entirely
above `minimum_effect` supports a recipe; entirely below its negative discourages
it. Otherwise evidence is exploratory. The bound closest to zero is the score.
Intervals assume independent dataset means and are unadjusted for selection or
multiple comparisons. Distinct hashes alone do not establish independence.

Local contradictions mark a supported direction `under_review`. Source weight
becomes `1 / (1 + contradictions)`; the remaining weight goes to the local mean.
Event IDs are retained and the imported pack remains frozen. Matched delay
outcomes can also supply dataset-weighted pass/fail pseudocounts to the planner,
discounted by local disagreement. These are adaptation heuristics.

Scores feed a softmax with positive uniform exploration. Records contain every
variant, score, evidence summary, conditional probability, and selected recipe.
`selection_probability` remains the root action probability;
`joint_selection_probability` multiplies it by the conditional recipe probability.
Seed and action sequence reproduce dispatch. No off-policy causal estimate is
reported. Local memory is rebuilt from committed history after resume.

The [transfer comparison](reproducibility.md#research-skill-transfer-comparison)
includes nearest-trajectory retrieval, null targets, and a hidden future break.
The coarse motif model groups all four edits as `structural_grow`. Skills remain
opt-in; descriptive confidence does not establish future validity or optimality.
