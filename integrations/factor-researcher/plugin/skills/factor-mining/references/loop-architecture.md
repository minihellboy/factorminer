# The mining loop — Ralph and Helix

FactorMiner discovers factors with a self-evolving loop. Two loops are exposed;
Helix is a strict superset of Ralph.

## Ralph loop (`factorminer mine`)

The paper-faithful Algorithm 1. Each iteration:

1. **Retrieve** — pull memory priors from experience memory (which formula
   families have worked, which have failed).
2. **Generate** — the LLM proposes a batch of candidate factor formulas,
   conditioned on the memory priors and the current library state.
3. **Evaluate** — candidates are scored: fast IC screen, redundancy-correlation
   check against the library, replacement-rule check, intra-batch dedup, then
   full validation.
4. **Update** — survivors are admitted to the factor library under the
   admission rules (IC / ICIR / correlation thresholds).
5. **Evolve memory** — successes and failures are written back so the next
   iteration's priors are sharper.

The loop stops when the library reaches `--target` size or `--iterations` is
exhausted.

## Helix loop (`factorminer helix`)

Helix re-frames the loop as five composable stages and adds optional Phase 2
research features. It falls back to exact Ralph behavior when no Phase 2
feature is enabled — a true drop-in superset.

| Stage | Does | Optional feature |
|---|---|---|
| **RETRIEVE** | Hybrid memory lookup | knowledge graph, embeddings |
| **PROPOSE** | Candidate generation | multi-specialist **debate** + critic |
| **SYNTHESIZE** | Normalize formulas | SymPy **canonicalization** |
| **VALIDATE** | Score candidates | **causal** (do-calculus), **regime**, capacity, significance |
| **DISTILL** | Admit + evolve memory | online regime memory with forgetting |

### When each feature earns its compute

- **debate** — diverse candidates; helps when single-shot generation stalls.
- **causal** — filters factors whose IC is correlation without intervention
  support; helps on noisy universes.
- **regime** — keeps factors that survive across market states, not just one.
- **canonicalization** — always cheap; prevents duplicate libraries.

Enable features per research question via `--causal / --no-causal` etc., or in
the YAML config under `phase2.*`. More features means more LLM calls and more
compute — turn on what the question needs, not everything.
