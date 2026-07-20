# Licensing

FactorMiner is dual-licensed. This document is the authoritative list of
which files are under which license; per-file headers and this document must
stay in sync (`scripts/check.py` enforces this — see below).

## MIT (default)

Every source file in this repository is MIT-licensed (see [LICENSE](LICENSE))
**except** the files listed below. This includes the DSL, operator registry,
`RalphLoop`/`HelixLoop`, the paper-protocol/dataset/evaluation-kernel/
geometry/memory-policy/stages contracts, data connectors, the canonical
benchmark runtime, base metrics/admission/CPCV/PBO/significance, and the MCP
server. The free tier is a complete, credible, unrestricted reproduction of
the underlying research — it is not deliberately crippled to force a
purchase.

## Business Source License 1.1 (advanced research modules)

The following files are licensed under [BUSL 1.1](LICENSE-BUSL), not MIT:

| File | What it does |
| --- | --- |
| `factorminer/architecture/sealed_joint_search.py` | Agora-style sealed multi-evaluator joint search |
| `factorminer/architecture/_sealed_evaluator_panel.py` | Private evaluator panel backing sealed joint search |
| `factorminer/architecture/island_model.py` | Island-model population mining with migration |
| `factorminer/architecture/rft_export.py` | Offline RFT/GRPO dataset export |
| `factorminer/architecture/model_stage.py` | Factor+model co-optimization loop stage |
| `factorminer/evaluation/crowding.py` | Factor-crowding diagnostics (Lou–Polk CoMetric, consensus-panel overlap) |
| `factorminer/evaluation/capacity.py` | Capacity-aware / market-impact backtesting |
| `factorminer/evaluation/model_zoo.py` | Downstream model co-optimization (RD-Agent(Q)-style) |
| `factorminer/evaluation/mrm_pack.py` | Model Risk Management (SR 26-2-shaped) validation pack |

Each file carries its own `SPDX-License-Identifier: BUSL-1.1` header. Where a
file's own header and this table ever disagree, the file's header controls
for that file, and `scripts/check.py` will fail until this table is
corrected to match.

BUSL 1.1's Additional Use Grant (full text in [LICENSE-BUSL](LICENSE-BUSL))
permits internal production research use, including by employees and
contractors; it restricts only (a) offering these modules to third parties
as a hosted/managed service, and (b) using them to build a competing product
for external customers. Each listed file converts to plain MIT four years
after its introduction (Change Date `2030-07-20` for the initial set above).

## Why a split, not one license for everything

None of this repository's closest funded/exited comparables in the
"AI research copilot for finance" category shipped their product as
permissively-licensed open source — MIT alone has no mechanism to capture
enterprise value from a hosted product. Keeping the DSL, evaluation kernel,
canonical benchmark runtime, and CPCV/PBO rigor fully MIT preserves the
credibility and adoption case (anyone can verify the research reproduces
faithfully); gating the modules above preserves a commercial path. See
[issue #12](https://github.com/minihellboy/factorminer/issues/12) for the
full comps analysis behind this decision.

## For contributors

- New files default to MIT. A file only belongs on the BUSL list above if it
  implements genuinely differentiated research IP beyond faithful paper
  reproduction — not every "advanced" feature qualifies.
- If your change adds a file that should be BUSL-licensed, add the SPDX
  header (see any file in the table above for the exact text) and add a row
  to this table in the same PR. `scripts/check.py` fails the build if the
  two drift apart.
- MIT-licensed files must not gain a *hard, unconditional* import-time
  dependency on a BUSL-licensed file — optional/lazy imports guarded by an
  opt-in config flag (the existing pattern throughout `architecture/` and
  `benchmark/runtime.py`) are fine and expected.
