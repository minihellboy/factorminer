---
description: Rank a factor library and return the strongest signals
argument-hint: "[library path and count, e.g. 'factor_library.json top 10']"
---

Load the `factor-evaluation` skill and produce a ranked signal shortlist — the
top-K factors by out-of-sample IC, each with its formula and stats. This is the
handoff artifact for a research-idea workflow that needs to know which
quantitative signals are currently working.
