---
description: Evaluate a factor library's out-of-sample IC, ICIR, and decay
argument-hint: "[library path, e.g. 'output/run1/factor_library.json']"
---

Load the `factor-evaluation` skill and recompute the library's metrics on the
held-out `test` split. Lead with out-of-sample numbers, and run `--period both`
to surface train→test decay. Report honestly — an in-sample-only factor is a
rejected factor.
