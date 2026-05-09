# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

FactorMiner is a single Python package (not a monorepo). It is a CLI-driven research framework for LLM-guided quantitative alpha factor mining. There are no databases, Docker containers, web servers, or external services required.

### Prerequisites

- Python 3.10+ (CI tests 3.10 and 3.12)
- `uv` package manager (lockfile: `uv.lock`)

### Common commands

Commands are documented in `README.md` and `CONTRIBUTING.md`. Quick reference:

| Task | Command |
|------|---------|
| Install deps | `uv sync --group dev --extra llm` |
| Lint | `uv run ruff check .` |
| Tests | `uv run pytest -q factorminer/tests` |
| Health check | `uv run factorminer doctor` |
| Quick demo (no API keys) | `uv run factorminer quickstart` |
| Full demo script | `uv run python run_demo.py` |
| Mock mining | `uv run factorminer mine --mock -n 2 -b 8 -t 10` |
| Build wheel | `uv build` |

### Non-obvious notes

- The `-o` (output directory) flag is a **top-level** `factorminer` option, not a subcommand option. Place it before the subcommand: `uv run factorminer -o /tmp/out mine --mock`, not `uv run factorminer mine -o /tmp/out --mock`.
- The `MockProvider` is built-in for LLM-free testing. No API keys are needed when using `--mock` or when `llm.provider` is set to `mock` in config.
- `uv` must be on `PATH`. If installed via the official installer, it lands in `~/.local/bin`. Ensure `PATH` includes that directory.
- Repo-wide ruff is clean on `main`. The `CONTRIBUTING.md` notes that repo-wide ruff debt existed historically but should not be introduced in new changes.
- The `output/` directory is gitignored mutable runtime state.
- Tests take ~4 minutes on a standard VM (498 tests). Some `RuntimeWarning` messages about empty slices and NaN windows are expected and non-blocking.
- The `--all-extras` flag includes the `gpu` extra (`cupy-cuda12x`) which is Linux-only. In CI, `--group dev --extra llm` is used instead, which avoids GPU dependencies.
