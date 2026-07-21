"""Public CLI package.

The console entry point remains ``factorminer.cli:main`` while command
implementations are split by responsibility inside this package.
"""

from __future__ import annotations

from importlib import import_module

from factorminer.cli import app as _app

import_module("factorminer.cli.analysis_commands")
import_module("factorminer.cli.benchmark_commands")
import_module("factorminer.cli.data_commands")
import_module("factorminer.cli.integration_commands")
import_module("factorminer.cli.hosted_commands")
_mining_commands = import_module("factorminer.cli.mining_commands")
import_module("factorminer.cli.partner_review_commands")
import_module("factorminer.cli.receipt_commands")
import_module("factorminer.cli.public_data_commands")
import_module("factorminer.cli.research_commands")

main = _app.main
mine = _mining_commands.mine
_app.mine = mine

__all__ = ["main"]


def __getattr__(name: str):
    """Preserve programmatic access during command-module extraction."""
    try:
        return getattr(_app, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_app)))
