#!/usr/bin/env python3
"""Enforce FactorMiner's executable dependency boundaries.

The checker operates on imports rather than directory intent.  It protects the
stable domain from orchestration dependencies, keeps adapters independent of
application workflows, and prevents engine code from reaching interface
surfaces such as CLI, MCP, or benchmarks.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "factorminer"

DOMAIN_MODULES = frozenset(
    {
        "factorminer.core.canonicalizer",
        "factorminer.core.config",
        "factorminer.core.expression_tree",
        "factorminer.core.factor_library",
        "factorminer.core.parser",
        "factorminer.core.types",
    }
)
ADAPTER_PREFIXES = (
    "factorminer.agent",
    "factorminer.data",
    "factorminer.operators",
)
APPLICATION_PREFIXES = (
    "factorminer.application",
    "factorminer.architecture",
    "factorminer.evaluation",
    "factorminer.memory",
)
INTERFACE_PREFIXES = (
    "factorminer.benchmark",
    "factorminer.cli",
    "factorminer.mcp",
)
BANNED_INTERNAL_IMPORTS = {
    "factorminer.architecture.dependence": "factorminer.domain.dependence",
}
BENCHMARK_SERVICE_MODULES = frozenset(
    {
        "factorminer.benchmark.datasets",
        "factorminer.benchmark.frozen_evaluation",
        "factorminer.benchmark.mining_runtime",
        "factorminer.benchmark.provenance",
        "factorminer.benchmark.reporting",
        "factorminer.benchmark.runners",
        "factorminer.benchmark.runtime_contracts",
        "factorminer.benchmark.speed",
        "factorminer.benchmark.statistics",
    }
)


@dataclass(frozen=True, order=True)
class ImportViolation:
    """One forbidden source-to-target dependency."""

    source: str
    line: int
    target: str
    rule: str

    def render(self) -> str:
        return f"{self.source}:{self.line}: {self.rule}: import {self.target}"


def _matches(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


def layer_for(module: str) -> str:
    """Return the declared architectural layer for a module."""
    if module == "factorminer.domain" or module.startswith("factorminer.domain."):
        return "domain"
    if module in DOMAIN_MODULES:
        return "domain"
    if _matches(module, ADAPTER_PREFIXES):
        return "adapter"
    if _matches(module, APPLICATION_PREFIXES) or (
        module.startswith("factorminer.core.") and module not in DOMAIN_MODULES
    ):
        return "application"
    if _matches(module, INTERFACE_PREFIXES):
        return "interface"
    return "support"


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative(source: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = source.split(".")[:-1]
    keep = max(0, len(package_parts) - node.level + 1)
    base = package_parts[:keep]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def iter_imports(source: str, text: str):
    """Yield ``(target, line)`` pairs from one Python source string."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            target = _resolve_relative(source, node)
            if target:
                yield target, node.lineno


def violations_for_source(module: str, text: str) -> list[ImportViolation]:
    """Check one module against the declared directional rules."""
    source_layer = layer_for(module)
    violations: list[ImportViolation] = []
    for target, line in iter_imports(module, text):
        if not target.startswith("factorminer"):
            continue
        target_layer = layer_for(target)
        rule = ""
        if target in BANNED_INTERNAL_IMPORTS and module != target:
            replacement = BANNED_INTERNAL_IMPORTS[target]
            rule = f"compatibility import is internal-only; use {replacement}"
        elif module in BENCHMARK_SERVICE_MODULES and target == "factorminer.benchmark.runtime":
            rule = "benchmark services must not depend on the runtime orchestrator"
        elif source_layer == "domain" and target_layer != "domain":
            rule = "domain must not depend on higher layers"
        elif source_layer == "adapter" and target_layer in {"application", "interface"}:
            rule = "adapter must not depend on application or interface layers"
        elif source_layer == "application" and target_layer == "interface":
            rule = "application must not depend on interface layers"
        if rule:
            violations.append(ImportViolation(module, line, target, rule))
    return violations


def check_repository() -> list[ImportViolation]:
    """Check every production Python module in the package."""
    violations: list[ImportViolation] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "tests" in path.parts:
            continue
        module = _module_name(path)
        violations.extend(violations_for_source(module, path.read_text(encoding="utf-8")))
    return sorted(violations)


def main() -> int:
    violations = check_repository()
    if violations:
        for violation in violations:
            print(violation.render(), file=sys.stderr)
        print(f"Architecture check failed: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    print("Architecture check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
