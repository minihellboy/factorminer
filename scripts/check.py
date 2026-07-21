#!/usr/bin/env python3
"""Validate FactorMiner integration manifests and their local references."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS = ROOT / "integrations"

errors: list[str] = []
checked = 0


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def err(message: str) -> None:
    errors.append(message)


def load_yaml(path: Path) -> Any:
    global checked
    checked += 1
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        err(f"YAML parse: {rel(path)}: {exc}")
    return {}


def load_json(path: Path) -> Any:
    global checked
    checked += 1
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        err(f"JSON parse: {rel(path)}: {exc}")
    return {}


def read_frontmatter(path: Path) -> dict[str, Any]:
    global checked
    checked += 1
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        err(f"frontmatter: {rel(path)}: missing leading ---")
        return {}
    try:
        _, frontmatter, _ = text.split("---", 2)
        payload = yaml.safe_load(frontmatter) or {}
    except (ValueError, yaml.YAMLError) as exc:
        err(f"frontmatter: {rel(path)}: {exc}")
        return {}
    if not isinstance(payload, dict):
        err(f"frontmatter: {rel(path)}: must be a mapping")
        return {}
    return payload


def check_frontmatter() -> None:
    for path in sorted(INTEGRATIONS.glob("*/plugin/agents/*.md")):
        meta = read_frontmatter(path)
        for key in ("name", "description"):
            if key not in meta:
                err(f"frontmatter: {rel(path)}: missing '{key}'")

    for path in sorted(INTEGRATIONS.glob("*/plugin/commands/*.md")):
        meta = read_frontmatter(path)
        if "description" not in meta:
            err(f"frontmatter: {rel(path)}: missing 'description'")

    for path in sorted(INTEGRATIONS.glob("*/plugin/skills/**/SKILL.md")):
        meta = read_frontmatter(path)
        for key in ("name", "description"):
            if key not in meta:
                err(f"frontmatter: {rel(path)}: missing '{key}'")


def check_output_schema(path: Path, schema: Any) -> None:
    if not isinstance(schema, dict):
        err(f"output_schema: {rel(path)}: must be a mapping")
        return
    if schema.get("type") != "object":
        err(f"output_schema: {rel(path)}: type must be object")
    if schema.get("additionalProperties") is not False:
        err(f"output_schema: {rel(path)}: additionalProperties must be false")
    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, list):
        err(f"output_schema: {rel(path)}: required must be a list")
        required = []
    if not isinstance(properties, dict):
        err(f"output_schema: {rel(path)}: properties must be a mapping")
        properties = {}
    for key in required:
        if key not in properties:
            err(f"output_schema: {rel(path)}: required key '{key}' has no property")


def check_refs(path: Path, data: dict[str, Any]) -> None:
    base = path.parent

    system = data.get("system")
    if isinstance(system, dict) and "file" in system:
        target = (base / system["file"]).resolve()
        if not target.is_file():
            err(f"ref: {rel(path)}: system.file -> {system['file']} (not found)")

    for skill in data.get("skills") or []:
        if not isinstance(skill, dict):
            continue
        if "path" in skill:
            target = (base / skill["path"]).resolve()
            if not target.exists():
                err(f"ref: {rel(path)}: skills.path -> {skill['path']} (not found)")
        if "from_plugin" in skill:
            target = (base / skill["from_plugin"]).resolve()
            if not (target / "skills").is_dir():
                err(
                    f"ref: {rel(path)}: skills.from_plugin -> {skill['from_plugin']} "
                    "(no skills/ dir)"
                )

    for agent in data.get("callable_agents") or []:
        if not isinstance(agent, dict) or "manifest" not in agent:
            continue
        target = (base / agent["manifest"]).resolve()
        if not target.is_file():
            err(f"ref: {rel(path)}: callable_agents.manifest -> {agent['manifest']} (not found)")

    if "output_schema" in data:
        check_output_schema(path, data["output_schema"])


def check_managed_agents() -> None:
    for directory in sorted(INTEGRATIONS.glob("*/managed-agent")):
        for path in sorted(directory.rglob("*.yaml")):
            data = load_yaml(path)
            if isinstance(data, dict):
                check_refs(path, data)

        for required in ("agent.yaml", "steering-examples.json"):
            if not (directory / required).is_file():
                err(f"missing: {rel(directory)}/{required}")
        if not (directory.parent / "README.md").is_file():
            err(f"missing: {rel(directory.parent)}/README.md")


def check_json_files() -> None:
    patterns = [
        ".claude-plugin/marketplace.json",
        "integrations/*/plugin/.claude-plugin/plugin.json",
        "integrations/*/plugin/.mcp.json",
        "integrations/*/plugin/hooks/*.json",
        "integrations/*/managed-agent/steering-examples.json",
    ]
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            load_json(path)


def check_marketplace_sources() -> None:
    marketplace = ROOT / ".claude-plugin" / "marketplace.json"
    data = load_json(marketplace)
    for plugin in data.get("plugins", []) if isinstance(data, dict) else []:
        source = plugin.get("source")
        name = plugin.get("name", "<unknown>")
        if not source:
            err(f"marketplace: {name}: missing source")
            continue
        target = (ROOT / source).resolve()
        if not (target / ".claude-plugin" / "plugin.json").is_file():
            err(f"marketplace: {name} source -> {source} (no plugin.json)")


def check_mit_license_policy() -> None:
    global checked
    license_files = sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING"))
    )
    checked += 1
    if license_files != ["LICENSE"]:
        err(f"license: expected only LICENSE at repository root, found {license_files}")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    checked += 1
    if not license_text.startswith("MIT License\n"):
        err("license: LICENSE does not contain the MIT License text")

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    checked += 1
    if project.get("license") != "MIT":
        err("license: pyproject.toml must declare license = \"MIT\"")
    if project.get("license-files") != ["LICENSE"]:
        err('license: pyproject.toml must declare license-files = ["LICENSE"]')

    data_manifest = json.loads(
        (ROOT / "data/binance_crypto_5m.manifest.json").read_text(encoding="utf-8")
    )
    checked += 1
    if data_manifest.get("license") != "MIT":
        err("license: bundled data manifest must declare MIT")

    marker = "SPDX-License-" + "Identifier:"
    source_roots = (ROOT / "factorminer", ROOT / "scripts", ROOT / "integrations")
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue
            for line in lines:
                if marker not in line:
                    continue
                checked += 1
                identifier = line.partition(marker)[2].strip()
                if identifier != "MIT":
                    err(f"license: {rel(path)} declares SPDX identifier {identifier!r}")


def main() -> int:
    check_json_files()
    check_frontmatter()
    check_managed_agents()
    check_marketplace_sources()
    check_mit_license_policy()

    if errors:
        print(f"FAIL - {len(errors)} issue(s) across {checked} check(s):", file=sys.stderr)
        for message in errors:
            print(f"  - {message}", file=sys.stderr)
        return 1
    print(f"OK - {checked} check(s), 0 issues.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
