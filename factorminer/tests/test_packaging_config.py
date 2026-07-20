"""Packaging-facing config tests."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from factorminer.configs import load_default_yaml
from factorminer.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_package_requires_python_312_or_newer():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    classifiers = pyproject["project"]["classifiers"]
    minor_version_classifiers = [
        classifier
        for classifier in classifiers
        if classifier.startswith("Programming Language :: Python :: 3.")
    ]

    assert pyproject["project"]["requires-python"] == ">=3.12"
    assert minor_version_classifiers == ["Programming Language :: Python :: 3.12"]
    assert pyproject["tool"]["ruff"]["target-version"] == "py312"
    assert pyproject["tool"]["mypy"]["python_version"] == "3.12"


def test_ci_uses_single_python_312_runtime():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert 'python-version: "3.12"' in workflow
    assert "matrix.python-version" not in workflow


def test_load_default_yaml_returns_packaged_config():
    data = load_default_yaml()

    assert data
    assert data["evaluation"]["backend"] == "numpy"


def test_binance_sample_config_matches_bundled_manifest():
    cfg = load_config(config_path=ROOT / "factorminer/configs/binance_sample.yaml")
    manifest = json.loads((ROOT / "data/binance_crypto_5m.manifest.json").read_text())

    assert cfg.llm.provider == "mock"
    assert cfg.data.market == "crypto"
    assert cfg.data.universe == "Binance"
    assert cfg.data.frequency == manifest["frequency"]
    assert cfg.data.default_target == "paper"
    assert cfg.data.train_period[0].startswith("2026-02-14")
    assert cfg.data.test_period[1].startswith("2026-02-18")
    assert manifest["matching_config"] == "factorminer/configs/binance_sample.yaml"
