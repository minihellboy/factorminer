"""Packaging-facing config tests."""

from __future__ import annotations

import json
from pathlib import Path

from factorminer.configs import load_default_yaml
from factorminer.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


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
