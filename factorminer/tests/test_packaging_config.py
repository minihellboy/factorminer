"""Packaging-facing config tests."""

from __future__ import annotations

from factorminer.configs import load_default_yaml


def test_load_default_yaml_returns_packaged_config():
    data = load_default_yaml()

    assert data
    assert data["evaluation"]["backend"] == "numpy"
