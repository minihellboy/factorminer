"""Tests for the canonical config-to-runtime boundary."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from factorminer.application.runtime_context import (
    MiningRunContext,
    MiningSettings,
    build_run_context,
)
from factorminer.utils.config import load_config


def test_hierarchical_config_is_the_settings_source(tmp_path) -> None:
    cfg = load_config(
        overrides={
            "mining": {"batch_size": 17, "ic_threshold": 0.03},
            "evaluation": {"num_workers": 3, "redundancy_metric": "pearson"},
            "memory": {"policy": "family_aware"},
        }
    )
    settings = MiningSettings(
        cfg,
        MiningRunContext(output_dir=tmp_path),
    )

    assert settings.batch_size == 17
    assert settings.ic_threshold == 0.03
    assert settings.num_workers == 3
    assert settings.redundancy_metric == "pearson"
    assert settings.memory_policy == "family_aware"
    assert settings.output_dir == str(tmp_path)


def test_dataset_materialization_lives_in_run_context(tmp_path) -> None:
    cfg = load_config()
    panel = np.ones((3, 5))
    dataset = SimpleNamespace(
        target_panels={"paper": panel},
        target_specs={"paper": SimpleNamespace(holding_bars=2)},
    )

    context = build_run_context(cfg, output_dir=tmp_path, dataset=dataset)
    settings = MiningSettings(cfg, context)

    assert settings.target_panels == {"paper": panel}
    assert settings.target_horizons == {"paper": 2}
    assert not hasattr(cfg, "target_panels")


def test_mock_override_changes_runtime_policy_not_canonical_config(tmp_path) -> None:
    cfg = load_config()
    normal = MiningSettings(cfg, build_run_context(cfg, output_dir=tmp_path / "normal"))
    mock = MiningSettings(
        cfg,
        build_run_context(cfg, output_dir=tmp_path / "mock", mock=True),
    )

    assert cfg.evaluation.signal_failure_policy == "reject"
    assert normal.signal_failure_policy == "reject"
    assert mock.signal_failure_policy == "synthetic"


def test_invalid_runtime_override_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="signal_failure_policy"):
        MiningRunContext(output_dir=tmp_path, signal_failure_policy="ignore")
