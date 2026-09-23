"""The isolated Qlib lane must derive receipted metrics from frozen rows."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from factorminer.benchmark.datasets import _cfg_with_overrides, load_benchmark_dataset
from factorminer.benchmark.evidence_run import run_evidence_benchmark
from factorminer.benchmark.qlib_runner import (
    _selectors,
    run_isolated_qlib,
    verify_qlib_receipt_artifacts,
)
from factorminer.benchmark.receipt import verify_research_receipt
from factorminer.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "factorminer" / "configs" / "binance_sample.yaml"
DATA = ROOT / "data" / "binance_crypto_5m.csv"


def _qlib_python() -> str:
    configured = os.environ.get("FACTORMINER_QLIB_PYTHON")
    if configured:
        return configured
    if importlib.util.find_spec("qlib") is not None:
        return sys.executable
    pytest.skip("pyqlib is not installed in the test environment")


def test_qlib_receipt_metrics_are_computed_from_frozen_test_rows(tmp_path):
    cfg = load_config(CONFIG)
    result = run_evidence_benchmark(
        cfg, tmp_path, data_path=str(DATA), baseline_names=["alpha101_adapted"],
        qlib_model="ridge", qlib_python=_qlib_python(),
    )
    release = Path(result["receipt_path"]).parent
    manifest = json.loads((release / "manifest.json").read_text())
    inventory = {f"phase2/{name}": release / path
                 for name, path in manifest["artifact_paths"].items()}
    assert manifest["qlib"]["protocol"] == "frozen_panel_qlib_ridge"
    assert manifest["qlib"]["metrics"]["scored_rows"] > 0
    assert verify_qlib_receipt_artifacts(manifest, inventory) == []
    assert verify_research_receipt(release, commitment_input=DATA).passed
    assert "qlib_panel" not in manifest["artifact_paths"]

    relocated = tmp_path / "copied" / release.name
    shutil.copytree(release, relocated)
    assert verify_research_receipt(relocated, commitment_input=DATA).passed
    metrics_path = inventory["phase2/qlib_metrics"]
    metrics = json.loads(metrics_path.read_text())
    metrics["IC"] += 0.1
    metrics_path.write_text(json.dumps(metrics))
    assert verify_qlib_receipt_artifacts(manifest, inventory)
    assert not verify_research_receipt(release, commitment_input=DATA).passed


def test_qlib_worker_failure_cannot_create_an_audited_result(tmp_path):
    cfg = load_config(CONFIG)
    dataset, _ = load_benchmark_dataset(
        _cfg_with_overrides(cfg, cfg.benchmark.freeze_universe), data_path=str(DATA)
    )
    with pytest.raises(RuntimeError, match="could not start"):
        run_isolated_qlib(dataset, cfg, tmp_path / "qlib", python=str(tmp_path / "missing-python"))
    assert not (tmp_path / "qlib" / "audit.json").exists()


def test_qlib_training_selector_respects_label_maturity():
    def dataset(train, test):
        return SimpleNamespace(
            get_split=lambda name: SimpleNamespace(indices=np.asarray(
                train if name == "train" else test, dtype=np.int64)),
            target_specs={"paper": SimpleNamespace(entry_delay_bars=1, holding_bars=2)},
            default_target="paper",
        )

    train, test, maturity = _selectors(dataset([0, 1, 2, 3, 4], [6, 7]))
    assert train.tolist() == [0, 1, 2]
    assert test.tolist() == [6, 7]
    assert maturity == 3
    with pytest.raises(ValueError, match="chronological"):
        _selectors(dataset([0, 1, 2], [2, 3]))
    with pytest.raises(ValueError, match="mature"):
        _selectors(dataset([0, 1], [3, 4]))
