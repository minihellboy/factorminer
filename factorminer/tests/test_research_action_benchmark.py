"""The policy benchmark must preserve information access and resource comparability."""

from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from factorminer.benchmark.research_actions import compare_research_actions, controlled_panel
from factorminer.cli import main


def test_matched_allowances_real_dispatch_and_frozen_heldout_access(tmp_path):
    data, returns, split = controlled_panel(93, "persistent_signal", periods=80)
    catalog = ("$close", "$open", "$volume", "Mean($close, 3)")
    first = compare_research_actions(data, returns, split, tmp_path / "first",
                                     seeds=(1, 2), evaluation_horizon=6, catalog=catalog, purge_bars=2)
    assert first["protocol"]["discovery_end_exclusive"] == split - 2
    assert first["protocol"]["base_config"]["research"]["planner"]["enabled"]
    assert len(first["protocol"]["benchmark_source_sha256"]) == 64
    assert len(first["runs"]) == 8
    assert all(row["evaluations"] <= 6 for row in first["runs"])
    assert all(row["evaluations"] == 6 for row in first["runs"] if row["policy"] != "decision_value")
    assert not first["default_policy_changed"]
    assert all(pair["paired_bootstrap_95_interval"] is not None for pair in first["paired_comparisons"].values())
    changed_returns = returns.copy()
    changed_returns[:, split:] = np.random.default_rng(811).normal(size=changed_returns[:, split:].shape)
    second = compare_research_actions(data, changed_returns, split, tmp_path / "second",
                                      seeds=(1,), evaluation_horizon=6, catalog=catalog, purge_bars=2)
    assert first["protocol"]["data_sha256"] != second["protocol"]["data_sha256"]
    for policy in first["summaries"]:
        left = json.loads((tmp_path / "first" / "seed-1" / policy / "frozen_discovery.json").read_text())
        right = json.loads((tmp_path / "second" / "seed-1" / policy / "frozen_discovery.json").read_text())
        assert left["formulas"] == right["formulas"]
        assert [r["decision"] for r in left["actions"]] == [r["decision"] for r in right["actions"]]
    assert first["runs"][0]["heldout_delay_utility"] != second["runs"][0]["heldout_delay_utility"]
    with pytest.raises(ValueError, match="already exists"):
        compare_research_actions(data, returns, split, tmp_path / "first")


def test_cli_research_action_benchmark_writes_protocol_and_results(tmp_path):
    result = CliRunner().invoke(main, ["-o", str(tmp_path), "benchmark", "research-actions", "--seeds", "8", "--evaluations", "2"])
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "summary.json").read_text())
    assert len(report["scenarios"]) == 4
    assert not report["default_policy_changed"]
    assert (tmp_path / "null" / "protocol.json").is_file()


@pytest.mark.parametrize("seeds,horizon", [((), 6), ((1, 1), 6), ((-1,), 6), ((1,), 1)])
def test_invalid_comparison_protocol_rejected(tmp_path, seeds, horizon):
    data, returns, split = controlled_panel(9, "null", periods=40)
    with pytest.raises(ValueError):
        compare_research_actions(data, returns, split, tmp_path, seeds=seeds, evaluation_horizon=horizon)
