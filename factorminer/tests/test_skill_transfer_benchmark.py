"""Frozen source/target boundaries and fair dispatch in the transfer comparison."""

from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from factorminer.benchmark import research_skills as benchmark
from factorminer.cli import main


def test_cli_freezes_source_pack_before_disjoint_equal_work_comparison(tmp_path, monkeypatch):
    episode = benchmark._episode

    def check_freeze(output, **kwargs):
        if not kwargs.get("source"):
            freeze = json.loads((tmp_path/"transfer_freeze.json").read_text())
            pack = json.loads((tmp_path/"skills.json").read_text())
            assert freeze["pack_id"] == pack["pack_id"]
        return episode(output, **kwargs)

    monkeypatch.setattr(benchmark, "_episode", check_freeze)
    result = CliRunner().invoke(main, ["-o", str(tmp_path), "benchmark", "research-skills",
                                       "--source-seeds", "71", "--target-seeds", "81"])
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path/"results.json").read_text())
    assert len(report["runs"]) == 20
    assert all(r["evaluations"] == 2 for r in report["runs"])
    assert not report["default_changed"]
    source_ids = {r["dataset_id"] for r in report["source_runs"]}
    target_ids = {r["dataset_id"] for r in report["runs"]}
    assert not source_ids & target_ids
    freeze = json.loads((tmp_path/"transfer_freeze.json").read_text())
    assert freeze["pack_id"] == report["pack_id"]
    assert freeze["source_cost_evaluations"] > 0
    with pytest.raises(FileExistsError):
        benchmark.run_skill_transfer_benchmark(tmp_path, source_seeds=(71,), target_seeds=(81,))


def test_future_observations_cannot_change_transferred_procedure_selection(tmp_path, monkeypatch):
    source = tmp_path/"source"
    benchmark._episode(source, seed=75, family="noisy_level", mode="none", source=True)
    from factorminer.architecture.research_skills import compile_skill_pack, write_skill_pack
    path = write_skill_pack(compile_skill_pack([source]), tmp_path/"skills.json")
    first = benchmark._episode(tmp_path/"first", seed=85, family="nonlinear_level",
                                mode="structured", pack_path=str(path))
    original = benchmark.transfer_panel

    def changed(*args, **kwargs):
        data, returns, split, parent = original(*args, **kwargs)
        returns[:, split:] = np.random.default_rng(86).normal(size=returns[:, split:].shape)
        return data, returns, split, parent

    monkeypatch.setattr(benchmark, "transfer_panel", changed)
    second = benchmark._episode(tmp_path/"second", seed=85, family="nonlinear_level",
                                 mode="structured", pack_path=str(path))
    left = json.loads((tmp_path/"first"/"frozen_discovery.json").read_text())
    right = json.loads((tmp_path/"second"/"frozen_discovery.json").read_text())
    assert [r["decision"] for r in left["actions"]] == [r["decision"] for r in right["actions"]]
    assert first["selected_formula"] == second["selected_formula"]
    assert first["heldout_utility"] != second["heldout_utility"]


@pytest.mark.parametrize("source,target", [((), (1,)), ((1,), (1,)), ((1, 1), (2,)), ((-1,), (2,))])
def test_invalid_or_overlapping_campaign_partitions_fail_before_dispatch(tmp_path, source, target):
    with pytest.raises(ValueError):
        benchmark.run_skill_transfer_benchmark(tmp_path, source_seeds=source, target_seeds=target)
