"""Behavioral tests of action selection, runtime dispatch, and durable resume."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict

import numpy as np
import pytest

from factorminer.application.runtime_context import MiningRunContext
from factorminer.architecture.research_actions import (
    POLICIES,
    ResearchAction,
    ResearchActionLedger,
    ResearchActionPlanner,
    ResearchPlannerConfig,
)
from factorminer.architecture.stages import GenerateStage, IterationPayload
from factorminer.core.helix_loop import HelixLoop
from factorminer.core.ralph_loop import BudgetTracker, RalphLoop
from factorminer.core.types import get_features
from factorminer.utils.config import Config, load_config


def action_menu():
    return [ResearchAction("generate", 2),
            ResearchAction("refine", 1, "$close", "Mean($close, 3)", 0.3),
            ResearchAction("challenge", 2, "$close", "Delay($close, 1)", 0.3),
            ResearchAction("stop", 0)]


def plan(planner, records=None, offers=None):
    return planner.plan(offers=offers or action_menu(), records=records or [],
                        context={"bucket": "growing"}, iteration=1, quality_threshold=0.04)


@pytest.mark.parametrize("policy", POLICIES)
def test_probabilities_are_reproducible_and_describe_actual_selection(policy):
    planner = ResearchActionPlanner(ResearchPlannerConfig(policy=policy))
    first = plan(planner)
    assert first == plan(planner)
    assert sum(first["probabilities"].values()) == pytest.approx(1)
    assert first["selection_probability"] == first["probabilities"][first["chosen"]["kind"]]
    assert first["selection_probability"] > 0
    assert first["config"]["policy"] == policy
    json.dumps(first, allow_nan=False)


def test_decision_value_prefers_informative_challenge_and_stops_when_value_is_negative():
    planner = ResearchActionPlanner(ResearchPlannerConfig(exploration_probability=0))
    decision = plan(planner)
    assert decision["chosen"]["kind"] == "challenge"
    challenge = next(row for row in decision["estimates"] if row["action"]["kind"] == "challenge")
    assert challenge["expected_benefit"] == pytest.approx(0.5 * (0.3 - 0.04))
    planner.config.evaluation_cost = 1
    stopped = plan(planner)
    assert stopped["chosen"]["kind"] == "stop"
    assert stopped["selection_probability"] == 1


def test_empty_generation_teaches_policy_without_fabricating_success():
    planner = ResearchActionPlanner(ResearchPlannerConfig(policy="heuristic"))
    decision = plan(planner)
    history = [{"decision": decision, "status": "completed", "outcome": {"candidates": [], "library_gain": 0}}]
    updated = plan(planner, history)
    before = next(row for row in decision["estimates"] if row["action"]["kind"] == "generate")
    after = next(row for row in updated["estimates"] if row["action"]["kind"] == "generate")
    assert after["success_probability"] < before["success_probability"]
    # Neither failed actions nor unrelated contexts supply positive evidence.
    history[0]["status"] = "failed"
    assert plan(planner, history)["estimates"] == decision["estimates"]


@pytest.mark.parametrize("field,value", [("prior_gain", float("nan")), ("evaluation_cost", -1),
                                         ("delay_bars", 0), ("policy", "unknown"),
                                         ("seed", 1.1), ("exploration_probability", 2)])
def test_invalid_planner_configuration_fails_early(field, value):
    cfg = ResearchPlannerConfig()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_nested_yaml_config_round_trip_and_unlimited_defaults(tmp_path):
    cfg = load_config(overrides={"research": {"planner": {"enabled": True, "policy": "random"}}})
    assert cfg.mining.max_iterations == 0
    assert cfg.research.planner.enabled
    cfg.save(tmp_path / "config.yaml")
    reloaded = load_config(tmp_path / "config.yaml")
    assert asdict(reloaded.research.planner) == asdict(cfg.research.planner)
    assert not Config().research.planner.enabled


def test_ledger_preserves_decisions_rejects_stale_writes_and_recovers_unknown_outcomes(tmp_path):
    ledger = ResearchActionLedger(tmp_path, {"data": "one"})
    decision = plan(ResearchActionPlanner(ResearchPlannerConfig()))
    ledger.begin(decision)
    second = ResearchActionLedger(tmp_path, {"data": "one"})
    with pytest.raises(ValueError, match="stale"):
        second.begin(decision)
    second.recover_interrupted()
    assert second.records()[0]["status"] == "interrupted"
    assert second.records()[0]["decision"] == decision
    assert "unknown" in second.records()[0]["outcome"]["reason"]
    next_decision = plan(ResearchActionPlanner(ResearchPlannerConfig()), second.records())
    second.begin(next_decision)
    second.finish(2, {"evaluations": 2, "elapsed_seconds": 0.25})
    with pytest.raises(ValueError, match="finalized"):
        second.finish(2, {})
    assert second.summary()["evaluations"] == 2
    assert not second.summary()["resource_quotas"]
    assert len(second.export().read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="identity"):
        ResearchActionLedger(tmp_path, {"data": "changed"})


def make_loop(tmp_path, cls=RalphLoop, *, config=None):
    cfg = config or Config()
    cfg.mining.batch_size = 2
    cfg.mining.target_library_size = 1000
    cfg.mining.icir_threshold = 0.01
    cfg.mining.correlation_threshold = 0.95
    cfg.evaluation.num_workers = 1
    cfg.memory.policy = "none"
    cfg.research.planner.enabled = True
    rng = np.random.default_rng(19)
    features = get_features()
    data = rng.normal(size=(20, 80, len(features)))
    slow = rng.normal(size=(20, 80))
    for t in range(1, 80):
        slow[:, t] = 0.95 * slow[:, t - 1] + 0.2 * slow[:, t]
    data[:, :, features.index("$close")] = slow
    returns = slow + rng.normal(size=(20, 80)) * 0.05
    loop = cls(cfg, data, returns, checkpoint_interval=0,
               run_context=MiningRunContext(output_dir=tmp_path))
    loop.stages["generate"] = GenerateStage(lambda _loop, _payload: [("slow", "$close")])
    return loop


def force_actions(loop, kinds):
    original = loop.research_actions.planner.plan
    choices = iter(kinds)

    def choose(**kwargs):
        decision = original(**kwargs)
        kind = next(choices)
        offer = next(offer for offer in kwargs["offers"] if offer.kind == kind)
        decision["chosen"] = asdict(offer)
        decision["probabilities"] = {offer.kind: float(offer.kind == kind) for offer in kwargs["offers"]}
        decision["selection_probability"] = 1.0
        decision["rationale"] = "Controlled dispatch for integration test"
        return decision

    loop.research_actions.planner.plan = choose


@pytest.mark.parametrize("cls", [RalphLoop, HelixLoop])
def test_all_four_actions_use_real_loop_and_challenge_does_not_mutate_library(tmp_path, cls):
    loop = make_loop(tmp_path, cls)
    force_actions(loop, ["generate", "refine", "challenge", "stop"])
    snapshots = []
    loop.run(max_iterations=0, callback=lambda _i, stats: snapshots.append((loop.library.size, stats)))
    records = loop.research_actions.ledger.records()
    assert [r["decision"]["chosen"]["kind"] for r in records] == ["generate", "refine", "challenge", "stop"]
    assert snapshots[1][0] == snapshots[2][0] == snapshots[3][0]
    assert records[1]["outcome"]["candidates"][0]["parent_formula"] == "$close"
    assert records[2]["outcome"]["challenge"]["status"] == "measured"
    assert records[2]["outcome"]["evaluations"] == 2
    assert snapshots[-1][1]["research_stop"]
    assert loop.budget.llm_calls == 1
    assert (tmp_path / "checkpoint" / "loop_state.json").is_file()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["research_actions"]["actions"] == 4


def test_stop_dispatches_no_generation_or_evaluation(tmp_path):
    cfg = Config()
    cfg.research.planner.evaluation_cost = 10
    loop = make_loop(tmp_path, config=cfg)
    loop.stages["generate"] = GenerateStage(lambda *_: pytest.fail("Generation must not run"))
    loop.run(max_iterations=0)
    assert loop.iteration == 1
    assert loop.research_actions.ledger.summary()["evaluations"] == 0
    assert loop.budget.llm_calls == 0


def test_failures_and_empty_generations_are_recorded(tmp_path):
    loop = make_loop(tmp_path)
    force_actions(loop, ["generate", "generate"])
    loop.stages["generate"] = GenerateStage(lambda *_: [])
    loop.iteration = 1
    stats = loop._run_iteration(2)
    assert stats["research_evaluations"] == 0

    def fail(*_):
        raise RuntimeError("Provider failure")

    loop.stages["generate"] = GenerateStage(fail)
    loop.iteration = 2
    with pytest.raises(RuntimeError, match="Provider"):
        loop._run_iteration(2)
    records = loop.research_actions.ledger.records()
    assert [r["status"] for r in records] == ["completed", "failed"]
    assert records[1]["outcome"]["error_type"] == "RuntimeError"
    assert loop.research_actions.active is None


def test_resume_retains_actions_library_and_memory_without_resetting_limits(tmp_path):
    first = make_loop(tmp_path)
    force_actions(first, ["generate"])
    first.budget = BudgetTracker(max_llm_calls=1, max_wall_seconds=0.001)
    first.run(max_iterations=1)
    before = first.research_actions.ledger.records()
    second = make_loop(tmp_path)
    force_actions(second, ["refine", "stop"])
    second.run(max_iterations=0)  # Automatically resumes the same output directory.
    after = second.research_actions.ledger.records()
    assert after[0] == before[0]
    assert second.iteration == 3
    assert second.library.size >= 1
    assert len(after) == 3
    assert second.budget.max_llm_calls == second.budget.max_wall_seconds == 0


def test_delay_challenge_recomputes_and_rejects_sign_reversal_on_identical_support(tmp_path):
    loop = make_loop(tmp_path)
    service = loop.research_actions
    payload = IterationPayload(iteration=1, batch_size=1)
    decision = plan(service.planner)
    service.ledger.begin(decision)
    service.active = decision
    payload.research_action = decision
    x = np.tile(np.arange(20)[:, None], (1, 80)).astype(float)
    loop.returns = x.copy()
    delayed = -x
    delayed[:, :3] = np.nan
    calls = []

    def compute(**kwargs):
        calls.append(kwargs["formula"])
        return None, x if len(calls) == 1 else delayed

    loop.evaluation_kernel.compute_signals = compute
    service.challenge(payload)
    result = payload.stage_metrics["research_challenge"]
    assert len(calls) == 2
    assert result["paired_periods"] == 77
    assert result["retention_ratio"] == pytest.approx(-1)
    assert result["passed"] is False
    assert loop.library.size == 0


@pytest.mark.parametrize("cls", [RalphLoop, HelixLoop])
@pytest.mark.parametrize("torn", [False, True])
def test_committed_action_recovers_even_without_a_readable_regular_checkpoint(tmp_path, cls, torn):
    loop = make_loop(tmp_path, cls)
    force_actions(loop, ["generate"])
    loop.iteration = 1
    loop._run_iteration(2)  # Simulate process death before run() saves a checkpoint.
    original = loop.research_actions.ledger.records()
    assert loop.library.size == 1
    checkpoint = tmp_path / "checkpoint"
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    if torn:
        checkpoint.mkdir()
        (checkpoint / "loop_state.json").write_text('{"broken":')
        (checkpoint / "library.json").write_text("partial metadata")
    resumed = make_loop(tmp_path, cls)
    force_actions(resumed, ["stop"])
    resumed.run(max_iterations=0)
    assert resumed.library.size == 1
    assert resumed.iteration == 2
    assert resumed.research_actions.ledger.records()[0] == original[0]
    np.testing.assert_array_equal(resumed.library.list_factors()[0].signals,
                                  loop.library.list_factors()[0].signals)


def test_interrupted_inflight_action_keeps_committed_work_and_unknown_outcome(tmp_path):
    loop = make_loop(tmp_path)
    force_actions(loop, ["generate", "refine"])
    loop.iteration = 1
    loop._run_iteration(2)
    payload = IterationPayload(iteration=2, batch_size=2)
    loop.research_actions.prepare(payload)  # Dispatch persisted, no terminal outcome.
    resumed = make_loop(tmp_path)
    force_actions(resumed, ["stop"])
    resumed.run(max_iterations=0)
    records = resumed.research_actions.ledger.records()
    assert [row["status"] for row in records] == ["completed", "interrupted", "completed"]
    assert "unknown" in records[1]["outcome"]["reason"]
    assert resumed.library.size == 1


def test_discovery_dataset_excludes_future_targets_and_honors_generation_settings(tmp_path):
    from factorminer.application.research_actions import discovery_dataset
    from factorminer.data.tensor_builder import TargetSpec
    from factorminer.evaluation.runtime import build_runtime_dataset_from_arrays

    data = np.arange(20 * 80, dtype=float).reshape(20, 80)
    dataset = build_runtime_dataset_from_arrays(
        {name: data for name in get_features()}, data,
        target_panels={"paper": data},
        target_specs={"paper": TargetSpec(name="paper", entry_delay_bars=2, holding_bars=3)},
        default_target="paper", split_indices={"train": np.arange(50), "test": np.arange(50, 80)},
    )
    discovery = discovery_dataset(dataset)
    assert discovery.returns.shape == (20, 45)
    assert set(discovery.splits) == {"train"}
    np.testing.assert_array_equal(discovery.get_target("paper"), data[:, :45])
    cfg = Config()
    cfg.llm.max_tokens = 8192
    cfg.llm.temperature = 0.4
    loop = make_loop(tmp_path, config=cfg)
    assert loop.generator.max_tokens == 8192
    assert loop.generator.temperature == 0.4


def test_regular_checkpoint_rejects_different_data_before_restoring_factors(tmp_path):
    cfg = Config()
    data = np.random.default_rng(9).normal(size=(20, 40, len(get_features())))
    returns = data[:, :, 0]
    first = RalphLoop(cfg, data, returns, run_context=MiningRunContext(output_dir=tmp_path))
    first.save_session()
    second = RalphLoop(cfg, data + 1, returns, run_context=MiningRunContext(output_dir=tmp_path))
    with pytest.raises(ValueError, match="dataset changed"):
        second.run(max_iterations=1)
