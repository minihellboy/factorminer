"""Behavioral contracts for empirical skill construction, transfer, and recovery."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
from click.testing import CliRunner

from factorminer.architecture.memory_policy import NoMemoryPolicy
from factorminer.architecture.paper_protocol import PaperProtocol
from factorminer.architecture.research_actions import ResearchActionPlanner, ResearchPlannerConfig
from factorminer.architecture.research_skills import (
    EDIT_RECIPES,
    RECIPE_VERSION,
    ResearchSkillsConfig,
    compile_skill_pack,
    context_key,
    digest,
    load_skill_pack,
    observation_from_action,
    recipe_formula,
    recipe_variants,
    signal_profile,
    summarize_skill,
    write_skill_pack,
)
from factorminer.architecture.skill_memory import TransferableSkillMemoryPolicy
from factorminer.benchmark.research_skills import _episode
from factorminer.cli import main
from factorminer.core.helix_loop import HelixLoop
from factorminer.tests.test_research_actions import action_menu, force_actions, make_loop
from factorminer.utils.config import Config, load_config


def context():
    return {"scope": "target-one", "persistence": "moderate", "scale": "stable",
            "quality_band": "high", "parent_formula": "$close", "parent_quality": 0.2,
            "lag1": 0.3, "scale_cv": 0.2, "recipe_version": RECIPE_VERSION,
            "quality_scope": "full", "ic_threshold": 0.04}


def observed(dataset="one", reward=0.1, recipe="smooth_3"):
    ctx = context()
    return {"event_id": digest([dataset, reward, recipe]), "dataset_id": dataset,
            "campaign_id": digest(dataset), "sequence": 2, "recipe_id": recipe,
            "kind": "refine", "parent_formula": "$close", "formula": recipe_formula(recipe, "$close"),
            "parent_quality": 0.2, "child_quality": 0.2+reward, "context": ctx,
            "context_key": context_key(ctx), "reward": reward, "status": "measured",
            "propensity": 0.25, "admitted": True}


def policy(rows, *, mode="structured"):
    cfg = Config()
    protocol = PaperProtocol.from_config(cfg)
    result = TransferableSkillMemoryPolicy(NoMemoryPolicy(protocol), ResearchSkillsConfig(mode=mode), protocol)
    result.pack = {"pack_id": "frozen-test-pack", "observations": rows}
    result.dataset_id = "new-target"
    return result


@pytest.mark.parametrize("field,value", [("mode", "unknown"), ("minimum_datasets", 1),
    ("minimum_datasets", True), ("temperature", 0), ("minimum_effect", float("nan")),
    ("confidence_level", 1), ("exploration_probability", 0), ("recipes", ["untrusted_code"])])
def test_skill_config_rejects_unsupported_or_nonfinite_inputs(field, value):
    cfg = ResearchSkillsConfig()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.validate()


def test_skill_config_roundtrips_safe_yaml_and_requires_planner(tmp_path):
    cfg = load_config(overrides={"research": {"planner": {"enabled": True},
                                             "skills": {"enabled": True, "mode": "trajectory"}}})
    cfg.save(tmp_path/"config.yaml")
    loaded = load_config(tmp_path/"config.yaml")
    assert asdict(cfg.research.skills) == asdict(loaded.research.skills)
    loaded.research.planner.enabled = False
    with pytest.raises(ValueError, match="requires"):
        loaded.validate()


def test_procedures_are_exact_typed_transformations():
    variants = recipe_variants("$close", EDIT_RECIPES)
    assert len(variants) == 4
    assert recipe_formula("rank_smooth_3", "$close") == "Mean(CsRank($close), 3)"
    with pytest.raises(ValueError):
        recipe_formula("arbitrary", "$close")
    assert recipe_variants("not-a-formula", EDIT_RECIPES) == []


def test_signal_context_depends_on_signal_dynamics_and_handles_missing_support():
    rng = np.random.default_rng(88)
    walk = np.cumsum(rng.normal(size=(24, 100)), axis=1)
    noise = rng.normal(size=walk.shape)
    assert signal_profile(walk)["persistence"] == "persistent"
    assert signal_profile(noise)["persistence"] == "short"
    assert signal_profile(np.full_like(noise, np.nan))["persistence"] == "unknown"
    assert signal_profile(np.zeros_like(noise))["lag1"] is None


def test_uncertainty_counts_datasets_and_retains_contradictory_outcomes():
    cfg = ResearchSkillsConfig()
    same = [observed(reward=0.1+i/1000) for i in range(100)]
    result = summarize_skill(same, cfg)
    assert result["datasets"] == 1
    assert result["status"] == "exploratory"
    assert result["interval"] == [None, None]
    distinct = [observed(str(i), reward=0.1) for i in range(4)]
    assert summarize_skill(distinct, cfg)["status"] == "supported"
    distinct.append(observed("negative", -0.8))
    conflicting = summarize_skill(distinct, cfg)
    assert conflicting["status"] == "exploratory"
    assert len(conflicting["contradictory"]) == 1


def test_scoped_retrieval_changes_selection_and_local_failure_reduces_source_influence():
    rows = [observed(str(i), 0.15) for i in range(4)]
    memory = policy(rows)
    variants = recipe_variants("$close", EDIT_RECIPES)
    _, before = memory.select_research_variant(variants, context=context(), sequence=2, seed=7)
    chosen = before["variants"][0]
    assert chosen["probability"] > 0.9
    assert chosen["evidence"]["status"] == "supported"
    local = observed("new-target", -0.4)
    memory.local[local["event_id"]] = local
    _, after = memory.select_research_variant(variants, context=context(), sequence=3, seed=7)
    changed = after["variants"][0]
    assert changed["evidence"]["status"] == "under_review"
    assert changed["evidence"]["source_influence"] == 0.5
    assert changed["probability"] < chosen["probability"]
    assert all(v["probability"] > 0 for v in after["variants"])
    assert rows == memory.pack["observations"]  # Frozen evidence was not rewritten.
    foreign = {**context(), "scope": "different-target"}
    _, unsupported = memory.select_research_variant(variants, context=foreign, sequence=4, seed=7)
    assert [v["probability"] for v in unsupported["variants"]] == pytest.approx([0.25]*4)


def test_same_dataset_is_excluded_from_imported_support():
    memory = policy([observed("new-target", 0.2) for _ in range(100)])
    _, guidance = memory.select_research_variant(recipe_variants("$close", EDIT_RECIPES),
                                                 context=context(), sequence=2, seed=3)
    assert all(v["evidence"]["status"] == "out_of_scope" for v in guidance["variants"])


@pytest.mark.parametrize("mode", ["none", "motif", "trajectory", "structured"])
def test_each_memory_mode_logs_actual_reproducible_selection_probabilities(mode):
    memory = policy([observed(str(i)) for i in range(4)], mode=mode)
    variants = recipe_variants("$close", EDIT_RECIPES)
    first = memory.select_research_variant(variants, context=context(), sequence=2, seed=3)
    assert first == memory.select_research_variant(variants, context=context(), sequence=2, seed=3)
    assert sum(v["probability"] for v in first[1]["variants"]) == pytest.approx(1)
    assert first[1]["conditional_probability"] > 0


@pytest.mark.parametrize("cls", [None, HelixLoop])
def test_real_loop_records_lineage_propensity_and_recovers_frozen_skill_state(tmp_path, cls):
    cfg = Config()
    cfg.research.skills.enabled = True
    cfg.research.skills.recipes = ["difference_1"]
    cfg.evaluation.fast_screen_assets = 10000
    loop = make_loop(tmp_path, **({"cls": cls} if cls else {}), config=cfg)
    force_actions(loop, ["generate", "refine"])
    loop.run(max_iterations=2)
    records = loop.research_actions.ledger.records()
    selected = records[1]["decision"]
    assert selected["chosen"]["recipe_id"] == "difference_1"
    assert selected["chosen"]["formula"] == "Delta($close, 1)"
    assert selected["joint_selection_probability"] == pytest.approx(
        selected["selection_probability"] * selected["skill_selection"]["conditional_probability"])
    assert selected["skill_context"]["quality_scope"] == "full"
    scope = selected["skill_context"]["scope_description"]
    assert scope["delay_test"]["retention"] == cfg.research.planner.delay_retention
    assert scope["signal_failure_policy"] == cfg.evaluation.signal_failure_policy
    row = observation_from_action(records[1], dataset_id=loop.trial_dataset_id,
                                 campaign_id=loop.research_actions.ledger.campaign_id)
    assert row["status"] == "measured"
    assert row["reward"] < 0
    old = copy.deepcopy(loop.memory_policy.local)
    resumed = make_loop(tmp_path, **({"cls": cls} if cls else {}), config=cfg)
    resumed.load_session(str(tmp_path/"checkpoint"))
    assert resumed.memory_policy.local == old
    assert resumed.research_actions.ledger.records() == records


def test_compiler_deduplicates_sources_verifies_artifacts_and_cli_preserves_versions(tmp_path):
    source = tmp_path/"source"
    _episode(source, seed=102, family="noisy_level", mode="none", source=True)
    first = compile_skill_pack([source])
    assert first["observations"]
    assert first == compile_skill_pack([source, source])
    path = write_skill_pack(first, tmp_path/"skills.json")
    assert load_skill_pack(path) == first
    assert write_skill_pack(first, path) == path
    corrupted = copy.deepcopy(first)
    corrupted["skills"][0]["mean_effect"] += 1
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="modified"):
        load_skill_pack(path)
    with pytest.raises(ValueError, match="immutable"):
        write_skill_pack(first, path)
    result = CliRunner().invoke(main, ["research-skills", "compile", str(source),
                                      "--destination", str(tmp_path/"compiled.json")])
    assert result.exit_code == 0, result.output
    inspected = CliRunner().invoke(main, ["research-skills", "inspect", str(tmp_path/"compiled.json")])
    assert inspected.exit_code == 0, inspected.output
    invalid = CliRunner().invoke(main, ["research-skills", "compile", str(tmp_path),
                                       "--destination", str(tmp_path/"invalid.json")])
    assert invalid.exit_code == 1
    assert "Error:" in invalid.output


def test_unfinished_and_legacy_records_do_not_invent_scoped_evidence():
    assert observation_from_action({"status": "interrupted"}, dataset_id="d", campaign_id="c") is None
    assert observation_from_action({"status": "completed", "decision": {"chosen": {"kind": "refine"}}},
                                   dataset_id="d", campaign_id="c") is None


def test_motif_baseline_uses_existing_residual_model_in_a_matching_context():
    memory = policy([observed(str(i), 0.1) for i in range(12)], mode="motif")
    _, guidance = memory.select_research_variant(recipe_variants("$close", EDIT_RECIPES),
                                                 context=context(), sequence=2, seed=11)
    assert guidance["variants"][0]["evidence"]["confidence"] > 0
    assert all(v["score"] > 0 for v in guidance["variants"])
    # These wrapper edits share the existing structural-grow motif. The
    # faithful coarse baseline cannot distinguish their specific operators.
    assert len({v["score"] for v in guidance["variants"]}) == 1


def test_delay_test_knowledge_changes_prior_and_disagreement_starts_review():
    rows = []
    for i in range(4):
        row = observed(str(i), 0.16, recipe="delay_1")
        row.update(kind="challenge", passed=False)
        rows.append(row)
    memory = policy(rows)
    prior = memory.research_action_prior("challenge", context=context())
    assert prior["beta_add"] == 4
    planner = ResearchActionPlanner(ResearchPlannerConfig())
    decision = planner.plan(offers=action_menu(), records=[], iteration=1, quality_threshold=0.04,
                            context={"bucket": "growing", "research_action_priors": {"challenge": prior}})
    challenge = next(r for r in decision["estimates"] if r["action"]["kind"] == "challenge")
    # Delay tests start from the planner's neutral Beta(1, 1) prior.
    assert challenge["success_probability"] == pytest.approx(1/6)
    local = observed("target", 0, recipe="delay_1")
    local.update(kind="challenge", passed=True)
    memory.local[local["event_id"]] = local
    revised = memory.research_action_prior("challenge", context=context())
    assert revised["beta_add"] == 2
    assert memory.pack["observations"] == rows


def test_campaign_keeps_frozen_source_when_original_moves_and_rejects_new_version(tmp_path):
    source = tmp_path/"source"
    _episode(source, seed=108, family="noisy_level", mode="none", source=True)
    pack = compile_skill_pack([source])
    path = write_skill_pack(pack, tmp_path/"source-skills.json")
    cfg = Config()
    cfg.research.skills.enabled = True
    cfg.research.skills.source = str(path)
    loop = make_loop(tmp_path/"target", config=cfg)
    force_actions(loop, ["generate", "refine"])
    loop.run(max_iterations=2)
    previous = loop.research_actions.ledger.records()
    manifest = json.loads((tmp_path/"target"/"run_manifest.json").read_text())
    assert manifest["research_skills"] == loop.memory_policy.transfer_identity()
    assert manifest["artifact_paths"]["research_skill_pack"] == str(tmp_path/"target"/"research_skill_pack.json")
    path.rename(tmp_path/"archived.json")
    resumed = make_loop(tmp_path/"target", config=cfg)
    resumed.load_session(str(tmp_path/"target"/"checkpoint"))
    assert resumed.memory_policy.pack["pack_id"] == pack["pack_id"]
    assert resumed.research_actions.ledger.records() == previous
    changed = copy.deepcopy(pack)
    changed["limitations"].append("new version")
    changed["pack_id"] = digest({k: v for k, v in changed.items() if k != "pack_id"})
    write_skill_pack(changed, path)
    with pytest.raises(ValueError, match="changed"):
        make_loop(tmp_path/"target", config=cfg)


def test_imported_library_metadata_cannot_supply_measured_parent_evidence(tmp_path):
    donor = make_loop(tmp_path/"donor")
    force_actions(donor, ["generate"])
    donor.run(max_iterations=1)
    assert donor.library.size == 1
    cfg = Config()
    cfg.research.skills.enabled = True
    cfg.evaluation.fast_screen_assets = 10000
    receiver = make_loop(tmp_path/"receiver", config=cfg)
    receiver.library = donor.library
    force_actions(receiver, ["refine"])
    receiver.run(max_iterations=1)
    record = receiver.research_actions.ledger.records()[0]
    assert record["decision"]["skill_context"]["quality_scope"] == "unverified"
    assert record["decision"]["skill_context"]["persistence"] == "unknown"
    row = observation_from_action(record, dataset_id=receiver.trial_dataset_id,
                                 campaign_id=receiver.research_actions.ledger.campaign_id)
    assert row["status"] == "inconclusive"
    assert row["reward"] is None


def test_unknown_signal_context_never_supplies_structured_transfer():
    ctx = {**context(), "persistence": "unknown"}
    rows = [observed(str(i)) for i in range(4)]
    for row in rows:
        row.update(context=ctx, context_key=context_key(ctx))
    memory = policy(rows)
    _, guidance = memory.select_research_variant(recipe_variants("$close", EDIT_RECIPES),
                                                 context=ctx, sequence=2, seed=4)
    assert all(v["evidence"]["status"] == "out_of_scope" for v in guidance["variants"])
    assert memory.research_action_prior("challenge", context=ctx) == {}


def test_delay_reward_uses_recomputed_paired_quality_instead_of_stored_metadata():
    record = {"sequence": 1, "status": "completed", "decision": {
        "chosen": {"kind": "challenge", "recipe_id": "delay_1", "parent_formula": "$close",
                   "formula": "Delay($close, 1)", "parent_quality": 0.9},
        "skill_context": context(), "selection_probability": 1},
        "outcome": {"challenge": {"status": "measured", "passed": False, "parent_signed_ic": -0.2}}}
    row = observation_from_action(record, dataset_id="d", campaign_id="c")
    assert row["reward"] == pytest.approx(0.16)
    assert row["measured_parent_quality"] == 0.2
