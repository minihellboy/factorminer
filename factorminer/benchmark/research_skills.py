"""Frozen cross-campaign transfer experiments using actual mining stages."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from factorminer.application.runtime_context import MiningRunContext
from factorminer.architecture.research_actions import ResearchAction, ResearchActionPlanner
from factorminer.architecture.research_skills import (
    TRANSFER_MODES,
    compile_skill_pack,
    digest,
    write_skill_pack,
)
from factorminer.architecture.stages import GenerateStage
from factorminer.core.ralph_loop import RalphLoop
from factorminer.core.types import get_features
from factorminer.evaluation.metrics import compute_rank_ic
from factorminer.utils.config import Config

TRANSFER_BENCHMARK_VERSION = "research-skill-transfer-v1"
SOURCE_FAMILIES = ("noisy_level", "integrated_nuisance", "scale_mixture")
TARGET_FAMILIES = ("nonlinear_level", "drifting_nuisance", "scale_bursts", "independent_noise", "future_reversal")


def transfer_panel(seed: int, family: str, *, periods: int = 180, assets: int = 24):
    """Diagnostic processes with disjoint source/target mechanisms, not prices."""
    if family not in SOURCE_FAMILIES + TARGET_FAMILIES:
        raise ValueError("Unknown transfer family")
    family_index = (SOURCE_FAMILIES + TARGET_FAMILIES).index(family)
    rng = np.random.default_rng(np.random.SeedSequence([seed, family_index]))
    latent = rng.normal(size=(assets, periods))
    for i in range(1, periods):
        latent[:, i] = 0.94*latent[:, i-1] + 0.35*latent[:, i]
    target = latent + 0.8*rng.normal(size=latent.shape)
    if family in ("integrated_nuisance", "drifting_nuisance"):
        innovation = rng.normal(size=latent.shape)
        signal = np.cumsum(innovation, axis=1) + 0.25*rng.normal(size=latent.shape)
        if family == "drifting_nuisance":
            signal += np.linspace(0, 6, periods)[None, :] * rng.normal(size=(assets, 1))
        target = innovation + rng.normal(size=latent.shape)
    elif family in ("scale_mixture", "scale_bursts"):
        scale = np.exp(rng.normal(0, 1.4, size=(1, periods)))
        if family == "scale_bursts":
            scale *= 1 + 5*(np.arange(periods) % 13 < 3)
        signal = scale*(latent + 0.9*rng.normal(size=latent.shape))
    else:
        signal = latent + 1.1*rng.normal(size=latent.shape)
        if family == "nonlinear_level":
            signal = np.sign(signal)*np.sqrt(abs(signal))
        elif family == "independent_noise":
            target = rng.normal(size=latent.shape)
        elif family == "future_reversal":
            target[:, periods//2:] *= -1  # No discovery method can see this break.
    data = rng.normal(size=(assets, periods, len(get_features())))
    # Syntax and leaf families differ across source and evaluation campaigns.
    if family in SOURCE_FAMILIES:
        parent = "$close"
        data[:, :, get_features().index("$close")] = signal
    else:
        parent = "Add($high, $low)"
        nuisance = rng.normal(size=signal.shape)
        data[:, :, get_features().index("$high")] = nuisance
        data[:, :, get_features().index("$low")] = signal-nuisance
    return data, target, periods//2, parent


def _config(mode: str, pack_path: str, seed: int) -> Config:
    cfg = Config()
    cfg.mining.batch_size = 1
    cfg.mining.target_library_size = 10000
    cfg.mining.icir_threshold = 0.01
    cfg.mining.correlation_threshold = 0.9
    cfg.evaluation.fast_screen_assets = 1000000
    cfg.evaluation.num_workers = 1
    cfg.evaluation.signal_failure_policy = "raise"
    cfg.memory.policy = "none"
    cfg.research.planner.enabled = True
    cfg.research.planner.seed = seed
    cfg.research.skills.enabled = True
    cfg.research.skills.mode = mode
    cfg.research.skills.source = pack_path
    return cfg


class _ProcedureSchedule(ResearchActionPlanner):
    """Hold the experiment kind fixed to isolate memory's choice of procedure.

    This is an explicit benchmark intervention, not a claim about end-to-end
    planner superiority. Every method receives the same original parent.
    """

    def __init__(self, inner, loop, parent: str, *, source: bool):
        super().__init__(inner.config)
        self.inner, self.loop, self.parent, self.source = inner, loop, parent, source

    def plan(self, **kwargs):
        records = kwargs["records"]
        kind = "generate" if not records else "refine"
        if self.source and len(records) == 5:
            kind = "challenge"
        if kind == "refine":
            # Freeze the original parent for each recipe observation, avoiding
            # different parent trajectories as a confound in this comparison.
            parent_row = records[0]["outcome"]["candidates"][0]
            kwargs["offers"] = [ResearchAction("refine", 1, self.parent,
                                               f"Mean({self.parent}, 3)", parent_row["quality"]),
                                ResearchAction("stop", 0)]
        available = [offer for offer in kwargs["offers"] if offer.kind == kind]
        if not available:
            kind = "stop"
        decision = self.inner.plan(**kwargs)
        selected = next(offer for offer in kwargs["offers"] if offer.kind == kind)
        from dataclasses import asdict
        decision["chosen"] = asdict(selected)
        decision["probabilities"] = {offer.kind: float(offer.kind == kind) for offer in kwargs["offers"]}
        decision["selection_probability"] = 1.0
        decision["rationale"] = "Frozen benchmark kind/parent intervention; memory chooses among available procedures"
        decision["benchmark_intervention"] = True
        return decision


def _episode(output: Path, *, seed: int, family: str, mode: str, pack_path: str = "", source=False):
    data, returns, split, parent = transfer_panel(seed, family)
    cfg = _config(mode, pack_path, seed)
    loop = RalphLoop(cfg, data[:, :split-1], returns[:, :split-1],
                     checkpoint_interval=0, run_context=MiningRunContext(output_dir=output))
    actions = loop.research_actions
    assert actions is not None  # Enabled by the frozen benchmark configuration.
    loop.stages["generate"] = GenerateStage(lambda *_: [("initial_parent", parent)])
    actions.planner = _ProcedureSchedule(actions.planner, loop, parent, source=source)
    steps = 6 if source else 2  # All edits in source; one choice on each unseen task.
    started = time.monotonic()
    for i in range(1, steps+1):
        loop.iteration = i
        stats = loop._run_iteration(1)
        if stats.get("research_stop"):
            break
    records = actions.ledger.records()
    # Freeze selection before computing any final result.
    factors = sorted(loop.library.list_factors(), key=lambda f: (-float(f.ic_paper_mean or 0.0), f.formula))
    best = factors[0] if factors else None
    frozen = {"config": cfg.to_dict(), "selected_formula": best.formula if best else None,
              "selected_direction": (1 if best.ic_mean >= 0 else -1) if best else None,
              "dataset_id": loop.trial_dataset_id, "actions": records}
    (output/"frozen_discovery.json").write_text(json.dumps(frozen, indent=2, allow_nan=False))
    result = {"seed": seed, "family": family, "mode": mode, "source": source,
              "elapsed_seconds": time.monotonic()-started,
              "evaluations": actions.ledger.summary()["evaluations"],
              "selected_formula": frozen["selected_formula"], "dataset_id": loop.trial_dataset_id,
              "selected_recipe": next((r["decision"]["chosen"].get("recipe_id") for r in records
                                       if r["decision"]["chosen"]["kind"] == "refine"), None),
              "library_size": loop.library.size}
    if not source:
        result["heldout_utility"] = 0.0
        result["false_claim"] = False
        if best:
            _, signals = loop.evaluation_kernel.compute_signals(
                formula=best.formula, data_dict=loop.evaluation_kernel.build_data_dict(data, get_features()),
                returns_shape=returns.shape, signal_failure_policy="raise")
            series = compute_rank_ic(signals[:, split:], returns[:, split:]) * frozen["selected_direction"]
            series = series[np.isfinite(series)]
            quality = float(series.mean()) if series.size else 0.0
            result["heldout_utility"] = quality-cfg.mining.ic_threshold
            result["false_claim"] = quality < cfg.mining.ic_threshold
    (output/"result.json").write_text(json.dumps(result, indent=2))
    return result


def run_skill_transfer_benchmark(output_dir: str | Path, *, source_seeds: tuple[int, ...] = tuple(range(1000, 1008)),
                                 target_seeds: tuple[int, ...] = tuple(range(2000, 2010))) -> dict:
    if not source_seeds or not target_seeds or set(source_seeds) & set(target_seeds):
        raise ValueError("Source and evaluation seeds must be nonempty and disjoint")
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in source_seeds+target_seeds):
        raise ValueError("Seeds must be nonnegative integers")
    if len(set(source_seeds)) != len(source_seeds) or len(set(target_seeds)) != len(target_seeds):
        raise ValueError("Seeds must be unique within each phase")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"version": TRANSFER_BENCHMARK_VERSION, "source_families": SOURCE_FAMILIES,
                "target_families": TARGET_FAMILIES, "source_seeds": source_seeds, "target_seeds": target_seeds,
                "modes": TRANSFER_MODES, "base_config": _config("none", "", source_seeds[0]).to_dict(),
                "random_stream": "SeedSequence([seed, index in source_families + target_families])",
                "source_schedule": "one parent, four randomized edits without replacement, one available delay challenge",
                "target_schedule": "one fixed parent, one selected edit; two candidate evaluations per method",
                "primary_metric": "held-out direction-frozen IC minus 0.04 for the best discovery-admitted factor, or zero for abstention",
                "limitations": ["Controlled procedure-choice transfer; not an end-to-end planner or market benchmark",
                                "Source syntax and generating families excluded from target campaigns",
                                "One-step procedures; no learned multistep options",
                                "Unannounced future reversal is invisible in discovery",
                                "IC utility is not portfolio profitability; all final panels are diagnostic"]}
    protocol["protocol_sha256"] = digest(protocol)
    with (output/"protocol.json").open("x") as handle:
        json.dump(protocol, handle, indent=2)
    source_rows, paths = [], []
    for family in SOURCE_FAMILIES:
        for seed in source_seeds:
            path = output/"source"/family/str(seed)
            source_rows.append(_episode(path, seed=seed, family=family, mode="none", source=True))
            paths.append(path)
    pack = compile_skill_pack(paths)
    pack_path = write_skill_pack(pack, output/"skills.json")
    # This file exists before any evaluation-family episode is constructed.
    (output/"transfer_freeze.json").write_text(json.dumps({"pack_id": pack["pack_id"],
        "source_datasets": sorted({r["dataset_id"] for r in source_rows}),
        "source_cost_evaluations": sum(r["evaluations"] for r in source_rows),
        "source_seconds": sum(r["elapsed_seconds"] for r in source_rows)}, indent=2))
    runs = []
    for family in TARGET_FAMILIES:
        for seed in target_seeds:
            for mode in TRANSFER_MODES:
                row = _episode(output/"target"/family/str(seed)/mode, seed=seed, family=family,
                               mode=mode, pack_path=str(pack_path))
                assert row["dataset_id"] not in {r["dataset_id"] for r in source_rows}
                runs.append(row)
    summaries: dict[str, dict] = {}
    for family in TARGET_FAMILIES:
        summaries[family] = {}
        for mode in TRANSFER_MODES:
            rows = [r for r in runs if r["family"] == family and r["mode"] == mode]
            summaries[family][mode] = {"mean_heldout_utility": float(np.mean([r["heldout_utility"] for r in rows])),
                                      "false_claims": sum(r["false_claim"] for r in rows),
                                      "recipe_choices": dict(Counter(r["selected_recipe"] for r in rows)),
                                      "mean_evaluations": float(np.mean([r["evaluations"] for r in rows]))}
    pairs: dict[str, dict] = {}
    for family in TARGET_FAMILIES:
        pairs[family] = {}
        for baseline in TRANSFER_MODES[:-1]:
            differences = np.asarray([next(r["heldout_utility"] for r in runs if
                r["family"] == family and r["seed"] == s and r["mode"] == "structured") -
                next(r["heldout_utility"] for r in runs if
                r["family"] == family and r["seed"] == s and r["mode"] == baseline) for s in target_seeds])
            interval = None
            if len(differences) > 1:
                draws = np.random.default_rng(20260913).choice(differences, size=(10000, len(differences)))
                interval = np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()
            pairs[family][baseline] = {"differences": differences.tolist(), "mean": float(differences.mean()),
                                      "paired_bootstrap_95_interval": interval}
    report = {"protocol": protocol, "pack_id": pack["pack_id"], "skill_statuses": dict(Counter(s["status"] for s in pack["skills"])),
              "source_runs": source_rows, "runs": runs, "summaries": summaries, "paired_comparisons": pairs,
              "default_changed": False}
    (output/"results.json").write_text(json.dumps(report, indent=2))
    return report
