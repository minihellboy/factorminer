"""Execute research actions through the existing mining and evaluation services."""

from __future__ import annotations

import io
import math
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from factorminer.architecture.research_actions import (
    ResearchAction,
    ResearchActionLedger,
    ResearchActionPlanner,
)
from factorminer.architecture.research_planner import ResearchCyclePlanner
from factorminer.architecture.research_skills import (
    RECIPE_VERSION,
    ResearchSkillsConfig,
    digest,
    recipe_variants,
    signal_profile,
)
from factorminer.core.parser import try_parse
from factorminer.core.types import get_features
from factorminer.evaluation.metrics import compute_rank_ic


def finite_float(value: Any, default: float = 0.0) -> float:
    number = float(value)
    return number if math.isfinite(number) else default


class ResearchActionService:
    """Prepare, dispatch, and account for a research action without resource quotas."""

    def __init__(self, loop: Any, config: Any) -> None:
        self.loop = loop
        self.config = config
        self.planner = ResearchActionPlanner(config)
        identity = {
            "dataset_id": loop.trial_dataset_id, "planner": asdict(config),
            "loop_type": loop._loop_type(), "protocol": loop.protocol.runtime_contract(),
        }
        self.skills: ResearchSkillsConfig = getattr(loop.settings.research, "skills", None) or ResearchSkillsConfig()
        self.skills_enabled = self.skills.enabled
        if self.skills_enabled:
            identity["skills"] = loop.memory_policy.transfer_identity()
        self.ledger = ResearchActionLedger(loop.settings.output_dir, identity)
        loop.memory_policy.persist_research_skills(loop.settings.output_dir)
        self.active: dict[str, Any] | None = None
        self.started_at = 0.0
        self.evaluations = 0
        self.library_before = 0.0

    def checkpoint_snapshot(self) -> bytes:
        """Reuse the canonical serializers; store no executable object payloads."""
        buffer = io.BytesIO()
        with TemporaryDirectory(prefix="checkpoint-research-") as directory:
            self.loop.save_session(directory, _snapshot=True)
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
                for path in sorted(Path(directory).iterdir()):
                    archive.write(path, path.name)
        return buffer.getvalue()

    def restore_checkpoint(self, directory: Path) -> None:
        """Recover committed state even if a regular checkpoint was torn or missing."""
        snapshot = self.ledger.snapshot()
        if snapshot is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        allowed = {"library.json", "library_signals.npz", "memory.json", "loop_state.json",
                   "knowledge_graph.json", "helix_state.json"}
        with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
            if set(archive.namelist()) - allowed:
                raise ValueError("Unexpected research recovery snapshot member")
            for name in allowed:
                path = directory / name
                if name in archive.namelist():
                    path.write_bytes(archive.read(name))
                elif path.exists():
                    path.unlink()

    def library_utility(self) -> float:
        threshold = self.loop.settings.ic_threshold
        return sum(max(finite_float(f.ic_paper_mean) - threshold, 0)
                   for f in self.loop.library.list_factors())

    def prepare(self, payload: Any) -> str:
        records = self.ledger.records()
        self.loop.memory_policy.sync_research_history(
            records, dataset_id=self.loop.trial_dataset_id, campaign_id=self.ledger.campaign_id)
        observed: dict[str, dict] = {}
        seen: set[str] = set()
        challenged: set[str] = set()
        for record in records:
            outcome = record.get("outcome") or {}
            for row in outcome.get("candidates", []):
                seen.add(row["formula"])
                if row["parse_ok"]:
                    observed[row["formula"]] = row
            if record["status"] == "completed" and outcome.get("challenge"):
                challenged.add(record["decision"]["chosen"]["parent_formula"])
        factors = self.loop.library.list_factors()
        for factor in factors:
            if self.skills_enabled and factor.formula in observed:
                observed[factor.formula]["admitted"] = True
            else:
                observed[factor.formula] = {
                    "formula": factor.formula, "quality": finite_float(factor.ic_paper_mean),
                    "admitted": True, "parse_ok": True,
                }
                if self.skills_enabled:
                    # Imported library statistics need not describe this dataset.
                    observed[factor.formula].update(
                        profile=signal_profile(None), quality_scope="unverified")
            seen.add(factor.formula)
        offers = [ResearchAction("generate", payload.batch_size)]
        # All distinct observed parents remain eligible. No fixed search quota.
        for row in sorted(observed.values(), key=lambda r: (-r["quality"], r["formula"])):
            candidates = ([v["formula"] for v in recipe_variants(row["formula"], self.skills.recipes)]
                          if self.skills_enabled else [f"Mean({row['formula']}, {self.config.refinement_window})"])
            formula = next((f for f in candidates if f not in seen and try_parse(f) is not None), None)
            if formula is not None:
                offers.append(ResearchAction("refine", 1, row["formula"], formula, row["quality"]))
                break
        for factor in sorted(factors, key=lambda f: (-finite_float(f.ic_paper_mean), f.formula)):
            if factor.formula not in challenged:
                formula = f"Delay({factor.formula}, {self.config.delay_bars})"
                if try_parse(formula) is not None:
                    offers.append(ResearchAction("challenge", 2, factor.formula, formula,
                                                 finite_float(factor.ic_paper_mean)))
                    break
        offers.append(ResearchAction("stop", 0))
        recent = [r for r in records if r["status"] == "completed"
                  and r["decision"]["chosen"]["kind"] in ("generate", "refine")][-3:]
        stalled = len(recent) == 3 and all((r["outcome"] or {}).get("library_gain", 0) <= 0 for r in recent)
        bucket = "empty" if not factors else ("stalled" if stalled else "growing")
        theme = ResearchCyclePlanner().plan_cycle(payload.library_state, payload.memory_signal)
        context = {
            "bucket": bucket, "library_size": len(factors), "library_utility": self.library_utility(),
            "family_theme": theme.to_dict(),
            "library": [{"formula": f.formula, "quality": finite_float(f.ic_paper_mean)} for f in factors],
            "memory_directions": payload.memory_signal.get("recommended_directions", []),
            "evaluations_so_far": self.ledger.summary()["evaluations"],
        }
        if self.skills_enabled:
            priors = {}
            for offer in offers:
                if offer.kind == "challenge" and self.config.delay_bars == 1:
                    parent_context = self._skill_context(observed[offer.parent_formula])
                    prior = self.loop.memory_policy.research_action_prior("challenge", context=parent_context)
                    if prior:
                        priors["challenge"] = prior
            context["research_action_priors"] = priors
        decision = self.planner.plan(
            offers=offers, records=records, context=context, iteration=payload.iteration,
            quality_threshold=self.loop.settings.ic_threshold,
        )
        chosen = decision["chosen"]
        if self.skills_enabled and chosen["kind"] in ("refine", "challenge"):
            parent = observed[chosen["parent_formula"]]
            skill_context = self._skill_context(parent)
            recipes = (self.skills.recipes if chosen["kind"] == "refine"
                       else (("delay_1",) if self.config.delay_bars == 1 else ()))
            variants = recipe_variants(chosen["parent_formula"], recipes)
            if chosen["kind"] == "refine":
                variants = [v for v in variants if v["formula"] not in seen]
            if variants:
                selected, guidance = self.loop.memory_policy.select_research_variant(
                    variants, context=skill_context, sequence=decision["sequence"], seed=self.config.seed)
                chosen.update(formula=selected["formula"], recipe_id=selected["recipe_id"])
                decision["skill_context"] = skill_context
                decision["skill_selection"] = guidance
                decision["joint_selection_probability"] = decision["selection_probability"] * guidance["conditional_probability"]
                # Kind-level value is the same for these equal-cost variants.
                for estimate in decision["estimates"]:
                    if estimate["action"]["kind"] == chosen["kind"]:
                        estimate["action"] = dict(chosen)
        self.ledger.begin(decision)
        self.active = decision
        self.started_at = time.monotonic()
        self.evaluations = 0
        self.library_before = self.library_utility()
        payload.research_action = decision
        payload.stage_metrics["research_action"] = {
            "sequence": decision["sequence"], "kind": decision["chosen"]["kind"],
            "selection_probability": decision["selection_probability"],
            "rationale": decision["rationale"],
        }
        return decision["chosen"]["kind"]

    def _skill_context(self, parent: dict) -> dict:
        data_config = getattr(self.loop.config, "data", None)
        scope = {"market": getattr(data_config, "market", "unknown"),
                 "frequency": getattr(data_config, "frequency", "unknown"),
                 "targets": self.loop.protocol.runtime_contract()["targets"],
                 "default_target": self.loop.protocol.default_target,
                 "signal_failure_policy": self.loop.settings.signal_failure_policy,
                 "delay_test": {"bars": self.config.delay_bars, "retention": self.config.delay_retention,
                                "ic_threshold": self.loop.settings.ic_threshold},
                 "quality_metric": "absolute mean cross-sectional Spearman IC"}
        quality = parent["quality"]
        profile = parent.get("profile") or signal_profile(None)
        return {**profile, "scope": digest(scope), "scope_description": scope,
                "parent_formula": parent["formula"], "parent_quality": quality,
                "quality_band": "low" if quality < 0.04 else ("medium" if quality < 0.12 else "high"),
                "quality_scope": parent.get("quality_scope", "unknown"),
                "recipe_version": RECIPE_VERSION, "ic_threshold": self.loop.settings.ic_threshold}

    def refine(self, payload: Any) -> None:
        action = payload.research_action["chosen"]
        payload.candidates = [(f"refinement_{payload.research_action['sequence']}", action["formula"])]

    def record_evaluation_dispatch(self, payload: Any) -> None:
        """Record exact evaluated candidates, including failed/duplicate attempts."""
        self.evaluations += len(payload.candidates)
        for name, formula in payload.candidates:
            if try_parse(formula) is not None:
                self._contact(name, formula, payload.iteration, "dispatched")

    def _contact(self, name: str, formula: str, iteration: int, status: str,
                 ic_series: np.ndarray | None = None) -> None:
        self.loop.trial_ledger.record_data_contact(
            factor_name=name, formula=formula, dataset_id=self.loop.trial_dataset_id,
            target_name=self.loop.dataset_contract.default_target, iteration=iteration,
            stage="research_action", status=status, ic_series=ic_series,
            metadata={"action_sequence": self.active["sequence"],
                      "action": self.active["chosen"]["kind"],
                      "research_campaign_id": self.ledger.campaign_id},
        )

    def challenge(self, payload: Any) -> None:
        """Recompute parent and delayed signals on a common finite observation mask.

        The parent's direction is frozen for the delayed comparison. A sign flip
        cannot masquerade as retained predictive power. This is an advisory
        discovery-data stress test, never a library-admission operation.
        """
        action = payload.research_action["chosen"]
        kernel = self.loop.evaluation_kernel
        data = kernel.build_data_dict(self.loop.data_tensor, get_features())
        panels = []
        for formula in (action["parent_formula"], action["formula"]):
            self._contact("delay_challenge", formula, payload.iteration, "dispatched")
            self.evaluations += 1
            _, signals = kernel.compute_signals(
                formula=formula, data_dict=data, returns_shape=self.loop.returns.shape,
                signal_failure_policy="raise",
            )
            panels.append(signals)
        parent, delayed = panels
        mask = np.isfinite(parent) & np.isfinite(delayed) & np.isfinite(self.loop.returns)
        target = np.where(mask, self.loop.returns, np.nan)
        original_ic = compute_rank_ic(np.where(mask, parent, np.nan), target)
        delayed_ic = compute_rank_ic(np.where(mask, delayed, np.nan), target)
        usable = np.isfinite(original_ic) & np.isfinite(delayed_ic)
        for formula, series in zip((action["parent_formula"], action["formula"]),
                                   (original_ic, delayed_ic), strict=True):
            self._contact("delay_challenge", formula, payload.iteration, "measured", series)
        measured = {"parent_formula": action["parent_formula"], "formula": action["formula"],
                    "delay_bars": self.config.delay_bars, "paired_periods": int(usable.sum()),
                    "retention_threshold": self.config.delay_retention,
                    "evidence_scope": "supplied discovery panel; no independent confirmation",
                    "admission_changed": False}
        if usable.sum() < 3:
            measured.update(status="inconclusive", reason="Fewer than three paired IC observations")
        else:
            original_mean = float(original_ic[usable].mean())
            aligned_mean = float(delayed_ic[usable].mean()) * (1 if original_mean >= 0 else -1)
            original_quality = abs(original_mean)
            measured.update(parent_signed_ic=original_mean, delayed_aligned_ic=aligned_mean)
            if original_quality < max(self.loop.settings.ic_threshold, 1e-12):
                measured.update(status="inconclusive", reason="Parent fails quality gate on paired support")
            else:
                ratio = aligned_mean / original_quality
                passed = ratio >= self.config.delay_retention and aligned_mean >= self.loop.settings.ic_threshold
                measured.update(status="measured", retention_ratio=ratio, passed=bool(passed))
        payload.stage_metrics["research_challenge"] = measured

    def finish(self, payload: Any, elapsed: float) -> None:
        if self.active is None:
            return
        chosen = self.active["chosen"]
        candidates = [{
            "name": r.factor_name, "formula": r.formula, "parse_ok": bool(r.parse_ok),
            "quality": finite_float(r.ic_paper_mean), "signed_ic": finite_float(r.ic_mean),
            "admitted": bool(r.admitted), "rejection_reason": r.rejection_reason,
            "parent_formula": r.parent_formula,
            **({"profile": signal_profile(r.signals),
                "quality_scope": "full" if r.target_stats else "fast_or_unmeasured"}
               if self.skills_enabled else {}),
        } for r in payload.results]
        challenge = payload.stage_metrics.get("research_challenge", {})
        decision_changed = None
        if challenge.get("status") == "measured":
            estimate = next(row for row in self.active["estimates"] if row["action"]["kind"] == "challenge")
            decision_changed = (estimate["success_probability"] >= 0.5) != challenge["passed"]
        outcome = {
            "candidates": candidates, "challenge": challenge,
            "library_gain": self.library_utility() - self.library_before,
            "library_utility_before": self.library_before, "library_utility_after": self.library_utility(),
            "decision_changed": decision_changed, "evaluations": self.evaluations,
            "elapsed_seconds": max(float(elapsed), 0.0),
            "stop_reason": self.active["rationale"] if chosen["kind"] == "stop" else None,
        }
        self.ledger.finish(self.active["sequence"], outcome, snapshot=self.checkpoint_snapshot())
        payload.stage_metrics["research_evaluations"] = self.evaluations
        payload.stage_metrics["research_stop"] = chosen["kind"] == "stop"
        self.active = None
        self.ledger.export()
        self.loop.memory_policy.sync_research_history(
            self.ledger.records(), dataset_id=self.loop.trial_dataset_id, campaign_id=self.ledger.campaign_id)

    def fail(self, exc: BaseException) -> None:
        if self.active is None:
            return
        self.ledger.finish(self.active["sequence"], {
            "error_type": type(exc).__name__, "evaluations": self.evaluations,
            "elapsed_seconds": max(time.monotonic() - self.started_at, 0),
            "reason": "Action did not complete; partial work is retained without a success claim",
        }, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            snapshot=self.checkpoint_snapshot())
        self.active = None
        self.ledger.export()


def build_research_action_service(loop: Any) -> ResearchActionService | None:
    config = getattr(loop.settings.research, "planner", None)
    if config is None or not config.enabled:
        return None
    return ResearchActionService(loop, config)


def discovery_dataset(dataset: Any) -> Any:
    """Provide CLI action research only the configured, horizon-purged train panel."""
    from factorminer.evaluation.runtime import build_runtime_dataset_from_arrays

    indices = dataset.get_split("train").indices
    purge = max((spec.entry_delay_bars + spec.holding_bars
                 for spec in dataset.target_specs.values()), default=1)
    if len(indices) <= purge + 3 or np.any(np.diff(indices) != 1):
        raise ValueError("Research actions require a contiguous train split with enough periods after target purging")
    indices = indices[:-purge]
    return build_runtime_dataset_from_arrays(
        {name: panel[:, indices] for name, panel in dataset.data_dict.items()},
        dataset.returns[:, indices],
        target_panels={name: panel[:, indices] for name, panel in dataset.target_panels.items()},
        target_specs=dataset.target_specs, default_target=dataset.default_target,
        split_indices={"train": np.arange(len(indices))},
        timestamps=dataset.timestamps[indices], asset_ids=dataset.asset_ids,
    )
