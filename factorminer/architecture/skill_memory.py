"""MemoryPolicy integration for frozen procedures and campaign-local feedback."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from factorminer.architecture.memory_policy import (
    EditAwareMemoryPolicy,
    MemoryPolicy,
    extract_edit_motif,
)
from factorminer.architecture.research_skills import (
    ResearchSkillsConfig,
    context_key,
    digest,
    load_skill_pack,
    observation_from_action,
    summarize_skill,
    write_skill_pack,
)
from factorminer.memory.memory_store import ExperienceMemory


class TransferableSkillMemoryPolicy(MemoryPolicy):
    """Compose procedure retrieval with an existing memory policy.

    Imported observations stay frozen. Local contradictions reduce the influence
    of imported evidence in their own applicability cell; exploration stays open.
    """

    def __init__(self, inner: MemoryPolicy, config: ResearchSkillsConfig, protocol: Any, *, output_dir=None):
        config.validate()
        self.inner, self.config, self.protocol = inner, config, protocol
        self.pack = None
        frozen_path = Path(output_dir)/"research_skill_pack.json" if output_dir else None
        if config.source:
            if frozen_path is not None and frozen_path.exists():
                self.pack = load_skill_pack(frozen_path)
                if Path(config.source).exists() and load_skill_pack(config.source)["pack_id"] != self.pack["pack_id"]:
                    raise ValueError("Skill source changed during a frozen campaign; use a new output directory")
            else:
                self.pack = load_skill_pack(config.source)
        self.local: dict[str, dict] = {}
        self.dataset_id = ""
        self.last_guidance: dict[str, Any] = {}

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def transfer_identity(self) -> dict:
        return {"config": {k: v for k, v in asdict(self.config).items() if k != "source"},
                "pack_id": self.pack["pack_id"] if self.pack else None}

    def persist_research_skills(self, output_dir):
        if self.pack is not None:
            write_skill_pack(self.pack, Path(output_dir)/"research_skill_pack.json")

    def schema(self) -> dict:
        return {**self.inner.schema(), "research_skills": self.transfer_identity(),
                "local_procedures": len(self.local)}

    def retrieve(self, memory, *, library_state):
        signal = self.inner.retrieve(memory, library_state=library_state)
        signal["research_skills"] = {**self.transfer_identity(), "local_procedures": len(self.local)}
        return signal

    def form(self, memory, trajectory, *, iteration):
        return self.inner.form(memory, trajectory, iteration=iteration)

    def evolve(self, memory, formed):
        return self.inner.evolve(memory, formed)

    def serialize(self, memory):
        payload = self.inner.serialize(memory)
        payload["transferable_skills"] = {"identity": self.transfer_identity(),
                                         "local": list(self.local.values()), "dataset_id": self.dataset_id}
        return payload

    def restore(self, payload):
        state = payload.get("transferable_skills")
        if state:
            if digest(state["identity"]) != digest(self.transfer_identity()):
                raise ValueError("Frozen skill configuration changed since the checkpoint")
            self.local = {r["event_id"]: r for r in state["local"]}
            self.dataset_id = state["dataset_id"]
        return self.inner.restore(payload)

    def sync_research_history(self, records, *, dataset_id, campaign_id):
        self.dataset_id = dataset_id
        # Rebuild from the authoritative committed ledger, including removal of
        # any uncommitted in-memory observation after an interrupted transaction.
        self.local = {}
        for record in records:
            row = observation_from_action(record, dataset_id=dataset_id, campaign_id=campaign_id)
            if row is not None:
                self.local[row["event_id"]] = row

    def _source_rows(self, context, recipe=None):
        return [r for r in (self.pack or {}).get("observations", [])
                if r["dataset_id"] != self.dataset_id
                and r["context"]["scope"] == context["scope"]
                and (recipe is None or r["recipe_id"] == recipe)]

    def research_action_prior(self, kind, *, context):
        if self.config.mode != "structured" or kind != "challenge":
            return {}
        if context.get("persistence") == "unknown" or context.get("scale") == "unknown":
            return {}
        rows = [r for r in self._source_rows(context, "delay_1")
                if r["context_key"] == context_key(context) and r["status"] == "measured"]
        datasets = sorted({r["dataset_id"] for r in rows})
        if len(datasets) < self.config.minimum_datasets:
            return {}
        passed = [np.mean([r["passed"] for r in rows if r["dataset_id"] == dataset]) for dataset in datasets]
        _, review = self._structured("delay_1", context)
        weight = review.get("source_influence", 1.0)
        return {"alpha_add": float(sum(passed)*weight),
                "beta_add": float((len(passed)-sum(passed))*weight),
                "scope": context_key(context), "source_datasets": len(datasets),
                "pack_id": self.pack["pack_id"], "interpretation": "prior over discovery delay-test pass, not future validity"}

    def _structured(self, recipe, context):
        if context.get("persistence") == "unknown" or context.get("scale") == "unknown":
            return 0.0, {"status": "out_of_scope", "source_datasets": 0}
        key = context_key(context)
        source = [r for r in self._source_rows(context, recipe) if r["context_key"] == key]
        if not source:
            return 0.0, {"status": "out_of_scope", "source_datasets": 0}
        summary = summarize_skill(source, self.config)
        source_score = 0.0
        if summary["status"] == "supported":
            source_score = summary["interval"][0]
        elif summary["status"] == "discouraged":
            source_score = summary["interval"][1]
        local = [r for r in self.local.values() if r["context_key"] == key and r["recipe_id"] == recipe
                 and r["status"] == "measured"]
        contradictions = [r for r in local if
                          (source_score > 0 and r["reward"] <= self.config.minimum_effect)
                          or (source_score < 0 and r["reward"] >= -self.config.minimum_effect)]
        trust = 1 / (1 + len(contradictions))
        local_effect = float(np.mean([r["reward"] for r in local])) if local else 0.0
        score = trust * source_score + (1-trust) * local_effect
        return score, {"status": "under_review" if contradictions else summary["status"],
                       "skill_id": summary["skill_id"], "source_datasets": summary["datasets"],
                       "mean_effect": summary["mean_effect"], "interval": summary["interval"],
                       "negative_transfer_observations": len(contradictions),
                       "contradictory_event_ids": [r["event_id"] for r in contradictions],
                       "source_influence": trust, "local_effect": local_effect}

    def _trajectory(self, recipe, context):
        rows = self._source_rows(context, recipe) + [r for r in self.local.values()
                if r["recipe_id"] == recipe and r["context"]["scope"] == context["scope"]]
        rows = [r for r in rows if r["status"] == "measured"]

        def distance(row):
            total = 0.0
            for key, weight in (("lag1", 1.0), ("scale_cv", 0.25), ("parent_quality", 1.0)):
                left, right = context.get(key), row["context"].get(key)
                if left is None or right is None:
                    return float("inf")
                total += weight * abs(left-right)
            return total

        nearest = [r for r in sorted(rows, key=lambda r: (distance(r), r["event_id"]))[:5]
                   if np.isfinite(distance(r))]
        score = float(np.mean([r["reward"] for r in nearest])) if nearest else 0.0
        return score, {"status": "retrieved" if nearest else "out_of_scope",
                       "event_ids": [r["event_id"] for r in nearest]}

    def _motif(self, variants, context):
        policy = EditAwareMemoryPolicy(self.protocol)
        rows = self._source_rows(context) + [r for r in self.local.values() if r["context"]["scope"] == context["scope"]]
        trajectory = [{"formula": r["formula"], "parent_formula": r["parent_formula"],
                       "parent_ic_paper_mean": r["parent_quality"], "ic_paper_mean": r["child_quality"]}
                      for r in rows if r["kind"] == "refine" and r["status"] == "measured"]
        policy.form(ExperienceMemory(), trajectory, iteration=0)
        scores = []
        for variant in variants:
            score = policy.score_action(context["parent_formula"],
                                        extract_edit_motif(context["parent_formula"], variant["formula"]),
                                        parent_quality=context["parent_quality"])
            scores.append((score.adjusted_score, {"status": "motif", "confidence": score.confidence,
                                                 "vetoed": score.vetoed, "residual": score.residual}))
        return scores

    def select_research_variant(self, variants, *, context, sequence, seed):
        mode = self.config.mode
        if mode == "motif" and variants[0]["kind"] == "refine":
            scored = self._motif(variants, context)
        else:
            scored = []
            for variant in variants:
                if mode == "structured":
                    scored.append(self._structured(variant["recipe_id"], context))
                elif mode == "trajectory":
                    scored.append(self._trajectory(variant["recipe_id"], context))
                else:
                    scored.append((0.0, {"status": "no_transfer"}))
        scores = np.asarray([s[0] for s in scored])
        probabilities = np.exp((scores-scores.max()) / self.config.temperature)
        probabilities /= probabilities.sum()
        probabilities = ((1-self.config.exploration_probability)*probabilities
                         + self.config.exploration_probability/len(variants))
        rng = np.random.default_rng(np.random.SeedSequence([seed, sequence, 271828]))
        chosen = int(rng.choice(len(variants), p=probabilities))
        self.last_guidance = {
            "mode": mode, "pack_id": self.pack["pack_id"] if self.pack else None,
            "variants": [{**v, "score": float(scores[i]), "probability": float(probabilities[i]),
                          "evidence": scored[i][1]} for i, v in enumerate(variants)],
            "selected_recipe": variants[chosen]["recipe_id"],
            "conditional_probability": float(probabilities[chosen]),
        }
        return variants[chosen], self.last_guidance
