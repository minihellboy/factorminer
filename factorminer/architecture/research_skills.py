"""Versioned empirical procedures compiled from auditable research actions.

Skills describe discovery-panel effects, not confirmed alpha. Dataset clusters,
rather than individual adaptive trials, supply uncertainty; interpretation
assumes independent source panels.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import t as student_t  # type: ignore[import-untyped]

from factorminer.architecture.research_actions import canonical_json
from factorminer.core.parser import try_parse

SKILL_SCHEMA = "research-skills-v1"
TRANSFER_MODES = ("none", "motif", "trajectory", "structured")
EDIT_RECIPES = ("smooth_3", "smooth_8", "difference_1", "rank_smooth_3")
RECIPE_VERSION = "research-recipes-v1"


@dataclass
class ResearchSkillsConfig:
    enabled: bool = False
    mode: str = "structured"
    source: str = ""
    minimum_datasets: int = 3
    minimum_effect: float = 0.002
    confidence_level: float = 0.9
    exploration_probability: float = 0.1
    temperature: float = 0.02
    recipes: list[str] = field(default_factory=lambda: list(EDIT_RECIPES))

    def validate(self) -> None:
        if self.mode not in TRANSFER_MODES:
            raise ValueError(f"research.skills.mode must be one of {TRANSFER_MODES}")
        if not isinstance(self.source, str):
            raise ValueError("research.skills.source must be a local path string")
        if isinstance(self.minimum_datasets, bool) or not isinstance(self.minimum_datasets, int) or self.minimum_datasets < 2:
            raise ValueError("research.skills.minimum_datasets must be an integer >= 2")
        for name in ("minimum_effect", "confidence_level", "exploration_probability", "temperature"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"research.skills.{name} must be finite")
        if self.minimum_effect < 0 or self.temperature <= 0:
            raise ValueError("minimum_effect must be nonnegative and temperature positive")
        if not 0.5 < self.confidence_level < 1 or not 0 < self.exploration_probability <= 1:
            raise ValueError("confidence_level must be in (0.5, 1), exploration_probability in (0, 1]")
        if not isinstance(self.recipes, list) or not self.recipes or len(set(self.recipes)) != len(self.recipes) or set(self.recipes) - set(EDIT_RECIPES):
            raise ValueError(f"research.skills.recipes must be distinct members of {EDIT_RECIPES}")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def recipe_formula(recipe: str, parent: str) -> str:
    templates = {
        "smooth_3": "Mean({parent}, 3)", "smooth_8": "Mean({parent}, 8)",
        "difference_1": "Delta({parent}, 1)", "rank_smooth_3": "Mean(CsRank({parent}), 3)",
        "delay_1": "Delay({parent}, 1)",
    }
    if recipe not in templates or try_parse(parent) is None:
        raise ValueError("Unknown recipe or invalid parent formula")
    child = templates[recipe].format(parent=parent)
    if try_parse(child) is None:
        raise ValueError("Recipe does not compile for this parent")
    return child


def recipe_variants(parent: str, recipes) -> list[dict]:
    variants = []
    for recipe in recipes:
        try:
            formula = recipe_formula(recipe, parent)
        except ValueError:
            continue
        variants.append({"recipe_id": recipe, "formula": formula,
                         "kind": "challenge" if recipe.startswith("delay_") else "refine"})
    return variants


def signal_profile(signals: np.ndarray | None) -> dict[str, Any]:
    """Target-free dynamics, computed from signals already evaluated by the kernel."""
    unknown = {"persistence": "unknown", "scale": "unknown", "lag1": None, "scale_cv": None}
    if signals is None or signals.ndim != 2 or signals.shape[1] < 8:
        return unknown
    left, right = signals[:, :-1], signals[:, 1:]
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(axis=1, keepdims=True)
    x = np.where(valid, left, 0.0)
    y = np.where(valid, right, 0.0)
    x = np.where(valid, x - x.sum(axis=1, keepdims=True) / np.maximum(count, 1), 0)
    y = np.where(valid, y - y.sum(axis=1, keepdims=True) / np.maximum(count, 1), 0)
    denominator = np.sqrt((x*x).sum(axis=1) * (y*y).sum(axis=1))
    usable = (count[:, 0] >= 6) & (denominator > 1e-12)
    if not usable.any():
        return unknown
    lag = float(np.median((x*y).sum(axis=1)[usable] / denominator[usable]))
    # Cross-sectional RMS around the contemporaneous mean; no target access.
    finite = np.isfinite(signals)
    counts = finite.sum(axis=0)
    values = np.where(finite, signals, 0)
    centers = values.sum(axis=0) / np.maximum(counts, 1)
    variance = (np.where(finite, signals-centers, 0)**2).sum(axis=0) / np.maximum(counts, 1)
    scales = np.sqrt(variance[counts >= 3])
    cv = float(scales.std() / scales.mean()) if scales.size and scales.mean() > 1e-12 else 0.0
    if not math.isfinite(lag) or not math.isfinite(cv):
        return unknown
    persistence = "reversing" if lag < -0.15 else ("short" if lag < 0.15 else ("moderate" if lag < 0.7 else "persistent"))
    return {"persistence": persistence, "scale": "variable" if cv > 0.75 else "stable",
            "lag1": lag, "scale_cv": cv}


def context_key(context: dict[str, Any]) -> str:
    # Formula family is deliberately not a precondition: behavior can transfer
    # across syntax. Target/timing scope remains exact and explicit.
    return digest({k: context[k] for k in ("scope", "persistence", "scale", "quality_band")})


def observation_from_action(record: dict, *, dataset_id: str, campaign_id: str) -> dict | None:
    if record.get("status") != "completed":
        return None
    decision, outcome = record["decision"], record.get("outcome") or {}
    chosen = decision["chosen"]
    recipe = chosen.get("recipe_id")
    context = decision.get("skill_context")
    if not recipe or not context or context.get("recipe_version") != RECIPE_VERSION:
        return None  # Legacy records lack measured applicability, never invent it.
    if chosen["kind"] not in ("refine", "challenge"):
        return None
    if recipe_formula(recipe, chosen["parent_formula"]) != chosen["formula"]:
        raise ValueError("Recorded recipe contradicts executable parent/child lineage")
    propensity = float(decision.get("joint_selection_probability", decision["selection_probability"]))
    if not math.isfinite(propensity) or not 0 < propensity <= 1:
        raise ValueError("Invalid recorded selection probability")
    row = {"event_id": digest([campaign_id, record["sequence"]]), "dataset_id": dataset_id,
           "campaign_id": campaign_id, "sequence": record["sequence"], "recipe_id": recipe,
           "kind": chosen["kind"], "parent_formula": chosen["parent_formula"],
           "formula": chosen["formula"], "parent_quality": chosen["parent_quality"],
           "context": context, "context_key": context_key(context), "propensity": propensity,
           "record_sha256": digest(record), "admitted": False}
    if chosen["kind"] == "challenge":
        measured = outcome.get("challenge", {})
        if measured.get("status") != "measured":
            row.update(status="inconclusive", reward=None)
        else:
            # Reward detecting fragility, not obtaining a flattering stress result.
            parent_quality = abs(measured["parent_signed_ic"])
            reward = float(not measured["passed"]) * max(parent_quality - context["ic_threshold"], 0)
            row.update(status="measured", reward=reward, passed=measured["passed"],
                       measured_parent_quality=parent_quality)
    else:
        candidates = [r for r in outcome.get("candidates", []) if r["formula"] == chosen["formula"]]
        if len(candidates) != 1:
            row.update(status="inconclusive", reward=None)
        elif not candidates[0]["parse_ok"]:
            row.update(status="implementation_failure", reward=None)
        elif context.get("quality_scope") != "full" or candidates[0].get("quality_scope") != "full":
            row.update(status="inconclusive", reward=None)
        else:
            child = candidates[0]
            row.update(status="measured", reward=float(child["quality"] - chosen["parent_quality"]),
                       child_quality=child["quality"], admitted=child["admitted"])
    if row["reward"] is not None and not math.isfinite(row["reward"]):
        raise ValueError("Nonfinite skill outcome")
    return row


def summarize_skill(observations: list[dict], config: ResearchSkillsConfig) -> dict:
    measured = [r for r in observations if r["status"] == "measured"]
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in measured:
        clusters[row["dataset_id"]].append(row["reward"])
    values = np.asarray([np.mean(v) for _, v in sorted(clusters.items())])
    mean = float(values.mean()) if values.size else 0.0
    low = high = None
    if len(values) >= 2:
        radius = float(student_t.ppf((1+config.confidence_level)/2, len(values)-1)
                       * values.std(ddof=1) / np.sqrt(len(values)))
        low, high = mean-radius, mean+radius
    status = "exploratory"
    if len(values) >= config.minimum_datasets and low is not None and high is not None:
        if low > config.minimum_effect:
            status = "supported"
        elif high < -config.minimum_effect:
            status = "discouraged"
    first = observations[0]
    return {"skill_id": digest([RECIPE_VERSION, first["recipe_id"], first["context_key"]]),
            "recipe_id": first["recipe_id"], "context_key": first["context_key"],
            "preconditions": {k: first["context"][k] for k in ("scope", "persistence", "scale", "quality_band")},
            "status": status, "mean_effect": mean, "interval": [low, high],
            "datasets": len(clusters), "observations": len(observations),
            "supporting": [r["event_id"] for r in measured if r["reward"] > config.minimum_effect],
            "contradictory": [r["event_id"] for r in measured if r["reward"] < -config.minimum_effect],
            "inconclusive": [r["event_id"] for r in observations if r["status"] != "measured"],
            "uncertainty_scope": "Student interval over independent-dataset mean effects; descriptive, unadjusted for selection"}


def compile_skill_pack(sources: Sequence[str | Path], *, config: ResearchSkillsConfig | None = None) -> dict:
    """Read terminal snapshots without mutating source ledgers; deduplicate events."""
    cfg = config or ResearchSkillsConfig()
    cfg.validate()
    observations: dict[str, dict] = {}
    manifests: dict[str, dict] = {}
    skipped: Counter[str] = Counter()
    for source in sources:
        path = Path(source)
        if path.is_dir():
            path = path / "research_actions.sqlite3"
        with closing(sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)) as db:
            db.execute("BEGIN")
            identity_json = db.execute("SELECT identity FROM metadata").fetchone()[0]
            identity = json.loads(identity_json)
            records = [{"sequence": n, "decision": json.loads(d), "status": s,
                        "outcome": json.loads(o) if o else None}
                       for n, d, s, o in db.execute("SELECT sequence,decision,status,outcome FROM actions ORDER BY sequence")]
        campaign = hashlib.sha256(identity_json.encode()).hexdigest()
        if any(r["status"] == "running" for r in records):
            raise ValueError("Freeze source campaigns after their in-flight action finishes")
        manifest = {"campaign_id": campaign, "dataset_id": identity["dataset_id"],
                    "records_sha256": digest(records), "actions": len(records),
                    "evaluations": sum((r.get("outcome") or {}).get("evaluations", 0) for r in records)}
        if campaign in manifests:
            if manifests[campaign] != manifest:
                raise ValueError("Conflicting snapshots of the same source campaign")
            continue
        manifests[campaign] = manifest
        for record in records:
            row = observation_from_action(record, dataset_id=identity["dataset_id"], campaign_id=campaign)
            if row is None:
                skipped["no_completed_scoped_procedure"] += 1
                continue
            observations[row["event_id"]] = row
    groups = defaultdict(list)
    for row in observations.values():
        groups[(row["recipe_id"], row["context_key"])].append(row)
    payload = {"schema_version": SKILL_SCHEMA, "recipe_version": RECIPE_VERSION,
               "source_campaigns": [manifests[k] for k in sorted(manifests)],
               "observations": [observations[k] for k in sorted(observations)],
               "skills": [summarize_skill(groups[k], cfg) for k in sorted(groups)],
               "construction": {"minimum_datasets": cfg.minimum_datasets,
                                "minimum_effect": cfg.minimum_effect, "confidence_level": cfg.confidence_level},
               "skipped": dict(skipped),
               "limitations": ["Discovery evidence, not confirmation or a causal estimate",
                               "Propensities retained; no unsupported off-policy estimate is reported",
                               "Distinct hashes do not prove independence; source data provenance remains the caller's responsibility"]}
    return {**payload, "pack_id": digest(payload)}


def write_skill_pack(pack: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(pack)+"\n"
    if target.exists():
        if target.read_text() != encoded:
            raise ValueError("Skill versions are immutable; use a new destination")
        return target
    with target.open("x") as handle:
        handle.write(encoded)
    return target


def load_skill_pack(path: str | Path) -> dict:
    pack = cast(dict[str, Any], json.loads(Path(path).read_text()))
    payload = {k: v for k, v in pack.items() if k != "pack_id"}
    if pack.get("schema_version") != SKILL_SCHEMA or pack.get("recipe_version") != RECIPE_VERSION or pack.get("pack_id") != digest(payload):
        raise ValueError("Invalid or modified skill pack")
    for row in pack["observations"]:
        if recipe_formula(row["recipe_id"], row["parent_formula"]) != row["formula"]:
            raise ValueError("Skill observation contradicts its recipe")
    return pack
