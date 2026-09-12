"""Research-action decisions and durable, quota-free campaign accounting.

The utility model is deliberately narrow: excess discovery IC for additions,
and avoided decision loss for a specified delay stress. It is not a model of
future profitability. The ledger measures work; it never imposes resource caps.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ACTION_SCHEMA = "research-actions-v1"
POLICIES = ("decision_value", "heuristic", "random", "contextual_bandit")


@dataclass
class ResearchPlannerConfig:
    """Opt-in finite-action policy; costs influence choices, never impose quotas."""

    enabled: bool = False
    policy: str = "decision_value"
    seed: int = 42
    exploration_probability: float = 0.1
    prior_success: float = 1.0
    prior_failure: float = 3.0
    prior_gain: float = 0.02
    evaluation_cost: float = 0.0005
    generation_cost: float = 0.001
    refinement_window: int = 3
    delay_bars: int = 1
    delay_retention: float = 0.5

    def validate(self) -> None:
        if self.policy not in POLICIES:
            raise ValueError(f"research.planner.policy must be one of {POLICIES}")
        for name in ("seed", "refinement_window", "delay_bars"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"research.planner.{name} must be an integer")
        if self.seed < 0:
            raise ValueError("research.planner.seed must be nonnegative")
        if not 2 <= self.refinement_window <= 60 or not 1 <= self.delay_bars <= 60:
            raise ValueError("refinement_window must be 2..60 and delay_bars 1..60")
        for name in ("exploration_probability", "prior_success", "prior_failure", "prior_gain",
                     "evaluation_cost", "generation_cost", "delay_retention"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"research.planner.{name} must be finite")
        if not 0 <= self.exploration_probability <= 1 or not 0 <= self.delay_retention <= 1:
            raise ValueError("exploration_probability and delay_retention must be in [0, 1]")
        if min(self.prior_success, self.prior_failure, self.prior_gain) <= 0:
            raise ValueError("research planner priors must be positive")
        if min(self.evaluation_cost, self.generation_cost) < 0:
            raise ValueError("research planner utility costs must be nonnegative")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ResearchAction:
    """One executable action, with exact parent and expected evaluation count."""

    kind: str
    evaluations: int
    parent_formula: str = ""
    formula: str = ""
    parent_quality: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("generate", "refine", "challenge", "stop"):
            raise ValueError(f"Unknown research action: {self.kind}")
        if isinstance(self.evaluations, bool) or not isinstance(self.evaluations, int):
            raise ValueError("Action evaluations must be an integer")
        expected = {"refine": 1, "challenge": 2, "stop": 0}.get(self.kind)
        if self.evaluations < 0 or (expected is not None and self.evaluations != expected):
            raise ValueError("Incorrect evaluation count for action")
        if self.kind == "generate" and self.evaluations < 1:
            raise ValueError("Generation requires a positive batch size")
        if self.kind in ("refine", "challenge") and not (self.parent_formula and self.formula):
            raise ValueError("Refinement and challenge require exact parent and child formulas")
        if not math.isfinite(self.parent_quality) or self.parent_quality < 0:
            raise ValueError("Parent quality must be finite and nonnegative")


class ResearchActionPlanner:
    """Select actions using explicit, inspectable one-step decision estimates.

    Generate/refine use a beta admission model and observed positive library
    gains. Challenge value is the Bayes risk of a binary *discovery-panel*
    retention decision under a perfect observation of that particular test.
    This is not uncertainty reduction about unseen market returns.
    """

    def __init__(self, config: ResearchPlannerConfig) -> None:
        config.validate()
        self.config = config

    def _posterior(self, kind: str, context: str, records: list[dict]) -> tuple[float, float, float]:
        cfg = self.config
        alpha, beta = (1.0, 1.0) if kind == "challenge" else (cfg.prior_success, cfg.prior_failure)
        gains = []
        for record in records:
            decision, outcome = record["decision"], record.get("outcome")
            if record["status"] != "completed" or not outcome:
                continue
            if decision["chosen"]["kind"] != kind or decision["context"]["bucket"] != context:
                continue
            if kind == "challenge":
                challenge = outcome.get("challenge", {})
                if challenge.get("status") != "measured":
                    continue
                success = float(challenge["passed"])
                alpha += success
                beta += 1 - success
            else:
                observations = outcome.get("candidates", [])
                # Empty generations count as failed proposals, rather than disappearing.
                attempts = max(len(observations), decision["chosen"]["evaluations"])
                successes = sum(bool(row["admitted"]) for row in observations)
                alpha += successes
                beta += attempts - successes
                gain = float(outcome.get("library_gain", 0.0))
                if successes and gain > 0:
                    gains.append(gain / successes)
        mean_gain = (cfg.prior_gain + sum(gains)) / (1 + len(gains))
        return alpha, beta, mean_gain

    def plan(
        self, *, offers: list[ResearchAction], records: list[dict], context: dict[str, Any],
        iteration: int, quality_threshold: float,
    ) -> dict[str, Any]:
        if len({offer.kind for offer in offers}) != len(offers):
            raise ValueError("Exactly one offer per action kind is supported")
        if not offers or offers[-1].kind != "stop":
            raise ValueError("Every action menu must end with stop")
        cfg = self.config
        sequence = len(records) + 1
        # Separate, reproducible streams for policy selection and posterior samples.
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, sequence]))
        estimates = []
        for offer in offers:
            a, b, gain = self._posterior(offer.kind, context["bucket"], records)
            p = a / (a + b)
            cost = offer.evaluations * cfg.evaluation_cost
            if offer.kind == "generate":
                cost += cfg.generation_cost
            if offer.kind == "stop":
                benefit, cost = 0.0, 0.0
            elif offer.kind == "challenge":
                benefit = min(p, 1 - p) * max(offer.parent_quality - quality_threshold, 0)
            else:
                benefit = offer.evaluations * p * gain
            estimates.append({
                "action": asdict(offer), "expected_benefit": benefit,
                "utility_cost": cost, "net_value": benefit - cost,
                "success_probability": p, "posterior_alpha": a, "posterior_beta": b,
                "positive_gain_mean": gain,
            })

        probabilities = np.zeros(len(offers), dtype=float)
        active = [i for i, offer in enumerate(offers) if offer.kind != "stop"]
        policy = cfg.policy
        if not active:
            probabilities[-1] = 1
            reason = "No executable research actions remain."
        elif policy == "heuristic":
            index = next((i for i in active if offers[i].kind == "generate"), len(offers) - 1)
            probabilities[index] = 1
            reason = "Existing generate/evaluate cycle with its family-routing heuristic."
        elif policy == "random":
            probabilities[active] = 1 / len(active)
            reason = "Uniform allocation over executable research actions."
        elif policy == "contextual_bandit":
            # Estimate Thompson selection probabilities, then draw from that exact
            # finite-mixture policy. Log these probabilities, not a sampled one-hot.
            scores = []
            for i in active:
                row = estimates[i]
                draws = rng.beta(row["posterior_alpha"], row["posterior_beta"], size=512)
                scale = row["positive_gain_mean"] * offers[i].evaluations
                if offers[i].kind == "challenge":
                    draws = np.minimum(draws, 1 - draws)
                    scale = max(offers[i].parent_quality - quality_threshold, 0)
                scores.append(draws * scale - row["utility_cost"])
            winners = np.argmax(np.asarray(scores), axis=0)
            for j, i in enumerate(active):
                probabilities[i] = np.mean(winners == j)
            probabilities[active] *= 1 - cfg.exploration_probability
            probabilities[active] += cfg.exploration_probability / len(active)
            reason = "Contextual posterior sampling with an explicitly logged finite-mixture policy."
        else:
            best = int(np.argmax([row["net_value"] for row in estimates]))
            if estimates[best]["net_value"] <= 0:
                probabilities[-1] = 1
                reason = "No available action has positive estimated net decision value."
            else:
                probabilities[active] = cfg.exploration_probability / len(active)
                probabilities[best] += 1 - cfg.exploration_probability
                reason = "One-step estimated decision value, with declared randomized exploration."
        chosen = int(rng.choice(len(offers), p=probabilities))
        return {
            "schema_version": ACTION_SCHEMA, "sequence": sequence, "iteration": iteration,
            "policy": policy, "seed": cfg.seed, "context": context,
            "chosen": asdict(offers[chosen]), "estimates": estimates,
            "probabilities": {offer.kind: float(probabilities[i]) for i, offer in enumerate(offers)},
            "selection_probability": float(probabilities[chosen]), "rationale": reason,
            "utility_model": "excess discovery IC plus resolved delay-test decision loss; not future utility",
            "config": asdict(cfg),
        }


class ResearchActionLedger:
    """Transactional decisions/outcomes; one immutable decision per dispatched action.

    A pending action is retained after interruption. Completed history is never
    erased to continue a campaign. This is an audit store, not a security or
    statistical independence boundary.
    """

    def __init__(self, output_dir: str | Path, identity: dict[str, Any]) -> None:
        self.path = Path(output_dir) / "research_actions.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        identity_json = canonical_json(identity)
        self.campaign_id = hashlib.sha256(identity_json.encode()).hexdigest()
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)")
            db.execute("""CREATE TABLE IF NOT EXISTS actions (
                sequence INTEGER PRIMARY KEY, decision TEXT NOT NULL,
                status TEXT NOT NULL, outcome TEXT)
            """)
            db.execute("CREATE TABLE IF NOT EXISTS recovery (id INTEGER PRIMARY KEY, snapshot BLOB NOT NULL)")
            row = db.execute("SELECT identity FROM metadata").fetchone()
            if row is None:
                db.execute("INSERT INTO metadata VALUES (?)", (identity_json,))
            elif row[0] != identity_json:
                raise ValueError("Research campaign identity changed; use a separate output directory")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def records(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT sequence, decision, status, outcome FROM actions ORDER BY sequence").fetchall()
        return [
            {"sequence": seq, "decision": json.loads(decision), "status": status,
             "outcome": json.loads(outcome) if outcome else None}
            for seq, decision, status, outcome in rows
        ]

    def begin(self, decision: dict[str, Any]) -> None:
        encoded = canonical_json(decision)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            last = db.execute("SELECT COALESCE(MAX(sequence), 0) FROM actions").fetchone()[0]
            if decision["sequence"] != last + 1:
                raise ValueError("Research decision used stale action history")
            if db.execute("SELECT 1 FROM actions WHERE status='running'").fetchone():
                raise RuntimeError("A research action is still running; resume its checkpoint first")
            db.execute("INSERT INTO actions VALUES (?, ?, 'running', NULL)", (last + 1, encoded))

    def finish(self, sequence: int, outcome: dict[str, Any], *, status: str = "completed",
               snapshot: bytes | None = None) -> None:
        if status not in ("completed", "failed", "interrupted"):
            raise ValueError("Invalid research action terminal status")
        encoded = canonical_json(outcome)
        with self._connect() as db:
            changed = db.execute(
                "UPDATE actions SET status=?, outcome=? WHERE sequence=? AND status='running'",
                (status, encoded, sequence),
            ).rowcount
            if changed != 1:
                raise ValueError("Research outcome already finalized or action does not exist")
            if snapshot is not None:
                # Outcome and latest recoverable state commit together.
                db.execute("INSERT OR REPLACE INTO recovery VALUES (1, ?)", (snapshot,))

    def snapshot(self) -> bytes | None:
        with self._connect() as db:
            row = db.execute("SELECT snapshot FROM recovery WHERE id=1").fetchone()
        return row[0] if row else None

    def recover_interrupted(self) -> None:
        """Mark unfinished dispatches unknown; never fabricate a successful outcome."""
        with self._connect() as db:
            db.execute(
                "UPDATE actions SET status='interrupted', outcome=? WHERE status='running'",
                (canonical_json({"reason": "Resumed after interruption; outcome unknown"}),),
            )

    def summary(self) -> dict[str, Any]:
        records = self.records()
        completed = [r for r in records if r["status"] == "completed"]
        return {
            "schema_version": ACTION_SCHEMA, "campaign_id": self.campaign_id,
            "actions": len(records), "completed": len(completed),
            "evaluations": sum((r.get("outcome") or {}).get("evaluations", 0) for r in records),
            "elapsed_seconds": sum((r.get("outcome") or {}).get("elapsed_seconds", 0.0) for r in records),
            "stopped": bool(records and records[-1]["decision"]["chosen"]["kind"] == "stop"),
            "path": str(self.path), "resource_quotas": False,
        }

    def export(self) -> Path:
        path = self.path.with_suffix(".jsonl")
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(canonical_json(record) + "\n" for record in self.records()))
        temporary.replace(path)
        return path
