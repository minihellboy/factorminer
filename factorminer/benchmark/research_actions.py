"""Frozen, matched-evaluation comparisons of the real research-action loop.

Episode horizons belong to this benchmark design, not production mining.
The planner sees discovery data only. Final evaluation is performed after all
actions and recommendations have been frozen and is never fed back to it.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from factorminer.application.runtime_context import MiningRunContext
from factorminer.architecture.research_actions import POLICIES, canonical_json
from factorminer.architecture.stages import GenerateStage
from factorminer.core.ralph_loop import RalphLoop
from factorminer.core.types import get_features
from factorminer.evaluation.metrics import compute_rank_ic
from factorminer.utils.config import Config

BENCHMARK_VERSION = "research-actions-controlled-v1"
SCENARIOS = ("persistent_signal", "null", "transient_signal", "unannounced_shift")
CATALOG = (
    "$close", "$open", "$volume", "$high", "$low", "$vwap", "$amt", "$returns",
    "Mean($close, 3)", "Mean($open, 3)", "Mean($volume, 3)", "Delta($close, 3)",
    "CsRank($close)", "CsRank($open)", "Std($close, 5)", "Std($volume, 5)",
    "Mean($close, 5)", "Mean($open, 5)", "Sub($close, $open)", "Add($close, $open)",
    "Corr($close, $volume, 5)", "Delay($close, 1)", "Neg($volume)", "Mean($returns, 3)",
)


def controlled_panel(seed: int, scenario: str, assets: int = 24, periods: int = 240):
    """Gaussian diagnostic worlds, not simulated tradable market prices."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    rng = np.random.default_rng(seed)
    features = get_features()
    data = rng.normal(size=(assets, periods, len(features)))
    latent = rng.normal(size=(assets, periods))
    if scenario != "transient_signal":
        for t in range(1, periods):
            latent[:, t] = 0.9 * latent[:, t - 1] + 0.4 * latent[:, t]
    data[:, :, features.index("$close")] = latent + 0.3 * rng.normal(size=latent.shape)
    data[:, :, features.index("$open")] = latent + 1.5 * rng.normal(size=latent.shape)
    returns = latent + 0.8 * rng.normal(size=latent.shape)
    split = periods // 2
    if scenario == "null":
        returns = rng.normal(size=latent.shape)
    elif scenario == "unannounced_shift":
        # Delay stability cannot anticipate this independent future break.
        returns[:, split:] = rng.normal(size=returns[:, split:].shape)
    return data, returns, split


class _EpisodePlanner:
    """Apply a frozen experimental horizon equally to every policy's action menu."""

    def __init__(self, inner, loop: RalphLoop, horizon: int):
        self.inner = inner
        self.loop = loop
        self.horizon = horizon

    def plan(self, **kwargs):
        remaining = self.horizon - self.loop.research_actions.ledger.summary()["evaluations"]
        kwargs["offers"] = [a for a in kwargs["offers"] if a.evaluations <= remaining]
        kwargs["context"] = {**kwargs["context"], "benchmark_evaluations_remaining": remaining}
        return self.inner.plan(**kwargs)


def _frozen_quality(loop, data, returns, split: int) -> dict[str, Any]:
    records = loop.research_actions.ledger.records()
    rejected = {
        r["decision"]["chosen"]["parent_formula"]
        for r in records if (r.get("outcome") or {}).get("challenge", {}).get("passed") is False
    }
    data_dict = loop.evaluation_kernel.build_data_dict(data, get_features())
    rows = []
    for factor in loop.library.list_factors():
        _, signals = loop.evaluation_kernel.compute_signals(
            formula=factor.formula, data_dict=data_dict, returns_shape=returns.shape,
            signal_failure_policy="raise",
        )
        direction = 1 if factor.ic_mean >= 0 else -1
        original = compute_rank_ic(signals[:, split:], returns[:, split:]) * direction
        # Freeze original discovery direction for all stress and final metrics.
        lag = loop.research_actions.config.delay_bars
        delayed = np.full_like(signals, np.nan)
        delayed[:, lag:] = signals[:, :-lag]
        stressed = compute_rank_ic(delayed[:, split:], returns[:, split:]) * direction
        original = original[np.isfinite(original)]
        stressed = stressed[np.isfinite(stressed)]
        rows.append({
            "formula": factor.formula, "recommended": factor.formula not in rejected,
            "discovery_direction": direction,
            "heldout_signed_ic": float(original.mean()) if original.size else 0.0,
            "heldout_delay_signed_ic": float(stressed.mean()) if stressed.size else 0.0,
        })
    recommended = [r for r in rows if r["recommended"]]
    threshold = loop.settings.ic_threshold
    # Primary score values the declared delay-tolerant research claim. Also
    # publish the unstressed result to expose harm to genuinely short-lived alpha.
    return {
        "factor_results": rows,
        "recommended_count": len(recommended),
        "heldout_delay_utility": sum(r["heldout_delay_signed_ic"] - threshold for r in recommended),
        "heldout_unstressed_utility": sum(r["heldout_signed_ic"] - threshold for r in recommended),
        "raw_library_delay_utility": sum(r["heldout_delay_signed_ic"] - threshold for r in rows),
        "failed_delay_claims": sum(r["heldout_delay_signed_ic"] < threshold for r in recommended),
    }


def _episode_config(seed: int, policy: str) -> Config:
    cfg = Config()
    cfg.mining.batch_size = 2
    cfg.mining.target_library_size = 1000000
    cfg.mining.icir_threshold = 0.01
    cfg.mining.correlation_threshold = 0.9
    cfg.evaluation.num_workers = 1
    cfg.memory.policy = "none"
    cfg.research.planner.enabled = True
    cfg.research.planner.policy = policy
    cfg.research.planner.seed = seed
    return cfg


def compare_research_actions(
    data: np.ndarray, returns: np.ndarray, split: int, output_dir: str | Path, *,
    seeds: tuple[int, ...] = (0, 1, 2), evaluation_horizon: int = 24,
    catalog: tuple[str, ...] = CATALOG, label: str = "supplied_panel", purge_bars: int = 1,
) -> dict[str, Any]:
    """Run real-loop policies on identical data and a frozen proposal tape."""
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Benchmark seeds must be distinct nonnegative integers")
    if evaluation_horizon < 2 or not catalog:
        raise ValueError("Benchmark requires at least two evaluations and a nonempty catalog")
    if data.ndim != 3 or returns.shape != data.shape[:2] or not 4 < split < data.shape[1] - 3:
        raise ValueError("Invalid panel or discovery/held-out split")
    if not isinstance(purge_bars, int) or not 1 <= purge_bars < split - 3:
        raise ValueError("purge_bars must leave at least four discovery periods")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "protocol.json"
    if manifest_path.exists():
        raise ValueError("Benchmark output already exists; use a fresh directory to preserve the frozen run")
    digest = hashlib.sha256(np.ascontiguousarray(data).tobytes() + np.ascontiguousarray(returns).tobytes()).hexdigest()
    manifest = {
        "version": BENCHMARK_VERSION, "label": label, "data_sha256": digest,
        "shape": list(data.shape), "discovery_end_exclusive": split - purge_bars,
        "heldout_start": split, "purged_periods": purge_bars, "seeds": list(seeds),
        "evaluation_horizon": evaluation_horizon, "catalog": list(catalog), "policies": list(POLICIES),
        "base_config": _episode_config(seeds[0], POLICIES[0]).to_dict(),
        "per_episode_overrides": "research.planner.seed and policy from the declared Cartesian product",
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "primary_metric": "sum of direction-frozen held-out delayed IC minus admission threshold over recommendations",
        "secondary_metrics": ["unstressed utility", "raw library delay utility", "failed delay claims", "elapsed seconds"],
        "limitations": [
            "IC-based diagnostic utility, not portfolio returns or a profitability claim",
            "The primary metric explicitly targets delay-tolerant claims; transient alpha can be valuable under another objective",
            "A supplied historical panel is not independent prospective validation",
            "Final results are not exposed to the policy; no parameter tuning occurs inside this comparison",
            "Equal evaluation allowances do not imply equal CPU, token, or wall-time costs; actual costs are reported",
        ],
    }
    manifest["protocol_sha256"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    runs = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(catalog))
        tape = tuple(catalog[i] for i in order)
        for policy in POLICIES:
            cfg = _episode_config(seed, policy)
            run_dir = output / f"seed-{seed}" / policy
            loop = RalphLoop(cfg, data[:, :split - purge_bars], returns[:, :split - purge_bars],
                             checkpoint_interval=0, run_context=MiningRunContext(output_dir=run_dir))
            cursor = 0

            def generate(_loop, payload):
                nonlocal cursor
                candidates = [(f"frozen_{cursor + j}", tape[(cursor + j) % len(tape)])
                              for j in range(payload.batch_size)]
                cursor += len(candidates)
                return candidates

            loop.stages["generate"] = GenerateStage(generate)
            loop.research_actions.planner = _EpisodePlanner(loop.research_actions.planner, loop, evaluation_horizon)
            started = time.monotonic()
            while loop.research_actions.ledger.summary()["evaluations"] < evaluation_horizon:
                remaining = evaluation_horizon - loop.research_actions.ledger.summary()["evaluations"]
                loop.iteration += 1
                stats = loop._run_iteration(min(2, remaining))
                if stats.get("research_stop"):
                    break
            elapsed = time.monotonic() - started
            records = loop.research_actions.ledger.records()
            frozen = {"formulas": [f.formula for f in loop.library.list_factors()],
                      "actions": records, "config": cfg.to_dict()}
            (run_dir / "frozen_discovery.json").write_text(json.dumps(frozen, indent=2, allow_nan=False))
            row = {"seed": seed, "policy": policy, "evaluations": loop.research_actions.ledger.summary()["evaluations"],
                   "actions": dict(Counter(r["decision"]["chosen"]["kind"] for r in records)),
                   "elapsed_seconds": elapsed, "generated_candidates": cursor,
                   "planner_config": asdict(cfg.research.planner),
                   **_frozen_quality(loop, data, returns, split)}
            runs.append(row)
            (run_dir / "result.json").write_text(json.dumps(row, indent=2, allow_nan=False))
    summaries = {}
    for policy in POLICIES:
        rows = [r for r in runs if r["policy"] == policy]
        summaries[policy] = {metric: float(np.mean([r[metric] for r in rows])) for metric in (
            "heldout_delay_utility", "heldout_unstressed_utility", "failed_delay_claims", "evaluations", "elapsed_seconds")}
    comparisons = {}
    for baseline in POLICIES[1:]:
        differences = np.asarray([
            next(r["heldout_delay_utility"] for r in runs if r["seed"] == seed and r["policy"] == "decision_value")
            - next(r["heldout_delay_utility"] for r in runs if r["seed"] == seed and r["policy"] == baseline)
            for seed in seeds
        ])
        comparisons[baseline] = {"paired_differences": differences.tolist(), "mean_difference": float(differences.mean())}
        if len(differences) >= 2:
            draws = np.random.default_rng(20260912).choice(differences, size=(10000, len(differences)))
            interval = np.quantile(draws.mean(axis=1), [0.025, 0.975])
            comparisons[baseline]["paired_bootstrap_95_interval"] = interval.tolist()
        else:
            comparisons[baseline]["paired_bootstrap_95_interval"] = None
        comparisons[baseline]["uncertainty_scope"] = "search-seed variation conditional on this fixed panel"
    result = {"protocol": manifest, "summaries": summaries, "paired_comparisons": comparisons,
              "runs": runs, "default_policy_changed": False}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def run_research_action_benchmark(
    output_dir: str | Path, *, seeds: tuple[int, ...] = (0, 1, 2),
    evaluation_horizon: int = 24,
) -> dict[str, Any]:
    """Run the preregistered four-world diagnostic suite."""
    output = Path(output_dir)
    scenarios = {}
    for scenario in SCENARIOS:
        data, returns, split = controlled_panel(1729, scenario)
        result = compare_research_actions(data, returns, split, output / scenario,
                                          seeds=seeds, evaluation_horizon=evaluation_horizon,
                                          label=scenario)
        scenarios[scenario] = {"summaries": result["summaries"], "paired_comparisons": result["paired_comparisons"]}
    report = {"version": BENCHMARK_VERSION, "scenarios": scenarios,
              "default_policy_changed": False, "evidence_level": "controlled diagnostic; not market validation"}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    return report
