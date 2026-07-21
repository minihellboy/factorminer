"""Strict paper/research benchmark runners built on runtime recomputation."""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factorminer.benchmark.contracts import (
    BenchmarkManifest,
)
from factorminer.benchmark.datasets import (
    _cfg_with_overrides,
    _clone_cfg,
    _factors_from_entries,
    _get_baseline_entries,
    build_benchmark_library,
    load_benchmark_dataset,
)
from factorminer.benchmark.frozen_evaluation import (
    evaluate_frozen_set,
    select_frozen_top_k,
)
from factorminer.benchmark.mining_runtime import (
    RUNTIME_LOOP_BASELINES as RUNTIME_LOOP_BASELINES,
)
from factorminer.benchmark.mining_runtime import (
    _build_phase2_runtime_kwargs as _build_phase2_runtime_kwargs,
)
from factorminer.benchmark.mining_runtime import (
    _build_runtime_provider as _build_runtime_provider,
)
from factorminer.benchmark.mining_runtime import (
    _cfg_for_runtime_baseline as _cfg_for_runtime_baseline,
)
from factorminer.benchmark.mining_runtime import (
    _filter_dataclass_kwargs as _filter_dataclass_kwargs,
)
from factorminer.benchmark.mining_runtime import (
    _prepare_runtime_loop as _prepare_runtime_loop,
)
from factorminer.benchmark.mining_runtime import (
    _real_mining_loop_type as _real_mining_loop_type,
)
from factorminer.benchmark.mining_runtime import (
    _run_runtime_mining_loop as _run_runtime_mining_loop,
)
from factorminer.benchmark.mining_runtime import (
    _runtime_loop_provenance as _runtime_loop_provenance,
)
from factorminer.benchmark.mining_runtime import (
    _runtime_strategy_backends as _runtime_strategy_backends,
)
from factorminer.benchmark.provenance import (
    _baseline_kind as _baseline_kind,
)
from factorminer.benchmark.provenance import (
    _baseline_provenance as _baseline_provenance,
)
from factorminer.benchmark.provenance import (
    _catalog_provenance as _catalog_provenance,
)
from factorminer.benchmark.provenance import (
    _file_sha256 as _file_sha256,
)
from factorminer.benchmark.provenance import (
    _json_safe as _json_safe,
)
from factorminer.benchmark.provenance import (
    _json_summary as _json_summary,
)
from factorminer.benchmark.provenance import (
    _saved_library_provenance as _saved_library_provenance,
)
from factorminer.benchmark.provenance import (
    _session_summary as _session_summary,
)
from factorminer.benchmark.reporting import _ensure_dir, _save_manifest, _write_json
from factorminer.benchmark.runtime_contracts import (
    _benchmark_dataset_contract as _benchmark_dataset_contract,
)
from factorminer.benchmark.runtime_contracts import (
    _build_strategy_grid_contract as _build_strategy_grid_contract,
)
from factorminer.benchmark.runtime_contracts import (
    _build_stress_contract as _build_stress_contract,
)
from factorminer.benchmark.runtime_contracts import (
    _build_walk_forward_contract as _build_walk_forward_contract,
)
from factorminer.benchmark.runtime_contracts import (
    _runtime_manifest_value as _runtime_manifest_value,
)
from factorminer.benchmark.runtime_contracts import (
    _safe_len as _safe_len,
)
from factorminer.benchmark.runtime_contracts import (
    _strategy_ablation_raw_config as _strategy_ablation_raw_config,
)
from factorminer.benchmark.runtime_contracts import (
    build_benchmark_runtime_contract as build_benchmark_runtime_contract,
)
from factorminer.benchmark.speed import (
    SpeedBenchmark as SpeedBenchmark,
)
from factorminer.benchmark.speed import (
    _build_mock_data_dict as _build_mock_data_dict,
)
from factorminer.benchmark.statistics import (
    AblationResult,
    BenchmarkResult,
    MethodResult,
    StatisticalComparisonTests,
)
from factorminer.benchmark.statistics import (
    DMTestResult as DMTestResult,
)
from factorminer.benchmark.statistics import (
    OperatorSpeedResult as OperatorSpeedResult,
)
from factorminer.benchmark.statistics import (
    PipelineSpeedResult as PipelineSpeedResult,
)
from factorminer.evaluation.metrics import METRIC_VERSION
from factorminer.evaluation.runtime import (
    evaluate_factors,
)
from factorminer.operators.c_backend import backend_available as c_backend_available

logger = logging.getLogger(__name__)


def _mean_universe_metric(
    payload: dict[str, Any],
    metric_group: str,
    metric_name: str,
) -> float | None:
    values: list[float] = []
    for universe_payload in payload.get("universes", {}).values():
        group = universe_payload.get(metric_group, {})
        value = group.get(metric_name)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def run_table1_benchmark(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    baseline_names: list[str] | None = None,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
    use_runtime_loops: bool = False,
) -> dict:
    """Run the strict Top-K freeze benchmark across all configured universes."""
    if runtime_manifests is None:
        runtime_manifests = getattr(cfg.benchmark, "runtime_manifests", None)
    use_runtime_loops = bool(
        use_runtime_loops or getattr(cfg.benchmark, "runtime_loops", False) or runtime_manifests
    )
    benchmark_dir = _ensure_dir(output_dir / "benchmark" / "table1")
    baseline_names = baseline_names or list(cfg.benchmark.baselines)
    freeze_cfg = _cfg_with_overrides(cfg, cfg.benchmark.freeze_universe)
    freeze_dataset, freeze_hash = load_benchmark_dataset(
        freeze_cfg,
        data_path=data_path,
        raw_df=raw_df,
        universe=cfg.benchmark.freeze_universe,
        mock=mock,
    )
    freeze_dataset_contract = _benchmark_dataset_contract(freeze_cfg, freeze_dataset)

    summary: dict[str, dict] = {}
    for baseline in baseline_names:
        runtime_manifest = _runtime_manifest_value(runtime_manifests, baseline)
        runtime_contract = build_benchmark_runtime_contract(
            cfg,
            freeze_dataset_contract,
            baseline=baseline,
            runtime_manifest=runtime_manifest,
        )
        runtime_baseline = bool(runtime_manifest) or (
            use_runtime_loops
            and baseline in (RUNTIME_LOOP_BASELINES | {"factor_miner", "factor_miner_no_memory"})
        )

        if runtime_baseline:
            runtime_result = _run_runtime_mining_loop(
                cfg,
                baseline=baseline,
                dataset=freeze_dataset,
                output_dir=output_dir,
                runtime_manifest=runtime_manifest,
                mock=mock,
            )
            factors = list(runtime_result["library"].list_factors())
            provenance = runtime_result["provenance"]
            candidate_count = len(factors)
        else:
            entries = _get_baseline_entries(
                baseline,
                cfg.benchmark.seed,
                factor_miner_library_path=factor_miner_library_path,
                factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
            )
            factors = _factors_from_entries(entries)
            provenance = _baseline_provenance(
                baseline,
                factor_miner_library_path=factor_miner_library_path,
                factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
                candidate_count=len(entries),
                seed=cfg.benchmark.seed,
            )
            candidate_count = len(entries)

        artifacts = evaluate_factors(
            factors,
            freeze_dataset,
            signal_failure_policy="reject",
        )

        library_cfg = _cfg_with_overrides(cfg, cfg.benchmark.freeze_universe)
        if baseline == "factor_miner_no_memory":
            library_cfg.mining.ic_threshold = 0.02
            library_cfg.mining.correlation_threshold = 0.85
        library, library_stats = build_benchmark_library(
            artifacts,
            library_cfg,
            split_name="train",
            ic_threshold=library_cfg.mining.ic_threshold,
            correlation_threshold=library_cfg.mining.correlation_threshold,
        )
        frozen = select_frozen_top_k(
            artifacts,
            library,
            top_k=cfg.benchmark.freeze_top_k,
            split_name="train",
        )

        baseline_result = {
            "baseline": baseline,
            "mode": cfg.benchmark.mode,
            "metric_version": METRIC_VERSION,
            "freeze_universe": cfg.benchmark.freeze_universe,
            "candidate_count": candidate_count,
            "runtime_contract": runtime_contract.to_dict(),
            "paper_protocol": runtime_contract.paper_protocol,
            "walk_forward_contract": runtime_contract.walk_forward.to_dict(),
            "stress_contract": runtime_contract.stress.to_dict(),
            "strategy_grid_contract": runtime_contract.strategy_grid.to_dict(),
            "freeze_dataset_contract": dict(freeze_dataset_contract),
            "freeze_library_size": library.size,
            "freeze_stats": library_stats,
            "frozen_top_k": [
                {
                    "name": artifact.name,
                    "formula": artifact.formula,
                    "category": artifact.category,
                    "train_ic": artifact.split_stats["train"]["ic_paper_mean"],
                    "train_ic_mean": artifact.split_stats["train"]["ic_mean"],
                    "train_ic_abs_mean": artifact.split_stats["train"]["ic_abs_mean"],
                    "train_icir": artifact.split_stats["train"]["ic_paper_icir"],
                }
                for artifact in frozen
            ],
            "universes": {},
        }

        dataset_hashes = {cfg.benchmark.freeze_universe: freeze_hash}
        for universe in cfg.benchmark.report_universes:
            universe_cfg = _cfg_with_overrides(cfg, universe)
            dataset, dataset_hash = load_benchmark_dataset(
                universe_cfg,
                data_path=data_path,
                raw_df=raw_df,
                universe=universe,
                mock=mock,
            )
            dataset_hashes[universe] = dataset_hash
            baseline_result["universes"][universe] = evaluate_frozen_set(
                frozen,
                dataset,
                split_name="test",
                fit_split="train",
                cost_bps=list(runtime_contract.stress.cost_bps),
                capacity_levels=list(runtime_contract.stress.capacity_levels),
            )

        result_path = benchmark_dir / f"{baseline}.json"
        manifest_path = benchmark_dir / f"{baseline}_manifest.json"
        baseline_result["provenance"] = provenance
        _write_json(result_path, baseline_result)
        manifest = BenchmarkManifest(
            benchmark_name="table1",
            mode=cfg.benchmark.mode,
            seed=cfg.benchmark.seed,
            metric_version=METRIC_VERSION,
            baseline=baseline,
            freeze_universe=cfg.benchmark.freeze_universe,
            report_universes=list(cfg.benchmark.report_universes),
            train_period=list(cfg.data.train_period),
            test_period=list(cfg.data.test_period),
            freeze_top_k=cfg.benchmark.freeze_top_k,
            signal_failure_policy="reject",
            default_target=cfg.data.default_target,
            target_stack=[target.get("name", "") for target in cfg.data.targets],
            primary_objective=cfg.research.primary_objective,
            dataset_hashes=dataset_hashes,
            artifact_paths={
                "result": str(result_path),
                "manifest": str(manifest_path),
            },
            runtime_contract=runtime_contract.to_dict(),
            baseline_provenance={baseline: provenance},
            warnings=[],
        )
        _save_manifest(manifest_path, manifest)
        summary[baseline] = baseline_result

    return summary


def run_ablation_memory_benchmark(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Compare the default FactorMiner lane to the relaxed no-memory lane."""
    use_runtime_loops = bool(runtime_manifests or getattr(cfg.benchmark, "runtime_loops", False))
    comparison = run_table1_benchmark(
        cfg,
        output_dir,
        data_path=data_path,
        raw_df=raw_df,
        mock=mock,
        baseline_names=["factor_miner", "factor_miner_no_memory"],
        factor_miner_library_path=factor_miner_library_path,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
        runtime_manifests=runtime_manifests,
        use_runtime_loops=use_runtime_loops,
    )
    result = {}
    for baseline, payload in comparison.items():
        freeze_stats = payload["freeze_stats"]
        succeeded = max(freeze_stats.get("succeeded", 0), 1)
        result[baseline] = {
            "library_size": payload["freeze_library_size"],
            "high_quality_yield": freeze_stats.get("admitted", 0) / succeeded,
            "redundancy_rejection_rate": freeze_stats.get("correlation_rejections", 0) / succeeded,
            "replacements": freeze_stats.get("replaced", 0),
        }
    out_path = _ensure_dir(output_dir / "benchmark" / "ablation") / "memory_ablation.json"
    _write_json(out_path, result)
    return result


def run_ablation_strategy_benchmark(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    baseline: str | None = None,
    memory_policies: list[str] | None = None,
    dependence_metrics: list[str] | None = None,
    backends: list[str] | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Compare runtime loop variants across memory policy × dependence × backend."""
    strategy_cfg = _strategy_ablation_raw_config(cfg)
    baseline_name = str(baseline or strategy_cfg.get("baseline", "factor_miner"))
    memory_policies = list(
        memory_policies
        or strategy_cfg.get(
            "memory_policies",
            ["paper", "none", "kg", "family_aware", "regime_aware"],
        )
    )
    dependence_metrics = list(
        dependence_metrics
        or strategy_cfg.get(
            "dependence_metrics",
            ["spearman", "pearson", "distance_correlation"],
        )
    )
    backends = _runtime_strategy_backends(
        backends or strategy_cfg.get("backends"),
        default_backend=getattr(cfg.evaluation, "backend", "numpy"),
    )
    base_runtime_manifest = _runtime_manifest_value(runtime_manifests, baseline_name)
    freeze_cfg = _cfg_with_overrides(cfg, cfg.benchmark.freeze_universe)
    freeze_dataset, _ = load_benchmark_dataset(
        freeze_cfg,
        data_path=data_path,
        raw_df=raw_df,
        universe=cfg.benchmark.freeze_universe,
        mock=mock,
    )
    freeze_dataset_contract = _benchmark_dataset_contract(freeze_cfg, freeze_dataset)
    strategy_contract = _build_strategy_grid_contract(
        cfg,
        baseline=baseline_name,
        runtime_manifest={
            "memory_policies": memory_policies,
            "dependence_metrics": dependence_metrics,
            "backends": backends,
            **base_runtime_manifest,
        },
    )

    comparisons: list[dict[str, Any]] = []
    for memory_policy in memory_policies:
        for dependence_metric in dependence_metrics:
            for backend in backends:
                combo_name = f"{memory_policy}__{dependence_metric}__{backend}"
                combo_output_dir = (
                    output_dir / "benchmark" / "ablation" / "strategy_grid" / combo_name
                )
                combo_manifest = {
                    **base_runtime_manifest,
                    "memory_policy": memory_policy,
                    "redundancy_metric": dependence_metric,
                    "backend": backend,
                }
                combo_payload = run_table1_benchmark(
                    cfg,
                    combo_output_dir,
                    data_path=data_path,
                    raw_df=raw_df,
                    mock=mock,
                    baseline_names=[baseline_name],
                    runtime_manifests={baseline_name: combo_manifest},
                    use_runtime_loops=True,
                )[baseline_name]
                freeze_stats = combo_payload.get("freeze_stats", {})
                succeeded = max(int(freeze_stats.get("succeeded", 0)), 1)
                comparisons.append(
                    {
                        "name": combo_name,
                        "baseline": baseline_name,
                        "memory_policy": memory_policy,
                        "dependence_metric": dependence_metric,
                        "backend": backend,
                        "runtime_contract": build_benchmark_runtime_contract(
                            cfg,
                            freeze_dataset_contract,
                            baseline=baseline_name,
                            runtime_manifest=combo_manifest,
                        ).to_dict(),
                        "artifacts_root": str(combo_output_dir),
                        "freeze_library_size": int(combo_payload.get("freeze_library_size", 0)),
                        "freeze_stats": freeze_stats,
                        "high_quality_yield": float(freeze_stats.get("admitted", 0)) / succeeded,
                        "redundancy_rejection_rate": (
                            float(freeze_stats.get("correlation_rejections", 0)) / succeeded
                        ),
                        "mean_test_library_ic": _mean_universe_metric(
                            combo_payload,
                            "library",
                            "ic",
                        ),
                        "mean_test_library_icir": _mean_universe_metric(
                            combo_payload,
                            "library",
                            "icir",
                        ),
                        "mean_test_library_rho": _mean_universe_metric(
                            combo_payload,
                            "library",
                            "avg_abs_rho",
                        ),
                        "universes": combo_payload.get("universes", {}),
                        "provenance": combo_payload.get("provenance", {}),
                    }
                )

    leaderboard = sorted(
        comparisons,
        key=lambda item: (
            item["mean_test_library_ic"] is not None,
            item["mean_test_library_ic"] or float("-inf"),
            item["mean_test_library_icir"] or float("-inf"),
            -float(item["mean_test_library_rho"] or 1.0),
        ),
        reverse=True,
    )
    result = {
        "baseline": baseline_name,
        "memory_policies": memory_policies,
        "dependence_metrics": dependence_metrics,
        "backends": backends,
        "strategy_grid_contract": strategy_contract.to_dict(),
        "comparisons": comparisons,
        "leaderboard": leaderboard,
        "best": leaderboard[0] if leaderboard else None,
    }
    out_path = _ensure_dir(output_dir / "benchmark" / "ablation") / "strategy_grid.json"
    _write_json(out_path, result)
    return result


def run_cost_pressure_benchmark(
    cfg,
    output_dir: Path,
    *,
    baseline: str = "factor_miner",
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    factor_miner_library_path: str | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Run cost-pressure analysis for one baseline on the configured universes."""
    use_runtime_loops = bool(runtime_manifests or getattr(cfg.benchmark, "runtime_loops", False))
    payload = run_table1_benchmark(
        cfg,
        output_dir,
        data_path=data_path,
        raw_df=raw_df,
        mock=mock,
        baseline_names=[baseline],
        factor_miner_library_path=factor_miner_library_path,
        runtime_manifests=runtime_manifests,
        use_runtime_loops=use_runtime_loops,
    )[baseline]
    result = {
        universe: {
            "combinations": {
                name: {
                    "cost_pressure": metrics.get("cost_pressure", {}),
                    "capacity_pressure": metrics.get("capacity_pressure"),
                }
                for name, metrics in universe_payload["combinations"].items()
            }
        }
        for universe, universe_payload in payload["universes"].items()
    }
    out_path = _ensure_dir(output_dir / "benchmark" / "cost_pressure") / f"{baseline}.json"
    _write_json(out_path, result)
    return result


def run_cpcv_benchmark(
    cfg,
    *,
    data_path: str | None = None,
    mock: bool = False,
    baseline: str = "factor_miner",
) -> dict:
    """Run Combinatorial Purged CV + Probability of Backtest Overfitting diagnostics.

    Companion to `run_table1_benchmark`'s Top-K freeze evaluation: rather
    than one frozen train/test split, every candidate in `baseline`'s
    catalog is scored on every `CombinatorialPurgedCV` path built from the
    freeze universe, producing an ``(n_trials, n_paths)`` out-of-sample
    long-short performance matrix. That matrix drives
    `ProbabilityOfBacktestOverfitting` (Bailey, Borwein, Lopez de Prado &
    Zhu, 2017) and, for the best-performing trial's per-path OOS series,
    `DeflatedSharpeCalculator` -- the same paper's companion statistic,
    already implemented in `evaluation.significance`. Together they answer
    "is the *best* candidate's edge real, or an artifact of trying many
    candidates against a validation scheme that leaks information".

    Parameters
    ----------
    cfg
        Hierarchical benchmark config (see `factorminer.utils.config.Config`).
    data_path : str, optional
        Market data path; ignored when `mock=True`.
    mock : bool
        Use deterministic mock market data instead of `data_path`.
    baseline : str
        Candidate catalog id understood by `_get_baseline_entries`
        (e.g. ``"factor_miner"``, ``"alpha101_classic"``, ``"gplearn"``).

    Returns
    -------
    dict
        JSON-safe manifest with `cpcv_config`, `cpcv`, `pbo`, and
        `deflated_sharpe` sections plus best-trial provenance.
    """
    from factorminer.evaluation.cross_validation import (
        CombinatorialPurgedCV,
        CrossValidationConfig,
        ProbabilityOfBacktestOverfitting,
    )
    from factorminer.evaluation.portfolio import PortfolioBacktester
    from factorminer.evaluation.significance import (
        DeflatedSharpeCalculator,
        SignificanceConfig,
    )

    freeze_cfg = _cfg_with_overrides(cfg, cfg.benchmark.freeze_universe)
    dataset, dataset_hash = load_benchmark_dataset(
        freeze_cfg,
        data_path=data_path,
        universe=cfg.benchmark.freeze_universe,
        mock=mock,
    )

    entries = _get_baseline_entries(baseline, cfg.benchmark.seed)
    factors = _factors_from_entries(entries)
    artifacts = evaluate_factors(factors, dataset, signal_failure_policy="reject")
    succeeded = [artifact for artifact in artifacts if artifact.succeeded]

    cv_config = CrossValidationConfig()
    manifest: dict[str, Any] = {
        "benchmark_name": "cpcv",
        "baseline": baseline,
        "freeze_universe": cfg.benchmark.freeze_universe,
        "dataset_hash": dataset_hash,
        "candidate_count": len(factors),
        "succeeded_count": len(succeeded),
        "cpcv_config": asdict(cv_config),
        "cpcv": None,
        "pbo": None,
        "deflated_sharpe": None,
        "best_trial": None,
        "warnings": [],
    }

    if len(succeeded) < 2:
        manifest["warnings"].append(
            "Fewer than 2 factors recomputed successfully; PBO/DSR require multiple trials."
        )
        return _json_safe(manifest)

    full_split = dataset.get_split("full")
    n_samples = int(full_split.returns.shape[1])
    cv = CombinatorialPurgedCV(cv_config)
    try:
        splits = cv.split(n_samples, label_horizon=1)
    except ValueError as exc:
        manifest["warnings"].append(f"CPCV split failed: {exc}")
        return _json_safe(manifest)

    backtester = PortfolioBacktester()
    returns_panel = full_split.returns.T  # (T, N assets)
    trial_names = [artifact.name for artifact in succeeded]
    n_trials = len(succeeded)
    n_paths = len(splits)
    is_oos_matrix = np.full((n_trials, n_paths), np.nan, dtype=np.float64)

    for trial_idx, artifact in enumerate(succeeded):
        stats = backtester.quintile_backtest(artifact.signals_full.T, returns_panel)
        ls_series = np.asarray(stats["ls_net_series"], dtype=np.float64)
        for split in splits:
            test_values = ls_series[split.test_indices]
            finite = test_values[np.isfinite(test_values)]
            if finite.size:
                is_oos_matrix[trial_idx, split.path_id] = float(np.mean(finite))

    valid_path_mask = np.all(np.isfinite(is_oos_matrix), axis=0)
    clean_matrix = is_oos_matrix[:, valid_path_mask]

    manifest["cpcv"] = {
        "n_samples": n_samples,
        "n_paths": n_paths,
        "n_valid_paths": int(clean_matrix.shape[1]),
        "label_horizon": 1,
    }

    if clean_matrix.shape[1] < cv_config.min_paths_for_pbo:
        manifest["warnings"].append(
            f"Only {clean_matrix.shape[1]} usable CPCV paths (< "
            f"min_paths_for_pbo={cv_config.min_paths_for_pbo}); skipping PBO."
        )
    else:
        pbo_result = ProbabilityOfBacktestOverfitting(cv_config).compute(clean_matrix)
        manifest["pbo"] = {
            "pbo": pbo_result.pbo,
            "n_combinations": pbo_result.n_combinations,
            "passes": pbo_result.passes,
            "logit_values": [float(value) for value in pbo_result.logit_values],
        }

    mean_oos = np.nanmean(is_oos_matrix, axis=1)
    best_idx = int(np.nanargmax(mean_oos))
    best_name = trial_names[best_idx]
    best_series = is_oos_matrix[best_idx][np.isfinite(is_oos_matrix[best_idx])]

    # `best_series` is one performance value per CPCV path (a subset of test
    # groups), not one value per calendar period -- annualization_factor=1.0
    # keeps the Deflated Sharpe on the same path-level scale as `pbo`
    # instead of implying a false calendar annualization.
    dsr_result = DeflatedSharpeCalculator(SignificanceConfig()).compute(
        best_name, best_series, n_trials=n_trials, annualization_factor=1.0
    )
    manifest["deflated_sharpe"] = {
        "factor_name": dsr_result.factor_name,
        "raw_sharpe": dsr_result.raw_sharpe,
        "deflated_sharpe": dsr_result.deflated_sharpe,
        "haircut": dsr_result.haircut,
        "p_value": dsr_result.p_value,
        "n_trials": dsr_result.n_trials,
        "passes": dsr_result.passes,
    }
    manifest["best_trial"] = {
        "name": best_name,
        "mean_oos_performance": float(mean_oos[best_idx]),
    }

    return _json_safe(manifest)


def _time_callable(fn, repeats: int = 3) -> float:
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return min(timings) * 1000.0


def run_efficiency_benchmark(cfg, output_dir: Path) -> dict:
    """Benchmark operator-level and factor-level compute time."""
    periods, assets = cfg.benchmark.efficiency_panel_shape
    matrix = np.random.RandomState(cfg.benchmark.seed).randn(assets, periods).astype(np.float64)
    other = np.random.RandomState(cfg.benchmark.seed + 1).randn(assets, periods).astype(np.float64)

    from factorminer.operators import torch_available
    from factorminer.operators.gpu_backend import to_tensor
    from factorminer.operators.registry import execute_operator
    from factorminer.utils.visualization import plot_efficiency_benchmark

    operator_bench: dict[str, dict[str, float | None]] = {"numpy": {}, "c": {}, "gpu": {}}

    def _backend_inputs(backend: str):
        if backend == "gpu":
            return to_tensor(matrix), to_tensor(other)
        return matrix, other

    operators = {
        "Add": lambda backend: execute_operator("Add", *_backend_inputs(backend), backend=backend),
        "Mean": lambda backend: execute_operator(
            "Mean", _backend_inputs(backend)[0], params={"window": 20}, backend=backend
        ),
        "Delta": lambda backend: execute_operator(
            "Delta", _backend_inputs(backend)[0], params={"window": 5}, backend=backend
        ),
        "TsRank": lambda backend: execute_operator(
            "TsRank", _backend_inputs(backend)[0], params={"window": 20}, backend=backend
        ),
        "Corr": lambda backend: execute_operator(
            "Corr", *_backend_inputs(backend), params={"window": 20}, backend=backend
        ),
        "CsRank": lambda backend: execute_operator(
            "CsRank", _backend_inputs(backend)[0], backend=backend
        ),
    }
    for op_name, runner in operators.items():
        operator_bench["numpy"][op_name] = _time_callable(lambda r=runner: r("numpy"))
        operator_bench["c"][op_name] = (
            _time_callable(lambda r=runner: r("c")) if c_backend_available() else None
        )
        if torch_available():
            operator_bench["gpu"][op_name] = _time_callable(lambda r=runner: r("gpu"))
        else:
            operator_bench["gpu"][op_name] = None

    factor_bench: dict[str, dict[str, float | None]] = {"numpy": {}, "c": {}, "gpu": {}}
    factor_specs = {
        "momentum_volume": lambda backend: execute_operator(
            "CsRank",
            execute_operator(
                "Mul",
                execute_operator(
                    "Return", _backend_inputs(backend)[0], params={"window": 5}, backend=backend
                ),
                execute_operator(
                    "Div",
                    _backend_inputs(backend)[1],
                    execute_operator(
                        "Mean", _backend_inputs(backend)[1], params={"window": 20}, backend=backend
                    ),
                    backend=backend,
                ),
                backend=backend,
            ),
            backend=backend,
        ),
        "vwap_gap": lambda backend: execute_operator(
            "Neg",
            execute_operator(
                "CsRank",
                execute_operator(
                    "Div",
                    execute_operator("Sub", *_backend_inputs(backend), backend=backend),
                    execute_operator(
                        "Add",
                        _backend_inputs(backend)[1],
                        to_tensor(np.full_like(other, 1e-8))
                        if backend == "gpu"
                        else np.full_like(other, 1e-8),
                        backend=backend,
                    ),
                    backend=backend,
                ),
                backend=backend,
            ),
            backend=backend,
        ),
    }
    for formula_name, runner in factor_specs.items():
        factor_bench["numpy"][formula_name] = _time_callable(lambda r=runner: r("numpy"))
        factor_bench["c"][formula_name] = (
            _time_callable(lambda r=runner: r("c")) if c_backend_available() else None
        )
        if torch_available():
            factor_bench["gpu"][formula_name] = _time_callable(lambda r=runner: r("gpu"))
        else:
            factor_bench["gpu"][formula_name] = None

    bench_dir = _ensure_dir(output_dir / "benchmark" / "efficiency")
    plot_efficiency_benchmark(
        {
            backend: {k: v for k, v in values.items() if v is not None}
            for backend, values in operator_bench.items()
        },
        save_path=str(bench_dir / "operator_efficiency.png"),
    )
    plot_efficiency_benchmark(
        {
            backend: {k: v for k, v in values.items() if v is not None}
            for backend, values in factor_bench.items()
        },
        save_path=str(bench_dir / "factor_efficiency.png"),
    )
    result = {
        "panel_shape": {"periods": periods, "assets": assets},
        "operator_level_ms": operator_bench,
        "factor_level_ms": factor_bench,
        "available_backends": {
            "numpy": True,
            "c": c_backend_available(),
            "gpu": torch_available(),
        },
    }
    _write_json(bench_dir / "efficiency.json", result)
    return result


def run_benchmark_suite(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Run the benchmark suite and return the artifact index."""
    if runtime_manifests is None:
        runtime_manifests = getattr(cfg.benchmark, "runtime_manifests", None)
    use_runtime_loops = bool(runtime_manifests or getattr(cfg.benchmark, "runtime_loops", False))
    strategy_cfg = _strategy_ablation_raw_config(cfg)
    run_strategy_ablation = bool(use_runtime_loops or strategy_cfg.get("enabled", False))
    results = {
        "table1": run_table1_benchmark(
            cfg,
            output_dir,
            data_path=data_path,
            raw_df=raw_df,
            mock=mock,
            factor_miner_library_path=factor_miner_library_path,
            factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
            runtime_manifests=runtime_manifests,
            use_runtime_loops=use_runtime_loops,
        ),
        "ablation_memory": run_ablation_memory_benchmark(
            cfg,
            output_dir,
            data_path=data_path,
            raw_df=raw_df,
            mock=mock,
            factor_miner_library_path=factor_miner_library_path,
            factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
            runtime_manifests=runtime_manifests,
        ),
        "ablation_strategy": (
            run_ablation_strategy_benchmark(
                cfg,
                output_dir,
                data_path=data_path,
                raw_df=raw_df,
                mock=mock,
                runtime_manifests=runtime_manifests,
            )
            if run_strategy_ablation
            else {
                "skipped": True,
                "reason": (
                    "runtime loops disabled; enable benchmark.strategy_ablation.enabled "
                    "or supply runtime manifests"
                ),
            }
        ),
        "cost_pressure": run_cost_pressure_benchmark(
            cfg,
            output_dir,
            data_path=data_path,
            raw_df=raw_df,
            mock=mock,
            factor_miner_library_path=factor_miner_library_path,
            runtime_manifests=runtime_manifests,
        ),
        "efficiency": run_efficiency_benchmark(cfg, output_dir),
    }
    _write_json(_ensure_dir(output_dir / "benchmark") / "suite.json", results)
    return results


def run_runtime_mining_benchmark(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
    runtime_manifests: dict[str, dict[str, Any]] | None = None,
) -> dict:
    """Run the benchmark suite with explicit real-loop manifests when provided."""
    return run_benchmark_suite(
        cfg,
        output_dir,
        data_path=data_path,
        raw_df=raw_df,
        mock=mock,
        factor_miner_library_path=factor_miner_library_path,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
        runtime_manifests=runtime_manifests,
    )


# ---------------------------------------------------------------------------
# Consolidated Phase-2 reporting
# ---------------------------------------------------------------------------


def _reported_universe(payload: dict[str, Any], cfg) -> dict[str, Any]:
    universes = payload.get("universes", {})
    for name in getattr(cfg.benchmark, "report_universes", []):
        if name in universes:
            return universes[name]
    return next(iter(universes.values()), {})


def _method_result_from_runtime_payload(
    method: str,
    payload: dict[str, Any],
    cfg,
) -> MethodResult:
    evaluation = _reported_universe(payload, cfg)
    library = evaluation.get("library", {})
    combinations = evaluation.get("combinations", {})
    selections = evaluation.get("selections", {})
    equal_weight = combinations.get("equal_weight", {})
    ic_weighted = combinations.get("ic_weighted", {})
    lasso = selections.get("lasso", {})
    xgboost = selections.get("xgboost", {})
    freeze_stats = payload.get("freeze_stats", {})
    succeeded = max(int(freeze_stats.get("succeeded", 0)), 1)
    ic_series = np.asarray(equal_weight.get("ic_series", []), dtype=np.float64)
    return MethodResult(
        method=method,
        library_ic=float(library.get("ic", 0.0) or 0.0),
        library_icir=float(library.get("icir", 0.0) or 0.0),
        avg_abs_rho=float(library.get("avg_abs_rho", 0.0) or 0.0),
        ew_ic=float(equal_weight.get("ic", 0.0) or 0.0),
        ew_icir=float(equal_weight.get("icir", 0.0) or 0.0),
        icw_ic=float(ic_weighted.get("ic", 0.0) or 0.0),
        icw_icir=float(ic_weighted.get("icir", 0.0) or 0.0),
        lasso_ic=float(lasso.get("ic", 0.0) or 0.0),
        lasso_icir=float(lasso.get("icir", 0.0) or 0.0),
        xgb_ic=float(xgboost.get("ic", 0.0) or 0.0),
        xgb_icir=float(xgboost.get("icir", 0.0) or 0.0),
        n_factors=int(evaluation.get("factor_count", 0) or 0),
        admission_rate=float(freeze_stats.get("admitted", 0)) / succeeded,
        avg_turnover=float(equal_weight.get("turnover", 0.0) or 0.0),
        ic_series=ic_series if ic_series.size else None,
    )


def _comparison_frames(
    methods: list[str],
    results: dict[str, MethodResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    library = pd.DataFrame(
        [
            {
                "method": method,
                "ic_pct": results[method].library_ic * 100.0,
                "icir": results[method].library_icir,
                "avg_abs_rho": results[method].avg_abs_rho,
                "n_factors": results[method].n_factors,
                "avg_turnover": results[method].avg_turnover,
            }
            for method in methods
        ]
    )
    combinations = pd.DataFrame(
        [
            {
                "method": method,
                "ew_ic_pct": results[method].ew_ic * 100.0,
                "ew_icir": results[method].ew_icir,
                "icw_ic_pct": results[method].icw_ic * 100.0,
                "icw_icir": results[method].icw_icir,
            }
            for method in methods
        ]
    )
    selections = pd.DataFrame(
        [
            {
                "method": method,
                "lasso_ic_pct": results[method].lasso_ic * 100.0,
                "lasso_icir": results[method].lasso_icir,
                "xgb_ic_pct": results[method].xgb_ic * 100.0,
                "xgb_icir": results[method].xgb_icir,
                "best_ic_pct": max(results[method].lasso_ic, results[method].xgb_ic) * 100.0,
            }
            for method in methods
        ]
    )
    return library, combinations, selections


def run_phase2_comparison(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    baseline_methods: list[str] | None = None,
    n_target_factors: int = 40,
    n_runs: int = 1,
) -> tuple[BenchmarkResult, dict[str, Any]]:
    """Build the Phase-2 report entirely from canonical Table-1 artifacts."""
    if n_runs != 1:
        warnings.warn(
            "run_phase2_comparison now executes one provenance-complete runtime run per method",
            RuntimeWarning,
            stacklevel=2,
        )
    runtime_cfg = _clone_cfg(cfg)
    runtime_cfg.mining.target_library_size = n_target_factors
    methods = baseline_methods or [
        "random_exploration",
        "alpha101_classic",
        "alpha101_adapted",
        "ralph_loop",
        "helix_phase2",
    ]
    table1 = run_table1_benchmark(
        runtime_cfg,
        output_dir,
        data_path=data_path,
        raw_df=raw_df,
        mock=mock,
        baseline_names=methods,
        use_runtime_loops=True,
    )
    method_results = {
        method: _method_result_from_runtime_payload(method, table1[method], runtime_cfg)
        for method in methods
    }
    library_frame, combination_frame, selection_frame = _comparison_frames(
        methods,
        method_results,
    )
    efficiency = run_efficiency_benchmark(runtime_cfg, output_dir)
    speed_rows: list[dict[str, Any]] = []
    for backend, timings in efficiency.get("operator_level_ms", {}).items():
        for name, milliseconds in timings.items():
            if milliseconds is not None:
                speed_rows.append(
                    {"name": name, "time_ms": milliseconds, "type": f"operator/{backend}"}
                )
    speed_frame = pd.DataFrame(speed_rows)

    runtime_payloads: dict[str, list[dict[str, Any]]] = {}
    turnover_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    for method in methods:
        payload = table1[method]
        evaluation = _reported_universe(payload, runtime_cfg)
        combinations = evaluation.get("combinations", {})
        projected = {
            "method": method,
            "run_id": 0,
            "frozen_top_k": payload.get("frozen_top_k", []),
            "library": evaluation.get("library", {}),
            "combinations": combinations,
            "selections": evaluation.get("selections", {}),
            "provenance": payload.get("provenance", {}),
        }
        runtime_payloads[method] = [projected]
        turnover_rows.append(
            {
                "method": method,
                "run_id": 0,
                **{
                    f"{name}_turnover": float(metrics.get("turnover", 0.0) or 0.0)
                    for name, metrics in combinations.items()
                },
            }
        )
        for combination, metrics in combinations.items():
            for cost_bps, cost_metrics in metrics.get("cost_pressure", {}).items():
                cost_rows.append(
                    {
                        "method": method,
                        "run_id": 0,
                        "combination": combination,
                        "cost_bps": float(cost_bps),
                        **{
                            key: cost_metrics.get(key, 0.0)
                            for key in ("ic", "icir", "turnover", "long_short", "monotonicity")
                        },
                    }
                )

    statistical_tests: dict[str, Any] = {}
    helix = method_results.get("helix_phase2")
    ralph = method_results.get("ralph_loop")
    if (
        helix is not None
        and ralph is not None
        and helix.ic_series is not None
        and ralph.ic_series is not None
    ):
        statistical_tests = StatisticalComparisonTests(runtime_cfg.benchmark.seed).run_all_tests(
            helix.ic_series,
            ralph.ic_series,
        )

    artifacts = {
        "runtime_root": str((output_dir / "benchmark").resolve()),
        "runtime_payloads": runtime_payloads,
        "table1": table1,
        "efficiency": efficiency,
    }
    result = BenchmarkResult(
        methods=methods,
        factor_library_metrics=library_frame,
        combination_metrics=combination_frame,
        selection_metrics=selection_frame,
        speed_metrics=speed_frame,
        statistical_tests=statistical_tests,
        raw_method_results={method: [result] for method, result in method_results.items()},
        turnover_metrics=pd.DataFrame(turnover_rows),
        cost_pressure_metrics=pd.DataFrame(cost_rows),
        runtime_artifacts=artifacts,
    )
    return result, artifacts


def run_phase2_ablation_study(
    cfg,
    output_dir: Path,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    mock: bool = False,
    configs_to_run: list[str] | None = None,
    n_target_factors: int = 40,
    n_runs: int = 1,
) -> AblationResult:
    """Run Phase-2 variants through the canonical runtime-loop benchmark."""
    if n_runs != 1:
        warnings.warn("Phase-2 ablations now run once per variant", RuntimeWarning, stacklevel=2)
    runtime_cfg = _clone_cfg(cfg)
    runtime_cfg.mining.target_library_size = n_target_factors
    configs = configs_to_run or [
        "full",
        "no_debate",
        "no_causal",
        "no_canonicalize",
        "no_regime",
        "no_capacity",
        "no_significance",
        "no_memory",
    ]
    baselines = {
        "full": "helix_phase2",
        "no_debate": "helix_no_debate",
        "no_causal": "helix_no_causal",
        "no_canonicalize": "helix_no_canonicalize",
        "no_regime": "helix_no_regime",
        "no_capacity": "helix_no_capacity",
        "no_significance": "helix_no_significance",
        "no_memory": "helix_no_memory",
    }
    results: dict[str, MethodResult] = {}
    for config_name in configs:
        if config_name not in baselines:
            logger.warning("Unknown runtime ablation config: %s", config_name)
            continue
        baseline = baselines[config_name]
        payload = run_table1_benchmark(
            runtime_cfg,
            output_dir / "runtime_ablation" / config_name,
            data_path=data_path,
            raw_df=raw_df,
            mock=mock,
            baseline_names=[baseline],
            use_runtime_loops=True,
        )[baseline]
        result = _method_result_from_runtime_payload(config_name, payload, runtime_cfg)
        results[config_name] = result

    rows: list[dict[str, Any]] = []
    full = results.get("full")
    if full is not None:
        for config_name, result in results.items():
            if config_name == "full":
                continue
            rows.append(
                {
                    "config": config_name,
                    "method": result.method,
                    "delta_library_ic": result.library_ic - full.library_ic,
                    "delta_library_icir": result.library_icir - full.library_icir,
                    "delta_ew_ic": result.ew_ic - full.ew_ic,
                    "delta_icw_ic": result.icw_ic - full.icw_ic,
                    "delta_lasso_ic": result.lasso_ic - full.lasso_ic,
                    "delta_xgb_ic": result.xgb_ic - full.xgb_ic,
                    "delta_turnover": result.avg_turnover - full.avg_turnover,
                }
            )
    return AblationResult(configs=configs, results=results, contributions=pd.DataFrame(rows))
