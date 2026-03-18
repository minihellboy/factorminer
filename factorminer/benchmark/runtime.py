"""Strict paper/research benchmark runners built on runtime recomputation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from factorminer.benchmark.catalogs import (
    CandidateEntry,
    build_alpha101_adapted,
    build_alphaagent_style,
    build_alphaforge_style,
    build_factor_miner_catalog,
    build_gplearn_style,
    build_random_exploration,
    dedupe_entries,
    entries_from_library,
    ALPHA101_CLASSIC,
)
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import load_library
from factorminer.evaluation.runtime import (
    EvaluationDataset,
    FactorEvaluationArtifact,
    compute_correlation_matrix,
    evaluate_factors,
    load_runtime_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkManifest:
    """Serializable description of one benchmark run."""

    benchmark_name: str
    mode: str
    seed: int
    baseline: str
    freeze_universe: str
    report_universes: list[str]
    train_period: list[str]
    test_period: list[str]
    freeze_top_k: int
    signal_failure_policy: str
    dataset_hashes: dict[str, str]
    artifact_paths: dict[str, str]
    warnings: list[str]


def _clone_cfg(cfg):
    cloned = copy.deepcopy(cfg)
    cloned._raw = copy.deepcopy(getattr(cfg, "_raw", {}))
    return cloned


def _cfg_with_overrides(cfg, universe: str, mode: Optional[str] = None):
    cloned = _clone_cfg(cfg)
    cloned.data.universe = universe
    if mode is not None:
        cloned.benchmark.mode = mode
    if cloned.benchmark.mode == "paper":
        cloned.evaluation.signal_failure_policy = "reject"
        cloned.phase2.causal.enabled = False
        cloned.phase2.regime.enabled = False
        cloned.phase2.capacity.enabled = False
        cloned.phase2.significance.enabled = False
        cloned.phase2.debate.enabled = False
        cloned.phase2.auto_inventor.enabled = False
        cloned.phase2.helix.enabled = False
    return cloned


def _data_hash(df: pd.DataFrame) -> str:
    sample = df.sort_values(["datetime", "asset_id"]).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
    return digest.hexdigest()


def load_benchmark_dataset(
    cfg,
    *,
    data_path: Optional[str] = None,
    raw_df: Optional[pd.DataFrame] = None,
    universe: Optional[str] = None,
    mock: bool = False,
) -> tuple[EvaluationDataset, str]:
    """Load one universe into the canonical runtime dataset."""
    if universe is None:
        universe = cfg.data.universe

    if raw_df is None:
        if mock:
            from factorminer.data.mock_data import MockConfig, generate_mock_data

            mock_cfg = MockConfig(
                num_assets=64 if universe.lower() == "binance" else 80,
                num_periods=12_200,
                frequency="10min",
                start_date="2024-01-02 09:30:00",
                universe=universe,
                plant_alpha=True,
                seed=cfg.benchmark.seed,
            )
            raw_df = generate_mock_data(mock_cfg)
        else:
            path = data_path
            if path is None:
                path = getattr(cfg, "_raw", {}).get("data_path")
            if path is None:
                raise ValueError("No data path specified for benchmark run")
            from factorminer.data.loader import load_market_data

            raw_df = load_market_data(path, universe=universe)

    dataset_cfg = _cfg_with_overrides(cfg, universe)
    return load_runtime_dataset(raw_df, dataset_cfg), _data_hash(raw_df)


def _factors_from_entries(entries: Iterable[CandidateEntry]) -> list[Factor]:
    return [
        Factor(
            id=idx + 1,
            name=entry.name,
            formula=entry.formula,
            category=entry.category,
            ic_mean=0.0,
            icir=0.0,
            ic_win_rate=0.0,
            max_correlation=0.0,
            batch_number=0,
        )
        for idx, entry in enumerate(entries)
    ]


def _get_baseline_entries(
    baseline: str,
    seed: int,
    *,
    factor_miner_library_path: Optional[str] = None,
    factor_miner_no_memory_library_path: Optional[str] = None,
) -> list[CandidateEntry]:
    if baseline == "alpha101_classic":
        return dedupe_entries(ALPHA101_CLASSIC)
    if baseline == "alpha101_adapted":
        return dedupe_entries(build_alpha101_adapted())
    if baseline == "random_exploration":
        return dedupe_entries(build_random_exploration(seed))
    if baseline == "gplearn":
        return dedupe_entries(build_gplearn_style(seed))
    if baseline == "alphaforge_style":
        return dedupe_entries(build_alphaforge_style())
    if baseline == "alphaagent_style":
        return dedupe_entries(build_alphaagent_style())
    if baseline == "factor_miner":
        if factor_miner_library_path:
            return dedupe_entries(entries_from_library(load_library(_base_path(factor_miner_library_path))))
        return dedupe_entries(build_factor_miner_catalog())
    if baseline == "factor_miner_no_memory":
        if factor_miner_no_memory_library_path:
            return dedupe_entries(entries_from_library(load_library(_base_path(factor_miner_no_memory_library_path))))
        return dedupe_entries(build_random_exploration(seed + 101, count=200))
    raise KeyError(f"Unknown benchmark baseline: {baseline}")


def _base_path(path: str) -> str:
    p = Path(path)
    return str(p.with_suffix("")) if p.suffix == ".json" else str(p)


def build_benchmark_library(
    artifacts: Iterable[FactorEvaluationArtifact],
    cfg,
    *,
    split_name: str = "train",
    ic_threshold: Optional[float] = None,
    correlation_threshold: Optional[float] = None,
) -> tuple[FactorLibrary, dict[str, int]]:
    """Build a library from candidate artifacts under the paper admission rules."""
    ic_threshold = cfg.mining.ic_threshold if ic_threshold is None else ic_threshold
    correlation_threshold = (
        cfg.mining.correlation_threshold
        if correlation_threshold is None
        else correlation_threshold
    )
    library = FactorLibrary(
        correlation_threshold=correlation_threshold,
        ic_threshold=ic_threshold,
    )

    stats = {
        "succeeded": 0,
        "admitted": 0,
        "replaced": 0,
        "threshold_rejections": 0,
        "correlation_rejections": 0,
    }

    ordered = [artifact for artifact in artifacts if artifact.succeeded]
    ordered.sort(
        key=lambda artifact: artifact.split_stats[split_name]["ic_abs_mean"],
        reverse=True,
    )
    stats["succeeded"] = len(ordered)

    for artifact in ordered:
        split_stats = artifact.split_stats[split_name]
        candidate_ic = float(split_stats["ic_abs_mean"])
        candidate_signals = artifact.split_signals[split_name]
        if candidate_ic < ic_threshold:
            stats["threshold_rejections"] += 1
            continue

        max_corr = (
            library._max_correlation_with_library(candidate_signals)  # noqa: SLF001
            if library.size
            else 0.0
        )
        factor = Factor(
            id=0,
            name=artifact.name,
            formula=artifact.formula,
            category=artifact.category,
            ic_mean=candidate_ic,
            icir=abs(float(split_stats["icir"])),
            ic_win_rate=float(split_stats["ic_win_rate"]),
            max_correlation=max_corr,
            batch_number=0,
            signals=candidate_signals,
        )
        admitted, _ = library.check_admission(candidate_ic, candidate_signals)
        if admitted:
            library.admit_factor(factor)
            stats["admitted"] += 1
            continue

        replace, replace_id, _ = library.check_replacement(
            candidate_ic,
            candidate_signals,
            ic_min=cfg.mining.replacement_ic_min,
            ic_ratio=cfg.mining.replacement_ic_ratio,
        )
        if replace and replace_id is not None:
            library.replace_factor(replace_id, factor)
            stats["replaced"] += 1
            continue

        stats["correlation_rejections"] += 1

    return library, stats


def select_frozen_top_k(
    artifacts: Iterable[FactorEvaluationArtifact],
    library: FactorLibrary,
    *,
    top_k: int,
    split_name: str = "train",
    min_ic: float = 0.05,
    min_icir: float = 0.5,
) -> list[FactorEvaluationArtifact]:
    """Freeze the paper Top-K set from train-split recomputed metrics."""
    admitted_formulas = {factor.formula for factor in library.list_factors()}
    succeeded = [artifact for artifact in artifacts if artifact.succeeded]
    admitted = [
        artifact
        for artifact in succeeded
        if artifact.formula in admitted_formulas
        and artifact.split_stats[split_name]["ic_abs_mean"] >= min_ic
        and abs(artifact.split_stats[split_name]["icir"]) >= min_icir
    ]
    admitted.sort(
        key=lambda artifact: artifact.split_stats[split_name]["ic_abs_mean"],
        reverse=True,
    )
    selected: list[FactorEvaluationArtifact] = admitted[:top_k]
    selected_formulas = {artifact.formula for artifact in selected}

    if len(selected) < top_k:
        remainder = [
            artifact
            for artifact in succeeded
            if artifact.formula not in selected_formulas
        ]
        remainder.sort(
            key=lambda artifact: artifact.split_stats[split_name]["ic_abs_mean"],
            reverse=True,
        )
        selected.extend(remainder[: top_k - len(selected)])

    return selected


def _abs_icir_from_series(ic_series: np.ndarray) -> float:
    valid = ic_series[np.isfinite(ic_series)]
    if len(valid) < 3:
        return 0.0
    std = float(np.std(valid, ddof=1))
    if std < 1e-12:
        return 0.0
    return abs(float(np.mean(valid))) / std


def _normalize_backtest_stats(stats: dict) -> dict[str, float]:
    ic_series = np.asarray(stats.get("ic_series", []), dtype=np.float64)
    return {
        "ic": abs(float(stats.get("ic_mean", 0.0))),
        "icir": _abs_icir_from_series(ic_series),
        "ic_win_rate": float(stats.get("ic_win_rate", 0.0)),
        "long_short": float(stats.get("ls_return", 0.0)),
        "monotonicity": float(stats.get("monotonicity", 0.0)),
        "turnover": float(stats.get("avg_turnover", 0.0)),
    }


def _avg_abs_rho(artifacts: list[FactorEvaluationArtifact], split_name: str) -> float:
    if len(artifacts) < 2:
        return 0.0
    corr = np.abs(compute_correlation_matrix(artifacts, split_name))
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _weighted_composite(
    factor_signals: dict[int, np.ndarray],
    weights: dict[int, float],
) -> np.ndarray:
    ordered = [(fid, factor_signals[fid], weights.get(fid, 0.0)) for fid in factor_signals]
    if not ordered:
        raise ValueError("Cannot build weighted composite from zero factors")
    total = sum(abs(weight) for _, _, weight in ordered)
    if total < 1e-12:
        total = float(len(ordered))
        ordered = [(fid, signal, 1.0) for fid, signal, _ in ordered]
    composite = np.zeros_like(ordered[0][1], dtype=np.float64)
    for _, signal, weight in ordered:
        composite += signal * (weight / total)
    return composite


def evaluate_frozen_set(
    frozen: list[FactorEvaluationArtifact],
    dataset: EvaluationDataset,
    *,
    split_name: str = "test",
    fit_split: str = "train",
    cost_bps: Optional[list[float]] = None,
) -> dict:
    """Evaluate one frozen factor set on one universe."""
    if cost_bps is None:
        cost_bps = [1.0, 4.0, 7.0, 10.0, 11.0]

    factors = _factors_from_entries(
        CandidateEntry(
            name=artifact.name,
            formula=artifact.formula,
            category=artifact.category,
        )
        for artifact in frozen
    )
    artifacts = evaluate_factors(factors, dataset, signal_failure_policy="reject")
    succeeded = [artifact for artifact in artifacts if artifact.succeeded]

    result = {
        "factor_count": len(succeeded),
        "library": {
            "ic": 0.0,
            "icir": 0.0,
            "avg_abs_rho": 0.0,
        },
        "combinations": {},
        "selections": {},
        "warnings": [],
    }
    if not succeeded:
        result["warnings"].append("No frozen factors recomputed successfully on this universe")
        return result

    result["library"] = {
        "ic": float(np.mean([artifact.split_stats[split_name]["ic_abs_mean"] for artifact in succeeded])),
        "icir": float(np.mean([abs(artifact.split_stats[split_name]["icir"]) for artifact in succeeded])),
        "avg_abs_rho": _avg_abs_rho(succeeded, split_name),
    }

    artifact_map = {artifact.factor_id: artifact for artifact in succeeded}
    fit_signals = {artifact.factor_id: artifact.split_signals[fit_split].T for artifact in succeeded}
    eval_signals = {artifact.factor_id: artifact.split_signals[split_name].T for artifact in succeeded}
    fit_returns = dataset.get_split(fit_split).returns.T
    eval_returns = dataset.get_split(split_name).returns.T

    from factorminer.evaluation.combination import FactorCombiner
    from factorminer.evaluation.portfolio import PortfolioBacktester
    from factorminer.evaluation.selection import FactorSelector

    combiner = FactorCombiner()
    backtester = PortfolioBacktester()
    selector = FactorSelector()

    fit_ic_values = {
        artifact.factor_id: artifact.split_stats[fit_split]["ic_mean"]
        for artifact in succeeded
    }

    combos = {
        "equal_weight": combiner.equal_weight(eval_signals),
        "ic_weighted": combiner.ic_weighted(eval_signals, fit_ic_values),
        "orthogonal": combiner.orthogonal(eval_signals),
    }
    for name, composite in combos.items():
        stats = backtester.quintile_backtest(composite, eval_returns)
        result["combinations"][name] = _normalize_backtest_stats(stats)
        result["combinations"][name]["cost_pressure"] = {
            str(cost): _normalize_backtest_stats(
                backtester.quintile_backtest(
                    composite, eval_returns, transaction_cost_bps=float(cost)
                )
            )
            for cost in cost_bps
        }

    selection_specs = {}
    try:
        selection_specs["lasso"] = selector.lasso_selection(fit_signals, fit_returns)
    except Exception as exc:
        result["warnings"].append(f"lasso unavailable: {exc}")
    try:
        selection_specs["forward_stepwise"] = selector.forward_stepwise(fit_signals, fit_returns)
    except Exception as exc:
        result["warnings"].append(f"forward_stepwise unavailable: {exc}")
    try:
        selection_specs["xgboost"] = selector.xgboost_selection(fit_signals, fit_returns)
    except Exception as exc:
        result["warnings"].append(f"xgboost unavailable: {exc}")

    for name, ranking in selection_specs.items():
        if not ranking:
            result["selections"][name] = {"factor_count": 0}
            continue
        selected_ids = [factor_id for factor_id, _ in ranking]
        selected_eval = {factor_id: eval_signals[factor_id] for factor_id in selected_ids}
        if name == "lasso":
            weights = {factor_id: score for factor_id, score in ranking}
            composite = _weighted_composite(selected_eval, weights)
        elif name == "xgboost":
            weights = {
                factor_id: score * np.sign(artifact_map[factor_id].split_stats[fit_split]["ic_mean"] or 1.0)
                for factor_id, score in ranking
            }
            composite = _weighted_composite(selected_eval, weights)
        else:
            signs = {
                factor_id: np.sign(artifact_map[factor_id].split_stats[fit_split]["ic_mean"] or 1.0)
                for factor_id in selected_ids
            }
            composite = _weighted_composite(selected_eval, signs)
        stats = backtester.quintile_backtest(composite, eval_returns)
        result["selections"][name] = {
            "factor_count": len(selected_ids),
            **_normalize_backtest_stats(stats),
        }

    return result


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=2, sort_keys=False)


def _save_manifest(path: Path, manifest: BenchmarkManifest) -> None:
    _write_json(path, asdict(manifest))


def run_table1_benchmark(
    cfg,
    output_dir: Path,
    *,
    data_path: Optional[str] = None,
    mock: bool = False,
    baseline_names: Optional[list[str]] = None,
    factor_miner_library_path: Optional[str] = None,
    factor_miner_no_memory_library_path: Optional[str] = None,
) -> dict:
    """Run the strict Top-K freeze benchmark across all configured universes."""
    benchmark_dir = _ensure_dir(output_dir / "benchmark" / "table1")
    baseline_names = baseline_names or list(cfg.benchmark.baselines)
    freeze_cfg = _cfg_with_overrides(cfg, cfg.benchmark.freeze_universe)
    freeze_dataset, freeze_hash = load_benchmark_dataset(
        freeze_cfg,
        data_path=data_path,
        universe=cfg.benchmark.freeze_universe,
        mock=mock,
    )

    summary: dict[str, dict] = {}
    for baseline in baseline_names:
        entries = _get_baseline_entries(
            baseline,
            cfg.benchmark.seed,
            factor_miner_library_path=factor_miner_library_path,
            factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
        )
        factors = _factors_from_entries(entries)
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
            "freeze_universe": cfg.benchmark.freeze_universe,
            "candidate_count": len(entries),
            "freeze_library_size": library.size,
            "freeze_stats": library_stats,
            "frozen_top_k": [
                {
                    "name": artifact.name,
                    "formula": artifact.formula,
                    "category": artifact.category,
                    "train_ic": artifact.split_stats["train"]["ic_abs_mean"],
                    "train_icir": abs(artifact.split_stats["train"]["icir"]),
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
                universe=universe,
                mock=mock,
            )
            dataset_hashes[universe] = dataset_hash
            baseline_result["universes"][universe] = evaluate_frozen_set(
                frozen,
                dataset,
                split_name="test",
                fit_split="train",
                cost_bps=list(cfg.benchmark.cost_bps),
            )

        result_path = benchmark_dir / f"{baseline}.json"
        _write_json(result_path, baseline_result)
        manifest = BenchmarkManifest(
            benchmark_name="table1",
            mode=cfg.benchmark.mode,
            seed=cfg.benchmark.seed,
            baseline=baseline,
            freeze_universe=cfg.benchmark.freeze_universe,
            report_universes=list(cfg.benchmark.report_universes),
            train_period=list(cfg.data.train_period),
            test_period=list(cfg.data.test_period),
            freeze_top_k=cfg.benchmark.freeze_top_k,
            signal_failure_policy="reject",
            dataset_hashes=dataset_hashes,
            artifact_paths={"result": str(result_path)},
            warnings=[],
        )
        _save_manifest(benchmark_dir / f"{baseline}_manifest.json", manifest)
        summary[baseline] = baseline_result

    return summary


def run_ablation_memory_benchmark(
    cfg,
    output_dir: Path,
    *,
    data_path: Optional[str] = None,
    mock: bool = False,
    factor_miner_library_path: Optional[str] = None,
    factor_miner_no_memory_library_path: Optional[str] = None,
) -> dict:
    """Compare the default FactorMiner lane to the relaxed no-memory lane."""
    comparison = run_table1_benchmark(
        cfg,
        output_dir,
        data_path=data_path,
        mock=mock,
        baseline_names=["factor_miner", "factor_miner_no_memory"],
        factor_miner_library_path=factor_miner_library_path,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
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


def run_cost_pressure_benchmark(
    cfg,
    output_dir: Path,
    *,
    baseline: str = "factor_miner",
    data_path: Optional[str] = None,
    mock: bool = False,
    factor_miner_library_path: Optional[str] = None,
) -> dict:
    """Run cost-pressure analysis for one baseline on the configured universes."""
    payload = run_table1_benchmark(
        cfg,
        output_dir,
        data_path=data_path,
        mock=mock,
        baseline_names=[baseline],
        factor_miner_library_path=factor_miner_library_path,
    )[baseline]
    result = {
        universe: {
            "combinations": {
                name: metrics.get("cost_pressure", {})
                for name, metrics in universe_payload["combinations"].items()
            }
        }
        for universe, universe_payload in payload["universes"].items()
    }
    out_path = _ensure_dir(output_dir / "benchmark" / "cost_pressure") / f"{baseline}.json"
    _write_json(out_path, result)
    return result


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
        "Mean": lambda backend: execute_operator("Mean", _backend_inputs(backend)[0], params={"window": 20}, backend=backend),
        "Delta": lambda backend: execute_operator("Delta", _backend_inputs(backend)[0], params={"window": 5}, backend=backend),
        "TsRank": lambda backend: execute_operator("TsRank", _backend_inputs(backend)[0], params={"window": 20}, backend=backend),
        "Corr": lambda backend: execute_operator("Corr", *_backend_inputs(backend), params={"window": 20}, backend=backend),
        "CsRank": lambda backend: execute_operator("CsRank", _backend_inputs(backend)[0], backend=backend),
    }
    for op_name, runner in operators.items():
        operator_bench["numpy"][op_name] = _time_callable(lambda r=runner: r("numpy"))
        operator_bench["c"][op_name] = None
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
                execute_operator("Return", _backend_inputs(backend)[0], params={"window": 5}, backend=backend),
                execute_operator(
                    "Div",
                    _backend_inputs(backend)[1],
                    execute_operator("Mean", _backend_inputs(backend)[1], params={"window": 20}, backend=backend),
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
                        to_tensor(np.full_like(other, 1e-8)) if backend == "gpu" else np.full_like(other, 1e-8),
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
        factor_bench["c"][formula_name] = None
        if torch_available():
            factor_bench["gpu"][formula_name] = _time_callable(lambda r=runner: r("gpu"))
        else:
            factor_bench["gpu"][formula_name] = None

    bench_dir = _ensure_dir(output_dir / "benchmark" / "efficiency")
    plot_efficiency_benchmark(
        {backend: {k: v for k, v in values.items() if v is not None} for backend, values in operator_bench.items()},
        save_path=str(bench_dir / "operator_efficiency.png"),
    )
    plot_efficiency_benchmark(
        {backend: {k: v for k, v in values.items() if v is not None} for backend, values in factor_bench.items()},
        save_path=str(bench_dir / "factor_efficiency.png"),
    )
    result = {
        "panel_shape": {"periods": periods, "assets": assets},
        "operator_level_ms": operator_bench,
        "factor_level_ms": factor_bench,
        "available_backends": {
            "numpy": True,
            "c": False,
            "gpu": torch_available(),
        },
    }
    _write_json(bench_dir / "efficiency.json", result)
    return result


def run_benchmark_suite(
    cfg,
    output_dir: Path,
    *,
    data_path: Optional[str] = None,
    mock: bool = False,
    factor_miner_library_path: Optional[str] = None,
    factor_miner_no_memory_library_path: Optional[str] = None,
) -> dict:
    """Run the benchmark suite and return the artifact index."""
    results = {
        "table1": run_table1_benchmark(
            cfg,
            output_dir,
            data_path=data_path,
            mock=mock,
            factor_miner_library_path=factor_miner_library_path,
            factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
        ),
        "ablation_memory": run_ablation_memory_benchmark(
            cfg,
            output_dir,
            data_path=data_path,
            mock=mock,
            factor_miner_library_path=factor_miner_library_path,
            factor_miner_no_memory_library_path=factor_miner_no_memory_library_path,
        ),
        "cost_pressure": run_cost_pressure_benchmark(
            cfg,
            output_dir,
            data_path=data_path,
            mock=mock,
            factor_miner_library_path=factor_miner_library_path,
        ),
        "efficiency": run_efficiency_benchmark(cfg, output_dir),
    }
    _write_json(_ensure_dir(output_dir / "benchmark") / "suite.json", results)
    return results
