"""Dataset loading and baseline-catalog construction for benchmarks."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from factorminer.benchmark.catalogs import (
    ALPHA101_CLASSIC,
    CandidateEntry,
    build_alpha101_adapted,
    build_alphaagent_style,
    build_alphaforge_style,
    build_factor_miner_catalog,
    build_gplearn_style,
    build_random_exploration,
    dedupe_entries,
    entries_from_library,
)
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import load_library
from factorminer.evaluation.runtime import (
    EvaluationDataset,
    FactorEvaluationArtifact,
    load_runtime_dataset,
)


def _clone_cfg(cfg):
    cloned = copy.deepcopy(cfg)
    cloned._raw = copy.deepcopy(getattr(cfg, "_raw", {}))
    return cloned


def _cfg_with_overrides(cfg, universe: str, mode: str | None = None):
    cloned = _clone_cfg(cfg)
    cloned.data.universe = universe
    if mode is not None:
        cloned.benchmark.mode = mode
    if cloned.benchmark.mode == "paper":
        cloned.evaluation.signal_failure_policy = "reject"
        cloned.research.enabled = False
        cloned.phase2.causal.enabled = False
        cloned.phase2.regime.enabled = False
        cloned.phase2.capacity.enabled = False
        cloned.phase2.significance.enabled = False
        cloned.phase2.debate.enabled = False
        cloned.phase2.auto_inventor.enabled = False
        cloned.phase2.helix.enabled = False
    else:
        cloned.research.enabled = True
    return cloned


def _data_hash(df: pd.DataFrame) -> str:
    sample = df.sort_values(["datetime", "asset_id"]).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(sample, index=True).values.tobytes())
    return digest.hexdigest()


def load_benchmark_dataset(
    cfg,
    *,
    data_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    universe: str | None = None,
    mock: bool = False,
) -> tuple[EvaluationDataset, str]:
    """Load one universe into the canonical runtime dataset."""
    if universe is None:
        universe = cfg.data.universe

    if raw_df is None:
        if mock:
            from factorminer.data.mock_data import MockConfig, generate_mock_data

            mock_shape = getattr(cfg.benchmark, "mock_panel_shape", None)
            if mock_shape is None:
                num_periods = 12_200
                num_assets = 64 if universe.lower() == "binance" else 80
            else:
                num_periods = int(mock_shape[0])
                num_assets = int(mock_shape[1])
            mock_cfg = MockConfig(
                num_assets=num_assets,
                num_periods=num_periods,
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
    factor_miner_library_path: str | None = None,
    factor_miner_no_memory_library_path: str | None = None,
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
            return dedupe_entries(
                entries_from_library(load_library(_base_path(factor_miner_library_path)))
            )
        return dedupe_entries(build_factor_miner_catalog())
    if baseline == "factor_miner_no_memory":
        if factor_miner_no_memory_library_path:
            return dedupe_entries(
                entries_from_library(load_library(_base_path(factor_miner_no_memory_library_path)))
            )
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
    ic_threshold: float | None = None,
    correlation_threshold: float | None = None,
) -> tuple[FactorLibrary, dict[str, int]]:
    """Build a library from candidate artifacts under the paper admission rules."""
    ic_threshold = cfg.mining.ic_threshold if ic_threshold is None else ic_threshold
    correlation_threshold = (
        cfg.mining.correlation_threshold if correlation_threshold is None else correlation_threshold
    )
    library = FactorLibrary(
        correlation_threshold=correlation_threshold,
        ic_threshold=ic_threshold,
        dependence_metric=getattr(cfg.evaluation, "redundancy_metric", "spearman"),
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
        key=lambda artifact: artifact.split_stats[split_name]["ic_paper_mean"],
        reverse=True,
    )
    stats["succeeded"] = len(ordered)

    for artifact in ordered:
        split_stats = artifact.split_stats[split_name]
        candidate_ic = float(split_stats["ic_paper_mean"])
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
            ic_mean=float(split_stats["ic_mean"]),
            ic_paper_mean=candidate_ic,
            ic_abs_mean=float(split_stats["ic_abs_mean"]),
            icir=float(split_stats["icir"]),
            ic_paper_icir=float(split_stats["ic_paper_icir"]),
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
