"""Frozen-set selection and held-out benchmark evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from factorminer.benchmark.catalogs import CandidateEntry
from factorminer.benchmark.contracts import json_safe as _json_safe
from factorminer.benchmark.datasets import _factors_from_entries
from factorminer.core.factor_library import FactorLibrary
from factorminer.evaluation.metrics import METRIC_VERSION
from factorminer.evaluation.runtime import (
    EvaluationDataset,
    FactorEvaluationArtifact,
    compute_correlation_matrix,
    evaluate_factors,
)


def _default_capacity_levels() -> list[float]:
    from factorminer.evaluation.capacity import CapacityConfig as RuntimeCapacityConfig

    return list(RuntimeCapacityConfig().capacity_levels)


def _extract_volume_panel(dataset: EvaluationDataset) -> np.ndarray | None:
    """Best-effort extraction of a dollar-volume panel for Helix capacity checks."""
    for key in ("$amt", "$volume"):
        panel = dataset.data_dict.get(key)
        if panel is not None and np.any(np.isfinite(panel)):
            return np.asarray(panel, dtype=np.float64)
    return None


def _split_volume_panel(
    dataset: EvaluationDataset,
    split_name: str,
) -> np.ndarray | None:
    """Align the available volume panel to one dataset split."""
    panel = _extract_volume_panel(dataset)
    if panel is None:
        return None
    split = dataset.get_split(split_name)
    if panel.ndim != 2 or panel.shape[1] < len(split.indices):
        return None
    return np.asarray(panel[:, split.indices], dtype=np.float64)


def _capacity_pressure_summary(
    *,
    factor_name: str,
    signals: np.ndarray,
    returns: np.ndarray,
    volume: np.ndarray,
    capacity_levels: list[float],
) -> dict[str, Any]:
    """Compute a compact capacity-stress summary for one factor/composite."""
    from factorminer.evaluation.capacity import CapacityConfig as RuntimeCapacityConfig
    from factorminer.evaluation.capacity import CapacityEstimator

    cap_cfg = RuntimeCapacityConfig(capacity_levels=list(capacity_levels))
    estimate = CapacityEstimator(
        np.asarray(returns, dtype=np.float64).T,
        np.asarray(volume, dtype=np.float64),
        cap_cfg,
    ).estimate(
        factor_name,
        np.asarray(signals, dtype=np.float64).T,
    )
    return {
        "factor_name": factor_name,
        "max_capacity_usd": float(estimate.max_capacity_usd),
        "break_even_cost_bps": float(estimate.break_even_cost_bps),
        "capacity_curve": {
            str(capital): float(degradation)
            for capital, degradation in estimate.capacity_curve.items()
        },
    }


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
        and artifact.split_stats[split_name]["ic_paper_mean"] >= min_ic
        and artifact.split_stats[split_name]["ic_paper_icir"] >= min_icir
    ]
    admitted.sort(
        key=lambda artifact: artifact.split_stats[split_name]["ic_paper_mean"],
        reverse=True,
    )
    selected: list[FactorEvaluationArtifact] = admitted[:top_k]
    selected_formulas = {artifact.formula for artifact in selected}

    if len(selected) < top_k:
        remainder = [
            artifact for artifact in succeeded if artifact.formula not in selected_formulas
        ]
        remainder.sort(
            key=lambda artifact: artifact.split_stats[split_name]["ic_paper_mean"],
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
    valid_ic = ic_series[np.isfinite(ic_series)]
    signed_ic = float(stats.get("ic_mean", 0.0))
    paper_ic = abs(signed_ic)
    return {
        "metric_version": METRIC_VERSION,
        "ic_definition": "spearman_rank",
        "ic": paper_ic,
        "ic_mean": signed_ic,
        "ic_paper_mean": paper_ic,
        "ic_abs_mean": float(np.mean(np.abs(valid_ic))) if valid_ic.size else 0.0,
        "icir": _abs_icir_from_series(ic_series),
        "ic_win_rate": float(stats.get("ic_win_rate", 0.0)),
        "rank_ic_mean": float(stats.get("rank_ic_mean", signed_ic)),
        "rank_ic_paper_mean": abs(float(stats.get("rank_ic_mean", signed_ic))),
        "rank_icir": float(stats.get("rank_icir", stats.get("icir", 0.0))),
        "rank_ic_paper_icir": abs(float(stats.get("rank_icir", stats.get("icir", 0.0)))),
        "pearson_ic_mean": float(stats.get("pearson_ic_mean", 0.0)),
        "pearson_ic_paper_mean": abs(float(stats.get("pearson_ic_mean", 0.0))),
        "pearson_icir": float(stats.get("pearson_icir", 0.0)),
        "pearson_ic_paper_icir": abs(float(stats.get("pearson_icir", 0.0))),
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
    cost_bps: list[float] | None = None,
    capacity_levels: list[float] | None = None,
) -> dict:
    """Evaluate one frozen factor set on one universe."""
    if cost_bps is None:
        cost_bps = [1.0, 4.0, 7.0, 10.0, 11.0]
    if capacity_levels is None:
        capacity_levels = _default_capacity_levels()

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
            "rank_ic": 0.0,
            "rank_ic_mean": 0.0,
            "rank_icir": 0.0,
            "rank_ic_paper_icir": 0.0,
            "pearson_ic": 0.0,
            "pearson_ic_mean": 0.0,
            "pearson_icir": 0.0,
            "pearson_ic_paper_icir": 0.0,
            "avg_abs_rho": 0.0,
        },
        "combinations": {},
        "selections": {},
        "stress": {
            "cost_bps": [float(value) for value in cost_bps],
            "capacity_levels": [float(value) for value in capacity_levels],
        },
        "warnings": [],
    }
    if not succeeded:
        result["warnings"].append("No frozen factors recomputed successfully on this universe")
        return result

    result["library"] = {
        "ic": float(
            np.mean([artifact.split_stats[split_name]["ic_paper_mean"] for artifact in succeeded])
        ),
        "icir": float(
            np.mean([artifact.split_stats[split_name]["ic_paper_icir"] for artifact in succeeded])
        ),
        "metric_version": METRIC_VERSION,
        "ic_definition": "spearman_rank",
        "rank_ic": float(
            np.mean(
                [artifact.split_stats[split_name]["rank_ic_paper_mean"] for artifact in succeeded]
            )
        ),
        "rank_ic_mean": float(
            np.mean([artifact.split_stats[split_name]["rank_ic_mean"] for artifact in succeeded])
        ),
        "rank_icir": float(
            np.mean([artifact.split_stats[split_name]["rank_icir"] for artifact in succeeded])
        ),
        "rank_ic_paper_icir": float(
            np.mean(
                [artifact.split_stats[split_name]["rank_ic_paper_icir"] for artifact in succeeded]
            )
        ),
        "pearson_ic": float(
            np.mean(
                [
                    artifact.split_stats[split_name]["pearson_ic_paper_mean"]
                    for artifact in succeeded
                ]
            )
        ),
        "pearson_ic_mean": float(
            np.mean([artifact.split_stats[split_name]["pearson_ic_mean"] for artifact in succeeded])
        ),
        "pearson_icir": float(
            np.mean([artifact.split_stats[split_name]["pearson_icir"] for artifact in succeeded])
        ),
        "pearson_ic_paper_icir": float(
            np.mean(
                [
                    artifact.split_stats[split_name]["pearson_ic_paper_icir"]
                    for artifact in succeeded
                ]
            )
        ),
        "avg_abs_rho": _avg_abs_rho(succeeded, split_name),
    }

    artifact_map = {artifact.factor_id: artifact for artifact in succeeded}
    fit_signals = {
        artifact.factor_id: artifact.split_signals[fit_split].T for artifact in succeeded
    }
    eval_signals = {
        artifact.factor_id: artifact.split_signals[split_name].T for artifact in succeeded
    }
    fit_returns = dataset.get_split(fit_split).returns.T
    eval_returns = dataset.get_split(split_name).returns.T
    eval_volume = _split_volume_panel(dataset, split_name)

    from factorminer.evaluation.combination import FactorCombiner
    from factorminer.evaluation.portfolio import PortfolioBacktester
    from factorminer.evaluation.selection import FactorSelector

    combiner = FactorCombiner()
    backtester = PortfolioBacktester()
    selector = FactorSelector()

    fit_ic_values = {
        artifact.factor_id: artifact.split_stats[fit_split]["ic_mean"] for artifact in succeeded
    }

    combos = {
        "equal_weight": combiner.equal_weight(eval_signals),
        "ic_weighted": combiner.ic_weighted(eval_signals, fit_ic_values),
        "orthogonal": combiner.orthogonal(eval_signals),
    }
    for name, composite in combos.items():
        stats = backtester.quintile_backtest(composite, eval_returns)
        result["combinations"][name] = _normalize_backtest_stats(stats)
        result["combinations"][name]["ic_series"] = _json_safe(
            np.asarray(stats.get("ic_series", []), dtype=np.float64).tolist()
        )
        result["combinations"][name]["turnover_series"] = _json_safe(
            np.asarray(stats.get("turnover_series", []), dtype=np.float64).tolist()
        )
        result["combinations"][name]["cost_pressure"] = {
            str(cost): _normalize_backtest_stats(
                backtester.quintile_backtest(
                    composite, eval_returns, transaction_cost_bps=float(cost)
                )
            )
            for cost in cost_bps
        }
        if eval_volume is not None:
            result["combinations"][name]["capacity_pressure"] = _capacity_pressure_summary(
                factor_name=name,
                signals=composite,
                returns=eval_returns,
                volume=eval_volume,
                capacity_levels=capacity_levels,
            )

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
                factor_id: score
                * np.sign(artifact_map[factor_id].split_stats[fit_split]["ic_mean"] or 1.0)
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
            "ic_series": _json_safe(
                np.asarray(stats.get("ic_series", []), dtype=np.float64).tolist()
            ),
            "turnover_series": _json_safe(
                np.asarray(stats.get("turnover_series", []), dtype=np.float64).tolist()
            ),
        }

    return result
