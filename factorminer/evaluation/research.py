"""Research-first multi-horizon scoring and model evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from factorminer.evaluation.backtest import rolling_splits
from factorminer.evaluation.metrics import compute_factor_stats, compute_pairwise_correlation
from factorminer.evaluation.portfolio import PortfolioBacktester
from factorminer.evaluation.regime import RegimeAwareEvaluator, RegimeConfig, RegimeDetector
from factorminer.evaluation.selection import FactorSelector
from factorminer.evaluation.significance import BootstrapICTester, SignificanceConfig


@dataclass
class FactorGeometryDiagnostics:
    """How much new information a factor adds beyond the current library."""

    max_abs_correlation: float = 0.0
    mean_abs_correlation: float = 0.0
    projection_loss: float = 0.0
    marginal_span_gain: float = 1.0
    effective_rank_gain: float = 1.0
    residual_ic: float = 0.0


@dataclass
class FactorScoreVector:
    """Multi-horizon quality summary used in research mode."""

    primary_objective: str
    primary_score: float
    lower_confidence_bound: float
    weighted_score: float
    decay_slope: float
    cross_horizon_consistency: float
    average_turnover: float
    geometry: FactorGeometryDiagnostics
    per_horizon_ic_mean: dict[str, float] = field(default_factory=dict)
    per_horizon_icir: dict[str, float] = field(default_factory=dict)
    per_horizon_shrunk_ic: dict[str, float] = field(default_factory=dict)
    per_horizon_se: dict[str, float] = field(default_factory=dict)
    per_horizon_lcb: dict[str, float] = field(default_factory=dict)
    per_horizon_turnover: dict[str, float] = field(default_factory=dict)
    pareto_dominant: bool = True

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["geometry"] = asdict(self.geometry)
        return payload


def compute_factor_geometry(
    candidate_signals: np.ndarray,
    returns: np.ndarray,
    library_signals: Sequence[np.ndarray] | None = None,
) -> FactorGeometryDiagnostics:
    """Compute soft library geometry metrics for a candidate."""
    library_signals = list(library_signals or [])
    if not library_signals:
        return FactorGeometryDiagnostics(
            max_abs_correlation=0.0,
            mean_abs_correlation=0.0,
            projection_loss=0.0,
            marginal_span_gain=1.0,
            effective_rank_gain=1.0,
            residual_ic=float(compute_factor_stats(candidate_signals, returns)["ic_abs_mean"]),
        )

    corrs = [
        abs(compute_pairwise_correlation(candidate_signals, lib_signal))
        for lib_signal in library_signals
    ]
    flattened_candidate, valid_mask = _flatten_panel(candidate_signals)
    library_vectors = []
    for signal in library_signals:
        flattened_signal, _ = _flatten_panel(signal, valid_mask=valid_mask)
        library_vectors.append(flattened_signal)

    if not library_vectors:
        return FactorGeometryDiagnostics(
            max_abs_correlation=max(corrs, default=0.0),
            mean_abs_correlation=float(np.mean(corrs)) if corrs else 0.0,
            residual_ic=float(compute_factor_stats(candidate_signals, returns)["ic_abs_mean"]),
        )

    design = np.column_stack(library_vectors)
    response = flattened_candidate
    if design.size == 0 or response.size == 0 or np.nanstd(response) < 1e-12:
        projection_loss = 0.0
        marginal_span_gain = 1.0
        residual_matrix = candidate_signals
    else:
        beta, *_ = np.linalg.lstsq(design, response, rcond=None)
        fitted = design @ beta
        residual = response - fitted
        response_var = float(np.var(response))
        residual_var = float(np.var(residual))
        marginal_span_gain = residual_var / response_var if response_var > 1e-12 else 0.0
        projection_loss = 1.0 - marginal_span_gain
        residual_matrix = _unflatten_panel(residual, valid_mask, candidate_signals.shape)

    before_rank = _effective_rank(design)
    after_rank = _effective_rank(np.column_stack([design, response]))
    residual_ic = float(compute_factor_stats(residual_matrix, returns)["ic_abs_mean"])

    return FactorGeometryDiagnostics(
        max_abs_correlation=max(corrs, default=0.0),
        mean_abs_correlation=float(np.mean(corrs)) if corrs else 0.0,
        projection_loss=float(projection_loss),
        marginal_span_gain=float(max(marginal_span_gain, 0.0)),
        effective_rank_gain=float(after_rank - before_rank),
        residual_ic=residual_ic,
    )


def build_score_vector(
    target_stats: dict[str, dict],
    target_horizons: dict[str, int],
    research_cfg,
    geometry: FactorGeometryDiagnostics,
) -> FactorScoreVector:
    """Aggregate per-target metrics into one research-mode score vector."""
    weights = _normalized_weights(
        target_stats.keys(),
        explicit_weights=getattr(research_cfg, "horizon_weights", {}),
    )
    uncertainty_cfg = research_cfg.uncertainty
    admission_cfg = research_cfg.admission

    per_horizon_ic_mean: dict[str, float] = {}
    per_horizon_icir: dict[str, float] = {}
    per_horizon_shrunk_ic: dict[str, float] = {}
    per_horizon_se: dict[str, float] = {}
    per_horizon_lcb: dict[str, float] = {}
    per_horizon_turnover: dict[str, float] = {}

    for target_name, stats in target_stats.items():
        ic_series = np.asarray(stats.get("ic_series", np.array([])), dtype=np.float64)
        se = _bootstrap_standard_error(ic_series, uncertainty_cfg)
        ic_abs_mean = float(stats.get("ic_abs_mean", 0.0))
        shrunk_ic = max(ic_abs_mean - uncertainty_cfg.shrinkage_strength * se, 0.0)
        lcb = ic_abs_mean - uncertainty_cfg.lcb_zscore * se

        per_horizon_ic_mean[target_name] = float(stats.get("ic_mean", 0.0))
        per_horizon_icir[target_name] = float(stats.get("icir", 0.0))
        per_horizon_shrunk_ic[target_name] = float(shrunk_ic)
        per_horizon_se[target_name] = float(se)
        per_horizon_lcb[target_name] = float(lcb)
        per_horizon_turnover[target_name] = float(stats.get("turnover", 0.0))

    weighted_quality = float(
        sum(weights[name] * per_horizon_shrunk_ic.get(name, 0.0) for name in weights)
    )
    average_turnover = float(
        np.mean(list(per_horizon_turnover.values())) if per_horizon_turnover else 0.0
    )
    lower_confidence_bound = float(
        min(per_horizon_lcb.values()) if per_horizon_lcb else 0.0
    )
    redundancy_penalty = admission_cfg.redundancy_penalty * geometry.max_abs_correlation
    turnover_penalty = admission_cfg.turnover_penalty * average_turnover
    geometry_bonus = 0.0
    if admission_cfg.use_residual_ic:
        geometry_bonus += 0.5 * geometry.residual_ic
    if admission_cfg.use_effective_rank_gain:
        geometry_bonus += 0.05 * max(geometry.effective_rank_gain, 0.0)

    weighted_score = weighted_quality - redundancy_penalty - turnover_penalty + geometry_bonus
    decay_slope = _decay_slope(target_horizons, per_horizon_shrunk_ic)
    consistency = _cross_horizon_consistency(per_horizon_ic_mean)

    return FactorScoreVector(
        primary_objective=research_cfg.primary_objective,
        primary_score=weighted_score,
        lower_confidence_bound=lower_confidence_bound,
        weighted_score=weighted_score,
        decay_slope=decay_slope,
        cross_horizon_consistency=consistency,
        average_turnover=average_turnover,
        geometry=geometry,
        per_horizon_ic_mean=per_horizon_ic_mean,
        per_horizon_icir=per_horizon_icir,
        per_horizon_shrunk_ic=per_horizon_shrunk_ic,
        per_horizon_se=per_horizon_se,
        per_horizon_lcb=per_horizon_lcb,
        per_horizon_turnover=per_horizon_turnover,
    )


def passes_research_admission(
    score_vector: FactorScoreVector,
    research_cfg,
    correlation_threshold: float,
) -> tuple[bool, str]:
    """Apply research-mode admission rules on top of paper-style correlation."""
    admission_cfg = research_cfg.admission
    if score_vector.primary_score < admission_cfg.min_score:
        return False, (
            f"Research score {score_vector.primary_score:.4f} "
            f"< {admission_cfg.min_score:.4f}"
        )
    if score_vector.lower_confidence_bound < admission_cfg.min_lcb:
        return False, (
            f"Research LCB {score_vector.lower_confidence_bound:.4f} "
            f"< {admission_cfg.min_lcb:.4f}"
        )
    if score_vector.geometry.max_abs_correlation < correlation_threshold:
        return True, "Research score passes direct admission"
    if (
        admission_cfg.use_residual_ic
        and score_vector.geometry.residual_ic >= admission_cfg.min_score
        and score_vector.geometry.marginal_span_gain >= admission_cfg.min_span_gain
        and (
            (not admission_cfg.use_effective_rank_gain)
            or score_vector.geometry.effective_rank_gain >= admission_cfg.min_effective_rank_gain
        )
    ):
        return True, "Research geometry passes residual-span admission"
    return False, (
        "Too redundant under research geometry: "
        f"max|rho|={score_vector.geometry.max_abs_correlation:.4f}, "
        f"residual_ic={score_vector.geometry.residual_ic:.4f}, "
        f"span_gain={score_vector.geometry.marginal_span_gain:.4f}"
    )


def run_research_model_suite(
    factor_signals: dict[int, np.ndarray],
    returns: np.ndarray,
    research_cfg,
) -> dict[str, dict]:
    """Fit research-mode models on rolling windows and report net IR/stability."""
    if not factor_signals:
        return {}

    selector = FactorSelector()
    backtester = PortfolioBacktester()
    splits = rolling_splits(
        returns.shape[0],
        train_window=research_cfg.selection.rolling_train_window,
        test_window=research_cfg.selection.rolling_test_window,
        step=research_cfg.selection.rolling_step,
    )
    if not splits:
        return {}

    reports: dict[str, dict] = {}
    for model_name in research_cfg.selection.models:
        fold_reports = []
        selected_sets = []
        for split in splits:
            train_returns = returns[split.train_start:split.train_end]
            test_returns = returns[split.test_start:split.test_end]
            train_signals = {
                fid: signal[split.train_start:split.train_end]
                for fid, signal in factor_signals.items()
            }
            test_signals = {
                fid: signal[split.test_start:split.test_end]
                for fid, signal in factor_signals.items()
            }
            try:
                selected, weights = _fit_research_model(
                    selector,
                    model_name,
                    train_signals,
                    train_returns,
                )
            except ImportError as exc:
                reports[model_name] = {"available": False, "error": str(exc)}
                fold_reports = []
                break
            if not selected:
                continue
            selected_sets.append(set(selected))
            composite = _weighted_composite(test_signals, weights)
            stats = backtester.quintile_backtest(
                composite,
                test_returns,
                transaction_cost_bps=research_cfg.execution.cost_bps,
            )
            regime_report = None
            if research_cfg.regimes.enabled:
                regime_report = _composite_regime_report(composite, test_returns)
            fold_reports.append(
                {
                    "selected_ids": selected,
                    "weights": weights,
                    "test_ic_mean": float(stats["ic_mean"]),
                    "test_icir": float(stats["icir"]),
                    "test_net_ir": _series_ir(stats["ls_net_series"]),
                    "avg_turnover": float(stats["avg_turnover"]),
                    "regimes": regime_report,
                }
            )

        if not fold_reports:
            reports.setdefault(model_name, {"available": True, "folds": []})
            continue

        reports[model_name] = {
            "available": True,
            "folds": fold_reports,
            "mean_test_ic_mean": float(np.mean([fold["test_ic_mean"] for fold in fold_reports])),
            "mean_test_icir": float(np.mean([fold["test_icir"] for fold in fold_reports])),
            "mean_test_net_ir": float(np.mean([fold["test_net_ir"] for fold in fold_reports])),
            "mean_turnover": float(np.mean([fold["avg_turnover"] for fold in fold_reports])),
            "selection_stability": _selection_stability(selected_sets),
        }

    return reports


def _fit_research_model(
    selector: FactorSelector,
    model_name: str,
    factor_signals: dict[int, np.ndarray],
    returns: np.ndarray,
) -> tuple[list[int], dict[int, float]]:
    if model_name == "ridge":
        from sklearn.linear_model import RidgeCV

        ids, X, y = selector._prepare_panel(factor_signals, returns)  # noqa: SLF001
        if len(ids) == 0:
            return [], {}
        model = RidgeCV(alphas=np.logspace(-4, 2, 12))
        model.fit(X, y)
        weights = {ids[idx]: float(coef) for idx, coef in enumerate(model.coef_)}
        selected = [factor_id for factor_id, weight in weights.items() if abs(weight) > 1e-10]
        return selected, {factor_id: weights[factor_id] for factor_id in selected}

    if model_name == "elastic_net":
        from sklearn.linear_model import ElasticNetCV

        ids, X, y = selector._prepare_panel(factor_signals, returns)  # noqa: SLF001
        if len(ids) == 0:
            return [], {}
        model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, max_iter=10000)
        model.fit(X, y)
        weights = {ids[idx]: float(coef) for idx, coef in enumerate(model.coef_)}
        selected = [factor_id for factor_id, weight in weights.items() if abs(weight) > 1e-10]
        return selected, {factor_id: weights[factor_id] for factor_id in selected}

    if model_name == "lasso":
        results = selector.lasso_selection(factor_signals, returns)
        selected = [factor_id for factor_id, _ in results]
        return selected, {factor_id: score for factor_id, score in results}

    if model_name == "stepwise":
        results = selector.forward_stepwise(factor_signals, returns)
        selected = [factor_id for factor_id, _ in results]
        return selected, {factor_id: 1.0 for factor_id in selected}

    if model_name == "xgboost":
        results = selector.xgboost_selection(factor_signals, returns)
        selected = [factor_id for factor_id, _ in results[: max(1, min(10, len(results)))]]
        return selected, {factor_id: score for factor_id, score in results if factor_id in selected}

    raise ValueError(f"Unknown research model: {model_name}")


def _weighted_composite(
    factor_signals: dict[int, np.ndarray],
    weights: dict[int, float],
) -> np.ndarray:
    selected_signals = {fid: factor_signals[fid] for fid in weights if fid in factor_signals}
    if not selected_signals:
        raise ValueError("No selected signals available for composite")
    raw_weights = np.array([abs(weights[fid]) for fid in selected_signals], dtype=np.float64)
    if raw_weights.sum() < 1e-12:
        raw_weights = np.ones_like(raw_weights)
    normalized_weights = raw_weights / raw_weights.sum()

    composite = np.zeros_like(next(iter(selected_signals.values())), dtype=np.float64)
    for idx, fid in enumerate(selected_signals):
        signal = selected_signals[fid].astype(np.float64)
        cs_mean = np.nanmean(signal, axis=1, keepdims=True)
        cs_std = np.nanstd(signal, axis=1, keepdims=True)
        cs_std = np.where(cs_std == 0.0, 1.0, cs_std)
        standardized = (signal - cs_mean) / cs_std
        composite += normalized_weights[idx] * np.where(np.isnan(standardized), 0.0, standardized)
    return composite


def _bootstrap_standard_error(ic_series: np.ndarray, uncertainty_cfg) -> float:
    valid = ic_series[np.isfinite(ic_series)]
    if len(valid) < 3:
        return 0.0
    tester = BootstrapICTester(
        SignificanceConfig(
            bootstrap_n_samples=uncertainty_cfg.bootstrap_samples,
            bootstrap_block_size=uncertainty_cfg.block_size,
            seed=42,
        )
    )
    result = tester.compute_ci("research", valid)
    return float(result.ic_std_boot)


def _normalized_weights(
    target_names: Iterable[str],
    explicit_weights: dict[str, float],
) -> dict[str, float]:
    target_names = list(target_names)
    if not target_names:
        return {}
    if explicit_weights:
        weights = np.array([max(float(explicit_weights.get(name, 0.0)), 0.0) for name in target_names])
        if weights.sum() > 1e-12:
            normalized = weights / weights.sum()
            return {name: float(normalized[idx]) for idx, name in enumerate(target_names)}
    equal = 1.0 / len(target_names)
    return {name: equal for name in target_names}


def _decay_slope(target_horizons: dict[str, int], shrunk_ic: dict[str, float]) -> float:
    aligned = [
        (target_horizons[name], value)
        for name, value in shrunk_ic.items()
        if name in target_horizons
    ]
    if len(aligned) < 2:
        return 0.0
    horizons = np.array([item[0] for item in aligned], dtype=np.float64)
    scores = np.array([item[1] for item in aligned], dtype=np.float64)
    if np.std(horizons) < 1e-12:
        return 0.0
    slope, _ = np.polyfit(horizons, scores, 1)
    return float(slope)


def _cross_horizon_consistency(per_horizon_ic_mean: dict[str, float]) -> float:
    values = [value for value in per_horizon_ic_mean.values() if abs(value) > 1e-12]
    if not values:
        return 0.0
    signs = np.sign(values)
    majority = np.sign(np.sum(signs))
    if majority == 0:
        return 0.0
    return float(np.mean(signs == majority))


def _flatten_panel(panel: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(panel, dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.isfinite(matrix)
    centered = np.where(valid_mask, matrix, np.nan)
    cs_mean = np.nanmean(centered, axis=0, keepdims=True)
    cs_std = np.nanstd(centered, axis=0, keepdims=True)
    cs_std = np.where(cs_std < 1e-12, 1.0, cs_std)
    standardized = (centered - cs_mean) / cs_std
    filled = np.where(np.isfinite(standardized), standardized, 0.0)
    return filled.reshape(-1), valid_mask


def _unflatten_panel(flat: np.ndarray, valid_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.full(shape, np.nan, dtype=np.float64)
    matrix[valid_mask] = flat.reshape(shape)[valid_mask]
    return matrix


def _effective_rank(matrix: np.ndarray) -> float:
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        return 0.0
    cov = matrix.T @ matrix
    singular_values = np.linalg.svd(cov, compute_uv=False)
    singular_values = singular_values[singular_values > 1e-12]
    if len(singular_values) == 0:
        return 0.0
    probs = singular_values / singular_values.sum()
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def _selection_stability(selected_sets: Sequence[set[int]]) -> float:
    if len(selected_sets) < 2:
        return 1.0 if selected_sets else 0.0
    overlaps = []
    for idx in range(len(selected_sets) - 1):
        left = selected_sets[idx]
        right = selected_sets[idx + 1]
        union = left | right
        overlaps.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(overlaps))


def _series_ir(series: np.ndarray) -> float:
    valid = np.asarray(series, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if len(valid) < 2:
        return 0.0
    std = float(np.std(valid, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(valid) / std)


def _composite_regime_report(composite: np.ndarray, returns: np.ndarray) -> dict:
    detector = RegimeDetector(RegimeConfig())
    classification = detector.classify(returns.T)
    evaluator = RegimeAwareEvaluator(returns.T, classification, RegimeConfig())
    regime_result = evaluator.evaluate("composite", composite.T)
    regime_net_ir = {}
    backtester = PortfolioBacktester()
    stats = backtester.quintile_backtest(composite, returns)
    for regime, mask in classification.periods.items():
        regime_net_ir[regime.name] = _series_ir(stats["ls_net_series"][mask])
    return {
        "regime_score": regime_result.overall_regime_score,
        "n_regimes_passing": regime_result.n_regimes_passing,
        "regime_ic": {regime.name: value for regime, value in regime_result.regime_ic.items()},
        "regime_icir": {regime.name: value for regime, value in regime_result.regime_icir.items()},
        "regime_net_ir": regime_net_ir,
    }
