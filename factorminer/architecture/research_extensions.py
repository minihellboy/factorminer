"""High-value research extension services for family, regime, dependence, and utility."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.stats import rankdata

from factorminer.architecture.dependence import DistanceCorrelationMetric
from factorminer.architecture.families import FactorFamilyDiscovery, infer_family

try:  # pragma: no cover - optional import guarded for portability
    from sklearn.feature_selection import mutual_info_regression
except Exception:  # pragma: no cover - sklearn should be available, but keep a fallback
    mutual_info_regression = None  # type: ignore[assignment]


_REGIME_KEYWORDS = {
    "bull": ("momentum", "trend", "breakout", "strength", "volume"),
    "bear": ("reversal", "defensive", "quality", "volatility", "liquidity"),
    "sideways": ("mean reversion", "range", "oscillator", "dispersion", "spread"),
    "unknown": ("diversification", "robustness", "stability", "orthogonality"),
}


def _to_float_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return arr.reshape(1)
    return arr


def _as_panel(signal: Any) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D panel of shape (assets, periods); got {arr.shape}")
    return arr


def _safe_nanmean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if np.isfinite(value)]
    if not clean:
        return 0.0
    return float(np.mean(clean))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return 0.0
    x = x[valid]
    y = y[valid]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(valid.sum()) < 3:
        return 0.0
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total <= 1e-12:
        return 0.0
    residual = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - residual / total)


def _standardize_train_test(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if train_matrix.size == 0:
        return train_matrix, test_matrix
    mean = np.mean(train_matrix, axis=0, keepdims=True)
    std = np.std(train_matrix, axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (train_matrix - mean) / std, (test_matrix - mean) / std


def _stack_reference_collection(collection: Sequence[Any] | np.ndarray | None) -> list[np.ndarray]:
    if collection is None:
        return []
    if isinstance(collection, np.ndarray):
        if collection.ndim == 2:
            return [collection]
        if collection.ndim == 3:
            return [collection[index] for index in range(collection.shape[0])]
        raise ValueError(f"Expected 2D or 3D array for signal collection, got {collection.shape}")
    return [_as_panel(signal) for signal in collection]


def _flatten_period_samples(
    panels: Sequence[np.ndarray],
    returns: np.ndarray,
    period_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not panels:
        raise ValueError("At least one panel is required to build samples")
    assets, periods = panels[0].shape
    for panel in panels[1:]:
        if panel.shape != (assets, periods):
            raise ValueError("All panels must share the same (assets, periods) shape")
    if returns.shape != (assets, periods):
        raise ValueError("Returns must match signal panel shape")

    rows: list[list[float]] = []
    targets: list[float] = []
    for period in period_indices:
        period_panels = [panel[:, period] for panel in panels]
        period_returns = returns[:, period]
        valid = np.isfinite(period_returns)
        for panel in period_panels:
            valid &= np.isfinite(panel)
        if int(valid.sum()) < 3:
            continue
        for row_index in np.where(valid)[0]:
            rows.append([float(panel[row_index]) for panel in period_panels])
            targets.append(float(period_returns[row_index]))
    if not rows:
        return np.empty((0, len(panels)), dtype=np.float64), np.empty((0,), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


def _linear_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if train_x.shape[1] == 0:
        mean_y = float(np.mean(train_y)) if train_y.size else 0.0
        return (
            np.full(train_y.shape, mean_y, dtype=np.float64),
            np.full(test_x.shape[0], mean_y, dtype=np.float64),
        )

    train_x_std, test_x_std = _standardize_train_test(train_x, test_x)
    train_aug = np.column_stack([np.ones(train_x_std.shape[0], dtype=np.float64), train_x_std])
    test_aug = np.column_stack([np.ones(test_x_std.shape[0], dtype=np.float64), test_x_std])
    coef = np.linalg.lstsq(train_aug, train_y, rcond=None)[0]
    return train_aug @ coef, test_aug @ coef


def _gaussian_copula_mi_proxy(x: np.ndarray, y: np.ndarray) -> float:
    rho = abs(_safe_corr(rankdata(x), rankdata(y)))
    rho = min(max(rho, 0.0), 0.999999)
    if rho <= 0.0:
        return 0.0
    return float(-0.5 * np.log(max(1.0 - rho**2, 1e-12)))


def _mutual_information_proxy(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 5:
        return 0.0, "gaussian_copula"

    x = x[valid]
    y = y[valid]
    if mutual_info_regression is None:
        return _gaussian_copula_mi_proxy(x, y), "gaussian_copula"

    x_ranked = rankdata(x).reshape(-1, 1)
    y_ranked = rankdata(y)
    neighbors = max(2, min(5, x_ranked.shape[0] - 1))
    try:
        raw = float(
            mutual_info_regression(
                x_ranked,
                y_ranked,
                random_state=0,
                n_neighbors=neighbors,
            )[0]
        )
    except Exception:
        return _gaussian_copula_mi_proxy(x, y), "gaussian_copula"

    return float(max(raw, 0.0)), "sklearn_mutual_info_regression"


def _periodwise_nonlinear_scores(
    candidate: np.ndarray, reference: np.ndarray
) -> dict[str, float | int | str]:
    candidate = _as_panel(candidate)
    reference = _as_panel(reference)
    if candidate.shape != reference.shape:
        raise ValueError(f"Panel shapes must match: {candidate.shape} vs {reference.shape}")

    distance_metric = DistanceCorrelationMetric()
    distance_scores: list[float] = []
    pearson_scores: list[float] = []
    spearman_scores: list[float] = []
    mi_scores: list[float] = []
    sample_count = 0
    period_count = 0
    mi_method = "gaussian_copula"

    for period in range(candidate.shape[1]):
        col_candidate = candidate[:, period]
        col_reference = reference[:, period]
        valid = np.isfinite(col_candidate) & np.isfinite(col_reference)
        if int(valid.sum()) < 5:
            continue
        period_count += 1
        sample_count += int(valid.sum())
        col_candidate = col_candidate[valid]
        col_reference = col_reference[valid]
        distance_scores.append(
            float(
                distance_metric.compute(col_candidate.reshape(-1, 1), col_reference.reshape(-1, 1))
            )
        )
        pearson_scores.append(abs(_safe_corr(col_candidate, col_reference)))
        spearman_scores.append(abs(_safe_corr(rankdata(col_candidate), rankdata(col_reference))))
        mi_raw, mi_method = _mutual_information_proxy(col_candidate, col_reference)
        mi_scores.append(float(1.0 - np.exp(-max(mi_raw, 0.0))))

    distance_mean = _safe_nanmean(distance_scores)
    pearson_mean = _safe_nanmean(pearson_scores)
    spearman_mean = _safe_nanmean(spearman_scores)
    mi_proxy = _safe_nanmean(mi_scores)

    return {
        "sample_count": sample_count,
        "period_count": period_count,
        "distance_correlation": distance_mean,
        "pearson_abs": pearson_mean,
        "spearman_abs": spearman_mean,
        "mutual_information_proxy": mi_proxy,
        "mutual_information_method": mi_method,
        "combined_score": max(distance_mean, mi_proxy),
    }


def _normalize_formula_entry(
    entry: str | Mapping[str, Any],
    *,
    admitted: bool,
) -> dict[str, Any]:
    if isinstance(entry, str):
        payload = {"formula": entry, "name": entry, "ic_mean": 0.0}
    else:
        payload = dict(entry)
    payload["formula"] = str(payload.get("formula", "") or "")
    payload["name"] = str(payload.get("name", "") or payload["formula"] or "Unnamed")
    payload["ic_mean"] = float(payload.get("ic_mean", 0.0) or 0.0)
    payload["admitted"] = bool(admitted)
    category = str(payload.get("category", "") or "").strip()
    if category:
        payload["category"] = category
    elif payload["formula"]:
        payload["category"] = infer_family(payload["formula"])
    else:
        payload["category"] = "Other"
    return payload


@dataclass(frozen=True)
class FamilySummary:
    """Per-family summary with admission and rejection counts."""

    name: str
    count: int
    admitted_count: int
    rejected_count: int
    acceptance_rate: float
    average_ic: float
    operators: dict[str, int] = field(default_factory=dict)
    features: dict[str, int] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FamilyContextSummary:
    """Structured family context for prompt construction."""

    families: list[FamilySummary]
    admitted_count: int
    rejected_count: int
    saturated_families: list[str]
    underexplored_families: list[str]
    recommended_families: list[str]
    rejection_heavy_families: list[str]
    prompt_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegimeMemorySummary:
    """Structured regime-aware memory summary."""

    labels: list[str]
    counts: dict[str, int]
    frequencies: dict[str, float]
    dominant_regime: str
    recent_regime: str
    transition_matrix: dict[str, dict[str, float]]
    metrics: dict[str, dict[str, dict[str, float]]]
    regime_keywords: list[str]
    prompt_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DependenceProfile:
    """Nonlinear dependence profile between two signal panels."""

    candidate_name: str
    reference_name: str
    sample_count: int
    period_count: int
    distance_correlation: float
    pearson_abs: float
    spearman_abs: float
    mutual_information_proxy: float
    mutual_information_method: str
    combined_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnsembleUtilitySummary:
    """Marginal utility estimate for a candidate signal against an ensemble."""

    candidate_name: str
    reference_names: list[str]
    sample_count_train: int
    sample_count_test: int
    train_periods: list[int]
    test_periods: list[int]
    base_ic: float
    augmented_ic: float
    delta_ic: float
    base_r2: float
    augmented_r2: float
    delta_r2: float
    residual_ic: float
    max_existing_distance_correlation: float
    max_existing_mutual_information_proxy: float
    mean_existing_distance_correlation: float
    mean_existing_mutual_information_proxy: float
    prompt_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FamilyContextService:
    """Derive factor-family context from admitted and rejected formulas."""

    def __init__(self, discovery: FactorFamilyDiscovery | None = None) -> None:
        self.discovery = discovery or FactorFamilyDiscovery()

    def summarize(
        self,
        admitted_formulas: Sequence[str | Mapping[str, Any]] | None,
        rejected_formulas: Sequence[str | Mapping[str, Any]] | None,
        *,
        library_state: dict[str, Any] | None = None,
        memory_signal: dict[str, Any] | None = None,
    ) -> FamilyContextSummary:
        admitted_entries = [
            _normalize_formula_entry(entry, admitted=True) for entry in (admitted_formulas or [])
        ]
        rejected_entries = [
            _normalize_formula_entry(entry, admitted=False) for entry in (rejected_formulas or [])
        ]
        all_entries = admitted_entries + rejected_entries
        discovered = self.discovery.discover(all_entries)

        family_records: list[FamilySummary] = []
        rejected_by_family: dict[str, int] = {}
        admitted_by_family: dict[str, int] = {}
        ic_totals: dict[str, float] = {}
        for entry in all_entries:
            family_name = infer_family(entry["formula"]) if entry["formula"] else entry["category"]
            admitted_by_family[family_name] = admitted_by_family.get(family_name, 0) + int(
                bool(entry["admitted"])
            )
            rejected_by_family[family_name] = rejected_by_family.get(family_name, 0) + int(
                not bool(entry["admitted"])
            )
            ic_totals[family_name] = ic_totals.get(family_name, 0.0) + float(entry["ic_mean"])

        for family in discovered:
            rejected_count = rejected_by_family.get(family.name, 0)
            admitted_count = admitted_by_family.get(family.name, 0)
            family_records.append(
                FamilySummary(
                    name=family.name,
                    count=family.count,
                    admitted_count=admitted_count,
                    rejected_count=rejected_count,
                    acceptance_rate=float(admitted_count / max(admitted_count + rejected_count, 1)),
                    average_ic=float(ic_totals.get(family.name, 0.0) / max(family.count, 1)),
                    operators=dict(family.operators),
                    features=dict(family.features),
                    examples=list(family.examples),
                )
            )

        for family_name, rejected_count in rejected_by_family.items():
            if any(family.name == family_name for family in family_records):
                continue
            admitted_count = admitted_by_family.get(family_name, 0)
            count = admitted_count + rejected_count
            family_records.append(
                FamilySummary(
                    name=family_name,
                    count=count,
                    admitted_count=admitted_count,
                    rejected_count=rejected_count,
                    acceptance_rate=float(admitted_count / max(count, 1)),
                    average_ic=float(ic_totals.get(family_name, 0.0) / max(count, 1)),
                )
            )

        family_records.sort(
            key=lambda family: (family.admitted_count, family.count, family.average_ic),
            reverse=True,
        )

        saturated = self._saturated_families(library_state, family_records)
        recommended = self._recommended_families(memory_signal)
        underexplored = self._underexplored_families(recommended, family_records)
        rejection_heavy = sorted(
            family.name
            for family in family_records
            if family.rejected_count > family.admitted_count
        )
        prompt_text = self._prompt_text(
            family_records,
            saturated,
            underexplored,
            recommended,
            rejection_heavy,
        )

        return FamilyContextSummary(
            families=family_records,
            admitted_count=len(admitted_entries),
            rejected_count=len(rejected_entries),
            saturated_families=saturated,
            underexplored_families=underexplored,
            recommended_families=recommended,
            rejection_heavy_families=rejection_heavy,
            prompt_text=prompt_text,
        )

    def _saturated_families(
        self,
        library_state: dict[str, Any] | None,
        families: Sequence[FamilySummary],
    ) -> list[str]:
        category_counts = dict((library_state or {}).get("categories", {}) or {})
        if not category_counts and families:
            category_counts = {family.name: family.count for family in families}
        if not category_counts:
            return []
        avg_count = sum(category_counts.values()) / max(len(category_counts), 1)
        return sorted(
            name for name, count in category_counts.items() if count >= max(2.0, avg_count * 1.5)
        )

    def _recommended_families(self, memory_signal: dict[str, Any] | None) -> list[str]:
        families: set[str] = set()
        for pattern in (memory_signal or {}).get("recommended_directions", []):
            template = str(pattern.get("template", "") or "")
            name = str(pattern.get("name", "") or "")
            if template:
                families.add(infer_family(template))
            elif name:
                families.add(name)
        return sorted(families)

    def _underexplored_families(
        self,
        recommended: Sequence[str],
        families: Sequence[FamilySummary],
    ) -> list[str]:
        current = {family.name for family in families}
        missing = sorted(set(recommended) - current)
        if missing:
            return missing
        return sorted(family.name for family in families if family.count <= 1)

    def _prompt_text(
        self,
        families: Sequence[FamilySummary],
        saturated: Sequence[str],
        underexplored: Sequence[str],
        recommended: Sequence[str],
        rejection_heavy: Sequence[str],
    ) -> str:
        lines = ["=== FACTOR FAMILY CONTEXT ==="]
        if families:
            top = ", ".join(f"{family.name} ({family.count})" for family in families[:5])
            lines.append(f"Current family mix: {top}")
        if saturated:
            lines.append(f"Saturated families: {', '.join(saturated)}")
        if underexplored:
            lines.append(f"Underexplored families: {', '.join(underexplored)}")
        if rejection_heavy:
            lines.append(f"Rejection-heavy families: {', '.join(rejection_heavy)}")
        if recommended:
            lines.append(f"Recommended families from memory: {', '.join(recommended)}")
        return "\n".join(lines)


class RegimeMemoryService:
    """Summarize regime labels and metrics into prompt-facing memory context."""

    def summarize(
        self,
        regime_labels: Sequence[str] | np.ndarray,
        regime_metrics: Mapping[str, Sequence[float] | Mapping[str, Sequence[float] | float]]
        | None = None,
        *,
        lookback_window: int | None = None,
    ) -> RegimeMemorySummary:
        labels = [str(label).strip().lower() for label in list(regime_labels)]
        if not labels:
            labels = ["unknown"]

        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        frequencies = {label: count / max(total, 1) for label, count in counts.items()}
        dominant_regime = max(counts.items(), key=lambda item: (item[1], item[0]))[0]

        recent_labels = labels[-max(int(lookback_window or 1), 1) :]
        recent_regime = max(
            ((label, recent_labels.count(label)) for label in set(recent_labels)),
            key=lambda item: (item[1], item[0]),
        )[0]

        transition_counts: dict[str, dict[str, int]] = {}
        for previous, current in zip(labels[:-1], labels[1:]):
            transition_counts.setdefault(previous, {})[current] = (
                transition_counts.setdefault(previous, {}).get(current, 0) + 1
            )
        transition_matrix = {
            previous: {
                current: count / max(sum(targets.values()), 1) for current, count in targets.items()
            }
            for previous, targets in transition_counts.items()
        }

        metrics_summary: dict[str, dict[str, dict[str, float]]] = {}
        if regime_metrics:
            for metric_name, values in regime_metrics.items():
                metrics_summary[metric_name] = self._summarize_metric_by_regime(
                    labels,
                    values,
                )

        keywords = list(_REGIME_KEYWORDS.get(dominant_regime, _REGIME_KEYWORDS["unknown"]))
        prompt_text = self._prompt_text(
            labels=labels,
            counts=counts,
            frequencies=frequencies,
            dominant_regime=dominant_regime,
            recent_regime=recent_regime,
            metrics_summary=metrics_summary,
            keywords=keywords,
            lookback_window=lookback_window,
        )

        return RegimeMemorySummary(
            labels=labels,
            counts=counts,
            frequencies=frequencies,
            dominant_regime=dominant_regime,
            recent_regime=recent_regime,
            transition_matrix=transition_matrix,
            metrics=metrics_summary,
            regime_keywords=keywords,
            prompt_text=prompt_text,
        )

    def _summarize_metric_by_regime(
        self,
        labels: Sequence[str],
        values: Sequence[float] | Mapping[str, Sequence[float] | float],
    ) -> dict[str, dict[str, float]]:
        if isinstance(values, Mapping):
            summary: dict[str, dict[str, float]] = {}
            for regime, regime_values in values.items():
                arr = _to_float_array(regime_values)
                summary[str(regime).strip().lower()] = {
                    "mean": float(np.mean(arr)) if arr.size else 0.0,
                    "std": float(np.std(arr)) if arr.size > 1 else 0.0,
                    "count": float(arr.size),
                }
            return summary

        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.shape[0] != len(labels):
            raise ValueError("Aligned regime metrics must match the number of regime labels")
        summary: dict[str, dict[str, float]] = {}
        label_array = np.asarray(list(labels), dtype=object)
        for regime in sorted(set(labels)):
            mask = label_array == regime
            regime_values = arr[mask]
            if regime_values.size == 0:
                continue
            summary[regime] = {
                "mean": float(np.mean(regime_values)),
                "std": float(np.std(regime_values)) if regime_values.size > 1 else 0.0,
                "count": float(regime_values.size),
            }
        return summary

    def _prompt_text(
        self,
        *,
        labels: Sequence[str],
        counts: Mapping[str, int],
        frequencies: Mapping[str, float],
        dominant_regime: str,
        recent_regime: str,
        metrics_summary: Mapping[str, Mapping[str, Mapping[str, float]]],
        keywords: Sequence[str],
        lookback_window: int | None,
    ) -> str:
        lines = ["=== REGIME CONTEXT ==="]
        dominant_fraction = frequencies[dominant_regime]
        dominant_count = counts[dominant_regime]
        lines.append(
            f"Dominant regime: {dominant_regime} "
            f"({dominant_count}/{len(labels)}, {dominant_fraction:.1%})"
        )
        lines.append(f"Recent regime: {recent_regime}")
        if lookback_window:
            recent_slice = list(labels)[-max(int(lookback_window), 1) :]
            recent_counts = {
                label: recent_slice.count(label) for label in sorted(set(recent_slice))
            }
            recent_text = ", ".join(f"{label} ({count})" for label, count in recent_counts.items())
            lines.append(f"Recent window: {recent_text}")
        if metrics_summary:
            for metric_name, regime_stats in metrics_summary.items():
                compact = ", ".join(
                    f"{regime}={stats['mean']:.4f}"
                    for regime, stats in sorted(regime_stats.items())
                )
                lines.append(f"{metric_name}: {compact}")
        if keywords:
            lines.append(f"Regime theme cues: {', '.join(keywords)}")
        return "\n".join(lines)


class NonlinearDependenceService:
    """Score nonlinear dependence with distance correlation and an MI proxy."""

    def score_pair(
        self,
        candidate: Any,
        reference: Any,
        *,
        candidate_name: str = "candidate",
        reference_name: str = "reference",
    ) -> DependenceProfile:
        scores = _periodwise_nonlinear_scores(candidate, reference)
        return DependenceProfile(
            candidate_name=candidate_name,
            reference_name=reference_name,
            sample_count=int(scores["sample_count"]),
            period_count=int(scores["period_count"]),
            distance_correlation=float(scores["distance_correlation"]),
            pearson_abs=float(scores["pearson_abs"]),
            spearman_abs=float(scores["spearman_abs"]),
            mutual_information_proxy=float(scores["mutual_information_proxy"]),
            mutual_information_method=str(scores["mutual_information_method"]),
            combined_score=float(scores["combined_score"]),
        )

    def score_collection(
        self,
        candidate: Any,
        references: Sequence[Any] | np.ndarray | None,
        *,
        candidate_name: str = "candidate",
        reference_names: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        reference_panels = _stack_reference_collection(references)
        names = list(
            reference_names or [f"reference_{index}" for index in range(len(reference_panels))]
        )
        if len(names) != len(reference_panels):
            raise ValueError("reference_names must match the number of reference panels")

        profiles = [
            self.score_pair(
                candidate, reference, candidate_name=candidate_name, reference_name=name
            )
            for reference, name in zip(reference_panels, names)
        ]
        if not profiles:
            return {
                "profiles": [],
                "max_distance_correlation": 0.0,
                "max_mutual_information_proxy": 0.0,
                "mean_distance_correlation": 0.0,
                "mean_mutual_information_proxy": 0.0,
                "max_combined_score": 0.0,
            }

        payloads = [profile.to_dict() for profile in profiles]
        return {
            "profiles": payloads,
            "max_distance_correlation": float(
                max(profile.distance_correlation for profile in profiles)
            ),
            "max_mutual_information_proxy": float(
                max(profile.mutual_information_proxy for profile in profiles)
            ),
            "mean_distance_correlation": float(
                np.mean([profile.distance_correlation for profile in profiles])
            ),
            "mean_mutual_information_proxy": float(
                np.mean([profile.mutual_information_proxy for profile in profiles])
            ),
            "max_combined_score": float(max(profile.combined_score for profile in profiles)),
        }


class EnsembleMarginalUtilityService:
    """Estimate candidate utility against the current ensemble and returns."""

    def __init__(self, dependence_service: NonlinearDependenceService | None = None) -> None:
        self.dependence_service = dependence_service or NonlinearDependenceService()

    def estimate(
        self,
        candidate: Any,
        existing_signals: Sequence[Any] | np.ndarray | None,
        returns: Any,
        *,
        train_fraction: float = 0.7,
        candidate_name: str = "candidate",
        reference_names: Sequence[str] | None = None,
    ) -> EnsembleUtilitySummary:
        candidate_panel = _as_panel(candidate)
        return_panel = _as_panel(returns)
        if candidate_panel.shape != return_panel.shape:
            raise ValueError("Candidate signal and returns must share the same shape")

        references = _stack_reference_collection(existing_signals)
        for reference in references:
            if reference.shape != candidate_panel.shape:
                raise ValueError("All reference panels must match the candidate signal shape")

        if reference_names is None:
            reference_names = [f"reference_{index}" for index in range(len(references))]
        if len(reference_names) != len(references):
            raise ValueError("reference_names must match the number of reference panels")

        periods = candidate_panel.shape[1]
        if periods <= 1:
            train_periods = [0]
            test_periods = [0]
        else:
            split = int(round(periods * float(train_fraction)))
            split = min(max(split, 1), periods - 1)
            train_periods = list(range(split))
            test_periods = list(range(split, periods))

        train_x_base, train_y = (
            _flatten_period_samples(references, return_panel, train_periods)
            if references
            else (
                np.empty((0, 0), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        )
        train_x_aug, train_y_aug = _flatten_period_samples(
            references + [candidate_panel],
            return_panel,
            train_periods,
        )
        test_x_base, _ = (
            _flatten_period_samples(references, return_panel, test_periods)
            if references
            else (
                np.empty((0, 0), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )
        )
        test_x_aug, test_y_aug = _flatten_period_samples(
            references + [candidate_panel],
            return_panel,
            test_periods,
        )

        _, base_test_pred = _linear_fit_predict(train_x_base, train_y, test_x_base)
        _, aug_test_pred = _linear_fit_predict(train_x_aug, train_y_aug, test_x_aug)

        if not references:
            base_test_pred = np.full(
                test_y_aug.shape, float(np.mean(train_y_aug)) if train_y_aug.size else 0.0
            )

        base_ic = _safe_corr(base_test_pred, test_y_aug)
        augmented_ic = _safe_corr(aug_test_pred, test_y_aug)
        base_r2 = _safe_r2(test_y_aug, base_test_pred)
        augmented_r2 = _safe_r2(test_y_aug, aug_test_pred)
        delta_ic = float(augmented_ic - base_ic)
        delta_r2 = float(augmented_r2 - base_r2)
        candidate_feature_test = (
            test_x_aug[:, -1] if test_x_aug.size else np.empty((0,), dtype=np.float64)
        )

        residual_ic = _safe_corr(candidate_feature_test, test_y_aug - base_test_pred)

        dependence_summary = self.dependence_service.score_collection(
            candidate_panel,
            references,
            candidate_name=candidate_name,
            reference_names=reference_names,
        )

        prompt_text = self._prompt_text(
            candidate_name=candidate_name,
            dependence_summary=dependence_summary,
            base_ic=base_ic,
            augmented_ic=augmented_ic,
            base_r2=base_r2,
            augmented_r2=augmented_r2,
            delta_ic=delta_ic,
            delta_r2=delta_r2,
            residual_ic=residual_ic,
        )

        return EnsembleUtilitySummary(
            candidate_name=candidate_name,
            reference_names=list(reference_names),
            sample_count_train=int(train_y_aug.size),
            sample_count_test=int(test_y_aug.size),
            train_periods=train_periods,
            test_periods=test_periods,
            base_ic=float(base_ic),
            augmented_ic=float(augmented_ic),
            delta_ic=float(delta_ic),
            base_r2=float(base_r2),
            augmented_r2=float(augmented_r2),
            delta_r2=float(delta_r2),
            residual_ic=float(residual_ic),
            max_existing_distance_correlation=float(dependence_summary["max_distance_correlation"]),
            max_existing_mutual_information_proxy=float(
                dependence_summary["max_mutual_information_proxy"]
            ),
            mean_existing_distance_correlation=float(
                dependence_summary["mean_distance_correlation"]
            ),
            mean_existing_mutual_information_proxy=float(
                dependence_summary["mean_mutual_information_proxy"]
            ),
            prompt_text=prompt_text,
        )

    def _prompt_text(
        self,
        *,
        candidate_name: str,
        dependence_summary: Mapping[str, Any],
        base_ic: float,
        augmented_ic: float,
        base_r2: float,
        augmented_r2: float,
        delta_ic: float,
        delta_r2: float,
        residual_ic: float,
    ) -> str:
        lines = ["=== ENSEMBLE MARGINAL UTILITY ==="]
        lines.append(f"Candidate: {candidate_name}")
        lines.append(
            f"Base IC -> Augmented IC: {base_ic:.4f} -> {augmented_ic:.4f} (delta {delta_ic:.4f})"
        )
        lines.append(
            f"Base R2 -> Augmented R2: {base_r2:.4f} -> {augmented_r2:.4f} (delta {delta_r2:.4f})"
        )
        lines.append(f"Residual IC against base ensemble: {residual_ic:.4f}")
        lines.append(
            "Max existing dependence: "
            f"dcor={dependence_summary['max_distance_correlation']:.4f}, "
            f"mi_proxy={dependence_summary['max_mutual_information_proxy']:.4f}"
        )
        return "\n".join(lines)


class ResearchExtensionService:
    """Composes the research extension services into a single prompt-facing API."""

    def __init__(
        self,
        *,
        family_service: FamilyContextService | None = None,
        regime_service: RegimeMemoryService | None = None,
        dependence_service: NonlinearDependenceService | None = None,
        ensemble_service: EnsembleMarginalUtilityService | None = None,
    ) -> None:
        self.family_service = family_service or FamilyContextService()
        self.regime_service = regime_service or RegimeMemoryService()
        self.dependence_service = dependence_service or NonlinearDependenceService()
        self.ensemble_service = ensemble_service or EnsembleMarginalUtilityService(
            dependence_service=self.dependence_service,
        )

    def family_context(
        self,
        admitted_formulas: Sequence[str | Mapping[str, Any]] | None,
        rejected_formulas: Sequence[str | Mapping[str, Any]] | None,
        *,
        library_state: dict[str, Any] | None = None,
        memory_signal: dict[str, Any] | None = None,
    ) -> FamilyContextSummary:
        return self.family_service.summarize(
            admitted_formulas,
            rejected_formulas,
            library_state=library_state,
            memory_signal=memory_signal,
        )

    def regime_context(
        self,
        regime_labels: Sequence[str] | np.ndarray,
        regime_metrics: Mapping[str, Sequence[float] | Mapping[str, Sequence[float] | float]]
        | None = None,
        *,
        lookback_window: int | None = None,
    ) -> RegimeMemorySummary:
        return self.regime_service.summarize(
            regime_labels,
            regime_metrics,
            lookback_window=lookback_window,
        )

    def dependence_profile(
        self,
        candidate: Any,
        reference: Any,
        *,
        candidate_name: str = "candidate",
        reference_name: str = "reference",
    ) -> DependenceProfile:
        return self.dependence_service.score_pair(
            candidate,
            reference,
            candidate_name=candidate_name,
            reference_name=reference_name,
        )

    def ensemble_utility(
        self,
        candidate: Any,
        existing_signals: Sequence[Any] | np.ndarray | None,
        returns: Any,
        *,
        train_fraction: float = 0.7,
        candidate_name: str = "candidate",
        reference_names: Sequence[str] | None = None,
    ) -> EnsembleUtilitySummary:
        return self.ensemble_service.estimate(
            candidate,
            existing_signals,
            returns,
            train_fraction=train_fraction,
            candidate_name=candidate_name,
            reference_names=reference_names,
        )

    def build_prompt_context_extensions(
        self,
        *,
        admitted_formulas: Sequence[str | Mapping[str, Any]] | None = None,
        rejected_formulas: Sequence[str | Mapping[str, Any]] | None = None,
        regime_labels: Sequence[str] | np.ndarray | None = None,
        regime_metrics: Mapping[str, Sequence[float] | Mapping[str, Sequence[float] | float]]
        | None = None,
        candidate: Any | None = None,
        existing_signals: Sequence[Any] | np.ndarray | None = None,
        returns: Any | None = None,
        library_state: dict[str, Any] | None = None,
        memory_signal: dict[str, Any] | None = None,
        lookback_window: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        family_context = self.family_context(
            admitted_formulas,
            rejected_formulas,
            library_state=library_state,
            memory_signal=memory_signal,
        )
        payload["family_context"] = asdict(family_context)

        if regime_labels is not None:
            regime_context = self.regime_context(
                regime_labels,
                regime_metrics,
                lookback_window=lookback_window,
            )
            payload["regime_context"] = asdict(regime_context)
        if candidate is not None and returns is not None:
            payload["ensemble_utility"] = asdict(
                self.ensemble_utility(
                    candidate,
                    existing_signals,
                    returns,
                )
            )
        if candidate is not None and existing_signals is not None:
            score = self.dependence_service.score_collection(
                candidate,
                existing_signals,
            )
            payload["dependence_context"] = score

        prompt_texts = [family_context.prompt_text]
        if "regime_context" in payload:
            prompt_texts.append(payload["regime_context"]["prompt_text"])
        if "ensemble_utility" in payload:
            prompt_texts.append(payload["ensemble_utility"]["prompt_text"])
        payload["prompt_text"] = "\n\n".join(text for text in prompt_texts if text)
        return payload
