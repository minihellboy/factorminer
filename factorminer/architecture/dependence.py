"""Pluggable dependence metrics for library redundancy and replacement logic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


def _iter_valid_columns(
    signals_a: np.ndarray,
    signals_b: np.ndarray,
):
    if signals_a.shape != signals_b.shape:
        raise ValueError(
            f"Signal shapes must match: {signals_a.shape} vs {signals_b.shape}"
        )

    _, periods = signals_a.shape
    for period in range(periods):
        col_a = signals_a[:, period]
        col_b = signals_b[:, period]
        valid = ~(np.isnan(col_a) | np.isnan(col_b))
        if int(valid.sum()) < 3:
            continue
        yield col_a[valid], col_b[valid]


def _pearson_abs(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if float(denom) < 1e-12:
        return 0.0
    return float(abs(np.sum(x_centered * y_centered) / denom))


def _distance_correlation_abs(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    if x.shape[0] < 3:
        return 0.0

    dist_x = np.abs(x - x.T)
    dist_y = np.abs(y - y.T)

    ax = dist_x - dist_x.mean(axis=0) - dist_x.mean(axis=1, keepdims=True) + dist_x.mean()
    ay = dist_y - dist_y.mean(axis=0) - dist_y.mean(axis=1, keepdims=True) + dist_y.mean()

    dcov2 = float(np.mean(ax * ay))
    dvar_x = float(np.mean(ax * ax))
    dvar_y = float(np.mean(ay * ay))
    if dvar_x <= 0.0 or dvar_y <= 0.0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvar_x * dvar_y)))


@dataclass(frozen=True)
class DependenceMetric(ABC):
    """Abstract pairwise dependence metric over cross-sectional signals."""

    name: str

    @abstractmethod
    def compute(self, signals_a: np.ndarray, signals_b: np.ndarray) -> float:
        raise NotImplementedError

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "family": "pairwise_time_averaged_dependence"}


@dataclass(frozen=True)
class SpearmanDependenceMetric(DependenceMetric):
    """Mean absolute Spearman rank correlation across time."""

    name: str = "spearman"

    def compute(self, signals_a: np.ndarray, signals_b: np.ndarray) -> float:
        scores: list[float] = []
        for col_a, col_b in _iter_valid_columns(signals_a, signals_b):
            scores.append(_pearson_abs(rankdata(col_a), rankdata(col_b)))
        if not scores:
            return 0.0
        return float(np.mean(scores))


@dataclass(frozen=True)
class PearsonDependenceMetric(DependenceMetric):
    """Mean absolute Pearson correlation across time."""

    name: str = "pearson"

    def compute(self, signals_a: np.ndarray, signals_b: np.ndarray) -> float:
        scores: list[float] = []
        for col_a, col_b in _iter_valid_columns(signals_a, signals_b):
            scores.append(_pearson_abs(col_a, col_b))
        if not scores:
            return 0.0
        return float(np.mean(scores))


@dataclass(frozen=True)
class DistanceCorrelationMetric(DependenceMetric):
    """Mean distance-correlation dependence across time."""

    name: str = "distance_correlation"

    def compute(self, signals_a: np.ndarray, signals_b: np.ndarray) -> float:
        scores: list[float] = []
        for col_a, col_b in _iter_valid_columns(signals_a, signals_b):
            scores.append(_distance_correlation_abs(col_a, col_b))
        if not scores:
            return 0.0
        return float(np.mean(scores))


def build_dependence_metric(name: str | None) -> DependenceMetric:
    """Build one supported dependence metric from config/runtime names."""

    metric_name = str(name or "spearman").strip().lower()
    if metric_name in {"spearman", "rank_correlation"}:
        return SpearmanDependenceMetric()
    if metric_name in {"pearson", "linear_correlation"}:
        return PearsonDependenceMetric()
    if metric_name in {"distance_correlation", "distance", "dcor"}:
        return DistanceCorrelationMetric()
    raise ValueError(
        "Unsupported redundancy/dependence metric "
        f"'{name}'. Expected one of: spearman, pearson, distance_correlation"
    )


# ---------------------------------------------------------------------------
# Spectral span diagnostics (Hypothesis-Redundancy / LLM jump gate)
# ---------------------------------------------------------------------------
#
# These operate on *formula-signal* matrices (columns = explored formulas),
# not on the pairwise cross-factor dependence metrics above. Used by
# ``architecture.geometry.assess_llm_jump_worth``.


def library_span_basis(
    span: np.ndarray,
    *,
    energy_fraction: float = 0.99,
    min_singular: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """Orthonormal basis for the column span of ``span`` via thin SVD.

    Deliberately NOT column-centered: this basis spans ``{span @ w}``, the
    literal column span, because callers (:func:`orthogonal_escape_score`,
    :func:`architecture.geometry.assess_llm_jump_worth`) project a
    candidate vector onto it and score the *uncentered* residual energy.
    A candidate that is exactly a linear combination of ``span``'s columns
    must have zero residual against this basis. Centering here would make
    that false: a nonzero-mean in-span candidate would then show spurious
    residual energy in its own (unrepresentable) mean component, reporting
    a false "escape"/novelty score for coverage the library already has.
    (:func:`spectral_compression_score` intentionally centers its own
    independent SVD -- it measures library self-redundancy via co-movement
    structure, a different, legitimate use of centering that does not
    share this function.)

    Parameters
    ----------
    span : np.ndarray, shape (D, K)
        Columns are flattened explored-formula signals.
    energy_fraction : float
        Keep leading singular vectors until this fraction of total
        squared singular value is captured (numerical rank soft-cut).
    min_singular : float
        Absolute floor on singular values treated as nonzero.

    Returns
    -------
    basis : np.ndarray, shape (D, R)
        Orthonormal columns spanning the retained subspace (may be empty).
    rank : int
        Number of retained columns.
    """
    mat = np.asarray(span, dtype=np.float64)
    if mat.ndim != 2 or mat.size == 0 or mat.shape[0] == 0 or mat.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float64), 0

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        # economy SVD: U is (D, K_eff)
        u, s, _vt = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros((mat.shape[0], 0), dtype=np.float64), 0

    if s.size == 0:
        return np.zeros((mat.shape[0], 0), dtype=np.float64), 0

    energy = s * s
    total = float(np.sum(energy))
    if total < 1e-18:
        return np.zeros((mat.shape[0], 0), dtype=np.float64), 0

    frac = float(np.clip(energy_fraction, 0.5, 1.0))
    cdf = np.cumsum(energy) / total
    rank = int(np.searchsorted(cdf, frac) + 1)
    rank = max(1, min(rank, int(np.sum(s > min_singular))))
    if rank <= 0:
        return np.zeros((mat.shape[0], 0), dtype=np.float64), 0
    return np.asarray(u[:, :rank], dtype=np.float64), rank


def spectral_compression_score(
    span: np.ndarray,
    *,
    energy_fraction: float = 0.95,
) -> float:
    """How compressed is the explored-formula span?

    Returns a score in ``[0, 1]``: 1 means nearly rank-1 (highly redundant
    library — a non-local jump is more valuable); 0 means energy is spread
    across many directions (span already rich — local edits may suffice).

    Defined as ``1 - (effective_rank - 1) / (k - 1)`` clipped, where
    effective rank is the number of singular values needed to capture
    ``energy_fraction`` of total energy.
    """
    mat = np.asarray(span, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] <= 1 or mat.size == 0:
        return 0.0

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = mat - np.mean(mat, axis=0, keepdims=True)
    try:
        _u, s, _vt = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0

    if s.size == 0:
        return 0.0
    energy = s * s
    total = float(np.sum(energy))
    if total < 1e-18:
        return 1.0  # all-zero span is fully "compressed"

    frac = float(np.clip(energy_fraction, 0.5, 1.0))
    cdf = np.cumsum(energy) / total
    eff_rank = int(np.searchsorted(cdf, frac) + 1)
    k = int(mat.shape[1])
    if k <= 1:
        return 0.0
    # eff_rank == 1 → score 1; eff_rank == k → score 0
    score = 1.0 - (eff_rank - 1) / (k - 1)
    return float(np.clip(score, 0.0, 1.0))


def orthogonal_escape_score(
    span: np.ndarray,
    candidate: np.ndarray,
    *,
    energy_fraction: float = 0.99,
) -> float:
    """Fraction of candidate energy outside the library span (in ``[0, 1]``)."""
    mat = np.asarray(span, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if mat.ndim != 2 or mat.size == 0:
        return 1.0
    d = min(mat.shape[0], cand.shape[0])
    if d < 1:
        return 0.0
    mat = np.nan_to_num(mat[:d, :], nan=0.0, posinf=0.0, neginf=0.0)
    cand = np.nan_to_num(cand[:d], nan=0.0, posinf=0.0, neginf=0.0)
    basis, rank = library_span_basis(mat, energy_fraction=energy_fraction)
    energy = float(np.dot(cand, cand))
    if energy < 1e-18:
        return 0.0
    if rank == 0 or basis.size == 0:
        return 1.0
    coeffs = basis.T @ cand
    projection = basis @ coeffs
    residual = cand - projection
    return float(np.clip(np.dot(residual, residual) / energy, 0.0, 1.0))


def residual_alignment_score(
    candidate_or_residual: np.ndarray,
    target: np.ndarray,
    *,
    span_basis: np.ndarray | None = None,
) -> float:
    """Absolute cosine alignment of residual direction with target residual.

    If ``span_basis`` is supplied, both vectors are first orthogonalized
    against it so alignment reflects *new* explanatory power, not span
    rehashing. Returns a score in ``[0, 1]``.
    """
    a = np.asarray(candidate_or_residual, dtype=np.float64).reshape(-1)
    b = np.asarray(target, dtype=np.float64).reshape(-1)
    n = min(a.shape[0], b.shape[0])
    if n < 2:
        return 0.0
    a = np.nan_to_num(a[:n], nan=0.0, posinf=0.0, neginf=0.0)
    b = np.nan_to_num(b[:n], nan=0.0, posinf=0.0, neginf=0.0)

    if span_basis is not None and np.asarray(span_basis).size > 0:
        basis = np.asarray(span_basis, dtype=np.float64)
        if basis.ndim == 2 and basis.shape[0] >= n and basis.shape[1] > 0:
            b_use = basis[:n, :]
            a = a - b_use @ (b_use.T @ a)
            b = b - b_use @ (b_use.T @ b)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.clip(abs(np.dot(a, b)) / (na * nb), 0.0, 1.0))
