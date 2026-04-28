"""Factor Library: maintains the growing collection of admitted alpha factors.

Implements the admission rules from the paper (Eq. 10, 11):
- Admission: IC(alpha) >= tau_IC AND max_{g in L} |rho(alpha, g)| < theta
- Replacement: IC(alpha) >= 0.10 AND IC(alpha) >= 1.3 * IC(g) AND only 1 correlated factor

The library tracks pairwise Spearman correlations and supports incremental
updates as new factors are added or replaced.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from factorminer.architecture.dependence import (
    DependenceMetric,
    build_dependence_metric,
)

logger = logging.getLogger(__name__)


@dataclass
class Factor:
    """A single admitted alpha factor."""

    id: int
    name: str
    formula: str
    category: str  # e.g., "VWAP", "Regime-switching", "Momentum", etc.
    ic_mean: float  # Signed mean IC across evaluation period
    icir: float  # Signed IC Information Ratio
    ic_win_rate: float  # Fraction of periods with positive IC
    max_correlation: float  # Max |rho| with any other library factor at admission
    batch_number: int  # Which mining batch admitted this factor
    ic_paper_mean: float | None = None  # Paper metric: abs(mean(IC_t))
    ic_abs_mean: float | None = None  # Legacy diagnostic: mean(abs(IC_t))
    ic_paper_icir: float | None = None  # Paper metric: abs(mean(IC_t)) / std(IC_t)
    metric_version: str = "paper_ic_v2"
    admission_date: str = ""
    signals: np.ndarray | None = field(default=None, repr=False)  # (M, T)
    research_metrics: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.admission_date:
            self.admission_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.ic_paper_mean is None:
            self.ic_paper_mean = abs(float(self.ic_mean))
        if self.ic_abs_mean is None:
            self.ic_abs_mean = abs(float(self.ic_mean))
        if self.ic_paper_icir is None:
            self.ic_paper_icir = abs(float(self.icir))

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary (excludes signals)."""
        return {
            "id": self.id,
            "name": self.name,
            "formula": self.formula,
            "category": self.category,
            "ic_mean": self.ic_mean,
            "ic_paper_mean": self.ic_paper_mean,
            "ic_abs_mean": self.ic_abs_mean,
            "icir": self.icir,
            "ic_paper_icir": self.ic_paper_icir,
            "ic_win_rate": self.ic_win_rate,
            "max_correlation": self.max_correlation,
            "batch_number": self.batch_number,
            "metric_version": self.metric_version,
            "admission_date": self.admission_date,
            "research_metrics": self.research_metrics,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Factor:
        """Reconstruct a Factor from a dictionary."""
        return cls(
            id=d["id"],
            name=d["name"],
            formula=d["formula"],
            category=d["category"],
            ic_mean=d["ic_mean"],
            ic_paper_mean=d.get("ic_paper_mean", abs(float(d.get("ic_mean", 0.0)))),
            ic_abs_mean=d.get("ic_abs_mean", abs(float(d.get("ic_mean", 0.0)))),
            icir=d["icir"],
            ic_paper_icir=d.get("ic_paper_icir", abs(float(d.get("icir", 0.0)))),
            ic_win_rate=d["ic_win_rate"],
            max_correlation=d["max_correlation"],
            batch_number=d["batch_number"],
            metric_version=d.get("metric_version", "legacy_abs_ic"),
            admission_date=d.get("admission_date", ""),
            research_metrics=d.get("research_metrics", {}),
            provenance=d.get("provenance", {}),
        )


class FactorLibrary:
    """The factor library L that maintains admitted alpha factors.

    Parameters
    ----------
    correlation_threshold : float
        Maximum allowed |rho| for admission (theta). Default 0.5 for A-shares.
    ic_threshold : float
        Minimum IC for admission (tau_IC). Default 0.04.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.5,
        ic_threshold: float = 0.04,
        dependence_metric: str | DependenceMetric | None = None,
    ) -> None:
        self.factors: dict[int, Factor] = {}
        self.correlation_matrix: np.ndarray | None = None  # Pairwise |rho|
        self._next_id: int = 1
        self.correlation_threshold = correlation_threshold
        self.ic_threshold = ic_threshold
        self.metric_version = "paper_ic_v2"
        if isinstance(dependence_metric, DependenceMetric):
            self.dependence_metric = dependence_metric
        else:
            self.dependence_metric = build_dependence_metric(dependence_metric)
        # Maps factor_id -> index in the correlation matrix
        self._id_to_index: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Correlation computation
    # ------------------------------------------------------------------

    def compute_correlation(
        self, signals_a: np.ndarray, signals_b: np.ndarray
    ) -> float:
        """Compute time-average dependence rho(alpha, beta) under the active metric.

        By default this is the paper's mean absolute Spearman rank correlation,
        but the library can now be configured with alternative dependence metrics.

        Parameters
        ----------
        signals_a, signals_b : np.ndarray, shape (M, T)
            Cross-sectional signal matrices.

        Returns
        -------
        float
            Mean absolute Spearman rank correlation across time steps.
        """
        return float(self.dependence_metric.compute(signals_a, signals_b))

    def _compute_correlation_vectorized(
        self, signals_a: np.ndarray, signals_b: np.ndarray
    ) -> float:
        """Compatibility alias for the active dependence metric implementation."""
        return self.compute_correlation(signals_a, signals_b)

    # ------------------------------------------------------------------
    # Admission and replacement
    # ------------------------------------------------------------------

    def check_admission(
        self, candidate_ic: float, candidate_signals: np.ndarray
    ) -> tuple[bool, str]:
        """Check if candidate passes admission criteria (Eq. 10).

        Admission rule:
            IC(alpha) >= tau_IC  AND  max_{g in L} |rho(alpha, g)| < theta

        Parameters
        ----------
        candidate_ic : float
            The candidate factor's mean IC.
        candidate_signals : np.ndarray, shape (M, T)
            The candidate's realized signals.

        Returns
        -------
        (admitted, reason) : Tuple[bool, str]
        """
        if candidate_ic < self.ic_threshold:
            return False, (
                f"IC {candidate_ic:.4f} below threshold {self.ic_threshold}"
            )

        if self.size == 0:
            return True, "First factor in library"

        max_corr = self._max_correlation_with_library(candidate_signals)

        if max_corr >= self.correlation_threshold:
            return False, (
                f"Max correlation/{self.dependence_metric.name} dependence {max_corr:.4f} >= threshold "
                f"{self.correlation_threshold} with existing library factor"
            )

        return True, (
            f"Admitted: IC={candidate_ic:.4f}, max_corr={max_corr:.4f}"
        )

    def check_replacement(
        self,
        candidate_ic: float,
        candidate_signals: np.ndarray,
        ic_min: float = 0.10,
        ic_ratio: float = 1.3,
    ) -> tuple[bool, int | None, str]:
        """Check replacement mechanism (Eq. 11).

        Replacement rule:
            IC(alpha) >= 0.10
            AND IC(alpha) >= 1.3 * IC(g)
            AND |{g in L : |rho(alpha, g)| > theta}| = 1

        If exactly one library factor g is correlated above theta AND the
        candidate's IC dominates g's IC by the required ratio, replace g.

        Parameters
        ----------
        candidate_ic : float
            The candidate's mean IC.
        candidate_signals : np.ndarray, shape (M, T)
            The candidate's realized signals.
        ic_min : float
            Absolute IC floor for replacement (default 0.10).
        ic_ratio : float
            Required IC ratio over the correlated factor (default 1.3).

        Returns
        -------
        (should_replace, factor_to_replace_id, reason) : Tuple[bool, Optional[int], str]
        """
        if candidate_ic < ic_min:
            return False, None, (
                f"IC {candidate_ic:.4f} below replacement floor {ic_min}"
            )

        if self.size == 0:
            return False, None, "Library is empty, use admission instead"

        # Find all library factors correlated above theta
        correlated_factors = []
        for fid, factor in self.factors.items():
            if factor.signals is None:
                continue
            corr = self._compute_correlation_vectorized(
                candidate_signals, factor.signals
            )
            if corr >= self.correlation_threshold:
                existing_paper_ic = float(factor.ic_paper_mean or abs(factor.ic_mean))
                correlated_factors.append((fid, corr, existing_paper_ic))

        if len(correlated_factors) != 1:
            return False, None, (
                f"Found {len(correlated_factors)} correlated/conflicting factors "
                "(need exactly 1 for replacement)"
            )

        fid, corr, existing_ic = correlated_factors[0]
        if candidate_ic < ic_ratio * existing_ic:
            return False, None, (
                f"IC {candidate_ic:.4f} < {ic_ratio} * {existing_ic:.4f} = "
                f"{ic_ratio * existing_ic:.4f}"
            )

        return True, fid, (
            f"Replace factor {fid}: candidate IC {candidate_ic:.4f} > "
            f"{ic_ratio} * {existing_ic:.4f}, corr={corr:.4f}"
        )

    # ------------------------------------------------------------------
    # Library mutations
    # ------------------------------------------------------------------

    def admit_factor(self, factor: Factor) -> int:
        """Add a factor to the library and update the correlation matrix.

        Parameters
        ----------
        factor : Factor
            The factor to add. Its ``id`` field is overwritten with the
            next available ID.

        Returns
        -------
        int
            The assigned factor ID.
        """
        factor.id = self._next_id
        self._next_id += 1
        self.factors[factor.id] = factor

        # Update correlation matrix incrementally
        self._extend_correlation_matrix(factor)

        logger.info(
            "Admitted factor %d '%s' (paper IC=%.4f, max_corr=%.4f, category=%s)",
            factor.id, factor.name, float(factor.ic_paper_mean or abs(factor.ic_mean)),
            factor.max_correlation, factor.category,
        )
        return factor.id

    def replace_factor(self, old_id: int, new_factor: Factor) -> None:
        """Replace an existing factor with a better one.

        The new factor takes the old factor's position in the correlation
        matrix and receives a fresh ID.

        Parameters
        ----------
        old_id : int
            ID of the factor being replaced.
        new_factor : Factor
            The replacement factor.
        """
        if old_id not in self.factors:
            raise KeyError(f"Factor {old_id} not in library")

        self.factors[old_id]
        new_factor.id = self._next_id
        self._next_id += 1

        # Remove old factor and reuse its matrix slot
        old_index = self._id_to_index.pop(old_id)
        del self.factors[old_id]

        # Insert new factor at the same index
        self.factors[new_factor.id] = new_factor
        self._id_to_index[new_factor.id] = old_index

        # Recompute the row/column for this index
        if self.correlation_matrix is not None and new_factor.signals is not None:
            self._recompute_matrix_slot(old_index, new_factor)

        logger.info(
            "Replaced factor %d with %d '%s' (paper IC=%.4f)",
            old_id,
            new_factor.id,
            new_factor.name,
            float(new_factor.ic_paper_mean or abs(new_factor.ic_mean)),
        )

    def remove_factor(self, factor_id: int) -> None:
        """Remove a factor from the library and rebuild correlation state."""
        if factor_id not in self.factors:
            raise KeyError(f"Factor {factor_id} not in library")

        removed = self.factors.pop(factor_id)
        self.update_correlation_matrix()

        logger.info(
            "Removed factor %d '%s' from library",
            factor_id,
            removed.name,
        )

    # ------------------------------------------------------------------
    # Correlation matrix management
    # ------------------------------------------------------------------

    def _max_correlation_with_library(
        self, candidate_signals: np.ndarray
    ) -> float:
        """Compute max |rho| between candidate and all library factors."""
        max_corr = 0.0
        for factor in self.factors.values():
            if factor.signals is None:
                continue
            corr = self._compute_correlation_vectorized(
                candidate_signals, factor.signals
            )
            max_corr = max(max_corr, corr)
        return max_corr

    def _extend_correlation_matrix(self, new_factor: Factor) -> None:
        """Extend the correlation matrix by one row/column for the new factor."""
        n = len(self._id_to_index)
        new_index = n
        self._id_to_index[new_factor.id] = new_index

        if new_factor.signals is None:
            # No signals to correlate; expand with zeros
            if self.correlation_matrix is None:
                self.correlation_matrix = np.zeros((1, 1), dtype=np.float64)
            else:
                new_size = new_index + 1
                new_mat = np.zeros((new_size, new_size), dtype=np.float64)
                new_mat[:new_index, :new_index] = self.correlation_matrix
                self.correlation_matrix = new_mat
            return

        # Build a new (n+1) x (n+1) matrix
        new_size = new_index + 1
        new_mat = np.zeros((new_size, new_size), dtype=np.float64)

        if self.correlation_matrix is not None and self.correlation_matrix.size > 0:
            new_mat[:new_index, :new_index] = self.correlation_matrix

        # Compute correlations with all existing factors
        index_to_id = {idx: fid for fid, idx in self._id_to_index.items()}
        for idx in range(new_index):
            fid = index_to_id.get(idx)
            if fid is None:
                continue
            other = self.factors.get(fid)
            if other is None or other.signals is None:
                continue
            corr = self._compute_correlation_vectorized(
                new_factor.signals, other.signals
            )
            new_mat[new_index, idx] = corr
            new_mat[idx, new_index] = corr

        self.correlation_matrix = new_mat

    def _recompute_matrix_slot(self, idx: int, factor: Factor) -> None:
        """Recompute one row/column of the correlation matrix after replacement."""
        n = self.correlation_matrix.shape[0]
        index_to_id = {i: fid for fid, i in self._id_to_index.items()}

        for other_idx in range(n):
            if other_idx == idx:
                self.correlation_matrix[idx, idx] = 0.0
                continue
            other_fid = index_to_id.get(other_idx)
            if other_fid is None:
                continue
            other = self.factors.get(other_fid)
            if other is None or other.signals is None:
                self.correlation_matrix[idx, other_idx] = 0.0
                self.correlation_matrix[other_idx, idx] = 0.0
                continue
            corr = self._compute_correlation_vectorized(
                factor.signals, other.signals
            )
            self.correlation_matrix[idx, other_idx] = corr
            self.correlation_matrix[other_idx, idx] = corr

    def update_correlation_matrix(self) -> None:
        """Recompute the full pairwise correlation matrix from scratch.

        This is O(n^2) in the number of library factors and should only be
        called when the incremental updates may have drifted or after bulk
        operations.
        """
        ids = sorted(self.factors.keys())
        n = len(ids)
        if n == 0:
            self.correlation_matrix = None
            self._id_to_index.clear()
            return

        self._id_to_index = {fid: i for i, fid in enumerate(ids)}
        mat = np.zeros((n, n), dtype=np.float64)

        factors_list = [self.factors[fid] for fid in ids]
        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = factors_list[i], factors_list[j]
                if fi.signals is None or fj.signals is None:
                    continue
                corr = self._compute_correlation_vectorized(fi.signals, fj.signals)
                mat[i, j] = corr
                mat[j, i] = corr

        self.correlation_matrix = mat

    # ------------------------------------------------------------------
    # Queries and diagnostics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of factors currently in the library."""
        return len(self.factors)

    def get_factor(self, factor_id: int) -> Factor:
        """Retrieve a factor by ID."""
        if factor_id not in self.factors:
            raise KeyError(f"Factor {factor_id} not in library")
        return self.factors[factor_id]

    def list_factors(self) -> list[Factor]:
        """Return all factors sorted by ID."""
        return [self.factors[k] for k in sorted(self.factors)]

    def get_factors_by_category(self, category: str) -> list[Factor]:
        """Return all factors matching a given category."""
        return [
            f for f in self.factors.values()
            if f.category == category
        ]

    def get_diagnostics(self) -> dict:
        """Library diagnostics: avg |rho|, max tail correlations, per-category counts, saturation.

        Returns
        -------
        dict with keys:
            - size: int
            - avg_correlation: float (average off-diagonal |rho|)
            - max_correlation: float (maximum off-diagonal |rho|)
            - p95_correlation: float (95th percentile off-diagonal |rho|)
            - category_counts: dict[str, int]
            - category_avg_ic: dict[str, float]
            - saturation: float (fraction of max correlation slots above 0.3)
        """
        diag: dict = {"size": self.size}

        # Category breakdown
        cat_counts: dict[str, int] = defaultdict(int)
        cat_ic_sums: dict[str, float] = defaultdict(float)
        for f in self.factors.values():
            cat_counts[f.category] += 1
            cat_ic_sums[f.category] += float(f.ic_paper_mean or abs(f.ic_mean))

        diag["category_counts"] = dict(cat_counts)
        diag["category_avg_ic"] = {
            cat: cat_ic_sums[cat] / cat_counts[cat]
            for cat in cat_counts
        }

        # Correlation statistics
        if self.correlation_matrix is not None and self.size > 1:
            n = self.correlation_matrix.shape[0]
            # Extract upper triangle (off-diagonal)
            triu_idx = np.triu_indices(n, k=1)
            off_diag = self.correlation_matrix[triu_idx]
            valid = off_diag[~np.isnan(off_diag)]

            if len(valid) > 0:
                diag["avg_correlation"] = float(np.mean(valid))
                diag["max_correlation"] = float(np.max(valid))
                diag["p95_correlation"] = float(np.percentile(valid, 95))
                diag["saturation"] = float(np.mean(valid > 0.3))
            else:
                diag["avg_correlation"] = 0.0
                diag["max_correlation"] = 0.0
                diag["p95_correlation"] = 0.0
                diag["saturation"] = 0.0
        else:
            diag["avg_correlation"] = 0.0
            diag["max_correlation"] = 0.0
            diag["p95_correlation"] = 0.0
            diag["saturation"] = 0.0

        return diag

    def get_state_summary(self) -> dict:
        """Summary for memory retrieval: size, categories, recent admissions.

        Returns a lightweight dictionary suitable for inclusion in LLM prompts
        or memory store entries.
        """
        factors_sorted = sorted(
            self.factors.values(), key=lambda f: f.id, reverse=True
        )
        recent = factors_sorted[:5]  # Last 5 admissions

        categories = defaultdict(int)
        for f in self.factors.values():
            categories[f.category] += 1

        return {
            "library_size": self.size,
            "categories": dict(categories),
            "recent_admissions": [
                {
                    "id": f.id,
                    "name": f.name,
                    "formula": f.formula,
                    "category": f.category,
                    "ic_mean": f.ic_mean,
                    "ic_paper_mean": f.ic_paper_mean,
                    "batch": f.batch_number,
                }
                for f in recent
            ],
            "correlation_threshold": self.correlation_threshold,
            "ic_threshold": self.ic_threshold,
            "dependence_metric": self.dependence_metric.name,
        }
