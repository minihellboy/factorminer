"""Reusable library geometry diagnostics and novelty utilities.

Also hosts the Hypothesis-Redundancy geometric gate (arXiv:2606.14386 /
landscape §10 item 5): a cheap spectral diagnostic that decides whether an
expensive non-local (frontier LLM) proposal is likely to add real coverage
vs. a cheap local AST edit. See :class:`JumpWorthAssessment`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from factorminer.architecture.dependence import (
    library_span_basis,
    residual_alignment_score,
    spectral_compression_score,
)
from factorminer.core.factor_library import FactorLibrary


@dataclass(frozen=True)
class CandidateGeometry:
    """Geometric view of one candidate relative to the current library."""

    max_correlation: float
    max_dependence: float
    correlated_factor_ids: list[int]
    correlated_factor_names: list[str]
    novelty_score: float
    library_size: int
    dependence_metric: str


@dataclass(frozen=True)
class JumpWorthAssessment:
    """Decide when a frontier LLM call is worth its cost.

    Concrete implementation of the Hypothesis-Redundancy geometric gate
    (arXiv:2606.14386 "Discovery under Hypothesis Redundancy"): an LLM's
    non-local proposal is likely to help only when it (a) spectrally
    compresses the explored-formula span, (b) escapes that span in an
    orthogonal direction, and (c) aligns its residual with the actual
    target. When those conditions fail, a cheap local AST edit is the
    rational move.

    This is deliberately a **budget / routing diagnostic**, not a mining
    objective. Default consumers (e.g. island-model exploration schedule)
    keep the gate off unless explicitly enabled.

    Attributes
    ----------
    jump_worth : float
        Composite score in ``[0, 1]``. Higher → a non-local LLM jump is
        more likely to add real coverage.
    spectral_compression : float
        How much of the library span's energy lives in a few principal
        directions (high = compressed / redundant span).
    orthogonal_escape : float
        Fraction of candidate energy outside the library span
        (``1 - R^2`` of projection onto the span basis).
    residual_alignment : float
        Alignment of the candidate's span-orthogonal residual with the
        target residual (0 if no target supplied).
    recommend_llm_jump : bool
        ``True`` when ``jump_worth >= threshold``.
    rationale : str
        Human-readable summary for logs / CLI.
    library_rank : int
        Numerical rank of the explored-formula span.
    library_size : int
        Number of library columns used to form the span.
    """

    jump_worth: float
    spectral_compression: float
    orthogonal_escape: float
    residual_alignment: float
    recommend_llm_jump: bool
    rationale: str
    library_rank: int
    library_size: int
    threshold: float = 0.45

    def to_dict(self) -> dict[str, Any]:
        return {
            "jump_worth": self.jump_worth,
            "spectral_compression": self.spectral_compression,
            "orthogonal_escape": self.orthogonal_escape,
            "residual_alignment": self.residual_alignment,
            "recommend_llm_jump": self.recommend_llm_jump,
            "rationale": self.rationale,
            "library_rank": self.library_rank,
            "library_size": self.library_size,
            "threshold": self.threshold,
        }



class LibraryGeometry:
    """Centralizes correlation and saturation geometry around the library."""

    def __init__(self, library: FactorLibrary) -> None:
        self.library = library

    def candidate_geometry(
        self,
        signals: np.ndarray,
        *,
        crowding_novelty_modulation: float | None = None,
    ) -> CandidateGeometry:
        """Geometric view of one candidate relative to the current library.

        Parameters
        ----------
        signals:
            Candidate factor signals ``(M, T)``.
        crowding_novelty_modulation:
            Optional soft multiplier in ``[0, 1]`` from
            ``evaluation.crowding.CrowdingScore.novelty_modulation``. Extends
            (does not replace) the intra-library ``novelty_score``:
            ``novelty_score *= modulation``. ``None`` leaves novelty unchanged.
        """
        if self.library.size == 0:
            novelty = 1.0
            if crowding_novelty_modulation is not None:
                novelty = float(
                    max(0.0, min(1.0, novelty * float(crowding_novelty_modulation)))
                )
            return CandidateGeometry(
                max_correlation=0.0,
                max_dependence=0.0,
                correlated_factor_ids=[],
                correlated_factor_names=[],
                novelty_score=novelty,
                library_size=0,
                dependence_metric=self.library.dependence_metric.name,
            )

        correlated_ids: list[int] = []
        correlated_names: list[str] = []
        max_corr = 0.0

        for factor in self.library.list_factors():
            if factor.signals is None:
                continue
            corr = float(self.library.compute_correlation(signals, factor.signals))
            if corr > max_corr:
                max_corr = corr
            if corr >= self.library.correlation_threshold:
                correlated_ids.append(int(factor.id))
                correlated_names.append(str(factor.name))

        novelty = max(0.0, 1.0 - max_corr)
        if crowding_novelty_modulation is not None:
            novelty = float(
                max(0.0, min(1.0, novelty * float(crowding_novelty_modulation)))
            )

        return CandidateGeometry(
            max_correlation=max_corr,
            max_dependence=max_corr,
            correlated_factor_ids=correlated_ids,
            correlated_factor_names=correlated_names,
            novelty_score=novelty,
            library_size=self.library.size,
            dependence_metric=self.library.dependence_metric.name,
        )

    def replacement_target(self, signals: np.ndarray) -> int | None:
        geometry = self.candidate_geometry(signals)
        if len(geometry.correlated_factor_ids) != 1:
            return None
        return geometry.correlated_factor_ids[0]

    def check_admission(
        self,
        candidate_ic: float,
        candidate_signals: np.ndarray,
    ) -> tuple[bool, str]:
        if candidate_ic < self.library.ic_threshold:
            return False, (
                f"IC {candidate_ic:.4f} below threshold {self.library.ic_threshold}"
            )

        if self.library.size == 0:
            return True, "First factor in library"

        geometry = self.candidate_geometry(candidate_signals)
        if geometry.max_dependence >= self.library.correlation_threshold:
            return False, (
                f"Max {geometry.dependence_metric} dependence {geometry.max_dependence:.4f} "
                f">= threshold {self.library.correlation_threshold}"
            )

        return True, (
            f"Admitted: IC={candidate_ic:.4f}, "
            f"max_{geometry.dependence_metric}={geometry.max_dependence:.4f}"
        )

    def check_replacement(
        self,
        candidate_ic: float,
        candidate_signals: np.ndarray,
        *,
        ic_min: float,
        ic_ratio: float,
    ) -> tuple[bool, int | None, str]:
        if candidate_ic < ic_min:
            return False, None, f"IC {candidate_ic:.4f} below replacement floor {ic_min}"

        geometry = self.candidate_geometry(candidate_signals)
        if len(geometry.correlated_factor_ids) != 1:
            return False, None, (
                "Replacement requires exactly 1 conflicting library factor, "
                f"found {len(geometry.correlated_factor_ids)}"
            )

        target_id = geometry.correlated_factor_ids[0]
        target_factor = self.library.get_factor(target_id)
        target_ic = float(target_factor.ic_paper_mean or abs(target_factor.ic_mean))
        if candidate_ic < ic_ratio * target_ic:
            return False, None, (
                f"IC {candidate_ic:.4f} insufficient to replace factor {target_id} "
                f"(needs >= {ic_ratio} * {target_ic:.4f})"
            )

        return True, target_id, f"Replacement over factor {target_id}"

    def library_snapshot(self) -> dict[str, Any]:
        diagnostics = self.library.get_diagnostics()
        return {
            "size": self.library.size,
            "avg_correlation": diagnostics.get("avg_correlation", 0.0),
            "max_correlation": diagnostics.get("max_correlation", 0.0),
            "domain_saturation": diagnostics.get("domain_saturation", {}),
            "dependence_metric": self.library.dependence_metric.name,
        }


# ---------------------------------------------------------------------------
# Hypothesis-Redundancy geometric gate (LLM jump worth)
# ---------------------------------------------------------------------------


def _flatten_signal_matrix(signals: np.ndarray) -> np.ndarray:
    """Flatten ``(M, T)`` or ``(T,)`` signals to a 1-D column, NaNs → 0."""
    arr = np.asarray(signals, dtype=np.float64)
    flat = arr.reshape(-1)
    return np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)


def collect_library_span_matrix(
    library: FactorLibrary | Sequence[np.ndarray],
) -> np.ndarray:
    """Stack library factor signals as columns of a ``(D, K)`` span matrix."""
    columns: list[np.ndarray] = []
    if isinstance(library, FactorLibrary):
        for factor in library.list_factors():
            if factor.signals is None:
                continue
            columns.append(_flatten_signal_matrix(factor.signals))
    else:
        for sig in library:
            columns.append(_flatten_signal_matrix(sig))
    if not columns:
        return np.zeros((0, 0), dtype=np.float64)
    length = min(c.shape[0] for c in columns)
    if length < 1:
        return np.zeros((0, 0), dtype=np.float64)
    return np.column_stack([c[:length] for c in columns])


def assess_llm_jump_worth(
    library_span: np.ndarray | FactorLibrary | Sequence[np.ndarray],
    candidate: np.ndarray,
    *,
    target: np.ndarray | None = None,
    threshold: float = 0.45,
    energy_fraction: float = 0.95,
    compression_weight: float = 0.3,
    escape_weight: float = 0.45,
    alignment_weight: float = 0.25,
) -> JumpWorthAssessment:
    """Score whether a non-local LLM jump is worth its cost.

    Parameters
    ----------
    library_span:
        Either a ``(D, K)`` matrix (columns = explored formulas), a
        :class:`FactorLibrary`, or a sequence of signal arrays.
    candidate:
        Candidate direction — ``(M, T)`` signals or a flat vector. Compared
        against the library span after flattening.
    target:
        Optional target / residual direction (e.g. forward returns flattened).
        When omitted, residual alignment contributes 0 and the composite
        reweights compression + escape only.
    threshold:
        ``recommend_llm_jump`` fires when ``jump_worth >= threshold``.
    """
    if isinstance(library_span, FactorLibrary) or (
        not isinstance(library_span, np.ndarray)
    ):
        span = collect_library_span_matrix(library_span)  # type: ignore[arg-type]
    else:
        span = np.asarray(library_span, dtype=np.float64)
        if span.ndim == 1:
            span = span.reshape(-1, 1)

    cand = _flatten_signal_matrix(candidate)
    library_size = int(span.shape[1]) if span.ndim == 2 else 0

    # Empty library → any candidate is a full escape; jump is worth it only
    # if we also have no cheap local baseline (always recommend first jump).
    if span.size == 0 or library_size == 0:
        return JumpWorthAssessment(
            jump_worth=1.0,
            spectral_compression=0.0,
            orthogonal_escape=1.0,
            residual_alignment=0.0,
            recommend_llm_jump=True,
            rationale=(
                "Empty library span — no local redundancy to exploit; "
                "non-local proposal is the only coverage option."
            ),
            library_rank=0,
            library_size=0,
            threshold=threshold,
        )

    # Align dimensions.
    d = min(span.shape[0], cand.shape[0])
    if d < 2:
        return JumpWorthAssessment(
            jump_worth=0.0,
            spectral_compression=0.0,
            orthogonal_escape=0.0,
            residual_alignment=0.0,
            recommend_llm_jump=False,
            rationale="Insufficient dimension for spectral jump diagnostic.",
            library_rank=0,
            library_size=library_size,
            threshold=threshold,
        )
    span = np.nan_to_num(span[:d, :], nan=0.0, posinf=0.0, neginf=0.0)
    cand = cand[:d]

    compression = spectral_compression_score(span, energy_fraction=energy_fraction)
    basis, rank = library_span_basis(span, energy_fraction=energy_fraction)
    # Orthogonal escape = fraction of candidate energy outside the span.
    if rank == 0 or basis.size == 0:
        escape = 1.0
        residual = cand.copy()
    else:
        # Project candidate onto orthonormal basis columns.
        coeffs = basis.T @ cand
        projection = basis @ coeffs
        residual = cand - projection
        cand_energy = float(np.dot(cand, cand))
        if cand_energy < 1e-18:
            escape = 0.0
        else:
            escape = float(np.clip(np.dot(residual, residual) / cand_energy, 0.0, 1.0))

    alignment = 0.0
    # A residual that is a negligible fraction of the candidate's energy is
    # numerical noise, not a meaningful "novel direction" -- cosine
    # similarity of noise against the target is unstable (observed ~0.24
    # on in-span cases) and must not contribute to jump_worth.
    if target is not None and escape > 1e-6:
        tgt = _flatten_signal_matrix(target)
        td = min(d, tgt.shape[0])
        alignment = residual_alignment_score(
            residual[:td],
            tgt[:td],
            span_basis=basis[:td, :] if basis.size else basis,
        )

    # Composite. If no target, redistribute alignment weight to escape.
    cw = float(compression_weight)
    ew = float(escape_weight)
    aw = float(alignment_weight) if target is not None else 0.0
    if target is None:
        ew = ew + float(alignment_weight)
    total_w = cw + ew + aw
    if total_w <= 1e-12:
        cw, ew, aw, total_w = 0.3, 0.7, 0.0, 1.0
    cw, ew, aw = cw / total_w, ew / total_w, aw / total_w

    jump_worth = float(
        np.clip(cw * compression + ew * escape + aw * alignment, 0.0, 1.0)
    )
    recommend = jump_worth >= threshold
    rationale = (
        f"jump_worth={jump_worth:.3f} "
        f"(compression={compression:.3f}, escape={escape:.3f}, "
        f"alignment={alignment:.3f}, rank={rank}/{library_size}); "
        + (
            "recommend non-local LLM jump."
            if recommend
            else "prefer cheap local AST edit."
        )
    )
    return JumpWorthAssessment(
        jump_worth=jump_worth,
        spectral_compression=float(compression),
        orthogonal_escape=float(escape),
        residual_alignment=float(alignment),
        recommend_llm_jump=recommend,
        rationale=rationale,
        library_rank=int(rank),
        library_size=library_size,
        threshold=threshold,
    )
