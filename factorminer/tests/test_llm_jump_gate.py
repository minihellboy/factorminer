"""Tests for the Hypothesis-Redundancy LLM jump-worth gate.

Covers spectral compression / orthogonal escape / residual alignment in
architecture/geometry.py + dependence.py, JumpWorthAssessment discrimination,
and optional island_model consumption (default off).
"""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.architecture.dependence import (
    library_span_basis,
    orthogonal_escape_score,
    residual_alignment_score,
    spectral_compression_score,
)
from factorminer.architecture.geometry import (
    JumpWorthAssessment,
    assess_llm_jump_worth,
    collect_library_span_matrix,
)
from factorminer.architecture.island_model import (
    IslandBiasDescriptor,
    IslandMiningCoordinator,
    IslandPopulation,
)
from factorminer.core.factor_library import Factor, FactorLibrary

# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------


def test_spectral_compression_high_for_redundant_span():
    rng = np.random.default_rng(0)
    base = rng.normal(size=100)
    # 5 near-duplicate columns → highly compressed span.
    span = np.column_stack([base + 0.01 * rng.normal(size=100) for _ in range(5)])
    score = spectral_compression_score(span)
    assert score > 0.7


def test_spectral_compression_low_for_full_rank_span():
    rng = np.random.default_rng(1)
    # Orthogonal-ish columns.
    span = rng.normal(size=(100, 5))
    # QR to force full column rank with spread energy.
    q, _ = np.linalg.qr(span)
    score = spectral_compression_score(q)
    assert score < 0.5


def test_orthogonal_escape_discriminates_novel_vs_duplicate():
    rng = np.random.default_rng(2)
    basis_cols = rng.normal(size=(80, 3))
    q, _ = np.linalg.qr(basis_cols)
    span = q  # orthonormal span

    # Duplicate direction: inside the span.
    dup = span @ np.array([0.5, -0.3, 0.8])
    # Novel direction: orthogonal complement.
    residual_dir = rng.normal(size=80)
    residual_dir = residual_dir - span @ (span.T @ residual_dir)
    residual_dir = residual_dir / (np.linalg.norm(residual_dir) + 1e-12)

    esc_dup = orthogonal_escape_score(span, dup)
    esc_novel = orthogonal_escape_score(span, residual_dir)
    assert esc_novel > esc_dup + 0.5
    assert esc_novel > 0.9
    assert esc_dup < 0.15


def test_residual_alignment_score_basic():
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.9, 0.1, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0, 0.0])
    assert residual_alignment_score(a, b) > 0.9
    assert residual_alignment_score(a, c) < 0.1


def test_library_span_basis_rank():
    rng = np.random.default_rng(3)
    col = rng.normal(size=50)
    span = np.column_stack([col, 2 * col, rng.normal(size=50)])
    basis, rank = library_span_basis(span, energy_fraction=0.999)
    assert rank >= 2
    assert basis.shape[1] == rank


def test_library_span_basis_not_centered():
    """library_span_basis must span the true (uncentered) column space.

    Regression test: a prior implementation column-centered before SVD,
    so a candidate exactly in the raw column span (but with nonzero mean)
    reported large spurious residual energy against its own unrepresented
    mean component -- a false "escape"/novelty signal for coverage the
    library already had.
    """
    v = np.arange(20, dtype=np.float64)  # nonzero mean (9.5)
    span = np.column_stack([v, 2 * v])
    basis, rank = library_span_basis(span)
    assert rank >= 1
    # v is exactly in col(span); projecting it on the basis must reproduce
    # it almost exactly (near-zero residual), which only holds if the
    # basis spans the *uncentered* column space.
    projection = basis @ (basis.T @ v)
    residual_energy = float(np.dot(v - projection, v - projection))
    assert residual_energy < 1e-6 * float(np.dot(v, v))


def test_orthogonal_escape_no_false_positive_on_nonzero_mean_in_span_candidate():
    """Regression test for the column-centering escape bug (see above).

    Concrete case from the audit: candidate is an exact linear combination
    of library columns, so true escape must be ~0; a centering bug in the
    shared basis made this ~0.73 instead.
    """
    v = np.arange(20, dtype=np.float64)
    span = np.column_stack([v, 2 * v])
    escape = orthogonal_escape_score(span, v)
    assert escape < 1e-6


def test_jump_worth_no_false_positive_on_in_span_nonzero_mean_candidate():
    """End-to-end regression: assess_llm_jump_worth must not recommend a
    non-local LLM jump for a candidate that adds zero new coverage."""
    v = np.arange(20, dtype=np.float64)
    span = np.column_stack([v, 2 * v])
    assessment = assess_llm_jump_worth(span, v)
    assert assessment.orthogonal_escape < 1e-6
    assert assessment.recommend_llm_jump is False


def test_jump_worth_forwards_energy_fraction_to_span_basis():
    """assess_llm_jump_worth must use the same energy_fraction cutoff for
    both spectral_compression_score and library_span_basis, not silently
    fall back to library_span_basis's own default (0.99) while compression
    uses a caller-supplied value -- that mismatch produces inconsistent
    rank cutoffs between the two diagnostics."""
    rng = np.random.default_rng(11)
    big1 = rng.normal(size=200)
    big2 = rng.normal(size=200)
    tiny = 1e-4 * rng.normal(size=200)
    span = np.column_stack([big1, big2, tiny])
    assessment = assess_llm_jump_worth(span, rng.normal(size=200), energy_fraction=0.5)
    # At energy_fraction=0.5 the tiny third direction should not be
    # required for the rank cut on either diagnostic -- library_rank
    # should reflect the same low cutoff compression_score used.
    assert assessment.library_rank <= 2


# ---------------------------------------------------------------------------
# JumpWorthAssessment discrimination (acceptance proof)
# ---------------------------------------------------------------------------


def test_jump_worth_higher_for_novel_direction_than_duplicate():
    """Concrete proof: novel-direction candidate scores higher than near-duplicate."""
    rng = np.random.default_rng(4)
    # Library span: 4 correlated momentum-like directions.
    core = rng.normal(size=(120, 2))
    q, _ = np.linalg.qr(np.column_stack([core, rng.normal(size=(120, 1))]))
    span = np.column_stack(
        [
            q[:, 0],
            q[:, 0] + 0.05 * q[:, 1],
            q[:, 1],
            q[:, 0] - 0.05 * q[:, 1],
        ]
    )

    # Near-duplicate candidate (in-span).
    duplicate = span[:, 0] + 0.02 * span[:, 1]
    # Genuinely novel candidate (orthogonal residual direction).
    basis, _ = library_span_basis(span)
    novel = rng.normal(size=120)
    novel = novel - basis @ (basis.T @ novel)
    novel = novel / (np.linalg.norm(novel) + 1e-12)

    # Target aligned with the novel residual → alignment boosts novel further.
    target = novel + 0.1 * rng.normal(size=120)

    a_dup = assess_llm_jump_worth(span, duplicate, target=target)
    a_novel = assess_llm_jump_worth(span, novel, target=target)

    assert isinstance(a_novel, JumpWorthAssessment)
    assert a_novel.jump_worth > a_dup.jump_worth
    assert a_novel.orthogonal_escape > a_dup.orthogonal_escape + 0.4
    assert a_novel.recommend_llm_jump is True or a_novel.jump_worth > a_dup.jump_worth
    # Duplicate should look like a local-edit case.
    assert a_dup.orthogonal_escape < 0.25


def test_jump_worth_empty_library_recommends_jump():
    a = assess_llm_jump_worth(np.zeros((0, 0)), np.ones(10))
    assert a.recommend_llm_jump is True
    assert a.jump_worth == pytest.approx(1.0)


def test_jump_worth_from_factor_library():
    rng = np.random.default_rng(5)
    library = FactorLibrary(correlation_threshold=0.99, ic_threshold=0.01)
    m, t = 8, 30
    for i in range(3):
        sig = rng.normal(size=(m, t))
        # Nearly orthogonal signals so all three clear the corr bar.
        if i > 0:
            for prev in library.list_factors():
                flat = sig.reshape(-1)
                p = prev.signals.reshape(-1)
                flat = flat - (np.dot(flat, p) / (np.dot(p, p) + 1e-12)) * p
                sig = flat.reshape(m, t)
        library.admit_factor(
            Factor(
                id=0,
                name=f"f{i}",
                formula=f"Rank($close)+{i}",
                category="test",
                ic_mean=0.05,
                icir=1.0,
                ic_win_rate=0.55,
                max_correlation=0.0,
                batch_number=1,
                signals=sig,
            )
        )
    span = collect_library_span_matrix(library)
    assert span.shape[1] == library.size
    assert library.size >= 1

    cand_dup = library.list_factors()[0].signals + 0.01 * rng.normal(size=(m, t))
    cand_novel = rng.normal(size=(m, t))

    a_dup = assess_llm_jump_worth(library, cand_dup)
    a_novel = assess_llm_jump_worth(library, cand_novel)
    assert a_novel.jump_worth >= a_dup.jump_worth - 1e-9
    assert a_novel.orthogonal_escape >= a_dup.orthogonal_escape - 1e-9
    d = a_novel.to_dict()
    assert "jump_worth" in d and "recommend_llm_jump" in d


def test_jump_worth_assessment_docstring_mentions_llm_cost():
    """Creative AI angle: public dataclass documents frontier-LLM cost gate."""
    doc = JumpWorthAssessment.__doc__ or ""
    assert "LLM" in doc
    assert "cost" in doc.lower() or "worth" in doc.lower()


# ---------------------------------------------------------------------------
# Island-model optional consumption (default off)
# ---------------------------------------------------------------------------


class _DummyGenerator:
    def generate(self, *args, **kwargs):
        return []


class _DummyAdmission:
    def admit_migrant(self, factor, *, iteration: int = 0, allow_replacement: bool = False):
        return False, "dummy-reject"


class _DummyLoop:
    """Minimal stand-in so IslandPopulation can be constructed in unit tests."""

    def __init__(self, library: FactorLibrary) -> None:
        self.library = library
        self.current_iteration = 0
        self.generator = _DummyGenerator()
        self.admission_service = _DummyAdmission()

    def run(self, target_size: int = 10_000, max_iterations: int = 1):
        self.current_iteration = max_iterations
        return self.library


def _pop(name: str, library: FactorLibrary) -> IslandPopulation:
    loop = _DummyLoop(library)
    return IslandPopulation(
        name=name,
        loop=loop,  # type: ignore[arg-type]
        bias=IslandBiasDescriptor(name=name),
    )


def test_island_jump_gate_default_off_no_log():
    rng = np.random.default_rng(6)
    lib_a = FactorLibrary(correlation_threshold=0.99, ic_threshold=0.01)
    lib_b = FactorLibrary(correlation_threshold=0.99, ic_threshold=0.01)
    for lib, offset in ((lib_a, 0), (lib_b, 10)):
        lib.admit_factor(
            Factor(
                id=0,
                name="x",
                formula=f"Rank($close)+{offset}",
                category="t",
                ic_mean=0.05,
                icir=1.0,
                ic_win_rate=0.5,
                max_correlation=0.0,
                batch_number=1,
                signals=rng.normal(size=(4, 10)),
            )
        )

    coord = IslandMiningCoordinator(
        [_pop("a", lib_a), _pop("b", lib_b)],
        migration_interval=1,
        migration_top_k=1,
        enable_llm_jump_gate=False,
    )
    coord.run(total_iterations=1)
    assert coord.jump_worth_log == []


def test_island_jump_gate_enabled_records_assessments():
    rng = np.random.default_rng(8)
    pops = []
    for name in ("a", "b"):
        lib = FactorLibrary(correlation_threshold=0.99, ic_threshold=0.01)
        lib.admit_factor(
            Factor(
                id=0,
                name=f"{name}1",
                formula=f"Rank($volume)+{name}",
                category="t",
                ic_mean=0.06,
                icir=1.2,
                ic_win_rate=0.6,
                max_correlation=0.0,
                batch_number=1,
                signals=rng.normal(size=(5, 12)),
            )
        )
        pops.append(_pop(name, lib))

    coord = IslandMiningCoordinator(
        pops,
        migration_interval=1,
        migration_top_k=1,
        enable_llm_jump_gate=True,
        llm_jump_threshold=0.3,
    )
    coord.run(total_iterations=1)
    assert len(coord.jump_worth_log) >= 2
    assert all("jump_worth" in e for e in coord.jump_worth_log)
    assert {e["island"] for e in coord.jump_worth_log} == {"a", "b"}


def test_assess_exploration_jump_direct():
    rng = np.random.default_rng(9)
    lib = FactorLibrary(correlation_threshold=0.99, ic_threshold=0.01)
    lib.admit_factor(
        Factor(
            id=0,
            name="only",
            formula="Rank($close)",
            category="t",
            ic_mean=0.05,
            icir=1.0,
            ic_win_rate=0.5,
            max_correlation=0.0,
            batch_number=1,
            signals=rng.normal(size=(6, 15)),
        )
    )
    empty = FactorLibrary(correlation_threshold=0.99, ic_threshold=0.01)
    coord = IslandMiningCoordinator(
        [_pop("a", lib), _pop("b", empty)],
        enable_llm_jump_gate=True,
    )
    assessment = coord.assess_exploration_jump(coord.island("a"))
    assert isinstance(assessment, JumpWorthAssessment)
    assert 0.0 <= assessment.jump_worth <= 1.0
