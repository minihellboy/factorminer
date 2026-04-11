"""Tests for the factor library management system."""

from __future__ import annotations

import numpy as np
import pytest

from factorminer.core.factor_library import Factor, FactorLibrary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def empty_library():
    return FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)


def _make_factor(
    name="test",
    formula="Neg($close)",
    ic=0.06,
    signals=None,
    rng=None,
    M=20,
    T=60,
):
    """Helper to create a Factor with random signals."""
    if signals is None and rng is not None:
        signals = rng.normal(0, 1, (M, T))
    return Factor(
        id=0,
        name=name,
        formula=formula,
        category="test",
        ic_mean=ic,
        icir=1.0,
        ic_win_rate=0.6,
        max_correlation=0.0,
        batch_number=1,
        signals=signals,
    )


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------

class TestAdmission:
    """Test factor admission rules."""

    def test_admit_first_factor(self, empty_library, rng):
        factor = _make_factor(name="f1", ic=0.05, rng=rng)
        fid = empty_library.admit_factor(factor)
        assert fid == 1
        assert empty_library.size == 1
        assert factor.id == 1

    def test_admit_assigns_incremental_ids(self, empty_library, rng):
        f1 = _make_factor(name="f1", rng=rng)
        f2 = _make_factor(name="f2", rng=rng)
        id1 = empty_library.admit_factor(f1)
        id2 = empty_library.admit_factor(f2)
        assert id1 == 1
        assert id2 == 2

    def test_check_admission_ic_below_threshold(self, empty_library, rng):
        signals = rng.normal(0, 1, (20, 60))
        admitted, reason = empty_library.check_admission(0.03, signals)
        assert not admitted
        assert "below threshold" in reason

    def test_check_admission_first_factor(self, empty_library, rng):
        signals = rng.normal(0, 1, (20, 60))
        admitted, reason = empty_library.check_admission(0.05, signals)
        assert admitted
        assert "First factor" in reason

    def test_check_admission_rejects_high_correlation(self, rng):
        lib = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)
        # Add a factor
        f1 = _make_factor(name="f1", rng=rng)
        lib.admit_factor(f1)

        # Try to admit same signals (correlation = 1.0)
        admitted, reason = lib.check_admission(0.05, f1.signals)
        assert not admitted
        assert "correlation" in reason.lower()

    def test_check_admission_accepts_low_correlation(self, rng):
        lib = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)
        f1 = _make_factor(name="f1", rng=rng)
        lib.admit_factor(f1)

        # Independent signals
        independent_signals = rng.normal(0, 1, (20, 60))
        admitted, reason = lib.check_admission(0.05, independent_signals)
        assert admitted

    def test_factor_in_library_after_admission(self, empty_library, rng):
        factor = _make_factor(name="f1", rng=rng)
        fid = empty_library.admit_factor(factor)
        assert fid in empty_library.factors
        retrieved = empty_library.get_factor(fid)
        assert retrieved.name == "f1"


# ---------------------------------------------------------------------------
# Replacement
# ---------------------------------------------------------------------------

class TestReplacement:
    """Test the replacement mechanism (Eq. 11)."""

    def test_replacement_ic_below_floor(self, rng):
        lib = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)
        f1 = _make_factor(name="f1", ic=0.06, rng=rng)
        lib.admit_factor(f1)

        signals = rng.normal(0, 1, (20, 60))
        should, fid, reason = lib.check_replacement(0.05, signals)  # Below 0.10
        assert not should
        assert "below replacement floor" in reason

    def test_replacement_needs_exactly_one_correlated(self, rng):
        lib = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)
        f1 = _make_factor(name="f1", ic=0.06, rng=rng)
        lib.admit_factor(f1)

        # Independent signals -> 0 correlated factors
        signals = rng.normal(0, 1, (20, 60))
        should, fid, reason = lib.check_replacement(0.15, signals)
        assert not should
        assert "0 correlated" in reason

    def test_replacement_success(self, rng):
        lib = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)
        f1_signals = rng.normal(0, 1, (20, 60))
        f1 = _make_factor(name="f1", ic=0.06, signals=f1_signals)
        lib.admit_factor(f1)

        # Candidate highly correlated with f1 but much better IC
        candidate_signals = f1_signals + rng.normal(0, 0.1, (20, 60))
        should, old_id, reason = lib.check_replacement(
            0.15, candidate_signals, ic_min=0.10, ic_ratio=1.3
        )
        assert should
        assert old_id == 1

    def test_replace_factor(self, rng):
        lib = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.04)
        f1 = _make_factor(name="old_factor", ic=0.06, rng=rng)
        fid = lib.admit_factor(f1)

        new_factor = _make_factor(name="new_factor", ic=0.15, rng=rng)
        lib.replace_factor(fid, new_factor)

        assert fid not in lib.factors
        assert lib.size == 1
        remaining = lib.list_factors()
        assert remaining[0].name == "new_factor"

    def test_replace_nonexistent_raises(self, empty_library, rng):
        new_factor = _make_factor(name="new", rng=rng)
        with pytest.raises(KeyError):
            empty_library.replace_factor(999, new_factor)


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:
    """Test correlation matrix management."""

    def test_matrix_initialized_on_first_admit(self, empty_library, rng):
        f = _make_factor(name="f1", rng=rng)
        empty_library.admit_factor(f)
        assert empty_library.correlation_matrix is not None
        assert empty_library.correlation_matrix.shape == (1, 1)

    def test_matrix_grows_with_admissions(self, empty_library, rng):
        for i in range(3):
            f = _make_factor(name=f"f{i}", rng=rng)
            empty_library.admit_factor(f)
        assert empty_library.correlation_matrix.shape == (3, 3)

    def test_matrix_symmetric(self, rng):
        lib = FactorLibrary()
        for i in range(4):
            f = _make_factor(name=f"f{i}", rng=rng)
            lib.admit_factor(f)
        mat = lib.correlation_matrix
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_update_correlation_matrix_full(self, rng):
        lib = FactorLibrary()
        for i in range(3):
            f = _make_factor(name=f"f{i}", rng=rng)
            lib.admit_factor(f)
        # Full recompute
        lib.update_correlation_matrix()
        assert lib.correlation_matrix.shape == (3, 3)
        np.testing.assert_array_almost_equal(
            lib.correlation_matrix, lib.correlation_matrix.T
        )

    def test_compute_correlation_same_signals(self, rng):
        lib = FactorLibrary()
        signals = rng.normal(0, 1, (20, 60))
        corr = lib.compute_correlation(signals, signals)
        assert corr > 0.95


# ---------------------------------------------------------------------------
# Queries and diagnostics
# ---------------------------------------------------------------------------

class TestQueries:
    """Test library query methods."""

    def test_size_property(self, mock_library):
        assert mock_library.size == 3

    def test_list_factors(self, mock_library):
        factors = mock_library.list_factors()
        assert len(factors) == 3
        # Should be sorted by ID
        ids = [f.id for f in factors]
        assert ids == sorted(ids)

    def test_get_factor(self, mock_library):
        factors = mock_library.list_factors()
        fid = factors[0].id
        f = mock_library.get_factor(fid)
        assert f.id == fid

    def test_get_factor_nonexistent_raises(self, mock_library):
        with pytest.raises(KeyError):
            mock_library.get_factor(9999)

    def test_get_factors_by_category(self, mock_library):
        result = mock_library.get_factors_by_category("test")
        assert len(result) == 3

    def test_get_factors_by_nonexistent_category(self, mock_library):
        result = mock_library.get_factors_by_category("nonexistent")
        assert len(result) == 0

    def test_get_diagnostics(self, mock_library):
        diag = mock_library.get_diagnostics()
        assert "size" in diag
        assert diag["size"] == 3
        assert "avg_correlation" in diag
        assert "max_correlation" in diag
        assert "category_counts" in diag
        assert "saturation" in diag

    def test_get_state_summary(self, mock_library):
        summary = mock_library.get_state_summary()
        assert "library_size" in summary
        assert summary["library_size"] == 3
        assert "categories" in summary
        assert "recent_admissions" in summary


# ---------------------------------------------------------------------------
# Factor serialization
# ---------------------------------------------------------------------------

class TestFactorSerialization:
    """Test Factor to_dict / from_dict."""

    def test_factor_to_dict(self):
        f = Factor(
            id=1,
            name="test",
            formula="Neg($close)",
            category="momentum",
            ic_mean=0.06,
            icir=1.0,
            ic_win_rate=0.6,
            max_correlation=0.1,
            batch_number=1,
            admission_date="2024-01-01 00:00:00",
        )
        d = f.to_dict()
        assert d["id"] == 1
        assert d["name"] == "test"
        assert d["formula"] == "Neg($close)"
        assert "signals" not in d

    def test_factor_from_dict(self):
        d = {
            "id": 2,
            "name": "restored",
            "formula": "Add($open, $close)",
            "category": "arithmetic",
            "ic_mean": 0.08,
            "icir": 1.2,
            "ic_win_rate": 0.65,
            "max_correlation": 0.2,
            "batch_number": 3,
            "admission_date": "2024-06-15 12:00:00",
        }
        f = Factor.from_dict(d)
        assert f.id == 2
        assert f.name == "restored"
        assert f.formula == "Add($open, $close)"

    def test_factor_roundtrip(self):
        f = Factor(
            id=5,
            name="roundtrip",
            formula="CsRank($close)",
            category="cross_sectional",
            ic_mean=0.07,
            icir=0.9,
            ic_win_rate=0.58,
            max_correlation=0.15,
            batch_number=2,
        )
        restored = Factor.from_dict(f.to_dict())
        assert restored.name == f.name
        assert restored.ic_mean == f.ic_mean
        assert restored.formula == f.formula

    def test_factor_roundtrip_preserves_provenance(self):
        f = Factor(
            id=6,
            name="with_provenance",
            formula="Neg($close)",
            category="test",
            ic_mean=0.05,
            icir=0.9,
            ic_win_rate=0.55,
            max_correlation=0.1,
            batch_number=2,
            provenance={
                "run_id": "run_001",
                "generator_family": "MockProvider",
                "candidate_rank": 2,
            },
        )

        restored = Factor.from_dict(f.to_dict())

        assert restored.provenance["run_id"] == "run_001"
        assert restored.provenance["generator_family"] == "MockProvider"
        assert restored.provenance["candidate_rank"] == 2
