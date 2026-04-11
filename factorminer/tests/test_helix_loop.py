"""Tests for the Helix Loop (core/helix_loop.py)."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from factorminer.core.helix_loop import HelixLoop
    HAS_HELIX = True
except ImportError:
    HAS_HELIX = False

from factorminer.agent.llm_interface import MockProvider
from factorminer.core.config import MiningConfig
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.ralph_loop import EvaluationResult

pytestmark = pytest.mark.skipif(not HAS_HELIX, reason="helix_loop not yet built")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_tensor(rng):
    """Small data tensor and returns for HelixLoop tests."""
    M, T, F = 10, 50, 3
    data = rng.normal(0, 1, (M, T, F))
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, (M, T)), axis=1)
    returns = np.zeros((M, T))
    returns[:, 1:] = np.diff(close, axis=1) / close[:, :-1]
    return data, returns


# -----------------------------------------------------------------------
# HelixLoop can be instantiated with all defaults
# -----------------------------------------------------------------------

def test_helix_loop_instantiates_with_defaults(small_tensor):
    """HelixLoop with all features off should be instantiable."""
    data, returns = small_tensor
    config = MiningConfig(target_library_size=5, max_iterations=1)
    provider = MockProvider()

    loop = HelixLoop(
        config=config,
        data_tensor=data,
        returns=returns,
        llm_provider=provider,
        canonicalize=False,
        enable_knowledge_graph=False,
        enable_auto_inventor=False,
    )
    assert loop is not None


# -----------------------------------------------------------------------
# HelixLoop with canonicalize=True
# -----------------------------------------------------------------------

def test_helix_loop_canonicalize_flag(small_tensor):
    """HelixLoop with canonicalize=True should initialize the canonicalizer."""
    data, returns = small_tensor
    config = MiningConfig(target_library_size=5, max_iterations=1)
    provider = MockProvider()

    loop = HelixLoop(
        config=config,
        data_tensor=data,
        returns=returns,
        llm_provider=provider,
        canonicalize=True,
    )
    assert loop._canonicalize is True


# -----------------------------------------------------------------------
# HelixLoop with MockProvider runs 1 iteration
# -----------------------------------------------------------------------

def test_helix_loop_runs_one_iteration(small_tensor):
    """HelixLoop should complete 1 iteration without error using MockProvider."""
    data, returns = small_tensor
    config = MiningConfig(
        target_library_size=3,
        max_iterations=1,
        batch_size=5,
    )
    provider = MockProvider()

    loop = HelixLoop(
        config=config,
        data_tensor=data,
        returns=returns,
        llm_provider=provider,
        canonicalize=False,
        enable_knowledge_graph=False,
        enable_auto_inventor=False,
    )
    # Run the loop -- should not raise
    loop.run()
    assert loop.library is not None


def test_phase2_revocation_updates_stats_and_library_state(small_tensor):
    """Post-admission revocation should keep stats aligned with library state."""
    data, returns = small_tensor
    config = MiningConfig(
        target_library_size=3,
        max_iterations=1,
        batch_size=5,
        ic_threshold=0.0001,
        correlation_threshold=0.95,
    )
    provider = MockProvider()

    loop = HelixLoop(
        config=config,
        data_tensor=data,
        returns=returns,
        llm_provider=provider,
        canonicalize=False,
        enable_knowledge_graph=False,
        enable_auto_inventor=False,
    )

    original_validate = loop._helix_validate

    def force_one_revocation(results, admitted_results):
        rejected = original_validate(results, admitted_results)
        for admitted in admitted_results:
            if admitted.admitted:
                loop._revoke_admission(admitted, results, "forced test revocation")
                return rejected + 1
        return rejected

    loop._helix_validate = force_one_revocation

    stats = loop._run_iteration(batch_size=5)

    assert stats["admitted"] == loop.library.size
    if loop.library.correlation_matrix is not None:
        assert loop.library.correlation_matrix.shape[0] == loop.library.size


def test_revoke_admission_rebuilds_library_indices(small_tensor):
    """Revoking a factor should rebuild the library correlation bookkeeping."""
    data, returns = small_tensor
    config = MiningConfig(target_library_size=5, max_iterations=1)
    provider = MockProvider()

    loop = HelixLoop(
        config=config,
        data_tensor=data,
        returns=returns,
        llm_provider=provider,
        canonicalize=False,
        enable_knowledge_graph=False,
        enable_auto_inventor=False,
    )

    factor_a = Factor(
        id=0,
        name="factor_a",
        formula="Mean($close, 5)",
        category="test",
        ic_mean=0.1,
        icir=1.0,
        ic_win_rate=0.6,
        max_correlation=0.0,
        batch_number=1,
        signals=np.ones_like(returns),
    )
    factor_b = Factor(
        id=0,
        name="factor_b",
        formula="Std($close, 5)",
        category="test",
        ic_mean=0.08,
        icir=0.9,
        ic_win_rate=0.55,
        max_correlation=0.1,
        batch_number=1,
        signals=np.full_like(returns, 2.0),
    )

    loop.library.admit_factor(factor_a)
    loop.library.admit_factor(factor_b)

    result = EvaluationResult(
        factor_name="factor_a",
        formula="Mean($close, 5)",
        admitted=True,
    )
    loop._revoke_admission(result, [], "forced test revocation")

    assert loop.library.size == 1
    assert list(loop.library.factors.values())[0].name == "factor_b"
    assert loop.library._id_to_index == {list(loop.library.factors.keys())[0]: 0}
    assert loop.library.correlation_matrix is not None
    assert loop.library.correlation_matrix.shape == (1, 1)


def test_helix_embedding_screen_filters_library_duplicates(small_tensor):
    """Embedding-aware synthesis should drop near-duplicates of admitted factors."""
    data, returns = small_tensor
    config = MiningConfig(target_library_size=5, max_iterations=1)
    provider = MockProvider()

    library = FactorLibrary(correlation_threshold=0.95, ic_threshold=0.0001)
    library.admit_factor(
        Factor(
            id=0,
            name="existing_factor",
            formula="Mean($close, 5)",
            category="test",
            ic_mean=0.1,
            icir=1.0,
            ic_win_rate=0.6,
            max_correlation=0.0,
            batch_number=0,
            signals=np.ones_like(returns),
        )
    )

    loop = HelixLoop(
        config=config,
        data_tensor=data,
        returns=returns,
        llm_provider=provider,
        library=library,
        canonicalize=False,
        enable_embeddings=True,
        enable_knowledge_graph=False,
        enable_auto_inventor=False,
    )

    deduped, canon_dupes, semantic_dupes = loop._canonicalize_and_dedup(
        [
            ("dup_factor", "Mean($close, 5)"),
            ("novel_factor", "Std($close, 5)"),
        ]
    )

    assert canon_dupes == 0
    assert semantic_dupes == 1
    assert deduped == [("novel_factor", "Std($close, 5)")]
