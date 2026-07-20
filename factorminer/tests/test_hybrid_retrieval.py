"""Tests for hybrid BM25 + dense + heuristic RRF fusion (roadmap item 12)."""

from __future__ import annotations

from factorminer.memory.embeddings import FormulaEmbedder
from factorminer.memory.memory_store import (
    ExperienceMemory,
    ForbiddenDirection,
    SuccessPattern,
)
from factorminer.memory.retrieval import (
    HybridRetrievalConfig,
    bm25_scores,
    build_retrieval_query,
    dense_cosine_scores,
    fuse_hybrid_ranks,
    lightweight_cross_encoder_scores,
    reciprocal_rank_fusion,
    retrieval_quality_smoke,
    retrieve_memory,
    scores_to_ranks,
    tokenize_dsl,
)


def test_tokenize_dsl_extracts_operators_and_features() -> None:
    tokens = tokenize_dsl("CsRank(Corr($close, $volume, 20))")
    assert "csrank" in tokens
    assert "corr" in tokens
    assert "$close" in tokens or "close" in tokens
    assert "20" in tokens


def test_bm25_prefers_lexical_overlap() -> None:
    query = tokenize_dsl("TsRank $close momentum")
    docs = [
        tokenize_dsl("Div($volume, Mean($volume, 60))"),
        tokenize_dsl("TsRank($close, 20)"),
        tokenize_dsl("Neg(Std($returns, 10))"),
    ]
    scores = bm25_scores(query, docs)
    assert len(scores) == 3
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_rrf_is_rank_based_and_score_scale_free() -> None:
    # Two lists with very different raw score magnitudes but same order.
    ranks_a = scores_to_ranks([1000.0, 10.0, 1.0])
    ranks_b = scores_to_ranks([0.9, 0.5, 0.1])
    fused = reciprocal_rank_fusion([ranks_a, ranks_b], k=60)
    assert fused[0] > fused[1] > fused[2]


def test_fuse_hybrid_combines_three_signals() -> None:
    heuristic = [0.1, 5.0, 0.2]  # middle wins heuristically
    bm25 = [5.0, 0.1, 0.2]  # first wins lexically
    dense = [0.1, 0.2, 5.0]  # third wins densely
    fused, diag = fuse_hybrid_ranks(
        heuristic_scores=heuristic,
        bm25=bm25,
        dense=dense,
        config=HybridRetrievalConfig(enabled=True),
    )
    assert set(diag["signals_used"]) == {"heuristic", "bm25", "dense"}
    assert len(fused) == 3
    # With one first-place vote each, RRF should not collapse to a single
    # signal -- all three get non-zero mass.
    assert all(s > 0 for s in fused)


def test_hybrid_changes_ranking_vs_heuristic_only() -> None:
    """Concrete proof hybrid fusion changes output vs heuristic baseline.

    Heuristic alone elevates the High/high-occurrence volume pattern.
    A TsRank-focused query + BM25 should pull the TsRank pattern above it
    under hybrid fusion.
    """
    memory = ExperienceMemory()
    memory.success_patterns = [
        SuccessPattern(
            name="Volume Burst High Occ",
            description="volume ratio without ranking",
            template="Div($volume, Mean($volume, 60))",
            success_rate="High",
            example_factors=["Div($volume, Mean($volume, 60))"],
            occurrence_count=50,  # heuristic loves this
            confidence=0.9,
        ),
        SuccessPattern(
            name="TsRank Close Momentum",
            description="time series rank of close",
            template="TsRank($close, 20)",
            success_rate="Medium",
            example_factors=["TsRank($close, 20)"],
            occurrence_count=1,  # heuristic deprioritizes
            confidence=0.5,
        ),
        SuccessPattern(
            name="Noise Pattern",
            description="unrelated open range",
            template="Sub($high, $low)",
            success_rate="Low",
            example_factors=["Sub($high, $low)"],
            occurrence_count=1,
        ),
    ]
    memory.state.domain_saturation = {
        "Volume Burst High Occ": 0.1,
        "TsRank Close Momentum": 0.1,
        "Noise Pattern": 0.1,
    }
    memory.state.recent_admissions = [
        {"factor_id": "seed", "formula": "TsRank(Delta($close, 1), 10)"},
    ]

    library_state = {
        "library_size": 3,
        "domain_saturation": memory.state.domain_saturation,
        "query_formula": "TsRank($close, 20)",
    }

    heuristic_signal = retrieve_memory(
        memory,
        library_state=library_state,
        max_success=3,
        hybrid_config=HybridRetrievalConfig(enabled=False),
    )
    hybrid_signal = retrieve_memory(
        memory,
        library_state=library_state,
        max_success=3,
        embedder=FormulaEmbedder(use_faiss=False),
        hybrid_config=HybridRetrievalConfig(
            enabled=True,
            enable_bm25=True,
            enable_dense=True,
            enable_heuristic=True,
            enable_rerank=False,
        ),
    )

    heur_names = [p["name"] for p in heuristic_signal["recommended_directions"]]
    hybrid_names = [p["name"] for p in hybrid_signal["recommended_directions"]]

    assert heur_names[0] == "Volume Burst High Occ"
    assert hybrid_names[0] == "TsRank Close Momentum", (
        f"hybrid should promote TsRank over volume; got {hybrid_names}"
    )
    assert heur_names != hybrid_names

    diag = hybrid_signal["retrieval_diagnostics"]
    assert diag["success"]["mode"] == "hybrid_rrf"
    assert "bm25" in diag["success"]["signals_used"]


def test_dense_cosine_scores_with_hash_embedder() -> None:
    embedder = FormulaEmbedder(use_faiss=False)
    docs = [
        "TsRank($close, 20)",
        "Div($volume, Mean($volume, 60))",
        "TsRank($close, 10)",
    ]
    scores = dense_cosine_scores("TsRank($close, 20)", docs, embedder)
    assert len(scores) == 3
    # Exact match / near-paraphrase should beat unrelated volume formula.
    assert scores[0] >= scores[1]
    assert scores[2] >= scores[1]


def test_dense_cosine_scores_tfidf_backend_no_dimension_mismatch(monkeypatch) -> None:
    """Regression test: TF-IDF-backend dense scoring must not silently
    collapse to all-zero on independently-fit vectors.

    A prior implementation encoded the query and each document via
    separate single-text calls; the TF-IDF backend refits its vocabulary
    per call from whatever corpus is present, so a query and a
    not-yet-cached document could land in different-dimensional vector
    spaces. The resulting ``q_vec @ d_vec`` raised inside a broad
    ``except Exception`` and was swallowed, silently dropping the dense
    signal out of the hybrid RRF fusion (falling back to all-zero
    scores) with no error surfaced anywhere.
    """
    import factorminer.memory.embeddings as emb_mod

    monkeypatch.setattr(emb_mod, "_has_sentence_transformers", False)
    assert emb_mod._has_sklearn, "sklearn required for this regression test"

    embedder = FormulaEmbedder(use_faiss=False)
    query = "CsRank(Corr($close, $volume, 20))"
    docs = [
        "CsRank(Corr($close, $volume, 10))",  # lexically similar
        "Neg(Std($returns, 5))",  # unrelated
        "Mean($volume, 30)",  # unrelated
    ]

    scores = dense_cosine_scores(query, docs, embedder)

    assert scores != [0.0, 0.0, 0.0], (
        "dense scores silently collapsed to all-zero -- dimension mismatch bug regressed"
    )
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_lightweight_rerank_reorders_pool() -> None:
    query = "CsRank Corr $close $volume"
    candidates = [
        "Neg(Std($returns, 10))",
        "CsRank(Corr($close, $volume, 20))",
        "Div($volume, Mean($volume, 5))",
    ]
    scores = lightweight_cross_encoder_scores(query, candidates)
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_retrieval_quality_smoke_passes() -> None:
    """Repo previously had zero retrieval-quality instrumentation."""
    embedder = FormulaEmbedder(use_faiss=False)
    result = retrieval_quality_smoke(embedder=embedder)
    assert result["passed"] is True
    assert result["hybrid_ranking"][0] == "TsRank Momentum Close"
    assert "criterion" in result


def test_build_retrieval_query_includes_recent_formulas() -> None:
    q = build_retrieval_query(
        domain_saturation={"Momentum": 0.2, "Value": 0.9},
        recent_admissions=[{"formula": "TsRank($close, 20)"}],
        library_state={"query_formula": "Corr($close, $volume, 10)"},
    )
    assert "TsRank" in q or "tsrank" in q.lower() or "$close" in q
    assert "Momentum" in q or "explore" in q


def test_hybrid_diagnostics_present_on_empty_memory() -> None:
    signal = retrieve_memory(ExperienceMemory())
    assert "retrieval_diagnostics" in signal
    assert signal["recommended_directions"] == []


def test_forbidden_hybrid_prefers_matching_rejection_reason() -> None:
    memory = ExperienceMemory()
    memory.forbidden_directions = [
        ForbiddenDirection(
            name="Unrelated Open",
            description="open-only noise",
            reason="low ic",
            typical_correlation=0.3,
            occurrence_count=1,
        ),
        ForbiddenDirection(
            name="Raw Close Level",
            description="absolute close price",
            reason="highly correlated with existing price-level factors",
            typical_correlation=0.5,
            occurrence_count=1,
            correlated_factors=["$close"],
        ),
    ]
    memory.state.recent_rejections = [
        {"reason": "too correlated with price-level close factors"},
    ]
    signal = retrieve_memory(
        memory,
        library_state={"library_size": 1, "query_formula": "$close level"},
        max_forbidden=1,
        hybrid_config=HybridRetrievalConfig(enabled=True, enable_rerank=False),
    )
    names = [d["name"] for d in signal["forbidden_directions"]]
    assert names[0] == "Raw Close Level"
