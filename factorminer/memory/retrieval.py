"""Memory Retrieval operator R(M, L).

Context-dependent retrieval of experience memory, producing a structured
memory signal m for injection into the LLM generation prompt.

The retrieval considers the current library state (domain saturation,
recent rejections) to select the most relevant patterns and insights.

Hybrid ranking (roadmap item 12)
--------------------------------
Primary pattern selection combines three complementary signals via
Reciprocal Rank Fusion (RRF):

1. **Heuristic** -- success-rate / occurrence / saturation (historical default)
2. **BM25** -- lexical match over DSL operator/feature tokens
3. **Dense** -- cosine similarity via :class:`FormulaEmbedder` when available

An optional lightweight cross-encoder-style rerank can reorder the fused
top-K pool before it is formatted into the prompt payload. Dense and
cross-encoder paths are lazy/optional; ``--mock`` and CI stay on the
hash/TF-IDF embedder fallbacks and never force a model download.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from factorminer.memory.memory_store import (
    ExperienceMemory,
    ForbiddenDirection,
    MiningState,
    StrategicInsight,
    SuccessPattern,
)

logger = logging.getLogger(__name__)

# DSL tokens: operators (TsRank), features ($close), bare identifiers, numbers.
_DSL_TOKEN_RE = re.compile(r"\$?[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")

# Optional rank_bm25 -- never a hard dependency.
_has_rank_bm25 = False
try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

    _has_rank_bm25 = True
except ImportError:
    BM25Okapi = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Hybrid retrieval config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Feature-local config for hybrid BM25 + dense + heuristic fusion.

    Parameters
    ----------
    enabled :
        When False, fall back to pure heuristic ranking (legacy behaviour).
    rrf_k :
        RRF smoothing constant (standard default 60).
    bm25_k1, bm25_b :
        Okapi BM25 free parameters (used by the hand-rolled fallback).
    enable_dense :
        Include dense cosine ranks when an embedder is supplied.
    enable_bm25 :
        Include BM25/lexical ranks.
    enable_heuristic :
        Include the legacy success-rate/occurrence ranks.
    enable_rerank :
        Optional cross-encoder-style rerank over the fused top pool.
    rerank_pool_size :
        How many fused candidates enter the reranker before truncation.
    """

    enabled: bool = True
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    enable_dense: bool = True
    enable_bm25: bool = True
    enable_heuristic: bool = True
    enable_rerank: bool = False
    rerank_pool_size: int = 16

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "rrf_k": self.rrf_k,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "enable_dense": self.enable_dense,
            "enable_bm25": self.enable_bm25,
            "enable_heuristic": self.enable_heuristic,
            "enable_rerank": self.enable_rerank,
            "rerank_pool_size": self.rerank_pool_size,
        }


DEFAULT_HYBRID_CONFIG = HybridRetrievalConfig()


# ---------------------------------------------------------------------------
# Tokenization + BM25
# ---------------------------------------------------------------------------


def tokenize_dsl(text: str) -> list[str]:
    """Tokenize a formula/pattern string into DSL-aware lowercase tokens."""
    if not text:
        return []
    return [tok.lower() for tok in _DSL_TOKEN_RE.findall(text)]


def _pattern_document_text(pattern: SuccessPattern) -> str:
    parts = [
        pattern.name or "",
        pattern.description or "",
        pattern.template or "",
        " ".join(pattern.example_factors or ()),
    ]
    return " ".join(parts)


def _forbidden_document_text(direction: ForbiddenDirection) -> str:
    parts = [
        direction.name or "",
        direction.description or "",
        direction.reason or "",
        " ".join(direction.correlated_factors or ()),
    ]
    return " ".join(parts)


def bm25_scores(
    query_tokens: Sequence[str],
    documents: Sequence[Sequence[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Score *documents* against *query_tokens* with Okapi BM25.

    Prefers ``rank_bm25`` when installed; otherwise uses a pure-Python
    implementation so there is zero hard dependency.
    """
    n_docs = len(documents)
    if n_docs == 0:
        return []
    if not query_tokens:
        return [0.0] * n_docs

    if _has_rank_bm25 and BM25Okapi is not None:
        try:
            bm25 = BM25Okapi([list(doc) for doc in documents])
            raw = bm25.get_scores(list(query_tokens))
            return [float(x) for x in raw]
        except Exception:  # noqa: BLE001 - fall through to hand-rolled
            logger.debug("rank_bm25 failed; using hand-rolled BM25", exc_info=True)

    return _bm25_scores_python(query_tokens, documents, k1=k1, b=b)


def _bm25_scores_python(
    query_tokens: Sequence[str],
    documents: Sequence[Sequence[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Hand-rolled Okapi BM25 (Robertson / Zaragoza)."""
    n_docs = len(documents)
    doc_lens = [len(doc) for doc in documents]
    avgdl = (sum(doc_lens) / n_docs) if n_docs else 0.0

    df: dict[str, int] = {}
    tfs: list[dict[str, int]] = []
    for doc in documents:
        counts: dict[str, int] = {}
        for tok in doc:
            counts[tok] = counts.get(tok, 0) + 1
        tfs.append(counts)
        for tok in counts:
            df[tok] = df.get(tok, 0) + 1

    scores = [0.0] * n_docs
    for qtok in query_tokens:
        n_q = df.get(qtok, 0)
        if n_q == 0:
            continue
        idf = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
        for i, counts in enumerate(tfs):
            tf = counts.get(qtok, 0)
            if tf == 0:
                continue
            dl = doc_lens[i]
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl if avgdl > 0 else 0.0))
            scores[i] += idf * (tf * (k1 + 1.0)) / denom
    return scores


# ---------------------------------------------------------------------------
# Dense scoring + RRF + rerank
# ---------------------------------------------------------------------------


def dense_cosine_scores(
    query_text: str,
    document_texts: Sequence[str],
    embedder: Any | None,
) -> list[float]:
    """Cosine similarities between *query_text* and each document via embedder.

    Uses the embedder's private ``_encode_batch`` (falls back to
    ``_encode``) / ``_formula_to_text`` path when present (same backend as
    nearest-neighbour search). Returns zeros when no embedder is available
    so callers can skip the dense rank list.

    Batches the query and every document through one call when the
    embedder exposes ``_encode_batch``. This matters for the TF-IDF
    fallback backend specifically: its vocabulary is fit per call from
    whatever corpus is present, so encoding the query and each document
    via *separate* single-text calls can land them in different-sized
    vector spaces (a real, previously silent bug -- the dimension
    mismatch raised inside the ``try`` below and was swallowed, silently
    dropping the dense signal out of the hybrid RRF fusion).
    """
    n = len(document_texts)
    if embedder is None or n == 0 or not query_text.strip():
        return [0.0] * n

    to_text = getattr(embedder, "_formula_to_text", None)
    encode_batch = getattr(embedder, "_encode_batch", None)
    encode = getattr(embedder, "_encode", None)
    if encode_batch is None and encode is None:
        return [0.0] * n

    try:
        q_src = to_text(query_text) if callable(to_text) else query_text
        doc_srcs = [to_text(doc) if callable(to_text) else doc for doc in document_texts]

        if encode_batch is not None:
            vecs = encode_batch([q_src, *doc_srcs])
            q_vec, d_vecs = vecs[0], vecs[1:]
        else:
            q_vec = encode(q_src)
            d_vecs = [encode(d_src) for d_src in doc_srcs]

        # Both backends L2-normalise; dot == cosine.
        return [float(q_vec @ d_vec) for d_vec in d_vecs]
    except Exception:  # noqa: BLE001 - dense is best-effort
        logger.debug("dense scoring failed; treating dense ranks as ties", exc_info=True)
        return [0.0] * n


def scores_to_ranks(scores: Sequence[float], *, higher_is_better: bool = True) -> list[int]:
    """Convert scores to 1-based ranks (ties broken by stable original index)."""
    n = len(scores)
    if n == 0:
        return []
    order = sorted(
        range(n),
        key=lambda i: (-scores[i] if higher_is_better else scores[i], i),
    )
    ranks = [0] * n
    for rank_pos, idx in enumerate(order, start=1):
        ranks[idx] = rank_pos
    return ranks


def reciprocal_rank_fusion(
    rank_lists: Sequence[Sequence[int]],
    *,
    k: int = 60,
) -> list[float]:
    """Fuse multiple 1-based rank lists with Reciprocal Rank Fusion.

    ``score(d) = sum_r 1 / (k + rank_r(d))``. Rank-based and score-scale-free.
    Missing / zero ranks are ignored for that list.
    """
    if not rank_lists:
        return []
    n = len(rank_lists[0])
    fused = [0.0] * n
    for ranks in rank_lists:
        if len(ranks) != n:
            raise ValueError("all rank lists must have equal length")
        for i, rank in enumerate(ranks):
            if rank is None or int(rank) <= 0:
                continue
            fused[i] += 1.0 / (float(k) + float(rank))
    return fused


def lightweight_cross_encoder_scores(
    query_text: str,
    candidate_texts: Sequence[str],
) -> list[float]:
    """Interaction-style rerank scores without a heavy model dependency.

    Combines:
    - token Jaccard overlap (query ↔ candidate)
    - BM25 of candidates against the query (local corpus)
    - length-normalised shared operator/feature count

    Optionally upgraded by :func:`optional_transformer_cross_encoder_scores`
    when ``sentence_transformers.CrossEncoder`` is installed; that path is
    lazy and never runs during default tests.
    """
    n = len(candidate_texts)
    if n == 0:
        return []

    # Prefer a real CrossEncoder only when already importable -- never force
    # a HuggingFace download in --mock / CI. Same lazy pattern as MiniLM.
    ce_scores = optional_transformer_cross_encoder_scores(query_text, candidate_texts)
    if ce_scores is not None:
        return ce_scores

    q_tokens = tokenize_dsl(query_text)
    q_set = set(q_tokens)
    docs = [tokenize_dsl(t) for t in candidate_texts]
    bm25 = bm25_scores(q_tokens, docs)
    scores: list[float] = []
    for i, doc_toks in enumerate(docs):
        d_set = set(doc_toks)
        if not q_set and not d_set:
            jaccard = 0.0
        else:
            union = q_set | d_set
            jaccard = (len(q_set & d_set) / len(union)) if union else 0.0
        shared = float(len(q_set & d_set))
        # Weighted blend; BM25 can dominate on longer templates.
        scores.append(0.45 * jaccard + 0.40 * float(bm25[i]) + 0.15 * shared)
    return scores


def optional_transformer_cross_encoder_scores(
    query_text: str,
    candidate_texts: Sequence[str],
    *,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[float] | None:
    """Try a real CrossEncoder; return None if unavailable or on any error.

    Security / ops notes:
    - Checkpoint load is opt-in via import success only (no forced download
      in tests). Operators who install ``sentence-transformers`` may trigger
      a one-time HuggingFace fetch on first use -- same contract as
      :class:`FormulaEmbedder`'s MiniLM path.
    - Model name is a trusted local constant, never taken from remote input.
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
    except ImportError:
        return None

    try:
        # Lazy singleton on the function object avoids reload thrash.
        model = getattr(optional_transformer_cross_encoder_scores, "_model", None)
        loaded_name = getattr(
            optional_transformer_cross_encoder_scores, "_model_name", None
        )
        if model is None or loaded_name != model_name:
            model = CrossEncoder(model_name)
            optional_transformer_cross_encoder_scores._model = model  # type: ignore[attr-defined]
            optional_transformer_cross_encoder_scores._model_name = model_name  # type: ignore[attr-defined]
        pairs = [(query_text, text) for text in candidate_texts]
        raw = model.predict(pairs)
        return [float(x) for x in raw]
    except Exception:  # noqa: BLE001 - optional path must never break retrieval
        logger.debug("CrossEncoder rerank unavailable; using lexical rerank", exc_info=True)
        return None


def rerank_indices(
    query_text: str,
    candidate_texts: Sequence[str],
    candidate_indices: Sequence[int],
    *,
    top_k: int,
) -> list[int]:
    """Rerank *candidate_indices* by cross-encoder-style scores; return top_k."""
    if not candidate_indices:
        return []
    texts = [candidate_texts[i] for i in candidate_indices]
    scores = lightweight_cross_encoder_scores(query_text, texts)
    order = sorted(range(len(candidate_indices)), key=lambda j: (-scores[j], j))
    return [candidate_indices[j] for j in order[:top_k]]


# ---------------------------------------------------------------------------
# Heuristic scorers (legacy)
# ---------------------------------------------------------------------------


def _score_success_pattern(
    pattern: SuccessPattern,
    domain_saturation: dict[str, float],
    saturated_threshold: float = 0.7,
) -> float:
    """Score a success pattern for relevance given current library state.

    Patterns in saturated domains score lower; high success-rate patterns
    with many occurrences score higher.
    """
    base_score = 1.0

    # Success rate bonus
    rate_bonus = {"High": 2.0, "Medium": 1.0, "Low": 0.5}
    base_score *= rate_bonus.get(pattern.success_rate, 1.0)

    # Occurrence count bonus (log scale to avoid runaway)
    if pattern.occurrence_count > 0:
        base_score *= 1.0 + math.log1p(pattern.occurrence_count)

    # Domain saturation penalty
    saturation = domain_saturation.get(pattern.name, 0.0)
    if saturation >= saturated_threshold:
        base_score *= 0.2  # Heavily penalize saturated domains
    elif saturation >= 0.5:
        base_score *= 0.6

    return base_score


def _score_forbidden_direction(
    direction: ForbiddenDirection,
    recent_rejection_reasons: list[str],
) -> float:
    """Score a forbidden direction for relevance.

    Directions matching recent rejection reasons score higher (more
    important to communicate to the LLM).
    """
    base_score = 1.0

    # Higher correlation = more important to avoid
    base_score *= 1.0 + direction.typical_correlation

    # Occurrence count: frequently encountered = important warning
    if direction.occurrence_count > 0:
        base_score *= 1.0 + math.log1p(direction.occurrence_count)

    # Boost if matching recent rejections
    direction_lower = direction.name.lower()
    for reason in recent_rejection_reasons:
        if any(
            word in reason.lower()
            for word in direction_lower.split()
            if len(word) > 3
        ):
            base_score *= 1.5
            break

    return base_score


# ---------------------------------------------------------------------------
# Query construction + hybrid selection
# ---------------------------------------------------------------------------


def build_retrieval_query(
    *,
    domain_saturation: dict[str, float] | None = None,
    library_state: dict[str, Any] | None = None,
    recent_admissions: Sequence[dict[str, Any]] | None = None,
    recent_rejection_reasons: Sequence[str] | None = None,
) -> str:
    """Build a lexical/semantic query string from the current mining context."""
    parts: list[str] = []
    sat = domain_saturation or {}
    # Prefer unsaturated domains as exploration targets.
    unsaturated = [name for name, value in sat.items() if float(value) < 0.5]
    saturated = [name for name, value in sat.items() if float(value) >= 0.5]
    if unsaturated:
        parts.append("explore " + " ".join(unsaturated))
    if saturated:
        parts.append("avoid saturated " + " ".join(saturated))

    for admission in list(recent_admissions or [])[-5:]:
        formula = str(admission.get("formula", "") or "")
        if formula:
            parts.append(formula)

    lib = library_state or {}
    for key in ("focus_formula", "query_formula", "seed_formula"):
        val = lib.get(key)
        if val:
            parts.append(str(val))

    for reason in list(recent_rejection_reasons or [])[-5:]:
        if reason:
            parts.append(str(reason))

    return " ".join(parts).strip()


def fuse_hybrid_ranks(
    *,
    heuristic_scores: Sequence[float],
    bm25: Sequence[float] | None,
    dense: Sequence[float] | None,
    config: HybridRetrievalConfig,
) -> tuple[list[float], dict[str, Any]]:
    """Combine available rank signals with RRF; return fused scores + diagnostics."""
    n = len(heuristic_scores)
    rank_lists: list[list[int]] = []
    used: list[str] = []

    if config.enable_heuristic and n:
        rank_lists.append(scores_to_ranks(heuristic_scores))
        used.append("heuristic")
    if config.enable_bm25 and bm25 is not None and len(bm25) == n and any(
        s != 0 for s in bm25
    ):
        rank_lists.append(scores_to_ranks(bm25))
        used.append("bm25")
    if config.enable_dense and dense is not None and len(dense) == n and any(
        s != 0 for s in dense
    ):
        rank_lists.append(scores_to_ranks(dense))
        used.append("dense")

    if not rank_lists:
        # Degenerate: preserve input order.
        return list(heuristic_scores), {"signals_used": [], "rrf_k": config.rrf_k}

    fused = reciprocal_rank_fusion(rank_lists, k=config.rrf_k)
    return fused, {"signals_used": used, "rrf_k": config.rrf_k, "n_candidates": n}


def _select_relevant_success(
    patterns: list[SuccessPattern],
    domain_saturation: dict[str, float],
    max_patterns: int = 8,
    *,
    query_text: str = "",
    embedder: Any | None = None,
    hybrid_config: HybridRetrievalConfig | None = None,
) -> tuple[list[SuccessPattern], dict[str, Any]]:
    """Select the most relevant success patterns for the current context."""
    diagnostics: dict[str, Any] = {
        "mode": "heuristic",
        "signals_used": ["heuristic"],
        "reranked": False,
    }
    if not patterns:
        return [], diagnostics

    cfg = hybrid_config or DEFAULT_HYBRID_CONFIG
    heuristic = [_score_success_pattern(pat, domain_saturation) for pat in patterns]

    if not cfg.enabled:
        scored = sorted(
            zip(patterns, heuristic, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        return [pat for pat, _ in scored[:max_patterns]], diagnostics

    docs_text = [_pattern_document_text(p) for p in patterns]
    docs_tokens = [tokenize_dsl(t) for t in docs_text]
    q_tokens = tokenize_dsl(query_text)

    bm25 = (
        bm25_scores(q_tokens, docs_tokens, k1=cfg.bm25_k1, b=cfg.bm25_b)
        if cfg.enable_bm25
        else None
    )
    dense = (
        dense_cosine_scores(query_text, docs_text, embedder)
        if cfg.enable_dense
        else None
    )

    fused, fuse_diag = fuse_hybrid_ranks(
        heuristic_scores=heuristic,
        bm25=bm25,
        dense=dense,
        config=cfg,
    )
    diagnostics.update(fuse_diag)
    diagnostics["mode"] = "hybrid_rrf"

    order = sorted(range(len(patterns)), key=lambda i: (-fused[i], i))

    if cfg.enable_rerank and query_text:
        pool = order[: max(cfg.rerank_pool_size, max_patterns)]
        order = rerank_indices(
            query_text,
            docs_text,
            pool,
            top_k=max_patterns,
        )
        diagnostics["reranked"] = True
        diagnostics["rerank"] = "cross_encoder_style"
    else:
        order = order[:max_patterns]

    selected = [patterns[i] for i in order]
    diagnostics["selected_names"] = [p.name for p in selected]
    return selected, diagnostics


def _select_relevant_forbidden(
    directions: list[ForbiddenDirection],
    recent_rejections: list[dict],
    max_directions: int = 10,
    *,
    query_text: str = "",
    embedder: Any | None = None,
    hybrid_config: HybridRetrievalConfig | None = None,
) -> tuple[list[ForbiddenDirection], dict[str, Any]]:
    """Select the most relevant forbidden directions for the current context."""
    diagnostics: dict[str, Any] = {
        "mode": "heuristic",
        "signals_used": ["heuristic"],
        "reranked": False,
    }
    if not directions:
        return [], diagnostics

    recent_reasons = [r.get("reason", "") for r in recent_rejections]
    cfg = hybrid_config or DEFAULT_HYBRID_CONFIG
    heuristic = [_score_forbidden_direction(d, recent_reasons) for d in directions]

    if not cfg.enabled:
        scored = sorted(
            zip(directions, heuristic, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        return [d for d, _ in scored[:max_directions]], diagnostics

    docs_text = [_forbidden_document_text(d) for d in directions]
    docs_tokens = [tokenize_dsl(t) for t in docs_text]
    # Blend explicit query with recent rejection reasons for lexical match.
    forbid_query = " ".join(
        [query_text] + [str(r) for r in recent_reasons if r]
    ).strip()
    q_tokens = tokenize_dsl(forbid_query)

    bm25 = (
        bm25_scores(q_tokens, docs_tokens, k1=cfg.bm25_k1, b=cfg.bm25_b)
        if cfg.enable_bm25
        else None
    )
    dense = (
        dense_cosine_scores(forbid_query, docs_text, embedder)
        if cfg.enable_dense
        else None
    )

    fused, fuse_diag = fuse_hybrid_ranks(
        heuristic_scores=heuristic,
        bm25=bm25,
        dense=dense,
        config=cfg,
    )
    diagnostics.update(fuse_diag)
    diagnostics["mode"] = "hybrid_rrf"

    order = sorted(range(len(directions)), key=lambda i: (-fused[i], i))

    if cfg.enable_rerank and forbid_query:
        pool = order[: max(cfg.rerank_pool_size, max_directions)]
        order = rerank_indices(
            forbid_query,
            docs_text,
            pool,
            top_k=max_directions,
        )
        diagnostics["reranked"] = True
        diagnostics["rerank"] = "cross_encoder_style"
    else:
        order = order[:max_directions]

    selected = [directions[i] for i in order]
    diagnostics["selected_names"] = [d.name for d in selected]
    return selected, diagnostics


def _format_library_state(state: MiningState) -> dict[str, Any]:
    """Format mining state as structured context for LLM prompt."""
    # Identify saturated domains
    saturated = {
        domain: sat
        for domain, sat in state.domain_saturation.items()
        if sat >= 0.5
    }

    # Recent admission rate trend
    recent_logs = state.admission_log[-5:] if state.admission_log else []
    avg_rate = 0.0
    if recent_logs:
        avg_rate = sum(log.get("admission_rate", 0) for log in recent_logs) / len(recent_logs)

    return {
        "library_size": state.library_size,
        "recent_admission_rate": round(avg_rate, 3),
        "saturated_domains": saturated,
        "recent_admissions_count": len(state.recent_admissions),
        "recent_rejections_count": len(state.recent_rejections),
    }


def _format_for_prompt(
    success_patterns: list[SuccessPattern],
    forbidden_directions: list[ForbiddenDirection],
    insights: list[StrategicInsight],
    library_state: dict[str, Any],
) -> str:
    """Format the memory signal as structured text for LLM injection.

    Produces a human-readable prompt section that can be inserted into
    the factor generation prompt to guide the LLM.
    """
    sections = []

    # Library state
    sections.append("=== CURRENT LIBRARY STATE ===")
    sections.append(f"Library size: {library_state['library_size']} factors")
    sections.append(f"Recent admission rate: {library_state['recent_admission_rate']:.1%}")
    if library_state.get("saturated_domains"):
        sections.append("Saturated domains (avoid):")
        for domain, sat in library_state["saturated_domains"].items():
            sections.append(f"  - {domain}: {sat:.0%} saturated")
    sections.append("")

    # Recommended directions
    if success_patterns:
        sections.append("=== RECOMMENDED DIRECTIONS (P_succ) ===")
        for i, pat in enumerate(success_patterns, 1):
            sections.append(f"{i}. {pat.name} [{pat.success_rate}]")
            sections.append(f"   {pat.description}")
            sections.append(f"   Template: {pat.template}")
            if pat.example_factors:
                sections.append(f"   Examples: {', '.join(pat.example_factors[:3])}")
        sections.append("")

    # Forbidden directions
    if forbidden_directions:
        sections.append("=== FORBIDDEN DIRECTIONS (P_fail) ===")
        sections.append("DO NOT generate factors using these patterns:")
        for i, fd in enumerate(forbidden_directions, 1):
            sections.append(f"{i}. {fd.name} (rho > {fd.typical_correlation:.2f})")
            sections.append(f"   Reason: {fd.reason}")
            if fd.correlated_factors:
                sections.append(f"   Correlated with: {', '.join(fd.correlated_factors[:3])}")
        sections.append("")

    # Strategic insights
    if insights:
        sections.append("=== STRATEGIC INSIGHTS ===")
        for insight in insights:
            sections.append(f"- {insight.insight}")
            sections.append(f"  Evidence: {insight.evidence}")
        sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Public API: Memory Retrieval
# ---------------------------------------------------------------------------


def retrieve_memory(
    memory: ExperienceMemory,
    library_state: dict[str, Any] | None = None,
    max_success: int = 8,
    max_forbidden: int = 10,
    max_insights: int = 10,
    *,
    embedder: Any | None = None,
    hybrid_config: HybridRetrievalConfig | None = None,
) -> dict[str, Any]:
    """Memory Retrieval operator R(M, L).

    Performs context-dependent retrieval matching against the current
    library state, returning a memory signal m suitable for LLM prompt
    injection.

    Parameters
    ----------
    memory : ExperienceMemory
        The experience memory to retrieve from.
    library_state : dict, optional
        Current library diagnostics. If None, uses the state from memory.
        Expected keys: library_size, domain_saturation, etc.
    max_success : int
        Maximum number of success patterns to include.
    max_forbidden : int
        Maximum number of forbidden directions to include.
    max_insights : int
        Maximum number of insights to include.
    embedder : optional
        :class:`~factorminer.memory.embeddings.FormulaEmbedder` (or compatible)
        used for the dense rank list in hybrid fusion. Ignored when hybrid
        dense scoring is disabled.
    hybrid_config : HybridRetrievalConfig, optional
        Fusion / rerank controls. Defaults to hybrid RRF enabled.

    Returns
    -------
    dict
        Memory signal m with keys:
        - recommended_directions: list of success pattern dicts
        - forbidden_directions: list of forbidden direction dicts
        - insights: list of insight dicts
        - library_state: dict of library state info
        - prompt_text: str - formatted text for LLM prompt injection
        - retrieval_diagnostics: hybrid fusion metadata (signals, rerank)
    """
    cfg = hybrid_config if hybrid_config is not None else DEFAULT_HYBRID_CONFIG

    # Use provided library state or fall back to memory's state
    if library_state is not None:
        # Update memory state with external library info
        state = MiningState(
            library_size=library_state.get("library_size", memory.state.library_size),
            recent_admissions=memory.state.recent_admissions,
            recent_rejections=memory.state.recent_rejections,
            domain_saturation=library_state.get(
                "domain_saturation", memory.state.domain_saturation
            ),
            admission_log=memory.state.admission_log,
        )
    else:
        state = memory.state
        library_state = {}

    query_text = build_retrieval_query(
        domain_saturation=state.domain_saturation,
        library_state=library_state or {},
        recent_admissions=state.recent_admissions,
        recent_rejection_reasons=[
            r.get("reason", "") for r in state.recent_rejections
        ],
    )

    # Select relevant patterns (hybrid RRF when enabled)
    relevant_success, success_diag = _select_relevant_success(
        memory.success_patterns,
        state.domain_saturation,
        max_success,
        query_text=query_text,
        embedder=embedder,
        hybrid_config=cfg,
    )
    relevant_forbidden, forbid_diag = _select_relevant_forbidden(
        memory.forbidden_directions,
        state.recent_rejections,
        max_forbidden,
        query_text=query_text,
        embedder=embedder,
        hybrid_config=cfg,
    )

    # Select most recent insights (up to limit)
    sorted_insights = sorted(
        memory.insights, key=lambda i: i.batch_source, reverse=True
    )
    relevant_insights = sorted_insights[:max_insights]

    # Format library state
    lib_state_info = _format_library_state(state)

    # Format as prompt text
    prompt_text = _format_for_prompt(
        relevant_success, relevant_forbidden, relevant_insights, lib_state_info
    )

    return {
        "recommended_directions": [p.to_dict() for p in relevant_success],
        "forbidden_directions": [f.to_dict() for f in relevant_forbidden],
        "insights": [i.to_dict() for i in relevant_insights],
        "library_state": lib_state_info,
        "prompt_text": prompt_text,
        "retrieval_diagnostics": {
            "query_text": query_text,
            "hybrid": cfg.to_dict(),
            "success": success_diag,
            "forbidden": forbid_diag,
        },
    }


def retrieval_quality_smoke(
    *,
    embedder: Any | None = None,
    hybrid_config: HybridRetrievalConfig | None = None,
) -> dict[str, Any]:
    """Synthetic labeled-set check that hybrid ranking prefers success over fail.

    Builds a tiny corpus where one success pattern is lexically/semantically
    aligned with the query and one historically-forbidden pattern is not, then
    verifies the fused ranking puts the successful pattern first. Used as a
    retrieval-quality smoke metric (the repo previously had none).
    """
    cfg = hybrid_config or HybridRetrievalConfig(enabled=True, enable_rerank=False)

    success_good = SuccessPattern(
        name="TsRank Momentum Close",
        description="Time-series rank of close-price momentum",
        template="TsRank($close, 20)",
        success_rate="High",
        example_factors=["TsRank($close, 20)", "TsRank(Delta($close, 5), 20)"],
        occurrence_count=12,
        confidence=0.9,
    )
    success_weak = SuccessPattern(
        name="Volume Spike Raw",
        description="Unnormalized volume burst without ranking",
        template="Div($volume, Mean($volume, 60))",
        success_rate="Low",
        example_factors=["Div($volume, Mean($volume, 60))"],
        occurrence_count=2,
        confidence=0.3,
    )
    forbidden = ForbiddenDirection(
        name="Raw Close Level",
        description="Absolute price level without cross-sectional rank",
        correlated_factors=["$close"],
        typical_correlation=0.95,
        reason="highly correlated with existing price-level factors",
        occurrence_count=20,
    )

    memory = ExperienceMemory()
    memory.success_patterns = [success_weak, success_good]
    memory.forbidden_directions = [forbidden]
    memory.state.domain_saturation = {
        "TsRank Momentum Close": 0.1,
        "Volume Spike Raw": 0.2,
    }
    memory.state.recent_admissions = [
        {"factor_id": "seed", "formula": "TsRank(Delta($close, 1), 10)"},
    ]

    library_state = {
        "library_size": 3,
        "domain_saturation": memory.state.domain_saturation,
        "query_formula": "TsRank($close, 20)",
    }

    hybrid_signal = retrieve_memory(
        memory,
        library_state=library_state,
        max_success=2,
        max_forbidden=1,
        embedder=embedder,
        hybrid_config=cfg,
    )
    heuristic_only = retrieve_memory(
        memory,
        library_state=library_state,
        max_success=2,
        max_forbidden=1,
        embedder=None,
        hybrid_config=HybridRetrievalConfig(enabled=False),
    )

    hybrid_names = [p["name"] for p in hybrid_signal["recommended_directions"]]
    heuristic_names = [p["name"] for p in heuristic_only["recommended_directions"]]

    # Quality criterion: historically-successful TsRank pattern ranks above
    # the weak volume pattern under hybrid fusion on this labeled set.
    passed = bool(hybrid_names) and hybrid_names[0] == success_good.name

    return {
        "passed": passed,
        "hybrid_ranking": hybrid_names,
        "heuristic_ranking": heuristic_names,
        "hybrid_diagnostics": hybrid_signal.get("retrieval_diagnostics"),
        "criterion": (
            "fused ranking places historically-successful TsRank pattern "
            "above historically-weak volume pattern on synthetic labeled set"
        ),
    }
