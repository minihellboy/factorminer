"""Semantic formula embeddings for factor similarity and deduplication.

Converts DSL formulas into natural language descriptions and encodes
them as dense vectors. Supports:
- sentence-transformers for high-quality embeddings (optional)
- FAISS for fast k-NN search (optional)
- TF-IDF fallback when sentence-transformers is unavailable
- Brute-force cosine fallback when FAISS is unavailable
"""

from __future__ import annotations

import re

import numpy as np

# Optional dependency flags -- resolved at runtime
_has_sentence_transformers = False
_has_faiss = False
_has_sklearn = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

    _has_sentence_transformers = True
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]

try:
    import faiss  # type: ignore[import-untyped]

    _has_faiss = True
except ImportError:
    faiss = None  # type: ignore[assignment]

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]

    _has_sklearn = True
except ImportError:
    TfidfVectorizer = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Operator name -> natural-language expansion table
# ---------------------------------------------------------------------------

_OPERATOR_EXPANSIONS: dict[str, str] = {
    # Arithmetic
    "Add": "addition",
    "Sub": "subtraction",
    "Mul": "multiplication",
    "Div": "division",
    "Neg": "negation",
    "Abs": "absolute value",
    "Log": "logarithm",
    "Sqrt": "square root",
    "Power": "power",
    "Sign": "sign",
    "Max": "maximum",
    "Min": "minimum",
    # Rolling / time-series
    "Mean": "rolling mean",
    "Median": "rolling median",
    "Std": "rolling standard deviation",
    "Var": "rolling variance",
    "Skew": "rolling skewness",
    "Kurt": "rolling kurtosis",
    "Sum": "rolling sum",
    "TsMax": "time-series maximum",
    "TsMin": "time-series minimum",
    "TsRank": "time-series rank",
    "TsArgMax": "time-series argmax",
    "TsArgMin": "time-series argmin",
    "Delta": "change over period",
    "Delay": "lagged value",
    "Return": "return over period",
    "Corr": "rolling correlation",
    "Cov": "rolling covariance",
    "TsLinRegSlope": "linear regression slope",
    "TsLinRegResid": "linear regression residual",
    "TsLinRegIntercept": "linear regression intercept",
    # Smoothing
    "EMA": "exponential moving average",
    "WMA": "weighted moving average",
    "SMA": "simple moving average",
    "DEMA": "double exponential moving average",
    # Cross-sectional
    "CsRank": "cross-sectional rank",
    "CsZScore": "cross-sectional z-score",
    "CsDemean": "cross-sectional demeaning",
    "CsScale": "cross-sectional scaling",
    # Logical / conditional
    "IfElse": "conditional selection",
    "Greater": "greater-than comparison",
    "Less": "less-than comparison",
    "Equal": "equality comparison",
    "And": "logical and",
    "Or": "logical or",
    "Not": "logical not",
}

# Feature name -> natural-language
_FEATURE_EXPANSIONS: dict[str, str] = {
    "$close": "close price",
    "$open": "open price",
    "$high": "high price",
    "$low": "low price",
    "$volume": "volume",
    "$amt": "turnover amount",
    "$vwap": "volume-weighted average price",
    "$returns": "returns",
}


class FormulaEmbedder:
    """Embed DSL formulas as dense vectors for similarity search.

    Parameters
    ----------
    model_name : str
        Name of a sentence-transformers model (used only when the
        library is installed).
    use_faiss : bool
        Whether to use FAISS for approximate nearest-neighbour search.
        Falls back to brute-force cosine similarity if unavailable.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_faiss: bool = True,
    ) -> None:
        self._model_name = model_name
        self._use_faiss = use_faiss and _has_faiss

        # Lazy-loaded model / vectoriser
        self._model: SentenceTransformer | None = None  # type: ignore[type-arg]
        self._tfidf: TfidfVectorizer | None = None  # type: ignore[type-arg]
        self._tfidf_dirty: bool = False  # whether TF-IDF needs refit

        # Cache: factor_id -> (embedding, text)
        self._cache: dict[str, tuple[np.ndarray, str]] = {}
        # Ordered list mirroring cache for FAISS index alignment
        self._ids: list[str] = []

        # FAISS index (rebuilt lazily)
        self._index: object | None = None
        self._index_dirty: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, factor_id: str, formula: str) -> np.ndarray:
        """Compute (or retrieve cached) embedding for a formula.

        Parameters
        ----------
        factor_id : str
            Unique identifier used for caching.
        formula : str
            DSL formula to embed.

        Returns
        -------
        ndarray
            Embedding vector (float32).
        """
        if factor_id in self._cache:
            return self._cache[factor_id][0]

        text = self._formula_to_text(formula)
        vec = self._encode(text)
        self._cache[factor_id] = (vec, text)
        self._ids.append(factor_id)
        self._index_dirty = True
        self._tfidf_dirty = True
        return vec

    def remove(self, factor_id: str) -> bool:
        """Remove a cached embedding by factor id."""
        if factor_id not in self._cache:
            return False

        self._cache.pop(factor_id, None)
        self._ids = [fid for fid in self._ids if fid != factor_id]
        self._index = None
        self._index_dirty = True
        self._tfidf_dirty = True
        return True

    def clear(self) -> None:
        """Clear all cached embeddings and search state."""
        self._cache.clear()
        self._ids.clear()
        self._index = None
        self._index_dirty = True
        self._tfidf = None
        self._tfidf_dirty = False

    @property
    def cache_size(self) -> int:
        """Return the number of cached factor embeddings."""
        return len(self._cache)

    def find_nearest(
        self,
        formula: str,
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find the *k* most similar cached formulas.

        Parameters
        ----------
        formula : str
            Query formula (does not need to be cached).
        k : int
            Number of neighbours to return.

        Returns
        -------
        list of (factor_id, similarity)
            Sorted by descending similarity.
        """
        if not self._cache:
            return []

        query_vec = self._encode(self._formula_to_text(formula))
        k = min(k, len(self._cache))

        if self._use_faiss and _has_faiss:
            return self._faiss_search(query_vec, k)
        return self._brute_force_search(query_vec, k)

    def is_semantic_duplicate(
        self,
        formula: str,
        threshold: float = 0.92,
    ) -> str | None:
        """Check if *formula* is a near-duplicate of a cached factor.

        Returns the factor_id of the most similar cached factor if the
        cosine similarity exceeds *threshold*, or ``None``.
        """
        results = self.find_nearest(formula, k=1)
        if results and results[0][1] >= threshold:
            return results[0][0]
        return None

    # ------------------------------------------------------------------
    # Formula -> text conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _formula_to_text(formula: str) -> str:
        """Convert a DSL formula into a natural-language description.

        Expands operator and feature names for better semantic matching.
        """
        text = formula

        # Expand operators (longest-first to avoid partial matches)
        for op in sorted(_OPERATOR_EXPANSIONS, key=len, reverse=True):
            text = text.replace(op, _OPERATOR_EXPANSIONS[op])

        # Expand features
        for feat in _FEATURE_EXPANSIONS:
            text = text.replace(feat, _FEATURE_EXPANSIONS[feat])

        # Clean up punctuation into spaces
        text = re.sub(r"[(),]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.lower()

    # ------------------------------------------------------------------
    # Encoding backends
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a unit-norm vector."""
        if _has_sentence_transformers:
            return self._encode_transformer(text)
        if _has_sklearn:
            return self._encode_tfidf(text)
        # Absolute fallback: hash-based bag of words
        return self._encode_hash(text)

    def _encode_transformer(self, text: str) -> np.ndarray:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        vec = self._model.encode(text, convert_to_numpy=True)
        vec = np.asarray(vec, dtype=np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _encode_tfidf(self, text: str) -> np.ndarray:
        """Encode using TF-IDF over all cached texts + query.

        Because TF-IDF vocabulary can change when new documents are
        added, we refit when dirty. This is cheap for the expected
        document counts (hundreds to low thousands).
        """
        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(max_features=512)
            self._tfidf_dirty = True

        # Collect all known texts + this one
        corpus = [t for _, t in self._cache.values()]
        query_idx = len(corpus)
        corpus.append(text)

        # Always refit because vocab may have grown
        matrix = self._tfidf.fit_transform(corpus)
        vec = np.asarray(matrix[query_idx].toarray(), dtype=np.float32).flatten()

        # Re-encode cached entries with updated vocab
        for i, fid in enumerate(self._ids):
            updated = np.asarray(matrix[i].toarray(), dtype=np.float32).flatten()
            norm = np.linalg.norm(updated)
            if norm > 0:
                updated /= norm
            self._cache[fid] = (updated, self._cache[fid][1])

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        self._tfidf_dirty = False
        self._index_dirty = True
        return vec

    @staticmethod
    def _encode_hash(text: str, dim: int = 128) -> np.ndarray:
        """Ultra-simple hash-based embedding fallback."""
        vec = np.zeros(dim, dtype=np.float32)
        for token in text.split():
            idx = hash(token) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ------------------------------------------------------------------
    # Search backends
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS ``IndexFlatIP`` from cached embeddings."""
        if not self._cache or not _has_faiss:
            return
        vecs = np.stack([self._cache[fid][0] for fid in self._ids])
        dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)  # type: ignore[union-attr]
        self._index_dirty = False

    def _faiss_search(
        self,
        query: np.ndarray,
        k: int,
    ) -> list[tuple[str, float]]:
        if self._index_dirty:
            self._rebuild_index()
        if self._index is None:
            return self._brute_force_search(query, k)

        distances, indices = self._index.search(  # type: ignore[union-attr]
            query.reshape(1, -1), k
        )
        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            results.append((self._ids[idx], float(dist)))
        return results

    def _brute_force_search(
        self,
        query: np.ndarray,
        k: int,
    ) -> list[tuple[str, float]]:
        sims: list[tuple[str, float]] = []
        for fid in self._ids:
            vec = self._cache[fid][0]
            sim = float(np.dot(query, vec))
            sims.append((fid, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]
