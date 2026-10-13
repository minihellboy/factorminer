"""Exact, cached Spearman dependence for library admission and replacement.

:class:`DependenceIndex` computes the same mean absolute Spearman correlation
as :class:`~factorminer.domain.dependence.SpearmanDependenceMetric`, bit for
bit, with two savings:

* each signal is ranked once per period (its *prepared* form) instead of once
  per pair; a pair re-ranks only the periods where one signal's own NaN mask
  differs from the pair's joint mask, which is exactly where joint ranking
  changes the result;
* pair results are cached, so the repeated candidate-vs-library comparisons
  made by admission, replacement, diagnostics, and the library correlation
  matrix are computed once.

Arrays are identified by object identity and a content fingerprint. A changed
array invalidates its prepared ranks and pair results without changing caller
write permissions. A read-only array is fingerprinted once and then trusted
not to change (evaluation hands out signals as read-only views), so comparing
one candidate against a large library no longer re-hashes every library
member. Entries are dropped when arrays are garbage collected. Hashing,
ranking, and pair arithmetic run outside the index lock, so parallel
candidate workers do not serialize on it.
"""

from __future__ import annotations

import hashlib
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import rankdata  # type: ignore[import-untyped]

from factorminer.domain.dependence import (
    DependenceMetric,
    SpearmanDependenceMetric,
    build_dependence_metric,
)

_CHUNK = 256  # Must match SpearmanDependenceMetric's period chunking.


@dataclass
class _Prepared:
    valid: np.ndarray  # (M, T) bool: own non-NaN mask
    ranks: np.ndarray  # (M, T) float64: own-mask average ranks, NaN elsewhere
    nbytes: int


class DependenceIndex:
    """Prepared-rank and pair-result caches for exact Spearman dependence."""

    def __init__(self, *, max_prepared_bytes: int | None = 512 * 1024 * 1024) -> None:
        if max_prepared_bytes is not None and max_prepared_bytes < 0:
            raise ValueError("max_prepared_bytes must be non-negative")
        self.max_prepared_bytes = max_prepared_bytes
        self._prepared: OrderedDict[int, _Prepared] = OrderedDict()
        self._refs: dict[int, weakref.ref[np.ndarray]] = {}
        self._fingerprints: dict[int, bytes] = {}
        self._pairs: dict[tuple[int, int], float] = {}
        self._pairs_by_id: dict[int, set[tuple[int, int]]] = {}
        self._trusted: set[int] = set()  # read-only arrays already fingerprinted
        self._lock = threading.RLock()
        self.prepared_bytes = 0
        self.pair_hits = 0
        self.pair_misses = 0
        self.prepare_hits = 0
        self.prepare_misses = 0
        self.reranked_periods = 0

    # ------------------------------------------------------------------
    # Identity tracking
    # ------------------------------------------------------------------

    @staticmethod
    def _fingerprint(signals: np.ndarray) -> bytes:
        contiguous = np.ascontiguousarray(signals)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(str(signals.dtype).encode())
        digest.update(str(signals.shape).encode())
        digest.update(contiguous.view(np.uint8))
        return digest.digest()

    def _track(self, signals: np.ndarray) -> int:
        """Register *signals* and return its identity; hashes outside the lock."""
        ident = id(signals)
        read_only = _immutable(signals)
        if read_only:
            with self._lock:
                ref = self._refs.get(ident)
                if ident in self._trusted and ref is not None and ref() is signals:
                    return ident
        fingerprint = self._fingerprint(signals)
        with self._lock:
            ref = self._refs.get(ident)
            if ref is not None and ref() is signals and self._fingerprints[ident] == fingerprint:
                if read_only:
                    self._trusted.add(ident)
                return ident
            self._forget(ident)
            self._refs[ident] = weakref.ref(signals, self._collector(ident))
            self._fingerprints[ident] = fingerprint
            if read_only:
                self._trusted.add(ident)
        return ident

    def _collector(self, ident: int) -> Any:
        def on_collect(_ref: Any) -> None:
            with self._lock:
                self._forget(ident)

        return on_collect

    def _forget(self, ident: int) -> None:
        self._refs.pop(ident, None)
        self._trusted.discard(ident)
        self._fingerprints.pop(ident, None)
        prepared = self._prepared.pop(ident, None)
        if prepared is not None:
            self.prepared_bytes -= prepared.nbytes
        for pair in self._pairs_by_id.pop(ident, ()):
            self._pairs.pop(pair, None)
            other = pair[1] if pair[0] == ident else pair[0]
            peers = self._pairs_by_id.get(other)
            if peers is not None:
                peers.discard(pair)

    # ------------------------------------------------------------------
    # Prepared signals
    # ------------------------------------------------------------------

    def _prepare(self, ident: int, signals: np.ndarray) -> _Prepared:
        """Own-mask ranks of *signals*, cached; the ranking runs outside the lock."""
        with self._lock:
            prepared = self._prepared.get(ident)
            if prepared is not None:
                self._prepared.move_to_end(ident)
                self.prepare_hits += 1
                return prepared
            self.prepare_misses += 1
        valid = ~np.isnan(signals)
        ranks = rankdata(np.where(valid, signals, np.nan), axis=0, nan_policy="omit")
        prepared = _Prepared(valid=valid, ranks=ranks, nbytes=valid.nbytes + ranks.nbytes)
        with self._lock:
            if self._refs.get(ident) is None:
                return prepared  # forgotten meanwhile: use it, do not cache it
            existing = self._prepared.get(ident)
            if existing is not None:
                return existing
            self._prepared[ident] = prepared
            self.prepared_bytes += prepared.nbytes
            if self.max_prepared_bytes is not None:
                while self._prepared and self.prepared_bytes > self.max_prepared_bytes:
                    _, evicted = self._prepared.popitem(last=False)
                    self.prepared_bytes -= evicted.nbytes
        return prepared

    # ------------------------------------------------------------------
    # Pair dependence
    # ------------------------------------------------------------------

    def compute(self, signals_a: np.ndarray, signals_b: np.ndarray) -> float:
        """Mean absolute Spearman correlation, identical to the reference metric."""
        if signals_a.shape != signals_b.shape:
            raise ValueError(f"Signal shapes must match: {signals_a.shape} vs {signals_b.shape}")
        if signals_a.ndim != 2:
            raise ValueError("Spearman signals must be two-dimensional")
        ident_a = self._track(signals_a)
        ident_b = self._track(signals_b)
        pair = (ident_a, ident_b) if ident_a <= ident_b else (ident_b, ident_a)
        with self._lock:
            cached = self._pairs.get(pair)
            if cached is not None:
                self.pair_hits += 1
                return cached
            self.pair_misses += 1
        prepared_a = self._prepare(ident_a, signals_a)
        prepared_b = self._prepare(ident_b, signals_b)
        value, reranked = self._pair(signals_a, signals_b, prepared_a, prepared_b)
        with self._lock:
            self.reranked_periods += reranked
            # Cache only while both arrays are still the ones registered.
            ref_a, ref_b = self._refs.get(ident_a), self._refs.get(ident_b)
            if ref_a is not None and ref_a() is signals_a and ref_b is not None and ref_b() is signals_b:
                self._pairs[pair] = value
                self._pairs_by_id.setdefault(ident_a, set()).add(pair)
                self._pairs_by_id.setdefault(ident_b, set()).add(pair)
        return value

    def _pair(
        self,
        signals_a: np.ndarray,
        signals_b: np.ndarray,
        prepared_a: _Prepared,
        prepared_b: _Prepared,
    ) -> tuple[float, int]:
        correlation_sum = 0.0
        period_count = 0
        reranked = 0
        for start in range(0, signals_a.shape[1], _CHUNK):
            window = slice(start, start + _CHUNK)
            own_a = prepared_a.valid[:, window]
            own_b = prepared_b.valid[:, window]
            valid = own_a & own_b
            usable = valid.sum(axis=0) >= 3
            if not np.any(usable):
                continue
            mask = valid[:, usable]
            ranked_a, stale_a = self._joint_ranks(signals_a[:, window],
                                                  prepared_a.ranks[:, window], own_a, mask, usable)
            ranked_b, stale_b = self._joint_ranks(signals_b[:, window],
                                                  prepared_b.ranks[:, window], own_b, mask, usable)
            reranked += stale_a + stale_b
            # From here on this is SpearmanDependenceMetric.compute verbatim,
            # applied to identical arrays with identical memory layout.
            ranked_a -= np.nanmean(ranked_a, axis=0)
            ranked_b -= np.nanmean(ranked_b, axis=0)
            np.nan_to_num(ranked_a, copy=False, nan=0.0)
            np.nan_to_num(ranked_b, copy=False, nan=0.0)
            numerator = np.sum(ranked_a * ranked_b, axis=0)
            denominator = np.sqrt(
                np.sum(ranked_a**2, axis=0) * np.sum(ranked_b**2, axis=0)
            )
            correlations = np.divide(
                np.abs(numerator),
                denominator,
                out=np.zeros_like(numerator),
                where=denominator >= 1e-12,
            )
            correlation_sum += float(np.sum(correlations))
            period_count += int(correlations.size)
        return (correlation_sum / period_count if period_count else 0.0), reranked

    @staticmethod
    def _joint_ranks(
        signals: np.ndarray,
        own_ranks: np.ndarray,
        own_valid: np.ndarray,
        mask: np.ndarray,
        usable: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Ranks on the joint mask, in ``rankdata``'s (Fortran) layout, and re-ranked periods."""
        ranks = np.asfortranarray(own_ranks[:, usable])
        stale = np.any(own_valid[:, usable] != mask, axis=0)
        if np.any(stale):
            columns = signals[:, usable][:, stale]
            ranks[:, stale] = rankdata(
                np.where(mask[:, stale], columns, np.nan), axis=0, nan_policy="omit"
            )
        return ranks, int(stale.sum())

    def stats(self) -> dict[str, int | None]:
        with self._lock:
            return {
                "pair_hits": self.pair_hits,
                "pair_misses": self.pair_misses,
                "prepare_hits": self.prepare_hits,
                "prepare_misses": self.prepare_misses,
                "reranked_periods": self.reranked_periods,
                "prepared_signals": len(self._prepared),
                "prepared_bytes": self.prepared_bytes,
                "max_prepared_bytes": self.max_prepared_bytes,
                "cached_pairs": len(self._pairs),
            }

    def __deepcopy__(self, memo: dict[int, Any]) -> DependenceIndex:
        return DependenceIndex(max_prepared_bytes=self.max_prepared_bytes)

    def __getstate__(self) -> dict[str, Any]:
        return {"max_prepared_bytes": self.max_prepared_bytes}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(max_prepared_bytes=state.get("max_prepared_bytes"))  # type: ignore[misc]


@dataclass(frozen=True)
class IndexedSpearmanMetric(SpearmanDependenceMetric):
    """Spearman dependence served by a :class:`DependenceIndex`.

    Same name, description, and values as the reference metric; only the cost
    of repeated comparisons changes.
    """

    index: DependenceIndex = field(default_factory=DependenceIndex, compare=False, repr=False)

    def compute(self, signals_a: np.ndarray, signals_b: np.ndarray) -> float:
        return self.index.compute(signals_a, signals_b)


def indexed_dependence_metric(name: str | None) -> DependenceMetric:
    """Build a dependence metric, using the exact index for Spearman."""
    metric = build_dependence_metric(name)
    if type(metric) is SpearmanDependenceMetric:
        return IndexedSpearmanMetric()
    return metric


def _immutable(array: np.ndarray) -> bool:
    """Whether neither *array* nor any array it views can be written."""
    owner: Any = array
    while isinstance(owner, np.ndarray):
        if owner.flags.writeable:
            return False
        owner = owner.base
    return True
