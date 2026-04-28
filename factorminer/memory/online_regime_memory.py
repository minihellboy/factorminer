"""Online regime-aware memory system for FactorMiner.

Addresses FactorMiner's core limitation: static, offline-only memory that
ignores regime changes.  This module provides:

- ``RegimeSpecificPattern`` / ``RegimeSpecificPatternStore``
  — per-regime success/failure pattern storage with IC-based scoring

- ``OnlineMemoryUpdater``
  — streaming memory update with exponential forgetting and regime-change hooks

- ``RegimeTransitionForecaster``
  — logistic-regression-based next-regime predictor for proactive memory prep

- ``OnlineRegimeMemory``
  — top-level orchestrator integrating all components

- ``MemoryForgetCurve``
  — snapshot tracker for visualising and analysing memory decay

All components are:
  * Thread-safe (``threading.RLock``)
  * Serialisable (``to_dict`` / ``from_dict`` + ``pickle`` compatible)
  * Streaming-fast (< 1 ms per ``update`` call with normal loads)
  * Pure Python + NumPy + scikit-learn (no additional dependencies)
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from factorminer.evaluation.regime import (
    RegimeState,
    StreamingRegimeConfig,
    StreamingRegimeDetector,
)
from factorminer.memory.evolution import (
    apply_confidence_decay,
    bump_pattern_confidence,
    penalise_pattern_confidence,
)
from factorminer.memory.memory_store import (
    ExperienceMemory,
    StrategicInsight,
)
from factorminer.memory.retrieval import retrieve_memory

logger = logging.getLogger(__name__)


def _regime_labels(regime: RegimeState) -> tuple[str, ...]:
    """Return deterministic searchable labels for a composite regime."""
    label = regime.label()
    return tuple(part for part in label.split("/") if part)


# ---------------------------------------------------------------------------
# MemorySignal — returned by OnlineRegimeMemory.retrieve()
# ---------------------------------------------------------------------------

@dataclass
class MemorySignal:
    """Structured memory signal for LLM prompt injection.

    Wraps the standard retrieval result with regime-specific additions.
    """
    recommended_directions: list[dict]
    forbidden_directions: list[dict]
    insights: list[dict]
    library_state: dict
    prompt_text: str
    # Regime-specific additions
    current_regime: RegimeState = field(default_factory=RegimeState)
    regime_patterns: list[dict] = field(default_factory=list)
    cross_regime_patterns: list[dict] = field(default_factory=list)
    forecasted_regime: RegimeState | None = None
    forecast_confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "recommended_directions": self.recommended_directions,
            "forbidden_directions": self.forbidden_directions,
            "insights": self.insights,
            "library_state": self.library_state,
            "prompt_text": self.prompt_text,
            "current_regime": self.current_regime.to_dict(),
            "regime_patterns": self.regime_patterns,
            "cross_regime_patterns": self.cross_regime_patterns,
            "forecasted_regime": self.forecasted_regime.to_dict()
                if self.forecasted_regime else None,
            "forecast_confidence": self.forecast_confidence,
        }


# ---------------------------------------------------------------------------
# RegimeSpecificPattern & RegimeSpecificPatternStore
# ---------------------------------------------------------------------------

@dataclass
class RegimeSpecificPattern:
    """A formula pattern with per-regime performance statistics.

    Attributes
    ----------
    formula_template : str
        DSL formula template (may contain ``{w}`` style placeholders).
    regime : RegimeState
        The regime context in which this pattern was discovered.
    ic_in_regime : float
        Mean IC when the current market regime matches ``self.regime``.
    ic_out_of_regime : float
        Mean IC when the current regime does not match.
    regime_specificity : float
        ``ic_in_regime / (|ic_out_of_regime| + 1e-8)``.  Values >> 1 indicate
        strong regime specialisation.
    discovery_date : datetime
        UTC timestamp of first observation.
    confidence : float
        Normalised confidence in [0, 1] based on sample count.  Decays via
        forgetting.
    n_observations : int
        Number of times this pattern has been observed.
    n_in_regime : int
        Observations when regime matched.
    """
    formula_template: str
    regime: RegimeState
    ic_in_regime: float = 0.0
    ic_out_of_regime: float = 0.0
    regime_specificity: float = 1.0
    discovery_date: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    confidence: float = 1.0
    n_observations: int = 1
    n_in_regime: int = 0

    def update_ic(self, ic: float, in_regime: bool) -> None:
        """Online update of IC statistics using an EW running mean."""
        self.n_observations += 1
        alpha = 2.0 / (min(self.n_observations, 50) + 1)
        if in_regime:
            self.n_in_regime += 1
            self.ic_in_regime = (1 - alpha) * self.ic_in_regime + alpha * ic
        else:
            self.ic_out_of_regime = (1 - alpha) * self.ic_out_of_regime + alpha * ic
        # Recompute specificity
        self.regime_specificity = abs(self.ic_in_regime) / (
            abs(self.ic_out_of_regime) + 1e-8
        )

    def to_dict(self) -> dict:
        return {
            "formula_template": self.formula_template,
            "regime": self.regime.to_dict(),
            "ic_in_regime": round(self.ic_in_regime, 6),
            "ic_out_of_regime": round(self.ic_out_of_regime, 6),
            "regime_specificity": round(self.regime_specificity, 4),
            "discovery_date": self.discovery_date.isoformat(),
            "confidence": round(self.confidence, 6),
            "n_observations": self.n_observations,
            "n_in_regime": self.n_in_regime,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RegimeSpecificPattern:
        discovery_date = datetime.fromisoformat(
            d.get("discovery_date", datetime.now(tz=timezone.utc).isoformat())
        )
        if discovery_date.tzinfo is None:
            discovery_date = discovery_date.replace(tzinfo=timezone.utc)
        return cls(
            formula_template=d["formula_template"],
            regime=RegimeState.from_dict(d["regime"]),
            ic_in_regime=d.get("ic_in_regime", 0.0),
            ic_out_of_regime=d.get("ic_out_of_regime", 0.0),
            regime_specificity=d.get("regime_specificity", 1.0),
            discovery_date=discovery_date,
            confidence=d.get("confidence", 1.0),
            n_observations=d.get("n_observations", 1),
            n_in_regime=d.get("n_in_regime", 0),
        )


class RegimeSpecificPatternStore:
    """Thread-safe store for regime-specific formula patterns.

    Patterns are keyed by ``(formula_template, regime_str)`` and indexed
    for fast retrieval by regime similarity.

    Parameters
    ----------
    max_patterns : int
        Maximum total patterns retained.  When full, lowest-confidence
        patterns are evicted.
    min_ic : float
        Minimum |IC| threshold; patterns consistently below this are pruned.
    cross_regime_specificity_threshold : float
        A pattern with ``regime_specificity < threshold`` is classified as
        a cross-regime (general) pattern.
    """

    def __init__(
        self,
        max_patterns: int = 500,
        min_ic: float = 0.02,
        cross_regime_specificity_threshold: float = 1.5,
    ) -> None:
        self.max_patterns = max_patterns
        self.min_ic = min_ic
        self.cross_regime_threshold = cross_regime_specificity_threshold
        self._lock = threading.RLock()
        # key: (formula_template, regime_str)
        self._patterns: dict[tuple[str, str], RegimeSpecificPattern] = {}

    # --- public API ---

    def add_pattern(
        self,
        formula: str,
        regime: RegimeState,
        ic: float,
    ) -> None:
        """Add or update a pattern observation.

        If the (formula, regime) pair already exists, the IC statistics
        are updated online.  Otherwise a new entry is created.

        Parameters
        ----------
        formula : str
        regime : RegimeState
            The regime active when this IC was measured.
        ic : float
            Observed IC (signed).
        """
        with self._lock:
            key = (formula, str(regime))
            if key in self._patterns:
                pat = self._patterns[key]
                pat.update_ic(ic, in_regime=True)
                pat.confidence = min(1.0, pat.confidence + 0.05)
            else:
                # Also update out-of-regime IC for all *existing* patterns
                # with a different regime tag
                for existing_key, existing_pat in self._patterns.items():
                    if existing_key[0] == formula and existing_key[1] != str(regime):
                        existing_pat.update_ic(ic, in_regime=False)

                # Create new pattern
                pat = RegimeSpecificPattern(
                    formula_template=formula,
                    regime=regime,
                    ic_in_regime=ic,
                    ic_out_of_regime=0.0,
                    confidence=1.0,
                    n_observations=1,
                    n_in_regime=1,
                )
                pat.regime_specificity = abs(ic) / (abs(0.0) + 1e-8)
                self._patterns[key] = pat

            # Evict if over capacity
            if len(self._patterns) > self.max_patterns:
                self._evict_weakest()

    def retrieve_for_regime(
        self,
        current_regime: RegimeState,
        top_k: int = 10,
        min_confidence: float = 0.1,
    ) -> list[RegimeSpecificPattern]:
        """Retrieve patterns most relevant to the current regime.

        Patterns are scored as:
            score = confidence * ic_in_regime * regime_similarity

        where ``regime_similarity`` is the Jaccard similarity between the
        pattern's tagged regime and ``current_regime``.

        Parameters
        ----------
        current_regime : RegimeState
        top_k : int
        min_confidence : float
            Minimum confidence to include.

        Returns
        -------
        list[RegimeSpecificPattern]
            Sorted by descending relevance score.
        """
        with self._lock:
            scored: list[tuple[float, RegimeSpecificPattern]] = []
            for pat in self._patterns.values():
                if pat.confidence < min_confidence:
                    continue
                sim = pat.regime.similarity(current_regime)
                score = pat.confidence * abs(pat.ic_in_regime) * (0.2 + 0.8 * sim)
                scored.append((score, pat))
            scored.sort(key=lambda x: -x[0])
            return [p for _, p in scored[:top_k]]

    def get_cross_regime_patterns(
        self,
        top_k: int = 10,
        min_confidence: float = 0.1,
    ) -> list[RegimeSpecificPattern]:
        """Return patterns that generalise well across regimes.

        A pattern qualifies as cross-regime if its ``regime_specificity``
        is below ``cross_regime_specificity_threshold`` *and* its absolute
        IC is meaningfully positive (>= ``min_ic``).

        Returns
        -------
        list[RegimeSpecificPattern]
        """
        with self._lock:
            cross: list[tuple[float, RegimeSpecificPattern]] = []
            for pat in self._patterns.values():
                if pat.confidence < min_confidence:
                    continue
                if pat.regime_specificity < self.cross_regime_threshold:
                    avg_ic = (abs(pat.ic_in_regime) + abs(pat.ic_out_of_regime)) / 2.0
                    if avg_ic >= self.min_ic:
                        cross.append((avg_ic * pat.confidence, pat))
            cross.sort(key=lambda x: -x[0])
            return [p for _, p in cross[:top_k]]

    def apply_decay(self, decay_factor: float) -> None:
        """Multiply all pattern confidences by ``decay_factor`` and prune weak ones."""
        with self._lock:
            to_delete = []
            for key, pat in self._patterns.items():
                pat.confidence = max(0.0, pat.confidence * decay_factor)
                if pat.confidence < 0.01 and pat.n_observations > 3:
                    to_delete.append(key)
            for key in to_delete:
                del self._patterns[key]

    def boost_regime_patterns(self, regime: RegimeState, boost: float = 0.1) -> None:
        """Increase confidence of patterns tagged for ``regime``."""
        with self._lock:
            for pat in self._patterns.values():
                if pat.regime == regime:
                    pat.confidence = min(1.0, pat.confidence + boost)

    def penalise_regime_patterns(self, regime: RegimeState, penalty: float = 0.3) -> None:
        """Decrease confidence of patterns tagged for ``regime``."""
        with self._lock:
            for pat in self._patterns.values():
                if pat.regime == regime:
                    pat.confidence = max(0.0, pat.confidence - penalty)

    def get_stats(self) -> dict:
        """Return aggregate statistics."""
        with self._lock:
            n = len(self._patterns)
            if n == 0:
                return {
                    "total_patterns": 0,
                    "avg_confidence": 0.0,
                    "avg_ic_in_regime": 0.0,
                    "cross_regime_count": 0,
                }
            confs = [p.confidence for p in self._patterns.values()]
            ics = [p.ic_in_regime for p in self._patterns.values()]
            cross = len(self.get_cross_regime_patterns(top_k=n))
            return {
                "total_patterns": n,
                "avg_confidence": float(np.mean(confs)),
                "avg_ic_in_regime": float(np.mean(np.abs(ics))),
                "cross_regime_count": cross,
            }

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "max_patterns": self.max_patterns,
                "min_ic": self.min_ic,
                "cross_regime_threshold": self.cross_regime_threshold,
                "patterns": [p.to_dict() for p in self._patterns.values()],
            }

    @classmethod
    def from_dict(cls, d: dict) -> RegimeSpecificPatternStore:
        store = cls(
            max_patterns=d.get("max_patterns", 500),
            min_ic=d.get("min_ic", 0.02),
            cross_regime_specificity_threshold=d.get("cross_regime_threshold", 1.5),
        )
        for pd in d.get("patterns", []):
            pat = RegimeSpecificPattern.from_dict(pd)
            key = (pat.formula_template, str(pat.regime))
            store._patterns[key] = pat
        return store

    # --- internals ---

    def _evict_weakest(self) -> None:
        """Remove the single weakest (lowest confidence * ic) pattern."""
        if not self._patterns:
            return
        worst_key = min(
            self._patterns,
            key=lambda k: (
                self._patterns[k].confidence
                * (abs(self._patterns[k].ic_in_regime) + 1e-8)
            ),
        )
        del self._patterns[worst_key]


# ---------------------------------------------------------------------------
# OnlineMemoryUpdater
# ---------------------------------------------------------------------------

class OnlineMemoryUpdater:
    """Streaming experience-memory updater with exponential forgetting.

    Integrates with the base ``ExperienceMemory`` and the
    ``RegimeSpecificPatternStore`` to maintain an up-to-date picture of
    what works in the current market regime.

    Thread safety
    -------------
    All mutating operations acquire ``self._lock`` (``threading.RLock``).
    The ``base_memory`` is replaced atomically so readers always see a
    consistent snapshot.

    Parameters
    ----------
    base_memory : ExperienceMemory
        The underlying experience memory (will be mutated in place via
        evolution helpers).
    forgetting_rate : float
        Per-iteration confidence retention factor applied to pattern confidence.
    regime_sensitivity : float
        Weight given to regime-specific IC boosts vs generic boosts.
        0 = ignore regime, 1 = fully regime-sensitive.
    min_confidence : float
        Patterns with normalised confidence below this are pruned during
        forgetting.
    regime_boost : float
        Confidence increment when a pattern's regime matches the current one.
    regime_penalty : float
        Confidence decrement when the regime changes away from a pattern's home.
    """

    def __init__(
        self,
        base_memory: ExperienceMemory,
        forgetting_rate: float = 0.99,
        regime_sensitivity: float = 0.5,
        min_confidence: float = 0.05,
        regime_boost: float = 0.1,
        regime_penalty: float = 0.3,
    ) -> None:
        self.forgetting_rate = forgetting_rate
        self.regime_sensitivity = regime_sensitivity
        self.min_confidence = min_confidence
        self.regime_boost = regime_boost
        self.regime_penalty = regime_penalty

        self._lock = threading.RLock()
        self._base_memory: ExperienceMemory = base_memory

        # Counters
        self._iteration: int = 0
        self._last_decay_iteration: int = 0

        # Per-regime IC accumulators: regime_str -> deque of ICs
        self._regime_ic_history: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=200)
        )

        # Outcome stats
        self._outcome_counts: dict[str, int] = defaultdict(int)
        self._formula_regime_map: dict[str, RegimeState] = {}

    # --- public API ---

    @property
    def base_memory(self) -> ExperienceMemory:
        """Thread-safe read of the current base memory snapshot."""
        with self._lock:
            return self._base_memory

    def on_factor_evaluated(
        self,
        formula: str,
        ic: float,
        regime: RegimeState,
        outcome: str,
    ) -> None:
        """Called immediately after each factor evaluation.

        Parameters
        ----------
        formula : str
            DSL formula of the evaluated candidate.
        ic : float
            Observed IC (signed).
        regime : RegimeState
            Active market regime at evaluation time.
        outcome : str
            One of: ``'admitted'``, ``'rejected_ic'``,
            ``'rejected_correlation'``, ``'replaced'``.
        """
        t0 = time.perf_counter()
        with self._lock:
            self._iteration += 1
            self._outcome_counts[outcome] += 1
            self._formula_regime_map[formula] = regime
            regime_key = str(regime)
            self._regime_ic_history[regime_key].append(ic)

            # Boost success patterns that match admitted factors
            if outcome == "admitted" and abs(ic) >= 0.03:
                boost_factor = 1 + int(
                    self.regime_sensitivity * 2 * abs(ic) / 0.1
                )
                # Try to match formula against existing success pattern templates
                for pat in self._base_memory.success_patterns:
                    if _formula_matches_template(formula, pat.template):
                        self._base_memory = bump_pattern_confidence(
                            self._base_memory, pat.name, boost=boost_factor
                        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if elapsed_ms > 1.0:
            logger.debug(
                "on_factor_evaluated took %.2f ms (target < 1 ms)", elapsed_ms
            )

    def apply_forgetting(self, iterations_elapsed: int = 1) -> None:
        """Exponentially decay pattern confidence and prune stale entries.

        Parameters
        ----------
        iterations_elapsed : int
            Number of mining iterations since last call to this method.
        """
        with self._lock:
            elapsed = max(1, int(iterations_elapsed))
            self._base_memory = apply_confidence_decay(
                self._base_memory,
                decay_factor=self.forgetting_rate ** elapsed,
                min_confidence=self.min_confidence,
            )
            self._last_decay_iteration = self._iteration

    def on_regime_change(
        self,
        old_regime: RegimeState,
        new_regime: RegimeState,
    ) -> None:
        """React to a detected regime transition.

        Actions performed:
        1. Boost confidence of success patterns tagged for ``new_regime``.
        2. Down-weight success patterns tagged for ``old_regime``.
        3. Insert a regime-transition ``StrategicInsight`` into base memory.

        Parameters
        ----------
        old_regime : RegimeState
        new_regime : RegimeState
        """
        with self._lock:
            # Boost / penalise patterns in base memory by tag matching
            for pat in self._base_memory.success_patterns:
                str(new_regime)
                str(old_regime)
                # We tag patterns heuristically via their description keywords
                desc_lower = pat.description.lower()
                name_lower = pat.name.lower()
                new_labels_lower = {lbl.lower() for lbl in _regime_labels(new_regime)}
                old_labels_lower = {lbl.lower() for lbl in _regime_labels(old_regime)}

                if any(lbl in desc_lower or lbl in name_lower for lbl in new_labels_lower):
                    self._base_memory = bump_pattern_confidence(
                        self._base_memory, pat.name,
                        boost=int(self.regime_boost * 10)
                    )
                elif any(lbl in desc_lower or lbl in name_lower for lbl in old_labels_lower):
                    self._base_memory = penalise_pattern_confidence(
                        self._base_memory, pat.name,
                        penalty=self.regime_penalty,
                    )

            # Add a strategic insight about the regime transition
            insight_text = (
                f"Regime transition detected: {old_regime} -> {new_regime} "
                f"at iteration {self._iteration}"
            )
            evidence = (
                f"Based on EW streaming statistics. New regime labels: "
                f"{', '.join(_regime_labels(new_regime))}. Old: "
                f"{', '.join(_regime_labels(old_regime))}."
            )
            new_insight = StrategicInsight(
                insight=insight_text,
                evidence=evidence,
                batch_source=self._iteration,
            )
            # Avoid duplicate back-to-back transition insights
            if not self._base_memory.insights or (
                self._base_memory.insights[-1].insight != insight_text
            ):
                self._base_memory.insights.append(new_insight)
                # Cap insights at 50 to avoid unbounded growth
                if len(self._base_memory.insights) > 50:
                    self._base_memory.insights = self._base_memory.insights[-50:]

    def get_memory_health_stats(self) -> dict:
        """Return comprehensive health statistics for the memory system.

        Returns
        -------
        dict
            Keys: ``active_patterns``, ``avg_confidence``,
            ``regime_distribution``, ``staleness_score``,
            ``outcome_counts``, ``total_iterations``.
        """
        with self._lock:
            mem = self._base_memory
            all_counts = [
                p.occurrence_count for p in mem.success_patterns
            ] + [
                f.occurrence_count for f in mem.forbidden_directions
            ]
            max_c = max(all_counts) if all_counts else 1
            if max_c == 0:
                max_c = 1
            norm_confs = [c / max_c for c in all_counts]
            avg_conf = float(np.mean(norm_confs)) if norm_confs else 0.0

            # Regime distribution from IC history
            regime_dist = {
                k: len(v) for k, v in self._regime_ic_history.items()
            }

            # Staleness: fraction of patterns with count 0 (never updated)
            n_patterns = len(mem.success_patterns) + len(mem.forbidden_directions)
            n_zero = sum(1 for c in all_counts if c == 0)
            staleness = n_zero / max(n_patterns, 1)

            return {
                "active_patterns": n_patterns,
                "avg_confidence": round(avg_conf, 4),
                "regime_distribution": regime_dist,
                "staleness_score": round(staleness, 4),
                "outcome_counts": dict(self._outcome_counts),
                "total_iterations": self._iteration,
                "last_decay_iteration": self._last_decay_iteration,
                "version": mem.version,
            }

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "forgetting_rate": self.forgetting_rate,
                "regime_sensitivity": self.regime_sensitivity,
                "min_confidence": self.min_confidence,
                "regime_boost": self.regime_boost,
                "regime_penalty": self.regime_penalty,
                "iteration": self._iteration,
                "last_decay_iteration": self._last_decay_iteration,
                "outcome_counts": dict(self._outcome_counts),
                "base_memory": self._base_memory.to_dict(),
                # Regime IC history stores last N ICs per regime
                "regime_ic_history": {
                    k: list(v) for k, v in self._regime_ic_history.items()
                },
            }

    @classmethod
    def from_dict(cls, d: dict) -> OnlineMemoryUpdater:
        mem = ExperienceMemory.from_dict(d["base_memory"])
        updater = cls(
            base_memory=mem,
            forgetting_rate=d.get("forgetting_rate", 0.99),
            regime_sensitivity=d.get("regime_sensitivity", 0.5),
            min_confidence=d.get("min_confidence", 0.05),
            regime_boost=d.get("regime_boost", 0.1),
            regime_penalty=d.get("regime_penalty", 0.3),
        )
        updater._iteration = d.get("iteration", 0)
        updater._last_decay_iteration = d.get("last_decay_iteration", 0)
        updater._outcome_counts.update(d.get("outcome_counts", {}))
        for regime_key, ic_list in d.get("regime_ic_history", {}).items():
            updater._regime_ic_history[regime_key] = deque(ic_list, maxlen=200)
        return updater


# ---------------------------------------------------------------------------
# RegimeTransitionForecaster
# ---------------------------------------------------------------------------

class RegimeTransitionForecaster:
    """Logistic-regression forecaster for regime transitions.

    Trains on the sequence of (feature_vector, next_regime_label) pairs
    accumulated during live trading / mining.  Used to proactively load
    regime-specific patterns *before* a transition occurs.

    The feature vector is constructed inside ``_build_feature_vector`` and
    encodes recent EW statistics (mean, vol, Hurst proxy) concatenated with
    a one-hot encoding of the current regime dimensions.

    Parameters
    ----------
    n_regime_classes : int
        Number of distinct regime label combinations tracked.  Set to a
        small number (e.g. 8 or 16) to keep the model tractable.
    min_samples_to_fit : int
        Minimum labelled samples before the model is fitted.
    refit_every : int
        Re-train every N calls to ``predict_next_regime``.
    """

    # Feature dimension: 3 (ew stats) + 3 (trend one-hot) + 3 (vol one-hot)
    #                   + 3 (mean_rev one-hot) = 12
    _FEATURE_DIM = 12

    def __init__(
        self,
        min_samples_to_fit: int = 30,
        refit_every: int = 20,
    ) -> None:
        self.min_samples_to_fit = min_samples_to_fit
        self.refit_every = refit_every

        self._lock = threading.RLock()
        self._feature_history: list[np.ndarray] = []
        self._regime_history: list[RegimeState] = []
        self._next_regime_labels: list[str] = []  # shifted by 1

        self._model = None  # sklearn LogisticRegression, lazy init
        self._label_encoder: dict[str, int] = {}
        self._inv_label_encoder: dict[int, str] = {}
        self._predict_call_count: int = 0
        self._fitted: bool = False

        # Cache of unique regime states seen during training
        self._known_regimes: dict[str, RegimeState] = {}

    def record_observation(
        self,
        regime: RegimeState,
        features: np.ndarray,
    ) -> None:
        """Append one (features, regime) observation to the training buffer.

        Should be called once per bar/update with the current streaming
        feature vector and the corresponding regime.

        Parameters
        ----------
        regime : RegimeState
        features : np.ndarray, shape (``_FEATURE_DIM``,)
        """
        with self._lock:
            self._feature_history.append(features.copy())
            self._regime_history.append(regime)
            regime_str = str(regime)
            self._known_regimes[regime_str] = regime

            # Build (X, y) where y[t] = regime_str[t+1]
            if len(self._regime_history) >= 2:
                # The label for the *previous* observation is the current regime
                self._next_regime_labels.append(regime_str)

    def fit(
        self,
        regime_history: list[RegimeState] | None = None,
        feature_history: np.ndarray | None = None,
    ) -> None:
        """Fit (or re-fit) the logistic regression model.

        Can be called with external data (for back-testing) or with no
        arguments to use the internally accumulated buffer.

        Parameters
        ----------
        regime_history : list[RegimeState] or None
            Optional external regime sequence (length T).
        feature_history : np.ndarray or None, shape (T, _FEATURE_DIM)
            Optional external feature matrix.
        """
        with self._lock:
            if regime_history is not None and feature_history is not None:
                assert len(regime_history) == len(feature_history)
                feats = feature_history
                regimes = regime_history
                labels = [str(r) for r in regimes[1:]]
                X = feats[:-1]
            else:
                if len(self._next_regime_labels) < self.min_samples_to_fit:
                    return
                X = np.array(self._feature_history[:-1])
                labels = self._next_regime_labels

            unique_labels = list(set(labels))
            if len(unique_labels) < 2:
                return  # Cannot fit with only one class

            self._label_encoder = {lbl: i for i, lbl in enumerate(unique_labels)}
            self._inv_label_encoder = {i: lbl for lbl, i in self._label_encoder.items()}

            y = np.array([self._label_encoder[lbl] for lbl in labels])

            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = LogisticRegression(
                    max_iter=500,
                    solver="lbfgs",
                    C=1.0,
                    random_state=42,
                )
                model.fit(X_scaled, y)
                self._model = (scaler, model)
                self._fitted = True
            except Exception as e:
                logger.warning("RegimeTransitionForecaster fit failed: %s", e)
                self._fitted = False

    def predict_next_regime(
        self,
        current_features: np.ndarray,
    ) -> tuple[RegimeState, float]:
        """Predict the most probable next regime.

        Parameters
        ----------
        current_features : np.ndarray, shape (``_FEATURE_DIM``,)

        Returns
        -------
        (RegimeState, float)
            Predicted regime and probability.  Returns (current regime, 0.0)
            if the model is not yet fitted.
        """
        with self._lock:
            self._predict_call_count += 1
            if self._predict_call_count % self.refit_every == 0:
                self.fit()

            if not self._fitted or self._model is None:
                # Fall back to current regime
                current = (
                    self._regime_history[-1]
                    if self._regime_history
                    else RegimeState()
                )
                return current, 0.0

            scaler, model = self._model
            try:
                X = scaler.transform(current_features.reshape(1, -1))
                proba = model.predict_proba(X)[0]
                best_class = int(np.argmax(proba))
                best_prob = float(proba[best_class])
                best_label = self._inv_label_encoder.get(best_class, "")
                best_regime = self._known_regimes.get(best_label, RegimeState())
                return best_regime, best_prob
            except Exception as e:
                logger.warning("RegimeTransitionForecaster predict failed: %s", e)
                return RegimeState(), 0.0

    def prepare_memory_for_transition(
        self,
        predicted_regime: RegimeState,
        pattern_store: RegimeSpecificPatternStore,
        boost: float = 0.15,
    ) -> None:
        """Pre-load (boost confidence of) patterns for the predicted regime.

        Parameters
        ----------
        predicted_regime : RegimeState
        pattern_store : RegimeSpecificPatternStore
        boost : float
            Confidence boost applied to matching patterns.
        """
        pattern_store.boost_regime_patterns(predicted_regime, boost=boost)

    @staticmethod
    def build_feature_vector(
        ew_mean: float,
        ew_std: float,
        hurst_proxy: float,
        regime: RegimeState,
    ) -> np.ndarray:
        """Build a fixed-length feature vector from streaming statistics.

        Layout (12 elements):
        [0]  ew_mean
        [1]  ew_std
        [2]  hurst_proxy
        [3-5]  trend one-hot (BULL, BEAR, NEUTRAL)
        [6-8]  vol one-hot (HIGH_VOL, LOW_VOL, NORMAL_VOL)
        [9-11] mean_rev one-hot (TRENDING, MEAN_REVERTING, RANDOM_WALK)

        Parameters
        ----------
        ew_mean, ew_std, hurst_proxy : float
        regime : RegimeState

        Returns
        -------
        np.ndarray, shape (12,)
        """
        from factorminer.evaluation.regime import MeanRevRegime, TrendRegime, VolRegime

        trend_oh = [
            float(regime.trend == TrendRegime.BULL),
            float(regime.trend == TrendRegime.BEAR),
            float(regime.trend == TrendRegime.NEUTRAL),
        ]
        vol_oh = [
            float(regime.vol == VolRegime.HIGH_VOL),
            float(regime.vol == VolRegime.LOW_VOL),
            float(regime.vol == VolRegime.NORMAL_VOL),
        ]
        mr_oh = [
            float(regime.mean_rev == MeanRevRegime.TRENDING),
            float(regime.mean_rev == MeanRevRegime.MEAN_REVERTING),
            float(regime.mean_rev == MeanRevRegime.RANDOM_WALK),
        ]
        return np.array(
            [ew_mean, ew_std, hurst_proxy] + trend_oh + vol_oh + mr_oh,
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "min_samples_to_fit": self.min_samples_to_fit,
                "refit_every": self.refit_every,
                "predict_call_count": self._predict_call_count,
                "fitted": self._fitted,
                "feature_history": [f.tolist() for f in self._feature_history[-500:]],
                "regime_history": [r.to_dict() for r in self._regime_history[-500:]],
                "next_regime_labels": self._next_regime_labels[-500:],
                "known_regimes": {k: v.to_dict() for k, v in self._known_regimes.items()},
            }

    @classmethod
    def from_dict(cls, d: dict) -> RegimeTransitionForecaster:
        forecaster = cls(
            min_samples_to_fit=d.get("min_samples_to_fit", 30),
            refit_every=d.get("refit_every", 20),
        )
        forecaster._feature_history = [
            np.array(f, dtype=np.float64) for f in d.get("feature_history", [])
        ]
        forecaster._regime_history = [
            RegimeState.from_dict(r) for r in d.get("regime_history", [])
        ]
        forecaster._next_regime_labels = d.get("next_regime_labels", [])
        forecaster._known_regimes = {
            k: RegimeState.from_dict(v)
            for k, v in d.get("known_regimes", {}).items()
        }
        forecaster._predict_call_count = d.get("predict_call_count", 0)
        if d.get("fitted", False):
            forecaster.fit()
        return forecaster


# ---------------------------------------------------------------------------
# OnlineRegimeMemory — main orchestrator
# ---------------------------------------------------------------------------

class OnlineRegimeMemory:
    """Full online regime-aware memory system.

    Integrates:
    - ``StreamingRegimeDetector`` for bar-by-bar regime classification
    - ``RegimeSpecificPatternStore`` for per-regime IC tracking
    - ``OnlineMemoryUpdater`` for streaming forgetting and regime-change hooks
    - ``RegimeTransitionForecaster`` for proactive memory preparation

    Usage
    -----
    ::

        from factorminer.memory.online_regime_memory import OnlineRegimeMemory
        from factorminer.memory.memory_store import ExperienceMemory

        mem = OnlineRegimeMemory(base_memory=ExperienceMemory(), config={})

        # In the mining loop, after each bar of market data:
        mem.update_market(returns=bar_returns)

        # After each factor evaluation:
        mem.update(formula, signals, ic, market_data, outcome)

        # At generation time:
        signal = mem.retrieve(library_state, market_data)
        print(signal.prompt_text)

    Parameters
    ----------
    base_memory : ExperienceMemory
    config : dict
        Optional configuration overrides.  Keys and defaults:

        - ``forgetting_rate`` (0.99): per-iteration confidence retention
        - ``regime_sensitivity`` (0.5): how much to weight regime-specific patterns
        - ``min_confidence`` (0.05): pruning threshold
        - ``forget_every_n_iterations`` (10): call ``apply_forgetting`` every N evals
        - ``max_regime_patterns`` (500): capacity of regime pattern store
        - ``streaming_config`` ({}): forwarded to ``StreamingRegimeConfig``
    """

    def __init__(
        self,
        base_memory: ExperienceMemory | None = None,
        config: dict | None = None,
    ) -> None:
        cfg = config or {}
        if base_memory is None:
            base_memory = ExperienceMemory()

        streaming_cfg = StreamingRegimeConfig(
            **{k: v for k, v in cfg.get("streaming_config", {}).items()
               if k in StreamingRegimeConfig.__dataclass_fields__}
        )
        self._detector = StreamingRegimeDetector(config=streaming_cfg)
        self._pattern_store = RegimeSpecificPatternStore(
            max_patterns=cfg.get("max_regime_patterns", 500),
        )
        self._updater = OnlineMemoryUpdater(
            base_memory=base_memory,
            forgetting_rate=cfg.get("forgetting_rate", 0.99),
            regime_sensitivity=cfg.get("regime_sensitivity", 0.5),
            min_confidence=cfg.get("min_confidence", 0.05),
        )
        self._forecaster = RegimeTransitionForecaster()
        self._forget_every = cfg.get("forget_every_n_iterations", 10)
        self._iteration_count: int = 0
        self._current_regime: RegimeState = RegimeState()
        self._lock = threading.RLock()

        # Track last regime for change detection
        self._prev_regime: RegimeState = RegimeState()

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def update_market(
        self,
        returns: np.ndarray,
        volumes: np.ndarray | None = None,
    ) -> RegimeState:
        """Process one bar of market data and update the regime state.

        Call this *before* ``update()`` on any factors evaluated at this bar.

        Parameters
        ----------
        returns : np.ndarray, shape (M,)
        volumes : np.ndarray or None

        Returns
        -------
        RegimeState
            Updated current regime.
        """
        with self._lock:
            new_regime = self._detector.update(returns, volumes)
            prev = self._current_regime

            if new_regime != prev:
                self._updater.on_regime_change(prev, new_regime)
                self._pattern_store.boost_regime_patterns(new_regime, boost=0.1)
                self._pattern_store.penalise_regime_patterns(prev, penalty=0.15)

                # Prepare memory proactively
                feat = self._build_feature_vector(new_regime)
                predicted, prob = self._forecaster.predict_next_regime(feat)
                if prob > 0.5:
                    self._forecaster.prepare_memory_for_transition(
                        predicted, self._pattern_store
                    )

            self._prev_regime = prev
            self._current_regime = new_regime

            # Record for forecaster
            feat = self._build_feature_vector(new_regime)
            self._forecaster.record_observation(new_regime, feat)

            return new_regime

    def update(
        self,
        formula: str,
        signals: np.ndarray,
        ic: float,
        market_data: dict | None = None,
        outcome: str = "admitted",
    ) -> None:
        """Single update call: detect regime from market_data, update patterns.

        This is the main hook called inside the mining loop after each factor
        evaluation.  It orchestrates:
        1. Regime detection from ``market_data`` (if provided)
        2. Regime-specific pattern update
        3. Base memory update (online updater)
        4. Periodic forgetting

        Parameters
        ----------
        formula : str
            DSL formula string.
        signals : np.ndarray
            Factor signal matrix (used only for future extension).
        ic : float
            Observed IC.
        market_data : dict or None
            Optional dict with key ``'returns'`` (np.ndarray).
        outcome : str
        """
        with self._lock:
            regime = self._current_regime

            # If market_data provided, do an inline regime update
            if market_data is not None and "returns" in market_data:
                regime = self.update_market(
                    market_data["returns"],
                    market_data.get("volumes"),
                )

            # Update regime-specific pattern store
            if abs(ic) >= 0.02:
                self._pattern_store.add_pattern(formula, regime, ic)

            # Notify online updater
            self._updater.on_factor_evaluated(formula, ic, regime, outcome)

            self._iteration_count += 1

            # Periodic forgetting
            if self._iteration_count % self._forget_every == 0:
                self._updater.apply_forgetting(
                    iterations_elapsed=self._forget_every
                )
                decay = self._updater.forgetting_rate ** self._forget_every
                self._pattern_store.apply_decay(decay)

    def retrieve(
        self,
        library_state: dict | None = None,
        market_data: dict | None = None,
        max_success: int = 8,
        max_forbidden: int = 10,
        max_insights: int = 10,
        top_regime_patterns: int = 5,
    ) -> MemorySignal:
        """Regime-aware memory retrieval.

        Combines the standard base-memory retrieval with regime-specific
        pattern selection and a next-regime forecast.

        Parameters
        ----------
        library_state : dict or None
        market_data : dict or None
        max_success : int
        max_forbidden : int
        max_insights : int
        top_regime_patterns : int

        Returns
        -------
        MemorySignal
        """
        with self._lock:
            current_regime = self._current_regime

            # Update regime if market data provided
            if market_data is not None and "returns" in market_data:
                current_regime = self.update_market(
                    market_data["returns"], market_data.get("volumes")
                )

            # 1. Base retrieval
            base_result = retrieve_memory(
                self._updater.base_memory,
                library_state=library_state,
                max_success=max_success,
                max_forbidden=max_forbidden,
                max_insights=max_insights,
            )

            # 2. Regime-specific patterns
            regime_pats = self._pattern_store.retrieve_for_regime(
                current_regime, top_k=top_regime_patterns
            )
            cross_pats = self._pattern_store.get_cross_regime_patterns(
                top_k=top_regime_patterns // 2 + 1
            )

            # 3. Forecast next regime
            feat = self._build_feature_vector(current_regime)
            predicted_regime, forecast_conf = self._forecaster.predict_next_regime(feat)

            # 4. Build enriched prompt text
            regime_section = self._format_regime_section(
                current_regime, regime_pats, cross_pats, predicted_regime, forecast_conf
            )
            prompt_text = base_result["prompt_text"] + "\n" + regime_section

            return MemorySignal(
                recommended_directions=base_result["recommended_directions"],
                forbidden_directions=base_result["forbidden_directions"],
                insights=base_result["insights"],
                library_state=base_result["library_state"],
                prompt_text=prompt_text,
                current_regime=current_regime,
                regime_patterns=[p.to_dict() for p in regime_pats],
                cross_regime_patterns=[p.to_dict() for p in cross_pats],
                forecasted_regime=predicted_regime if forecast_conf > 0.0 else None,
                forecast_confidence=forecast_conf,
            )

    def get_full_status(self) -> dict:
        """Comprehensive status: regime, patterns, health, forecasts.

        Returns
        -------
        dict
            Keys: ``current_regime``, ``regime_history``, ``transition_probs``,
            ``pattern_store_stats``, ``memory_health``, ``forecasted_regime``,
            ``forecast_confidence``, ``iteration_count``.
        """
        with self._lock:
            current = self._current_regime
            feat = self._build_feature_vector(current)
            predicted, conf = self._forecaster.predict_next_regime(feat)
            history = self._detector.get_regime_history(lookback=20)
            return {
                "current_regime": current.to_dict(),
                "regime_history": [r.to_dict() for r in history],
                "transition_probs": self._detector.regime_transition_probability(),
                "pattern_store_stats": self._pattern_store.get_stats(),
                "memory_health": self._updater.get_memory_health_stats(),
                "forecasted_regime": predicted.to_dict() if conf > 0.0 else None,
                "forecast_confidence": round(conf, 4),
                "iteration_count": self._iteration_count,
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise to JSON.

        Parameters
        ----------
        path : str or Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str | Path) -> None:
        """Deserialise from JSON.

        Parameters
        ----------
        path : str or Path
        """
        with open(path) as f:
            data = json.load(f)
        with self._lock:
            self._from_dict_inplace(data)

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "_version": 1,
                "iteration_count": self._iteration_count,
                "current_regime": self._current_regime.to_dict(),
                "prev_regime": self._prev_regime.to_dict(),
                "forget_every": self._forget_every,
                "updater": self._updater.to_dict(),
                "pattern_store": self._pattern_store.to_dict(),
                "forecaster": self._forecaster.to_dict(),
            }

    @classmethod
    def from_dict(cls, d: dict) -> OnlineRegimeMemory:
        mem_data = d["updater"]["base_memory"]
        base_mem = ExperienceMemory.from_dict(mem_data)
        cfg = {"forget_every_n_iterations": d.get("forget_every", 10)}
        obj = cls(base_memory=base_mem, config=cfg)
        obj._from_dict_inplace(d)
        return obj

    def _from_dict_inplace(self, d: dict) -> None:
        self._iteration_count = d.get("iteration_count", 0)
        self._current_regime = RegimeState.from_dict(
            d.get("current_regime", {})
        )
        self._prev_regime = RegimeState.from_dict(
            d.get("prev_regime", {})
        )
        self._forget_every = d.get("forget_every", 10)
        self._updater = OnlineMemoryUpdater.from_dict(d["updater"])
        self._pattern_store = RegimeSpecificPatternStore.from_dict(
            d["pattern_store"]
        )
        self._forecaster = RegimeTransitionForecaster.from_dict(
            d["forecaster"]
        )

    # pickle support
    def __getstate__(self) -> dict:
        return self.to_dict()

    def __setstate__(self, state: dict) -> None:
        # Minimal init to avoid __init__ side effects
        self._lock = threading.RLock()
        self._from_dict_inplace(state)
        # Rebuild detector (streaming state is not persisted)
        self._detector = StreamingRegimeDetector()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_feature_vector(self, regime: RegimeState) -> np.ndarray:
        """Build a 12-element feature vector from the detector's EW state."""
        ew_mean = self._detector._ew_mean
        ew_std = float(np.sqrt(max(self._detector._ew_var, 0.0)))
        # Use the ratio of fast/slow variance as a Hurst proxy
        slow_var = max(self._detector._ew_var_slow, 1e-16)
        fast_var = max(self._detector._ew_var, 1e-16)
        hurst_proxy = float(np.clip(
            0.5 + 0.5 * math.log(fast_var / slow_var + 1e-10) / (math.log(20) + 1e-10),
            0.0, 1.0
        ))
        return RegimeTransitionForecaster.build_feature_vector(
            ew_mean, ew_std, hurst_proxy, regime
        )

    @staticmethod
    def _format_regime_section(
        current: RegimeState,
        regime_patterns: list[RegimeSpecificPattern],
        cross_patterns: list[RegimeSpecificPattern],
        predicted: RegimeState,
        forecast_conf: float,
    ) -> str:
        lines = [
            "=== REGIME-AWARE MEMORY ===",
            f"Current regime: {current}",
        ]
        if forecast_conf > 0.3:
            lines.append(
                f"Forecasted next regime: {predicted} "
                f"(confidence {forecast_conf:.1%})"
            )
        if regime_patterns:
            lines.append("\nTop patterns for current regime:")
            for i, p in enumerate(regime_patterns, 1):
                lines.append(
                    f"  {i}. {p.formula_template[:80]}  "
                    f"[IC={p.ic_in_regime:.3f}, "
                    f"spec={p.regime_specificity:.2f}, "
                    f"conf={p.confidence:.2f}]"
                )
        if cross_patterns:
            lines.append("\nCross-regime (universal) patterns:")
            for i, p in enumerate(cross_patterns, 1):
                lines.append(
                    f"  {i}. {p.formula_template[:80]}  "
                    f"[avg_IC={abs(p.ic_in_regime):.3f}, "
                    f"conf={p.confidence:.2f}]"
                )
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MemoryForgetCurve
# ---------------------------------------------------------------------------

@dataclass
class _MemorySnapshot:
    """Internal snapshot used by MemoryForgetCurve."""
    iteration: int
    timestamp: float
    active_patterns: int
    avg_confidence: float
    n_regime_patterns: int
    staleness_score: float
    pattern_confidences: list[float]


class MemoryForgetCurve:
    """Track and visualise how memory evolves (and forgets) over mining iterations.

    Parameters
    ----------
    max_snapshots : int
        Maximum snapshots to retain in memory.
    """

    def __init__(self, max_snapshots: int = 1000) -> None:
        self.max_snapshots = max_snapshots
        self._snapshots: list[_MemorySnapshot] = []
        self._lock = threading.RLock()

    def record_snapshot(
        self,
        memory: OnlineRegimeMemory,
        iteration: int,
    ) -> None:
        """Record a snapshot of the current memory state.

        Parameters
        ----------
        memory : OnlineRegimeMemory
        iteration : int
            Current mining iteration number (used as x-axis).
        """
        status = memory.get_full_status()
        health = status["memory_health"]
        ps = status["pattern_store_stats"]

        # Collect per-pattern confidences from the regime pattern store
        with memory._lock:
            confs = [
                p.confidence
                for p in memory._pattern_store._patterns.values()
            ]

        snap = _MemorySnapshot(
            iteration=iteration,
            timestamp=time.time(),
            active_patterns=health["active_patterns"],
            avg_confidence=health["avg_confidence"],
            n_regime_patterns=ps["total_patterns"],
            staleness_score=health["staleness_score"],
            pattern_confidences=confs,
        )
        with self._lock:
            self._snapshots.append(snap)
            if len(self._snapshots) > self.max_snapshots:
                self._snapshots = self._snapshots[-self.max_snapshots:]

    def get_pattern_lifetimes(self) -> list[float]:
        """Estimate pattern lifetimes (iterations survived) from snapshot series.

        Returns
        -------
        list[float]
            One entry per 'pattern birth' estimated from count increases.
            Uses the number of iterations between when a pattern first appears
            (count > 0) and drops below min_confidence.

        Note: this is an approximation based on the active count trajectory.
        """
        with self._lock:
            if len(self._snapshots) < 2:
                return []
            counts = [s.n_regime_patterns for s in self._snapshots]
            iterations = [s.iteration for s in self._snapshots]
            lifetimes = []
            # Simple heuristic: measure spans between count peaks and troughs
            for i in range(1, len(counts) - 1):
                if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
                    # Local peak: estimate lifetime as distance to next trough
                    for j in range(i + 1, len(counts)):
                        if counts[j] < counts[i] * 0.5:
                            lifetimes.append(float(iterations[j] - iterations[i]))
                            break
            return lifetimes

    def plot_confidence_decay(self) -> None:
        """Plot confidence decay and pattern count over iterations.

        Requires matplotlib to be installed.  If not available, prints a
        summary table instead.
        """
        with self._lock:
            snapshots = list(self._snapshots)

        if not snapshots:
            print("No snapshots recorded yet.")
            return

        iterations = [s.iteration for s in snapshots]
        avg_confs = [s.avg_confidence for s in snapshots]
        active = [s.active_patterns for s in snapshots]
        regime_counts = [s.n_regime_patterns for s in snapshots]
        staleness = [s.staleness_score for s in snapshots]

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Memory Forget Curve", fontsize=14)

            ax = axes[0, 0]
            ax.plot(iterations, avg_confs, "b-o", markersize=3)
            ax.set_title("Average Pattern Confidence")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Confidence")
            ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            ax.plot(iterations, active, "g-o", markersize=3)
            ax.set_title("Active Patterns (base memory)")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

            ax = axes[1, 0]
            ax.plot(iterations, regime_counts, "r-o", markersize=3)
            ax.set_title("Regime-Specific Patterns")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            ax.plot(iterations, staleness, "m-o", markersize=3)
            ax.set_title("Staleness Score (fraction of zero-count patterns)")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Staleness")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

        except ImportError:
            # Fallback: ASCII table
            print(
                f"{'Iter':>8} {'AvgConf':>10} {'Active':>8} "
                f"{'RegimePats':>12} {'Staleness':>10}"
            )
            print("-" * 52)
            for s in snapshots[::max(1, len(snapshots) // 20)]:
                print(
                    f"{s.iteration:>8} {s.avg_confidence:>10.4f} "
                    f"{s.active_patterns:>8} {s.n_regime_patterns:>12} "
                    f"{s.staleness_score:>10.4f}"
                )

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "max_snapshots": self.max_snapshots,
                "snapshots": [
                    {
                        "iteration": s.iteration,
                        "timestamp": s.timestamp,
                        "active_patterns": s.active_patterns,
                        "avg_confidence": s.avg_confidence,
                        "n_regime_patterns": s.n_regime_patterns,
                        "staleness_score": s.staleness_score,
                    }
                    for s in self._snapshots
                ],
            }

    @classmethod
    def from_dict(cls, d: dict) -> MemoryForgetCurve:
        curve = cls(max_snapshots=d.get("max_snapshots", 1000))
        for sd in d.get("snapshots", []):
            snap = _MemorySnapshot(
                iteration=sd["iteration"],
                timestamp=sd["timestamp"],
                active_patterns=sd["active_patterns"],
                avg_confidence=sd["avg_confidence"],
                n_regime_patterns=sd["n_regime_patterns"],
                staleness_score=sd["staleness_score"],
                pattern_confidences=[],
            )
            curve._snapshots.append(snap)
        return curve


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _formula_matches_template(formula: str, template: str) -> bool:
    """Heuristic check: does a formula share structural operators with a template?

    Extracts capitalised operator names from both strings and tests for
    meaningful overlap (>= 1 shared operator, or substring containment).
    """
    import re
    op_re = re.compile(r"\b([A-Z][a-zA-Z]+)\(")
    f_ops = set(op_re.findall(formula))
    t_ops = set(op_re.findall(template))
    if not f_ops or not t_ops:
        return False
    overlap = f_ops & t_ops
    # At least 1 operator shared AND at least half of template ops present
    return (
        len(overlap) >= 1
        and len(overlap) / max(len(t_ops), 1) >= 0.4
    )
