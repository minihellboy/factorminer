"""Offline RFT dataset export for external GRPO/Verl-style trainers.

QuantEvolver (arXiv:2605.15412) converts mining backtest scores into GRPO/RFT
policy updates on a miner LLM. FactorMiner freezes the LLM and externalizes
experience into memory policies; this module is the *orthogonal* axis: it
turns existing mining trajectories into a reward-annotated offline dataset
that an external GPU host can consume.

This environment ships the reward + export layer only. It does **not** train
a model, load Verl/vLLM, or run GRPO. Policy-weight training requires external
GPU infrastructure not available here.

Placement note
--------------
Lives under ``architecture/`` (not a new ``training/`` package) because the
deliverable is a pure offline artifact producer that composes
``EvaluationKernel`` rewards, ``dependence`` metrics, ``regime`` labels, and
``FactorLifecycleStore`` trajectories. There is no trainer, checkpoint, or
optimizer surface to own a top-level package.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from factorminer.architecture.dependence import (
    DependenceMetric,
    SpearmanDependenceMetric,
    build_dependence_metric,
)
from factorminer.architecture.lifecycle import FactorLifecycleStore
from factorminer.evaluation.regime import (
    MarketRegime,
    RegimeClassification,
    RegimeConfig,
    RegimeDetector,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema / versioning
# ---------------------------------------------------------------------------

RFT_DATASET_SCHEMA_VERSION = "rft_v1"
RFT_DATASET_FORMAT = "factorminer.rft_trajectory_jsonl"

# Explicit honesty banner reused by CLI --help and module docs.
RFT_EXPORT_HONESTY = (
    "Exports a reward-annotated training dataset for external reinforcement "
    "fine-tuning (e.g. GRPO via Verl/vLLM on a GPU host). This command does "
    "NOT train a model -- policy-weight training requires external GPU "
    "infrastructure not available in this environment."
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiCoRewardConfig:
    """Weights for the Diversity-Complementarity (DiCo) composite reward.

    QuantEvolver scores a candidate for being both individually predictive
    (IC-like) and complementary to the current library. FactorMiner already
    computes both primitives; DiCo multiplies them into one scalar.

    Attributes
    ----------
    predictive_weight :
        Exponent on the absolute predictive score (paper IC). 1.0 is linear.
    complementarity_weight :
        Exponent on ``(1 - max_dependence)``. Values > 1 penalize redundancy
        more aggressively.
    dependence_metric :
        Name accepted by :func:`build_dependence_metric` when signals are
        available. Ignored when only a precomputed ``max_correlation`` is
        supplied.
    min_predictive :
        Floor applied to |IC| before weighting (avoids zeroing a weakly
        predictive but highly novel candidate entirely when using product
        form with noisy near-zero IC).
    clip_reward :
        Optional ``(lo, hi)`` clamp on the final scalar. ``None`` disables.
    """

    predictive_weight: float = 1.0
    complementarity_weight: float = 1.0
    dependence_metric: str = "spearman"
    min_predictive: float = 0.0
    clip_reward: tuple[float, float] | None = None


@dataclass(frozen=True)
class RFTExportConfig:
    """Controls offline RFT dataset export.

    Attributes
    ----------
    schema_version :
        Written into every JSONL record and the sidecar manifest.
    include_failed_parses :
        When False, skip trajectory entries with empty formulas / parse fails.
    regime_lookback :
        Rolling window forwarded to :class:`RegimeDetector` when returns are
        supplied for regime-aware task bucketing.
    dico :
        DiCo reward hyper-parameters.
    """

    schema_version: str = RFT_DATASET_SCHEMA_VERSION
    include_failed_parses: bool = False
    regime_lookback: int = 60
    dico: DiCoRewardConfig = field(default_factory=DiCoRewardConfig)


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiCoRewardResult:
    """Decomposed DiCo reward for one candidate vs. the current library."""

    reward: float
    predictive_score: float
    complementarity: float
    max_dependence: float
    dependence_metric: str
    library_size: int
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": float(self.reward),
            "predictive_score": float(self.predictive_score),
            "complementarity": float(self.complementarity),
            "max_dependence": float(self.max_dependence),
            "dependence_metric": self.dependence_metric,
            "library_size": int(self.library_size),
            "components": {k: float(v) for k, v in self.components.items()},
        }


@dataclass(frozen=True)
class RegimeTaskBucket:
    """One regime-aware task-bank cell for an episode/iteration."""

    iteration: int
    dominant_regime: str
    regime_mix: dict[str, float]
    n_periods: int
    task_id: str
    stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "dominant_regime": self.dominant_regime,
            "regime_mix": dict(self.regime_mix),
            "n_periods": int(self.n_periods),
            "task_id": self.task_id,
            "stats": {
                k: {sk: float(sv) for sk, sv in v.items()} for k, v in self.stats.items()
            },
        }


@dataclass(frozen=True)
class RFTTrajectoryRecord:
    """One ``(state, action, reward, regime_context)`` export row.

    JSONL schema (``rft_v1``)
    -------------------------
    Every line is one JSON object with keys:

    - ``schema_version`` (str): always ``rft_v1`` for this module.
    - ``record_id`` (str): stable id ``iter{N}:{factor_id}``.
    - ``iteration`` (int): mining round index.
    - ``state`` (object): library/context snapshot at proposal time.
        - ``library_size`` (int)
        - ``admitted_formulas`` (list[str]): formulas already admitted
          *before* this candidate in export order (running library).
        - ``iteration`` (int)
    - ``action`` (object): the candidate formula / policy action.
        - ``factor_id`` (str)
        - ``formula`` (str)
        - ``admitted`` (bool | null)
        - ``rejection_reason`` (str)
    - ``reward`` (float): DiCo composite scalar.
    - ``reward_breakdown`` (object): predictive / complementarity / dependence.
    - ``regime_context`` (object): task-bank bucket metadata for the episode.
    - ``metrics`` (object): raw IC/ICIR/max_correlation carried from lifecycle.
    """

    schema_version: str
    record_id: str
    iteration: int
    state: dict[str, Any]
    action: dict[str, Any]
    reward: float
    reward_breakdown: dict[str, Any]
    regime_context: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "iteration": int(self.iteration),
            "state": dict(self.state),
            "action": dict(self.action),
            "reward": float(self.reward),
            "reward_breakdown": dict(self.reward_breakdown),
            "regime_context": dict(self.regime_context),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class RFTExportResult:
    """Summary of a completed offline RFT export."""

    path: str
    n_records: int
    n_iterations: int
    schema_version: str
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    regime_task_counts: dict[str, int]
    manifest_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# DiCo reward
# ---------------------------------------------------------------------------


def _clip(value: float, bounds: tuple[float, float] | None) -> float:
    if bounds is None:
        return float(value)
    lo, hi = bounds
    return float(min(max(value, lo), hi))


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return v


def compute_max_dependence(
    candidate_signals: np.ndarray,
    library_signals: Sequence[np.ndarray],
    metric: DependenceMetric | None = None,
) -> float:
    """Max pairwise dependence of a candidate against a library of signals.

    Parameters
    ----------
    candidate_signals :
        Factor signal panel, shape ``(M, T)`` (assets × time) — the same
        convention used by :class:`LibraryGeometry` and dependence metrics.
    library_signals :
        Iterable of already-admitted signal panels with the same shape.
    metric :
        Dependence metric instance; defaults to Spearman.

    Returns
    -------
    float
        Maximum dependence in ``[0, 1]`` (metrics return abs-valued scores).
        ``0.0`` when the library is empty.
    """
    if not library_signals:
        return 0.0
    dep = metric or SpearmanDependenceMetric()
    cand = np.asarray(candidate_signals, dtype=np.float64)
    scores: list[float] = []
    for lib in library_signals:
        try:
            scores.append(float(dep.compute(cand, np.asarray(lib, dtype=np.float64))))
        except Exception:  # noqa: BLE001 — fail closed on malformed panels
            logger.warning("Dependence compute failed; treating pair as independent")
            continue
    if not scores:
        return 0.0
    return float(max(0.0, min(1.0, max(scores))))


def compute_dico_reward(
    predictive_score: float,
    *,
    max_dependence: float | None = None,
    candidate_signals: np.ndarray | None = None,
    library_signals: Sequence[np.ndarray] | None = None,
    config: DiCoRewardConfig | None = None,
    metric: DependenceMetric | None = None,
) -> DiCoRewardResult:
    """Compute the Diversity-Complementarity (DiCo) composite reward.

    ``reward = |IC|^α * (1 - max_dep)^β``

    where ``α = predictive_weight``, ``β = complementarity_weight``.

    Either supply ``max_dependence`` directly (cheap path used when exporting
    from lifecycle logs that already recorded ``max_correlation``), *or*
    supply ``candidate_signals`` + ``library_signals`` so dependence is
    measured with :mod:`factorminer.architecture.dependence`.
    """
    cfg = config or DiCoRewardConfig()
    if metric is None:
        try:
            metric = build_dependence_metric(cfg.dependence_metric)
        except ValueError:
            metric = SpearmanDependenceMetric()

    if max_dependence is None:
        if candidate_signals is None:
            max_dep = 0.0
        else:
            max_dep = compute_max_dependence(
                candidate_signals,
                library_signals or (),
                metric=metric,
            )
    else:
        max_dep = float(max(0.0, min(1.0, _finite(max_dependence, 0.0))))

    pred = abs(_finite(predictive_score, 0.0))
    if cfg.min_predictive > 0.0:
        pred = max(pred, float(cfg.min_predictive))

    complementarity = max(0.0, 1.0 - max_dep)
    # Guard 0**positive → 0; 0**0 is conventionally 1 for empty products.
    pred_term = (
        1.0
        if cfg.predictive_weight == 0.0
        else (pred ** cfg.predictive_weight if pred > 0.0 else 0.0)
    )
    comp_term = (
        1.0
        if cfg.complementarity_weight == 0.0
        else (
            complementarity ** cfg.complementarity_weight
            if complementarity > 0.0
            else 0.0
        )
    )
    reward = _clip(pred_term * comp_term, cfg.clip_reward)

    return DiCoRewardResult(
        reward=reward,
        predictive_score=pred,
        complementarity=complementarity,
        max_dependence=max_dep,
        dependence_metric=metric.name,
        library_size=len(library_signals or ()),
        components={
            "predictive_term": float(pred_term),
            "complementarity_term": float(comp_term),
            "predictive_weight": float(cfg.predictive_weight),
            "complementarity_weight": float(cfg.complementarity_weight),
        },
    )


# ---------------------------------------------------------------------------
# Regime-aware task bank
# ---------------------------------------------------------------------------


def _regime_name(value: int | str | MarketRegime) -> str:
    if isinstance(value, MarketRegime):
        return value.name.lower()
    if isinstance(value, str):
        return value.strip().lower() or "unknown"
    try:
        return MarketRegime(int(value)).name.lower()
    except (ValueError, KeyError):
        return "unknown"


def build_regime_task_bucket(
    iteration: int,
    classification: RegimeClassification | None,
    *,
    n_periods_hint: int | None = None,
) -> RegimeTaskBucket:
    """Build one regime task-bank cell from a :class:`RegimeClassification`.

    When ``classification`` is None (no returns available at export time),
    emits an ``unknown`` bucket so the export still carries structured
    regime_context rather than omitting the field.
    """
    if classification is None:
        n = int(n_periods_hint or 0)
        return RegimeTaskBucket(
            iteration=iteration,
            dominant_regime="unknown",
            regime_mix={"unknown": 1.0},
            n_periods=n,
            task_id=f"iter{iteration}:unknown",
            stats={},
        )

    labels = np.asarray(classification.labels)
    n = int(labels.size)
    if n == 0:
        return RegimeTaskBucket(
            iteration=iteration,
            dominant_regime="unknown",
            regime_mix={"unknown": 1.0},
            n_periods=0,
            task_id=f"iter{iteration}:unknown",
            stats={},
        )

    counts: Counter[str] = Counter()
    for code in labels.tolist():
        counts[_regime_name(code)] += 1

    mix = {name: float(c) / float(n) for name, c in sorted(counts.items())}
    dominant = max(mix.items(), key=lambda kv: kv[1])[0]
    stats = {
        _regime_name(regime): {k: float(v) for k, v in s.items()}
        for regime, s in classification.stats.items()
    }
    return RegimeTaskBucket(
        iteration=iteration,
        dominant_regime=dominant,
        regime_mix=mix,
        n_periods=n,
        task_id=f"iter{iteration}:{dominant}",
        stats=stats,
    )


def build_regime_task_bank(
    iterations: Sequence[int],
    returns: np.ndarray | None = None,
    *,
    config: RegimeConfig | None = None,
    precomputed: Mapping[int, RegimeTaskBucket] | None = None,
) -> dict[int, RegimeTaskBucket]:
    """Bucket mining iterations by detected market regime.

    Parameters
    ----------
    iterations :
        Iteration indices present in the lifecycle store.
    returns :
        Optional ``(M, T)`` forward-return panel used by
        :class:`RegimeDetector`. When omitted, every iteration is labeled
        ``unknown`` unless ``precomputed`` supplies a bucket.
    config :
        Regime detection config (lookback etc.).
    precomputed :
        Optional per-iteration buckets that short-circuit detection (used
        by tests and by callers that already classified regimes).
    """
    bank: dict[int, RegimeTaskBucket] = {}
    classification: RegimeClassification | None = None
    if returns is not None:
        detector = RegimeDetector(config or RegimeConfig())
        try:
            classification = detector.classify(np.asarray(returns, dtype=np.float64))
        except Exception:  # noqa: BLE001
            logger.warning("Regime classification failed; using unknown buckets")
            classification = None

    for iteration in iterations:
        if precomputed is not None and iteration in precomputed:
            bank[int(iteration)] = precomputed[iteration]
            continue
        bank[int(iteration)] = build_regime_task_bucket(
            int(iteration), classification
        )
    return bank


# ---------------------------------------------------------------------------
# Trajectory extraction helpers
# ---------------------------------------------------------------------------


def _extract_predictive(entry: Mapping[str, Any], details: Mapping[str, Any] | None = None) -> float:
    """Pull a paper-style IC from a trajectory entry or lifecycle details."""
    details = details or {}
    for source in (entry, details):
        for key in (
            "ic_paper_mean",
            "paper_ic",
            "ic_mean",
            "ic",
            "predictive_score",
        ):
            if key in source and source[key] is not None:
                return abs(_finite(source[key], 0.0))
    return 0.0


def _extract_max_dependence(entry: Mapping[str, Any], details: Mapping[str, Any] | None = None) -> float:
    details = details or {}
    for source in (entry, details):
        for key in ("max_dependence", "max_correlation", "max_corr"):
            if key in source and source[key] is not None:
                return float(max(0.0, min(1.0, _finite(source[key], 0.0))))
    return 0.0


def _iter_lifecycle_candidates(
    store: FactorLifecycleStore,
) -> list[dict[str, Any]]:
    """Collapse lifecycle events into one candidate row per (iteration, factor, formula).

    Prefers evaluation-bearing stages (fast_screened / admitted / replaced /
    correlation_rejected) over proposed/parsed shells so IC and correlation
    details survive into the export.
    """
    by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    stage_rank = {
        "proposed": 0,
        "parsed": 1,
        "fast_screened": 2,
        "correlation_rejected": 3,
        "admitted": 4,
        "replaced": 4,
        "memory_distilled": 1,
    }

    for event in store.events:
        key = (int(event.iteration), str(event.factor_name), str(event.formula))
        record = by_key.get(key)
        if record is None:
            record = {
                "iteration": int(event.iteration),
                "factor_id": str(event.factor_name),
                "formula": str(event.formula),
                "ic": 0.0,
                "icir": 0.0,
                "max_correlation": 0.0,
                "admitted": False,
                "replaced": None,
                "rejection_reason": "",
                "stage": event.stage,
                "status": event.status,
                "_rank": stage_rank.get(event.stage, 0),
            }
            by_key[key] = record

        rank = stage_rank.get(event.stage, 0)
        if rank >= int(record["_rank"]):
            record["_rank"] = rank
            record["stage"] = event.stage
            record["status"] = event.status

        details = event.details or {}
        if "ic_mean" in details:
            record["ic"] = _finite(details["ic_mean"], record["ic"])
        if "icir" in details:
            record["icir"] = _finite(details["icir"], record["icir"])
        if "max_correlation" in details or "max_dependence" in details:
            record["max_correlation"] = _extract_max_dependence(details)
        if "replaced" in details:
            record["replaced"] = details.get("replaced")
        if event.stage in {"admitted", "replaced"} and event.status == "passed":
            record["admitted"] = True
        if event.stage == "correlation_rejected":
            record["rejection_reason"] = str(details.get("reason", "") or "")
            record["admitted"] = False
        if event.stage == "parsed" and event.status == "failed":
            record["rejection_reason"] = str(
                details.get("reason", "parse_failed") or "parse_failed"
            )
        if event.stage == "memory_distilled":
            if "admitted" in details:
                record["admitted"] = bool(details.get("admitted"))
            if details.get("rejection_reason"):
                record["rejection_reason"] = str(details.get("rejection_reason") or "")

    rows = list(by_key.values())
    for row in rows:
        row.pop("_rank", None)
    rows.sort(key=lambda r: (r["iteration"], r["factor_id"], r["formula"]))
    return rows

def _resolve_artifact_dir(store: FactorLifecycleStore, source: str | Path | None) -> Path | None:
    """Best-effort directory containing sibling mining artifacts."""
    if store._path is not None:
        return store._path.parent
    if source is None:
        return None
    path = Path(source)
    if path.is_dir():
        return path
    if path.is_file():
        return path.parent
    return None


def _load_session_factor_metrics(artifact_dir: Path | None) -> dict[str, dict[str, float | str | bool]]:
    """Load per-formula IC / correlation metrics from ``session_log.json``.

    ``FactorLifecycleStore.record_batch_results`` only attaches ``ic_mean`` on
    the ``fast_screened`` / ``admitted`` stages today, so IC-screen failures
    (the bulk of a typical mock run) leave the lifecycle log without a
    predictive score. The session logger *does* persist IC for every evaluated
    candidate — use it as a non-destructive enrichment source when present.
    """
    if artifact_dir is None:
        return {}
    path = Path(artifact_dir) / "session_log.json"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read session_log.json for RFT enrichment: %s", exc)
        return {}

    factors = payload.get("factors") if isinstance(payload, dict) else None
    if not isinstance(factors, list):
        return {}

    by_formula: dict[str, dict[str, float | str | bool]] = {}
    for entry in factors:
        if not isinstance(entry, Mapping):
            continue
        formula = str(entry.get("expression") or entry.get("formula") or "").strip()
        if not formula:
            continue
        metrics: dict[str, float | str | bool] = {}
        if entry.get("ic") is not None:
            metrics["ic"] = _finite(entry.get("ic"), 0.0)
        if entry.get("icir") is not None:
            metrics["icir"] = _finite(entry.get("icir"), 0.0)
        if entry.get("max_correlation") is not None:
            metrics["max_correlation"] = float(
                max(0.0, min(1.0, _finite(entry.get("max_correlation"), 0.0)))
            )
        if "admitted" in entry:
            metrics["admitted"] = bool(entry.get("admitted"))
        if entry.get("rejection_reason"):
            metrics["rejection_reason"] = str(entry.get("rejection_reason") or "")
        if metrics:
            # Last write wins; session_log is append-ordered per evaluation.
            by_formula[formula] = metrics
    return by_formula


def _enrich_candidates_with_session_metrics(
    candidates: list[dict[str, Any]],
    session_metrics: Mapping[str, Mapping[str, float | str | bool]],
) -> list[dict[str, Any]]:
    """Fill missing IC / dependence fields from session_log metrics."""
    if not session_metrics:
        return candidates
    enriched: list[dict[str, Any]] = []
    for cand in candidates:
        row = dict(cand)
        formula = str(row.get("formula", "") or "").strip()
        extra = session_metrics.get(formula)
        if extra:
            if _extract_predictive(row) == 0.0 and "ic" in extra:
                row["ic"] = _finite(extra["ic"], 0.0)
            if _finite(row.get("icir"), 0.0) == 0.0 and "icir" in extra:
                row["icir"] = _finite(extra["icir"], 0.0)
            if _extract_max_dependence(row) == 0.0 and "max_correlation" in extra:
                row["max_correlation"] = _finite(extra["max_correlation"], 0.0)
            if not row.get("rejection_reason") and extra.get("rejection_reason"):
                row["rejection_reason"] = str(extra.get("rejection_reason") or "")
            # Prefer lifecycle admission flag; only fill if still false/missing
            # and session_log says admitted.
            if not row.get("admitted") and bool(extra.get("admitted")):
                row["admitted"] = True
        enriched.append(row)
    return enriched




# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_rft_dataset(
    store: FactorLifecycleStore | str | Path,
    output_path: str | Path,
    *,
    returns: np.ndarray | None = None,
    config: RFTExportConfig | None = None,
    library_signals_by_formula: Mapping[str, np.ndarray] | None = None,
    reward_fn: Callable[..., DiCoRewardResult] | None = None,
    regime_task_bank: Mapping[int, RegimeTaskBucket] | None = None,
    write_manifest: bool = True,
) -> RFTExportResult:
    """Export a reward-annotated RFT trajectory dataset as JSONL.

    Walks :class:`FactorLifecycleStore` records and emits one JSON object per
    candidate with the documented ``rft_v1`` schema
    ``(state, action, reward, regime_context)``.

    This function does **not** train a model. It only materializes the offline
    dataset an external GRPO/Verl trainer would consume on a GPU host.

    Parameters
    ----------
    store :
        A :class:`FactorLifecycleStore` instance, an output directory
        containing ``factor_lifecycle.jsonl``, or a direct path to that file.
    output_path :
        Destination ``.jsonl`` path. Parent directories are created as needed.
    returns :
        Optional ``(M, T)`` return panel for regime-aware task bucketing.
    config :
        Export controls (schema version, DiCo weights, parse filtering).
    library_signals_by_formula :
        Optional map of formula → signal panel. When present, DiCo dependence
        is recomputed from signals for admitted formulas rather than trusting
        the lifecycle ``max_correlation`` field alone.
    reward_fn :
        Optional override of :func:`compute_dico_reward` (also the seam the
        evaluation-kernel reward hook can plug into).
    regime_task_bank :
        Optional precomputed per-iteration regime buckets.
    write_manifest :
        When True, also write ``{stem}.manifest.json`` beside the JSONL.
    """
    cfg = config or RFTExportConfig()
    reward_fn = reward_fn or compute_dico_reward

    source_hint: str | Path | None = None
    if not isinstance(store, FactorLifecycleStore):
        source_hint = store
        store = FactorLifecycleStore.load(store)

    candidates = _iter_lifecycle_candidates(store)
    artifact_dir = _resolve_artifact_dir(store, source_hint)
    candidates = _enrich_candidates_with_session_metrics(
        candidates,
        _load_session_factor_metrics(artifact_dir),
    )

    if not cfg.include_failed_parses:
        filtered: list[dict[str, Any]] = []
        for c in candidates:
            formula = str(c.get("formula", "") or "").strip()
            if not formula:
                continue
            if c.get("stage") == "parsed" and c.get("status") == "failed":
                continue
            stage = str(c.get("stage", "") or "")
            if (
                stage == "proposed"
                and _extract_predictive(c) == 0.0
                and not c.get("admitted")
            ):
                continue
            filtered.append(c)
        candidates = filtered


    iterations = sorted({int(c["iteration"]) for c in candidates})
    regime_cfg = RegimeConfig(lookback_window=cfg.regime_lookback)
    bank = dict(
        build_regime_task_bank(
            iterations,
            returns,
            config=regime_cfg,
            precomputed=regime_task_bank,
        )
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Running admitted-formula library reconstructed in export order so the
    # state snapshot reflects "what the policy saw" without needing live
    # library signals for the default lifecycle path.
    admitted_formulas: list[str] = []
    admitted_signals: list[np.ndarray] = []
    signal_map = dict(library_signals_by_formula or {})

    records: list[RFTTrajectoryRecord] = []
    rewards: list[float] = []
    regime_counts: Counter[str] = Counter()

    for cand in candidates:
        iteration = int(cand["iteration"])
        formula = str(cand.get("formula", "") or "")
        factor_id = str(cand.get("factor_id", "") or "")
        predictive = _extract_predictive(cand)
        max_dep = _extract_max_dependence(cand)

        cand_signals = signal_map.get(formula)
        lib_signals: list[np.ndarray] = list(admitted_signals)
        if cand_signals is not None and lib_signals:
            dico = reward_fn(
                predictive,
                candidate_signals=cand_signals,
                library_signals=lib_signals,
                config=cfg.dico,
            )
        else:
            dico = reward_fn(
                predictive,
                max_dependence=max_dep,
                library_signals=lib_signals,
                config=cfg.dico,
            )

        bucket = bank.get(iteration) or build_regime_task_bucket(iteration, None)
        regime_counts[bucket.dominant_regime] += 1

        record = RFTTrajectoryRecord(
            schema_version=cfg.schema_version,
            record_id=f"iter{iteration}:{factor_id}",
            iteration=iteration,
            state={
                "library_size": len(admitted_formulas),
                "admitted_formulas": list(admitted_formulas),
                "iteration": iteration,
            },
            action={
                "factor_id": factor_id,
                "formula": formula,
                "admitted": bool(cand.get("admitted", False)),
                "rejection_reason": str(cand.get("rejection_reason", "") or ""),
                "stage": str(cand.get("stage", "") or ""),
                "replaced": cand.get("replaced"),
            },
            reward=float(dico.reward),
            reward_breakdown=dico.to_dict(),
            regime_context=bucket.to_dict(),
            metrics={
                "ic": _finite(cand.get("ic"), predictive),
                "icir": _finite(cand.get("icir"), 0.0),
                "max_correlation": float(dico.max_dependence),
                "predictive_score": float(dico.predictive_score),
            },
        )
        records.append(record)
        rewards.append(float(dico.reward))

        if bool(cand.get("admitted", False)) and formula:
            if formula not in admitted_formulas:
                admitted_formulas.append(formula)
                if formula in signal_map:
                    admitted_signals.append(np.asarray(signal_map[formula], dtype=np.float64))

    with open(out_path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record.to_dict(), default=str, ensure_ascii=False) + "\n")

    reward_arr = np.asarray(rewards, dtype=np.float64) if rewards else np.zeros(0)
    manifest_path: str | None = None
    result = RFTExportResult(
        path=str(out_path),
        n_records=len(records),
        n_iterations=len(iterations),
        schema_version=cfg.schema_version,
        reward_mean=float(np.mean(reward_arr)) if reward_arr.size else 0.0,
        reward_std=float(np.std(reward_arr)) if reward_arr.size else 0.0,
        reward_min=float(np.min(reward_arr)) if reward_arr.size else 0.0,
        reward_max=float(np.max(reward_arr)) if reward_arr.size else 0.0,
        regime_task_counts=dict(regime_counts),
        manifest_path=None,
    )

    if write_manifest:
        man_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
        if out_path.suffix == ".jsonl":
            man_path = out_path.with_name(out_path.stem + ".manifest.json")
        manifest = {
            "format": RFT_DATASET_FORMAT,
            "schema_version": cfg.schema_version,
            "honesty": RFT_EXPORT_HONESTY,
            "n_records": result.n_records,
            "n_iterations": result.n_iterations,
            "reward_summary": {
                "mean": result.reward_mean,
                "std": result.reward_std,
                "min": result.reward_min,
                "max": result.reward_max,
            },
            "regime_task_counts": result.regime_task_counts,
            "dico": asdict(cfg.dico),
            "dataset_path": str(out_path),
            # No API keys / secrets ever written here.
            "trains_model": False,
            "gpu_required_for_training": True,
        }
        with open(man_path, "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2, default=str)
            fp.write("\n")
        manifest_path = str(man_path)
        result = RFTExportResult(
            path=result.path,
            n_records=result.n_records,
            n_iterations=result.n_iterations,
            schema_version=result.schema_version,
            reward_mean=result.reward_mean,
            reward_std=result.reward_std,
            reward_min=result.reward_min,
            reward_max=result.reward_max,
            regime_task_counts=result.regime_task_counts,
            manifest_path=manifest_path,
        )

    logger.info(
        "Exported %d RFT trajectory records to %s (reward std=%.6f); does NOT train a model",
        result.n_records,
        result.path,
        result.reward_std,
    )
    return result


def load_rft_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load an exported RFT JSONL dataset into a list of dict records."""
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed RFT JSONL line")
                continue
            if not isinstance(payload, dict):
                logger.warning("Skipping non-object RFT JSONL line")
                continue
            records.append(payload)
    return records


# ---------------------------------------------------------------------------
# Evaluation-kernel reward-hook protocol (re-exported seam)
# ---------------------------------------------------------------------------


@runtime_checkable
class RewardHook(Protocol):
    """Pluggable reward callback a future external training loop can register.

    Implementations receive the same arguments :func:`compute_dico_reward`
    accepts via keywords and must return a :class:`DiCoRewardResult` (or any
    object exposing a ``.reward`` float and ``.to_dict()``).
    """

    def __call__(
        self,
        predictive_score: float,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> DiCoRewardResult: ...


def default_reward_hook(
    predictive_score: float,
    /,
    *args: Any,
    **kwargs: Any,
) -> DiCoRewardResult:
    """Default DiCo reward hook used when no custom hook is registered."""
    return compute_dico_reward(predictive_score, *args, **kwargs)
