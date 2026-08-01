"""Append-only canonical ledger for every candidate contact with market data."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from factorminer.core.canonicalizer import FormulaCanonicalizer
from factorminer.core.parser import try_parse
from factorminer.core.provenance import stable_digest
from factorminer.domain.trials import TrialInferenceFamily

TRIAL_LEDGER_SCHEMA_VERSION = "factor-trial-ledger-v1"
TRIAL_LEDGER_FILENAME = "global_trial_ledger.jsonl"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def dataset_scope_identity(
    contract: Mapping[str, Any],
    *,
    data: Any,
    targets: Mapping[str, Any],
) -> str:
    """Hash the declared contract and exact numerical panels for trial scope."""
    digest = hashlib.sha256()
    digest.update(b"factorminer-trial-dataset-v1\0")
    digest.update(_canonical_json(dict(_json_safe(contract))).encode("utf-8"))

    def update_value(label: str, value: Any) -> None:
        digest.update(label.encode("utf-8") + b"\0")
        if isinstance(value, Mapping):
            for key in sorted(value, key=str):
                update_value(f"{label}/{key}", value[key])
            return
        array = np.asarray(value)
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
        digest.update(_canonical_json({"shape": list(contiguous.shape)}).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))

    update_value("data", data)
    update_value("targets", targets)
    return digest.hexdigest()


@lru_cache(maxsize=8192)
def canonical_formula_identity(formula: str) -> tuple[str, str]:
    """Return the canonical form and SHA-256 identity for a valid formula."""
    tree = try_parse(formula)
    if tree is None:
        raise ValueError(f"cannot ledger an unparsable formula: {formula!r}")
    canonical = FormulaCanonicalizer().get_canonical_form(tree)
    digest = hashlib.sha256(
        f"factorminer-canonical-form-v1\0{canonical}".encode()
    ).hexdigest()
    return canonical, digest


def _trial_id(
    *,
    campaign_id: str,
    dataset_id: str,
    target_name: str,
    canonical_formula_hash: str,
) -> str:
    return stable_digest(
        {
            "campaign_id": campaign_id,
            "dataset_id": dataset_id,
            "target_name": target_name,
            "canonical_formula_hash": canonical_formula_hash,
        }
    )


@dataclass(frozen=True)
class TrialObservation:
    """One immutable observation that a canonical candidate contacted data."""

    schema_version: str
    sequence: int
    trial_id: str
    campaign_id: str
    dataset_id: str
    target_name: str
    iteration: int
    stage: str
    status: str
    factor_name: str
    formula: str
    canonical_formula: str
    canonical_formula_hash: str
    ic_series: tuple[float | None, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = ""
    previous_record_hash: str = ""
    record_hash: str = ""


class TrialLedger:
    """Tamper-evident, append-only trial history for one or more campaigns."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        campaign_id: str,
    ) -> None:
        if not campaign_id:
            raise ValueError("campaign_id must not be empty")
        self.campaign_id = campaign_id
        self.observations: list[TrialObservation] = []
        self._path = self._resolve_path(output_dir) if output_dir is not None else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                self._read(self._path)

    @staticmethod
    def _resolve_path(source: str | Path) -> Path:
        path = Path(source)
        return path if path.name.endswith(".jsonl") else path / TRIAL_LEDGER_FILENAME

    @classmethod
    def load(
        cls,
        source: str | Path,
        *,
        campaign_id: str = "*",
    ) -> TrialLedger:
        """Load and verify an existing ledger; missing paths yield an empty one."""
        path = cls._resolve_path(source)
        ledger = cls(output_dir=None, campaign_id=campaign_id)
        ledger._path = path
        if path.exists():
            ledger._read(path)
        return ledger

    @property
    def path(self) -> Path | None:
        return self._path

    def _read(self, path: Path) -> None:
        self.observations.clear()
        previous_hash = ""
        with open(path, encoding="utf-8") as handle:
            for expected_sequence, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                payload["ic_series"] = tuple(payload.get("ic_series", ()))
                observation = TrialObservation(**payload)
                self._verify_observation(
                    observation,
                    expected_sequence=expected_sequence,
                    previous_hash=previous_hash,
                )
                self.observations.append(observation)
                previous_hash = observation.record_hash

    @staticmethod
    def _record_payload(observation: TrialObservation) -> dict[str, Any]:
        payload = asdict(observation)
        payload.pop("record_hash", None)
        return payload

    @classmethod
    def _compute_record_hash(cls, observation: TrialObservation) -> str:
        return hashlib.sha256(
            _canonical_json(cls._record_payload(observation)).encode("utf-8")
        ).hexdigest()

    @classmethod
    def _verify_observation(
        cls,
        observation: TrialObservation,
        *,
        expected_sequence: int,
        previous_hash: str,
    ) -> None:
        if observation.schema_version != TRIAL_LEDGER_SCHEMA_VERSION:
            raise ValueError(f"unsupported trial ledger schema: {observation.schema_version!r}")
        if observation.sequence != expected_sequence:
            raise ValueError("trial ledger sequence is not contiguous")
        if observation.previous_record_hash != previous_hash:
            raise ValueError("trial ledger hash chain is broken")
        canonical, canonical_hash = canonical_formula_identity(observation.formula)
        if canonical != observation.canonical_formula:
            raise ValueError("trial ledger canonical formula does not match formula")
        if canonical_hash != observation.canonical_formula_hash:
            raise ValueError("trial ledger canonical formula hash does not match formula")
        expected_trial_id = _trial_id(
            campaign_id=observation.campaign_id,
            dataset_id=observation.dataset_id,
            target_name=observation.target_name,
            canonical_formula_hash=observation.canonical_formula_hash,
        )
        if observation.trial_id != expected_trial_id:
            raise ValueError("trial ledger trial_id does not match its scope")
        if observation.record_hash != cls._compute_record_hash(observation):
            raise ValueError("trial ledger record hash mismatch")

    def record_data_contact(
        self,
        *,
        factor_name: str,
        formula: str,
        dataset_id: str,
        target_name: str,
        iteration: int,
        stage: str,
        status: str,
        ic_series: Sequence[float] | np.ndarray | None = None,
        metadata: Mapping[str, Any] | None = None,
        recorded_at: str | None = None,
    ) -> TrialObservation:
        """Append one data-contact observation after canonicalizing its formula."""
        if not dataset_id or not target_name or not stage or not status:
            raise ValueError("dataset_id, target_name, stage, and status must not be empty")
        if iteration < 0:
            raise ValueError("iteration must be >= 0")
        canonical, canonical_hash = canonical_formula_identity(formula)
        normalized_series: tuple[float | None, ...] = ()
        if ic_series is not None:
            array = np.asarray(ic_series, dtype=np.float64)
            if array.ndim != 1 or array.size == 0:
                raise ValueError("ic_series must be a non-empty one-dimensional series")
            normalized_series = tuple(
                float(value) if math.isfinite(float(value)) else None for value in array
            )

        previous_hash = self.observations[-1].record_hash if self.observations else ""
        partial = TrialObservation(
            schema_version=TRIAL_LEDGER_SCHEMA_VERSION,
            sequence=len(self.observations) + 1,
            trial_id=_trial_id(
                campaign_id=self.campaign_id,
                dataset_id=dataset_id,
                target_name=target_name,
                canonical_formula_hash=canonical_hash,
            ),
            campaign_id=self.campaign_id,
            dataset_id=dataset_id,
            target_name=target_name,
            iteration=iteration,
            stage=stage,
            status=status,
            factor_name=factor_name,
            formula=formula,
            canonical_formula=canonical,
            canonical_formula_hash=canonical_hash,
            ic_series=normalized_series,
            metadata=dict(_json_safe(dict(metadata or {}))),
            recorded_at=recorded_at or datetime.now(UTC).isoformat(),
            previous_record_hash=previous_hash,
        )
        observation = TrialObservation(
            **{
                **asdict(partial),
                "record_hash": self._compute_record_hash(partial),
            }
        )
        self.observations.append(observation)
        if self._path is not None:
            with open(self._path, "a", encoding="utf-8") as handle:
                handle.write(_canonical_json(asdict(observation)) + "\n")
        return observation

    def record_batch_results(
        self,
        *,
        iteration: int,
        results: Iterable[Any],
        dataset_id: str,
        target_name: str,
        stage: str = "candidate_evaluation",
    ) -> list[TrialObservation]:
        """Record parsed candidates; parse failures never contacted data."""
        recorded: list[TrialObservation] = []
        for result in results:
            if not bool(getattr(result, "parse_ok", False)):
                continue
            target_stats = getattr(result, "target_stats", {}) or {}
            stats = target_stats.get(target_name) or target_stats.get("paper") or {}
            series = stats.get("rank_ic_series", stats.get("ic_series"))
            status = "evaluated" if stats else "screened_or_failed"
            recorded.append(
                self.record_data_contact(
                    factor_name=str(getattr(result, "factor_name", "")),
                    formula=str(getattr(result, "formula", "")),
                    dataset_id=dataset_id,
                    target_name=target_name,
                    iteration=iteration,
                    stage=stage,
                    status=status,
                    ic_series=series,
                    metadata={
                        "stage_passed": int(getattr(result, "stage_passed", 0)),
                        "admitted": bool(getattr(result, "admitted", False)),
                        "rejection_reason": str(getattr(result, "rejection_reason", "")),
                    },
                )
            )
        return recorded

    def _matching(
        self,
        *,
        campaign_id: str | None,
        dataset_id: str,
        target_name: str,
    ) -> list[TrialObservation]:
        active_campaign = self.campaign_id if campaign_id is None else campaign_id
        return [
            observation
            for observation in self.observations
            if (active_campaign == "*" or observation.campaign_id == active_campaign)
            and observation.dataset_id == dataset_id
            and observation.target_name == target_name
        ]

    def build_inference_family(
        self,
        *,
        dataset_id: str,
        target_name: str,
        periods: int | None = None,
        campaign_id: str | None = None,
        estimate_effective: bool = True,
    ) -> TrialInferenceFamily:
        """Build an all-trials family, deduplicated by canonical identity."""
        matching = self._matching(
            campaign_id=campaign_id,
            dataset_id=dataset_id,
            target_name=target_name,
        )
        if not matching:
            raise ValueError("no trial observations match the requested inference scope")

        by_trial: dict[str, TrialObservation] = {}
        for observation in matching:
            current = by_trial.get(observation.trial_id)
            if current is None or (not current.ic_series and observation.ic_series):
                by_trial[observation.trial_id] = observation

        observed_lengths = {len(item.ic_series) for item in by_trial.values() if item.ic_series}
        if periods is None:
            if len(observed_lengths) != 1:
                raise ValueError("cannot infer one common IC-series length for the trial family")
            periods = next(iter(observed_lengths))
        if periods < 1:
            raise ValueError("periods must be >= 1")
        if any(length != periods for length in observed_lengths):
            raise ValueError("trial IC series do not match the requested period count")

        series_map: dict[str, tuple[float, ...]] = {}
        missing = 0
        for observation in by_trial.values():
            if observation.ic_series:
                series_map[observation.canonical_formula_hash] = tuple(
                    float(value) if value is not None else float("nan")
                    for value in observation.ic_series
                )
            else:
                missing += 1
                series_map[observation.canonical_formula_hash] = (0.0,) * periods

        effective_count: float | None = None
        effective_metadata: dict[str, Any] = {}
        if estimate_effective:
            from factorminer.evaluation.significance import estimate_effective_trials

            matrix = np.asarray(list(series_map.values()), dtype=np.float64)
            estimate = estimate_effective_trials(matrix)
            effective_count = estimate.effective_trials
            effective_metadata = asdict(estimate)

        active_campaign = self.campaign_id if campaign_id is None else campaign_id
        return TrialInferenceFamily(
            campaign_id=active_campaign,
            dataset_id=dataset_id,
            target_name=target_name,
            raw_trial_count=len(series_map),
            ic_series=series_map,
            effective_trial_count=effective_count,
            metadata={
                "schema_version": TRIAL_LEDGER_SCHEMA_VERSION,
                "observation_count": len(matching),
                "missing_series_filled_with_zero": missing,
                "effective_trials": effective_metadata,
            },
        )
