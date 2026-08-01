"""Inference contracts for complete, canonical trial families."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrialInferenceFamily:
    """One deduplicated family of every candidate that contacted data.

    ``ic_series`` is keyed by canonical formula hash, not by a mutable factor
    name.  Failed evaluations remain in the mapping as zero series so that an
    inference consumer cannot accidentally shrink the family to survivors.
    """

    campaign_id: str
    dataset_id: str
    target_name: str
    raw_trial_count: int
    ic_series: dict[str, tuple[float, ...]] = field(default_factory=dict)
    effective_trial_count: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise ValueError("campaign_id must not be empty")
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty")
        if not self.target_name:
            raise ValueError("target_name must not be empty")
        if self.raw_trial_count < 1:
            raise ValueError("raw_trial_count must be >= 1")
        if len(self.ic_series) != self.raw_trial_count:
            raise ValueError("ic_series must contain exactly one entry per canonical trial")
        lengths = {len(values) for values in self.ic_series.values()}
        if not lengths or 0 in lengths or len(lengths) != 1:
            raise ValueError("all trial IC series must have one common, non-zero length")
        if self.effective_trial_count is not None and not (
            1.0 <= self.effective_trial_count <= float(self.raw_trial_count)
        ):
            raise ValueError("effective_trial_count must be between 1 and raw_trial_count")

    @property
    def periods(self) -> int:
        """Number of periods in every family member's IC series."""
        return len(next(iter(self.ic_series.values())))

    @property
    def dsr_trial_count(self) -> float:
        """Correlation-adjusted count when available, otherwise the raw count."""
        return float(self.effective_trial_count or self.raw_trial_count)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """Return defensive float64 arrays suitable for inference services."""
        return {
            key: np.asarray(values, dtype=np.float64).copy()
            for key, values in self.ic_series.items()
        }
