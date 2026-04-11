"""Data structures for the FactorMiner experience memory system.

Implements the experience memory M = {S, P_succ, P_fail, I} where:
- S: Mining state tracking global evolution of the factor library
- P_succ: Success patterns (recommended mining directions)
- P_fail: Forbidden directions (directions to avoid)
- I: Strategic insights (high-level lessons)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class MiningState:
    """Tracks the global evolution of the factor library (S).

    Captures a snapshot of the current library status including size,
    recent admission/rejection history, and per-category saturation.
    """

    library_size: int = 0
    recent_admissions: list[dict] = field(default_factory=list)
    recent_rejections: list[dict] = field(default_factory=list)
    domain_saturation: dict[str, float] = field(default_factory=dict)
    admission_log: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MiningState:
        return cls(
            library_size=d.get("library_size", 0),
            recent_admissions=d.get("recent_admissions", []),
            recent_rejections=d.get("recent_rejections", []),
            domain_saturation=d.get("domain_saturation", {}),
            admission_log=d.get("admission_log", []),
        )


@dataclass
class SuccessPattern:
    """A recommended mining direction (P_succ).

    Encodes a known-effective pattern for factor construction, including
    a canonical formula template and tracked success rate.
    """

    name: str
    description: str
    template: str
    success_rate: str  # "High", "Medium", "Low"
    example_factors: list[str] = field(default_factory=list)
    occurrence_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SuccessPattern:
        return cls(
            name=d["name"],
            description=d["description"],
            template=d["template"],
            success_rate=d.get("success_rate", "Medium"),
            example_factors=d.get("example_factors", []),
            occurrence_count=d.get("occurrence_count", 0),
        )


@dataclass
class ForbiddenDirection:
    """A forbidden mining direction (P_fail).

    Encodes a pattern that consistently produces factors too correlated
    with existing library members or that fail quality thresholds.
    """

    name: str
    description: str
    correlated_factors: list[str] = field(default_factory=list)
    typical_correlation: float = 0.0
    reason: str = ""
    occurrence_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ForbiddenDirection:
        return cls(
            name=d["name"],
            description=d["description"],
            correlated_factors=d.get("correlated_factors", []),
            typical_correlation=d.get("typical_correlation", 0.0),
            reason=d.get("reason", ""),
            occurrence_count=d.get("occurrence_count", 0),
        )


@dataclass
class StrategicInsight:
    """High-level lesson from mining (I).

    Captures abstract observations about what works and what doesn't,
    derived from accumulated mining experience across batches.
    """

    insight: str
    evidence: str
    batch_source: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> StrategicInsight:
        return cls(
            insight=d["insight"],
            evidence=d["evidence"],
            batch_source=d.get("batch_source", 0),
        )


@dataclass
class ExperienceMemory:
    """The complete experience memory M = {S, P_succ, P_fail, I}.

    Persists across mining sessions and evolves with each batch of
    evaluated factor candidates.
    """

    state: MiningState = field(default_factory=MiningState)
    success_patterns: list[SuccessPattern] = field(default_factory=list)
    forbidden_directions: list[ForbiddenDirection] = field(default_factory=list)
    insights: list[StrategicInsight] = field(default_factory=list)
    version: int = 0

    def to_dict(self) -> dict:
        return {
            "state": self.state.to_dict(),
            "success_patterns": [p.to_dict() for p in self.success_patterns],
            "forbidden_directions": [f.to_dict() for f in self.forbidden_directions],
            "insights": [i.to_dict() for i in self.insights],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExperienceMemory:
        return cls(
            state=MiningState.from_dict(d.get("state", {})),
            success_patterns=[
                SuccessPattern.from_dict(p) for p in d.get("success_patterns", [])
            ],
            forbidden_directions=[
                ForbiddenDirection.from_dict(f) for f in d.get("forbidden_directions", [])
            ],
            insights=[
                StrategicInsight.from_dict(i) for i in d.get("insights", [])
            ],
            version=d.get("version", 0),
        )
