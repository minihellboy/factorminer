"""Immutable, content-addressed evidence contracts for admitted factors."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

type JSONScalar = None | bool | int | float | str
type FrozenValue = JSONScalar | tuple["FrozenValue", ...] | FrozenPayload


@dataclass(frozen=True, slots=True)
class FrozenPayload:
    """Recursively immutable representation of a JSON object."""

    items: tuple[tuple[str, FrozenValue], ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> FrozenPayload:
        return cls(
            tuple(
                sorted(
                    (str(key), freeze_value(item))
                    for key, item in (value or {}).items()
                )
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: thaw_value(value) for key, value in self.items}


def freeze_value(value: Any) -> FrozenValue:
    """Normalize a JSON-shaped value into an immutable representation."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return FrozenPayload.from_mapping(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(freeze_value(item) for item in value)
    if hasattr(value, "tolist"):
        return freeze_value(value.tolist())
    if hasattr(value, "item"):
        return freeze_value(value.item())
    return str(value)


def thaw_value(value: FrozenValue) -> Any:
    if isinstance(value, FrozenPayload):
        return value.to_dict()
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class HumanAttestation:
    """Explicit human review state; generation leaves this unattested."""

    attested: bool = False
    attestor: str = ""
    attested_at: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "attested": self.attested,
            "attestor": self.attestor,
            "attested_at": self.attested_at,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class EvidencePack:
    """Versioned evidence for one evaluated factor and admission decision."""

    evidence_id: str
    schema_version: str
    created_at: str
    content_hash: str
    dataset_hash: str
    config_hash: str
    code_hash: str
    source_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    factor_name: str
    formula: str
    formula_ast: FrozenPayload
    lineage: FrozenPayload
    split_metrics: FrozenPayload
    failure_evidence: FrozenPayload
    cost_results: FrozenPayload
    capacity_results: FrozenPayload
    regime_results: FrozenPayload
    significance_results: FrozenPayload
    model_risk_results: FrozenPayload
    generator_identity: str
    model_identity: str
    prompt_identity: str
    admission_decision: FrozenPayload
    replacement_decision: FrozenPayload
    human_attestation: HumanAttestation = field(default_factory=HumanAttestation)
    approval_state: str = "unreviewed"

    @classmethod
    def create(
        cls,
        *,
        content_hash: str,
        dataset_hash: str,
        config_hash: str,
        code_hash: str,
        factor_name: str,
        formula: str,
        formula_ast: Mapping[str, Any],
        lineage: Mapping[str, Any],
        split_metrics: Mapping[str, Any],
        failure_evidence: Mapping[str, Any] | None = None,
        cost_results: Mapping[str, Any] | None = None,
        capacity_results: Mapping[str, Any] | None = None,
        regime_results: Mapping[str, Any] | None = None,
        significance_results: Mapping[str, Any] | None = None,
        model_risk_results: Mapping[str, Any] | None = None,
        generator_identity: str = "",
        model_identity: str = "",
        prompt_identity: str = "",
        admission_decision: Mapping[str, Any] | None = None,
        replacement_decision: Mapping[str, Any] | None = None,
        source_ids: Sequence[str] = (),
        hypothesis_ids: Sequence[str] = (),
        human_attestation: HumanAttestation | None = None,
        approval_state: str = "unreviewed",
        created_at: str | None = None,
        schema_version: str = "1.0",
    ) -> EvidencePack:
        created_at = created_at or datetime.now(UTC).isoformat()
        values = {
            "schema_version": schema_version,
            "created_at": created_at,
            "content_hash": content_hash,
            "dataset_hash": dataset_hash,
            "config_hash": config_hash,
            "code_hash": code_hash,
            "source_ids": sorted(set(source_ids)),
            "hypothesis_ids": sorted(set(hypothesis_ids)),
            "factor_name": factor_name,
            "formula": formula,
            "formula_ast": dict(formula_ast),
            "lineage": dict(lineage),
            "split_metrics": dict(split_metrics),
            "failure_evidence": dict(failure_evidence or {}),
            "cost_results": dict(cost_results or {}),
            "capacity_results": dict(capacity_results or {}),
            "regime_results": dict(regime_results or {}),
            "significance_results": dict(significance_results or {}),
            "model_risk_results": dict(model_risk_results or {}),
            "generator_identity": generator_identity,
            "model_identity": model_identity,
            "prompt_identity": prompt_identity,
            "admission_decision": dict(admission_decision or {}),
            "replacement_decision": dict(replacement_decision or {}),
            "human_attestation": (human_attestation or HumanAttestation()).to_dict(),
            "approval_state": approval_state,
        }
        normalized = FrozenPayload.from_mapping(values).to_dict()
        evidence_id = _evidence_digest(normalized)
        return cls.from_dict({"evidence_id": evidence_id, **normalized}, verify=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, verify: bool = True) -> EvidencePack:
        attestation = payload.get("human_attestation") or {}
        pack = cls(
            evidence_id=str(payload["evidence_id"]),
            schema_version=str(payload.get("schema_version", "1.0")),
            created_at=str(payload.get("created_at", "")),
            content_hash=str(payload.get("content_hash", "")),
            dataset_hash=str(payload.get("dataset_hash", "")),
            config_hash=str(payload.get("config_hash", "")),
            code_hash=str(payload.get("code_hash", "")),
            source_ids=tuple(str(item) for item in payload.get("source_ids", ())),
            hypothesis_ids=tuple(str(item) for item in payload.get("hypothesis_ids", ())),
            factor_name=str(payload.get("factor_name", "")),
            formula=str(payload.get("formula", "")),
            formula_ast=FrozenPayload.from_mapping(payload.get("formula_ast")),
            lineage=FrozenPayload.from_mapping(payload.get("lineage")),
            split_metrics=FrozenPayload.from_mapping(payload.get("split_metrics")),
            failure_evidence=FrozenPayload.from_mapping(payload.get("failure_evidence")),
            cost_results=FrozenPayload.from_mapping(payload.get("cost_results")),
            capacity_results=FrozenPayload.from_mapping(payload.get("capacity_results")),
            regime_results=FrozenPayload.from_mapping(payload.get("regime_results")),
            significance_results=FrozenPayload.from_mapping(payload.get("significance_results")),
            model_risk_results=FrozenPayload.from_mapping(payload.get("model_risk_results")),
            generator_identity=str(payload.get("generator_identity", "")),
            model_identity=str(payload.get("model_identity", "")),
            prompt_identity=str(payload.get("prompt_identity", "")),
            admission_decision=FrozenPayload.from_mapping(payload.get("admission_decision")),
            replacement_decision=FrozenPayload.from_mapping(payload.get("replacement_decision")),
            human_attestation=HumanAttestation(
                attested=bool(attestation.get("attested", False)),
                attestor=str(attestation.get("attestor", "")),
                attested_at=str(attestation.get("attested_at", "")),
                note=str(attestation.get("note", "")),
            ),
            approval_state=str(payload.get("approval_state", "unreviewed")),
        )
        if verify and not pack.verify():
            raise ValueError(f"Evidence content hash mismatch: {pack.evidence_id}")
        return pack

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
            "dataset_hash": self.dataset_hash,
            "config_hash": self.config_hash,
            "code_hash": self.code_hash,
            "source_ids": list(self.source_ids),
            "hypothesis_ids": list(self.hypothesis_ids),
            "factor_name": self.factor_name,
            "formula": self.formula,
            "formula_ast": self.formula_ast.to_dict(),
            "lineage": self.lineage.to_dict(),
            "split_metrics": self.split_metrics.to_dict(),
            "failure_evidence": self.failure_evidence.to_dict(),
            "cost_results": self.cost_results.to_dict(),
            "capacity_results": self.capacity_results.to_dict(),
            "regime_results": self.regime_results.to_dict(),
            "significance_results": self.significance_results.to_dict(),
            "model_risk_results": self.model_risk_results.to_dict(),
            "generator_identity": self.generator_identity,
            "model_identity": self.model_identity,
            "prompt_identity": self.prompt_identity,
            "admission_decision": self.admission_decision.to_dict(),
            "replacement_decision": self.replacement_decision.to_dict(),
            "human_attestation": self.human_attestation.to_dict(),
            "approval_state": self.approval_state,
        }

    def content_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("evidence_id", None)
        return payload

    def verify(self) -> bool:
        return self.evidence_id == _evidence_digest(self.content_dict())


def _evidence_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
