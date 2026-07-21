"""Construction, persistence, and verification of factor evidence packs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from factorminer.application.mining_budget import EvaluationResult
from factorminer.core.expression_tree import ConstantNode, LeafNode, OperatorNode
from factorminer.core.parser import try_parse
from factorminer.core.provenance import stable_digest
from factorminer.domain.evidence import EvidencePack


class EvidenceStore:
    """Append-only content-addressed storage under one run directory."""

    def __init__(self, output_dir: str | Path) -> None:
        self.root = Path(output_dir) / "evidence"

    def put(self, pack: EvidencePack) -> Path:
        if not pack.verify():
            raise ValueError(f"Refusing invalid evidence pack {pack.evidence_id}")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{pack.evidence_id}.json"
        payload = pack.to_dict()
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise ValueError(f"Evidence ID collision at {path}")
            return path
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        return path

    def get(self, evidence_id: str) -> EvidencePack:
        path = self.root / f"{evidence_id}.json"
        return EvidencePack.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def verify_all(self) -> dict[str, Any]:
        checked = 0
        failures: list[str] = []
        if not self.root.exists():
            return {"ok": True, "checked": 0, "failures": []}
        for path in sorted(self.root.glob("*.json")):
            checked += 1
            try:
                pack = EvidencePack.from_dict(json.loads(path.read_text(encoding="utf-8")))
                if path.stem != pack.evidence_id:
                    raise ValueError("filename does not match evidence_id")
            except Exception as exc:  # noqa: BLE001 - verification reports all failures
                failures.append(f"{path.name}: {exc}")
        return {"ok": not failures, "checked": checked, "failures": failures}


class FactorEvidenceService:
    """Build complete evidence from live loop state and persist it once."""

    def __init__(self, loop: Any) -> None:
        self.loop = loop
        self.store = EvidenceStore(loop.settings.output_dir)
        self._code_hash: str | None = None

    def build_and_store(
        self,
        *,
        factor: Any,
        result: EvaluationResult,
        run_manifest: Mapping[str, Any],
        memory_signal: Mapping[str, Any],
        phase2_summary: Mapping[str, Any],
        generator_family: str,
    ) -> EvidencePack:
        lineage = {
            "parent_formula": result.parent_formula,
            "parent_ic_paper_mean": result.parent_ic_paper_mean,
            "edit_type": result.edit_type,
            "edit_motif": result.edit_motif,
            "secondary_parent_formula": result.secondary_parent_formula,
        }
        evaluation = {
            "paper": {
                "ic_mean": factor.ic_mean,
                "ic_paper_mean": factor.ic_paper_mean,
                "ic_abs_mean": factor.ic_abs_mean,
                "icir": factor.icir,
                "ic_paper_icir": factor.ic_paper_icir,
                "ic_win_rate": factor.ic_win_rate,
                "max_correlation": factor.max_correlation,
            },
            **dict(result.target_stats or {}),
        }
        admission = {
            "admitted": result.admitted,
            "stage_passed": result.stage_passed,
            "rejection_reason": result.rejection_reason,
            "correlated_with": result.correlated_with,
        }
        replacement = {
            "replaced_factor_id": result.replaced,
            "performed": result.replaced is not None,
        }
        research = dict(factor.research_metrics or {})
        provider = getattr(self.loop.generator, "llm_provider", None)
        model_identity = "/".join(
            value
            for value in (
                provider.__class__.__name__ if provider is not None else "",
                str(getattr(provider, "model", "") or ""),
            )
            if value
        )
        source_ids = _string_ids(memory_signal, "source_ids", "research_source_ids")
        hypothesis_ids = _string_ids(
            memory_signal,
            "hypothesis_ids",
            "research_hypothesis_ids",
        )
        config_hash = str(run_manifest.get("config_digest", "")) or stable_digest(
            self.loop._serialize_config()
        )
        dataset_hash = stable_digest(self.loop.dataset_contract.to_dict())
        content_hash = stable_digest(
            {
                "formula": factor.formula,
                "evaluation": evaluation,
                "admission": admission,
                "replacement": replacement,
            }
        )
        pack = EvidencePack.create(
            content_hash=content_hash,
            dataset_hash=dataset_hash,
            config_hash=config_hash,
            code_hash=self.code_hash(),
            source_ids=source_ids,
            hypothesis_ids=hypothesis_ids,
            factor_name=factor.name,
            formula=factor.formula,
            formula_ast=_formula_ast(factor.formula),
            lineage=lineage,
            split_metrics=evaluation,
            failure_evidence={
                "rejection_reason": result.rejection_reason,
                "parse_ok": result.parse_ok,
            },
            cost_results=_mapping_value(research, "cost", "cost_results"),
            capacity_results=_mapping_value(research, "capacity", "capacity_results"),
            regime_results=_mapping_value(research, "regime", "regime_results"),
            significance_results=_mapping_value(
                research,
                "significance",
                "significance_results",
            ),
            model_risk_results={
                **_mapping_value(research, "model_risk", "model_risk_results"),
                "phase2": dict(phase2_summary),
            },
            generator_identity=generator_family,
            model_identity=model_identity,
            prompt_identity=stable_digest(
                {
                    "memory_signal": dict(memory_signal),
                    "generator": generator_family,
                }
            ),
            admission_decision=admission,
            replacement_decision=replacement,
            approval_state="unreviewed",
        )
        self.store.put(pack)
        return pack

    def code_hash(self) -> str:
        if self._code_hash is not None:
            return self._code_hash
        package_root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for path in sorted(package_root.rglob("*.py")):
            if "tests" in path.parts:
                continue
            digest.update(str(path.relative_to(package_root)).encode("utf-8"))
            digest.update(path.read_bytes())
        self._code_hash = digest.hexdigest()
        return self._code_hash


def _formula_ast(formula: str) -> dict[str, Any]:
    tree = try_parse(formula)
    if tree is None:
        return {"type": "parse_failure", "formula": formula}

    def serialize(node) -> dict[str, Any]:
        if isinstance(node, LeafNode):
            return {"type": "leaf", "feature": node.feature_name}
        if isinstance(node, ConstantNode):
            return {"type": "constant", "value": node.value}
        if isinstance(node, OperatorNode):
            return {
                "type": "operator",
                "name": node.operator.name,
                "params": dict(node.params),
                "children": [serialize(child) for child in node.children],
            }
        return {"type": type(node).__name__}

    return serialize(tree.root)


def _mapping_value(payload: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _string_ids(payload: Mapping[str, Any], *keys: str) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        raw = payload.get(key, ())
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, Sequence):
            values.extend(str(item) for item in raw if item)
    return tuple(sorted(set(values)))
