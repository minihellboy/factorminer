"""Run-manifest persistence and factor-provenance attachment services."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from factorminer.application.evidence_service import FactorEvidenceService
from factorminer.application.mining_budget import EvaluationResult
from factorminer.core.provenance import build_factor_provenance, build_run_manifest


class MiningArtifactService:
    """Own mutable artifact concerns for one mining-loop composition."""

    def __init__(self, loop: Any) -> None:
        self.loop = loop
        self.evidence = FactorEvidenceService(loop)

    def refresh_manifest(
        self,
        *,
        output_dir: str,
        artifact_paths: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Build and cache the current run manifest."""
        loop = self.loop
        if loop._session is None:
            return {}

        target_stack = list((loop.settings.target_panels or {}).keys())
        pipeline_targets = getattr(loop.pipeline, "target_panels", None) or {}
        if pipeline_targets:
            target_stack = [
                name for name in pipeline_targets if name and name != "paper"
            ] or target_stack

        existing_paths = dict(
            (loop._run_manifest or {}).get("artifact_paths", {})
        )
        existing_paths.update(artifact_paths or {})
        manifest = build_run_manifest(
            run_id=loop._session.session_id,
            session_id=loop._session.session_id,
            loop_type=loop._loop_type(),
            benchmark_mode=loop.settings.benchmark_mode,
            created_at=loop._session.start_time,
            updated_at=datetime.now().isoformat(),
            iteration=loop.iteration,
            library_size=loop.library.size,
            output_dir=output_dir,
            config_summary=loop._serialize_config(),
            dataset_summary={
                "data_tensor_shape": list(loop.data_tensor.shape),
                "returns_shape": list(loop.returns.shape),
                "memory_version": loop.memory.version,
                "library_size": loop.library.size,
                "library_diagnostics": loop.library.get_diagnostics(),
            },
            phase2_features=loop._phase2_features(),
            target_stack=target_stack,
            artifact_paths=existing_paths,
            notes=[],
        )
        loop._run_manifest = manifest.to_dict()
        return loop._run_manifest

    def persist_manifest(self, path: Path) -> None:
        """Write the current manifest and mirror it into session state."""
        loop = self.loop
        if loop._session is None:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        if not loop._run_manifest:
            self.refresh_manifest(
                output_dir=str(path.parent.parent),
                artifact_paths={"run_manifest": str(path)},
            )
        loop._run_manifest.setdefault("artifact_paths", {})["run_manifest"] = str(path)
        with open(path, "w") as file:
            json.dump(loop._run_manifest, file, indent=2, default=str)

        loop._session.run_manifest_path = str(path)
        loop._session.run_manifest = loop._run_manifest

    def attach_factor_provenance(
        self,
        admitted_results: list[EvaluationResult],
        *,
        library_state: dict[str, Any],
        memory_signal: dict[str, Any],
        phase2_summary: dict[str, Any],
        generator_family: str | None = None,
    ) -> None:
        """Stamp provenance onto factors that survived admission."""
        loop = self.loop
        if not admitted_results or loop._session is None:
            return

        run_manifest = loop._run_manifest or self.refresh_manifest(
            output_dir=loop.settings.output_dir,
            artifact_paths={},
        )
        for rank, result in enumerate(admitted_results, start=1):
            if not result.admitted:
                continue
            factor = next(
                (
                    candidate
                    for candidate in reversed(loop.library.list_factors())
                    if candidate.name == result.factor_name
                    and candidate.formula == result.formula
                ),
                None,
            )
            if factor is None:
                continue

            factor.provenance = build_factor_provenance(
                run_manifest=run_manifest,
                factor_name=factor.name,
                formula=factor.formula,
                factor_category=factor.category,
                factor_id=factor.id,
                iteration=loop.iteration,
                batch_number=factor.batch_number,
                candidate_rank=rank,
                generator_family=generator_family or loop._generator_family(),
                memory_signal=memory_signal,
                library_state=library_state,
                evaluation={
                    "ic_mean": factor.ic_mean,
                    "ic_paper_mean": factor.ic_paper_mean,
                    "ic_abs_mean": factor.ic_abs_mean,
                    "icir": factor.icir,
                    "ic_paper_icir": factor.ic_paper_icir,
                    "ic_win_rate": factor.ic_win_rate,
                    "max_correlation": factor.max_correlation,
                    "research_metrics": factor.research_metrics,
                },
                admission={
                    "admitted": True,
                    "stage_passed": result.stage_passed,
                    "replaced": result.replaced,
                    "correlated_with": result.correlated_with,
                    "rejection_reason": result.rejection_reason,
                },
                phase2=phase2_summary,
                target_stack=run_manifest.get("target_stack", []),
                research_metrics=factor.research_metrics,
                parent_formula=result.parent_formula,
                parent_ic_paper_mean=result.parent_ic_paper_mean,
                edit_type=result.edit_type,
                edit_motif=result.edit_motif,
                secondary_parent_formula=result.secondary_parent_formula,
                draft_rationale=True,
                llm_provider=getattr(loop, "llm", None)
                or getattr(loop.generator, "llm", None),
                use_llm_rationale=False,
            ).to_dict()
            pack = self.evidence.build_and_store(
                factor=factor,
                result=result,
                run_manifest=run_manifest,
                memory_signal=memory_signal,
                phase2_summary=phase2_summary,
                generator_family=generator_family or loop._generator_family(),
            )
            factor.evidence_ids = tuple(dict.fromkeys((*factor.evidence_ids, pack.evidence_id)))
            factor.provenance["evidence_ids"] = list(factor.evidence_ids)
            loop._run_manifest.setdefault("artifact_paths", {})["evidence_dir"] = str(
                self.evidence.store.root
            )
