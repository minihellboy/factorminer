"""Model Risk Management (MRM) validation pack assembler.

Composes existing FactorMiner evidence (DSR/PBO/FDR, CPCV, causal, decay,
lineage, economic rationale) into an examiner-shaped artifact mapped to the
*themes* of Fed/OCC SR 26-2 (2026-04-17).

This module produces **evidence for a qualified human reviewer**. It does NOT
determine compliance, issue a pass/fail regulatory verdict, or claim that any
factor library is "SR 26-2 compliant."
"""

from __future__ import annotations

import html
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Explicit disclaimer embedded in every rendered pack.
MRM_DISCLAIMER = (
    "This pack assembles research evidence for a qualified reviewer. "
    "It is not a compliance determination, does not claim SR 26-2 / SR 11-7 / "
    "SEC AI compliance, and must not be presented as a regulatory certification."
)

UNATTESTED_RATIONALE_BANNER = "UNATTESTED -- LLM DRAFT, NOT REVIEWED"


@dataclass(frozen=True)
class MrmPackConfig:
    """Feature-local config for MRM pack assembly."""

    owner: str = "research"
    default_risk_tier: str = "research-prototype"
    intended_use: str = (
        "Offline formulaic alpha research and library construction. "
        "Not for automated trade execution or client portfolio management."
    )
    purpose: str = (
        "Discover and document interpretable cross-sectional equity/crypto "
        "factors expressed in a typed DSL for human review."
    )
    limitations: str = (
        "Historical IC/backtest evidence is in-sample/out-of-sample research "
        "only; no production monitoring SLA; LLM-drafted rationales are unattested "
        "until a human reviewer marks them attested."
    )
    include_empty_sections: bool = True


@dataclass
class ModelInventoryRow:
    """One model-inventory entry for a factor (examiner checklist shape)."""

    factor_id: str
    name: str
    formula: str
    purpose: str
    limitations: str
    intended_use: str
    owner: str
    risk_tier: str
    parent_formula: str = ""
    edit_type: str = ""
    economic_rationale_attested: bool = False
    category: str = ""
    ic_paper_mean: float | None = None
    ic_paper_icir: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MrmPack:
    """Examiner-shaped MRM evidence pack."""

    generated_at: str
    disclaimer: str
    library_source: str
    inventory: list[ModelInventoryRow] = field(default_factory=list)
    conceptual_soundness: dict[str, Any] = field(default_factory=dict)
    outcomes_analysis: dict[str, Any] = field(default_factory=dict)
    ongoing_monitoring: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "disclaimer": self.disclaimer,
            "library_source": self.library_source,
            "inventory": [row.to_dict() for row in self.inventory],
            "conceptual_soundness": dict(self.conceptual_soundness),
            "outcomes_analysis": dict(self.outcomes_analysis),
            "ongoing_monitoring": dict(self.ongoing_monitoring),
            "notes": list(self.notes),
        }


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _normalize_sequence_or_mapping(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    label: str,
) -> dict[str, Any]:
    if value is None:
        return {"status": "not_supplied"}
    if isinstance(value, Mapping):
        return dict(value)
    rows = [dict(item) for item in value if isinstance(item, Mapping)]
    return {"status": "supplied", "label": label, "rows": rows, "count": len(rows)}


def _factor_list(library: Any) -> list[dict[str, Any]]:
    """Normalize library inputs (FactorLibrary, dict payload, or sequence)."""
    if library is None:
        return []
    if hasattr(library, "list_factors"):
        factors = []
        for factor in library.list_factors():
            if hasattr(factor, "to_dict"):
                factors.append(factor.to_dict())
            elif isinstance(factor, Mapping):
                factors.append(dict(factor))
            else:
                factors.append(
                    {
                        "id": getattr(factor, "id", ""),
                        "name": getattr(factor, "name", ""),
                        "formula": getattr(factor, "formula", ""),
                        "category": getattr(factor, "category", ""),
                        "ic_paper_mean": getattr(factor, "ic_paper_mean", None),
                        "ic_paper_icir": getattr(factor, "ic_paper_icir", None),
                        "provenance": getattr(factor, "provenance", {}) or {},
                    }
                )
        return factors
    if isinstance(library, Mapping):
        raw = library.get("factors", [])
        return [dict(f) for f in raw if isinstance(f, Mapping)]
    if isinstance(library, Sequence) and not isinstance(library, (str, bytes)):
        out = []
        for item in library:
            if isinstance(item, Mapping):
                out.append(dict(item))
            elif hasattr(item, "to_dict"):
                out.append(item.to_dict())
        return out
    return []


def _provenance_of(factor: Mapping[str, Any]) -> dict[str, Any]:
    prov = factor.get("provenance") or {}
    return _as_mapping(prov)


def _rationale_of(factor: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    for source in (
        factor.get("economic_rationale"),
        provenance.get("economic_rationale"),
    ):
        if isinstance(source, Mapping) and source:
            return dict(source)
    return {}


def build_mrm_pack(
    library: Any,
    *,
    config: MrmPackConfig | None = None,
    significance_summary: Mapping[str, Any] | None = None,
    pbo_summary: Mapping[str, Any] | None = None,
    cpcv_summary: Mapping[str, Any] | None = None,
    causal_summary: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    decay_summary: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    library_source: str = "factor_library",
    extra_notes: Sequence[str] | None = None,
) -> MrmPack:
    """Assemble an MRM evidence pack from a factor library and optional metrics."""
    cfg = config or MrmPackConfig()
    factors = _factor_list(library)
    inventory: list[ModelInventoryRow] = []
    rationale_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []

    for factor in factors:
        prov = _provenance_of(factor)
        rationale = _rationale_of(factor, prov)
        attested = bool(rationale.get("attested", False)) and rationale.get("source") == "human"
        parent_formula = str(
            prov.get("parent_formula") or factor.get("parent_formula") or ""
        )
        edit_type = str(prov.get("edit_type") or factor.get("edit_type") or "")
        fid = str(factor.get("id", factor.get("factor_id", "")))
        name = str(factor.get("name", "") or f"factor_{fid}")
        formula = str(factor.get("formula", "") or "")
        inventory.append(
            ModelInventoryRow(
                factor_id=fid,
                name=name,
                formula=formula,
                purpose=cfg.purpose,
                limitations=cfg.limitations,
                intended_use=cfg.intended_use,
                owner=cfg.owner,
                risk_tier=cfg.default_risk_tier,
                parent_formula=parent_formula,
                edit_type=edit_type,
                economic_rationale_attested=attested,
                category=str(factor.get("category", "") or ""),
                ic_paper_mean=_maybe_float(factor.get("ic_paper_mean", factor.get("ic_mean"))),
                ic_paper_icir=_maybe_float(factor.get("ic_paper_icir", factor.get("icir"))),
            )
        )
        rationale_rows.append(
            {
                "factor_id": fid,
                "name": name,
                "attested": attested,
                "source": rationale.get("source", ""),
                "mathematical_structure": rationale.get("mathematical_structure", ""),
                "financial_semantics": rationale.get("financial_semantics", ""),
                "market_logic": rationale.get("market_logic", ""),
                "banner": "" if attested else UNATTESTED_RATIONALE_BANNER,
            }
        )
        lineage_rows.append(
            {
                "factor_id": fid,
                "name": name,
                "formula": formula,
                "parent_formula": parent_formula,
                "edit_type": edit_type,
                "edit_motif": prov.get("edit_motif", ""),
                "parent_ic_paper_mean": prov.get("parent_ic_paper_mean"),
                "iteration": prov.get("iteration"),
                "generator_family": prov.get("generator_family", ""),
            }
        )

    conceptual_soundness = {
        "title": "Conceptual soundness (evidence)",
        "summary": (
            "Developmental history (parent_formula lineage), structured economic "
            "rationale triples, and human attestation status. LLM-drafted text is "
            "evidence of a hypothesis only until attested by a human reviewer."
        ),
        "lineage": lineage_rows,
        "economic_rationales": rationale_rows,
        "attested_count": sum(1 for row in rationale_rows if row["attested"]),
        "unattested_count": sum(1 for row in rationale_rows if not row["attested"]),
    }

    causal_block = _normalize_sequence_or_mapping(causal_summary, label="causal")
    outcomes_analysis = {
        "title": "Outcomes analysis (evidence)",
        "summary": (
            "Statistical rigor artifacts assembled from significance testing "
            "(bootstrap/DSR/FDR), combinatorial purged CV / PBO, and optional "
            "causal validation results when supplied by the caller."
        ),
        "significance": _as_mapping(significance_summary)
        if significance_summary is not None
        else {"status": "not_supplied"},
        "pbo": _as_mapping(pbo_summary) if pbo_summary is not None else {"status": "not_supplied"},
        "cpcv": _as_mapping(cpcv_summary)
        if cpcv_summary is not None
        else {"status": "not_supplied"},
        "causal": causal_block,
        "library_metrics": {
            "factor_count": len(inventory),
            "mean_ic_paper_mean": _mean(
                [row.ic_paper_mean for row in inventory if row.ic_paper_mean is not None]
            ),
            "mean_ic_paper_icir": _mean(
                [row.ic_paper_icir for row in inventory if row.ic_paper_icir is not None]
            ),
        },
    }

    decay_block = _normalize_sequence_or_mapping(decay_summary, label="decay")
    ongoing_monitoring = {
        "title": "Ongoing monitoring (evidence)",
        "summary": (
            "Alpha-decay / half-life classifications and attestation gaps that a "
            "reviewer should re-check on a scheduled basis. This is a research "
            "monitoring checklist, not a production control."
        ),
        "decay": decay_block,
        "attestation_gaps": [
            {"factor_id": row["factor_id"], "name": row["name"]}
            for row in rationale_rows
            if not row["attested"]
        ],
        "lineage_coverage": {
            "with_parent": sum(1 for row in lineage_rows if row.get("parent_formula")),
            "fresh_or_unknown": sum(
                1
                for row in lineage_rows
                if not row.get("parent_formula")
                or row.get("edit_type") in {"fresh", "unknown", ""}
            ),
            "total": len(lineage_rows),
        },
    }

    notes = [
        MRM_DISCLAIMER,
        "Inventory risk_tier defaults to research-prototype; override per firm policy.",
    ]
    if extra_notes:
        notes.extend(str(n) for n in extra_notes)

    return MrmPack(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        disclaimer=MRM_DISCLAIMER,
        library_source=library_source,
        inventory=inventory,
        conceptual_soundness=conceptual_soundness,
        outcomes_analysis=outcomes_analysis,
        ongoing_monitoring=ongoing_monitoring,
        notes=notes,
    )


def _brief_json(value: Any, *, limit: int = 4000) -> str:
    try:
        text = json.dumps(value, indent=2, sort_keys=True, default=str)
    except TypeError:
        text = str(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def render_mrm_pack_markdown(pack: MrmPack | Mapping[str, Any]) -> str:
    """Render an MRM pack as Markdown (evidence packaging only)."""
    data = pack.to_dict() if isinstance(pack, MrmPack) else dict(pack)
    lines: list[str] = [
        "# MRM Validation Pack (Evidence for Reviewer)",
        f"Generated at: {data.get('generated_at', '-')}",
        f"Library source: `{data.get('library_source', '-')}`",
        "",
        f"> **Disclaimer:** {data.get('disclaimer', MRM_DISCLAIMER)}",
        "",
        "## Model Inventory",
    ]
    inventory = data.get("inventory") or []
    if not inventory:
        lines.append("_No factors in inventory._")
    else:
        lines.append(
            "| ID | Name | Formula | Owner | Risk tier | Parent | Edit | Rationale attested |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in inventory:
            lines.append(
                "| {id} | {name} | `{formula}` | {owner} | {tier} | `{parent}` | {edit} | {att} |".format(
                    id=row.get("factor_id", "-"),
                    name=row.get("name", "-"),
                    formula=row.get("formula", "-"),
                    owner=row.get("owner", "-"),
                    tier=row.get("risk_tier", "-"),
                    parent=row.get("parent_formula", "") or "-",
                    edit=row.get("edit_type", "") or "-",
                    att="yes" if row.get("economic_rationale_attested") else "no",
                )
            )
        lines.append("")
        for row in inventory:
            lines.append(f"### Inventory detail: {row.get('name', row.get('factor_id', '-'))}")
            lines.append(f"- **Purpose:** {row.get('purpose', '')}")
            lines.append(f"- **Limitations:** {row.get('limitations', '')}")
            lines.append(f"- **Intended use:** {row.get('intended_use', '')}")
            lines.append("")

    sound = data.get("conceptual_soundness") or {}
    lines.extend(["", "## Conceptual Soundness", sound.get("summary", ""), ""])
    lines.append(
        f"- Attested rationales: `{sound.get('attested_count', 0)}`  \n"
        f"- Unattested rationales: `{sound.get('unattested_count', 0)}`"
    )
    for row in sound.get("economic_rationales") or []:
        lines.append("")
        lines.append(f"### Rationale: {row.get('name', row.get('factor_id', '-'))}")
        if not row.get("attested"):
            lines.append(f"**{UNATTESTED_RATIONALE_BANNER}**")
        lines.append(f"- Mathematical structure: {row.get('mathematical_structure', '')}")
        lines.append(f"- Financial semantics: {row.get('financial_semantics', '')}")
        lines.append(f"- Market logic: {row.get('market_logic', '')}")

    lines.append("")
    lines.append("### Developmental history (lineage)")
    lineage = sound.get("lineage") or []
    if not lineage:
        lines.append("_No lineage rows._")
    else:
        lines.append("| Factor | Parent formula | Edit type | Motif |")
        lines.append("| --- | --- | --- | --- |")
        for row in lineage:
            lines.append(
                f"| {row.get('name', '-')} | `{row.get('parent_formula', '') or '-'}` | "
                f"{row.get('edit_type', '-') or '-'} | {row.get('edit_motif', '-') or '-'} |"
            )

    outcomes = data.get("outcomes_analysis") or {}
    lines.extend(["", "## Outcomes Analysis", outcomes.get("summary", ""), ""])
    for key in ("significance", "pbo", "cpcv", "causal", "library_metrics"):
        block = outcomes.get(key, {})
        lines.append(f"### {key}")
        lines.append(f"```json\n{_brief_json(block)}\n```")
        lines.append("")

    monitoring = data.get("ongoing_monitoring") or {}
    lines.extend(["", "## Ongoing Monitoring", monitoring.get("summary", ""), ""])
    lines.append(f"### decay\n```json\n{_brief_json(monitoring.get('decay', {}))}\n```")
    lines.append("")
    lines.append(
        f"### lineage coverage\n```json\n{_brief_json(monitoring.get('lineage_coverage', {}))}\n```"
    )
    gaps = monitoring.get("attestation_gaps") or []
    lines.append("")
    lines.append("### Attestation gaps")
    if not gaps:
        lines.append("_None._")
    else:
        for gap in gaps:
            lines.append(f"- {gap.get('name', gap.get('factor_id', '-'))}")

    notes = data.get("notes") or []
    if notes:
        lines.extend(["", "## Notes"])
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines).rstrip() + "\n"


def render_mrm_pack_html(pack: MrmPack | Mapping[str, Any]) -> str:
    """Render an MRM pack as HTML with mandatory escaping of free text."""
    data = pack.to_dict() if isinstance(pack, MrmPack) else dict(pack)

    def esc(value: Any) -> str:
        return html.escape(str(value if value is not None else ""))

    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>MRM Validation Pack (Evidence)</title>",
        "<style>",
        "body{font-family:Arial,Helvetica,sans-serif;margin:32px;max-width:1200px;color:#111}",
        "section{margin:24px 0;padding:16px;border:1px solid #e5e7eb;border-radius:12px;background:#fff}",
        "table{border-collapse:collapse;width:100%;font-size:14px}",
        "th,td{border:1px solid #d1d5db;padding:8px;text-align:left;vertical-align:top}",
        "th{background:#f3f4f6}",
        ".disclaimer{background:#fff7ed;border:1px solid #fdba74;padding:12px;border-radius:8px}",
        ".banner{background:#fef2f2;border:1px solid #f87171;color:#7f1d1d;"
        "padding:10px;border-radius:8px;font-weight:700;margin:8px 0}",
        ".muted{color:#6b7280}",
        "pre{white-space:pre-wrap;background:#f9fafb;padding:10px;border-radius:8px}",
        "</style></head><body>",
        "<h1>MRM Validation Pack (Evidence for Reviewer)</h1>",
        f'<p class="muted">Generated at {esc(data.get("generated_at", "-"))}</p>',
        f'<p class="muted">Library source: <code>{esc(data.get("library_source", "-"))}</code></p>',
        f'<div class="disclaimer"><strong>Disclaimer:</strong> {esc(data.get("disclaimer", MRM_DISCLAIMER))}</div>',
        "<section><h2>Model Inventory</h2>",
    ]

    inventory = data.get("inventory") or []
    if not inventory:
        parts.append('<p class="muted">No factors in inventory.</p>')
    else:
        parts.append(
            "<table><thead><tr>"
            "<th>ID</th><th>Name</th><th>Formula</th><th>Owner</th>"
            "<th>Risk tier</th><th>Parent</th><th>Edit</th><th>Attested</th>"
            "</tr></thead><tbody>"
        )
        for row in inventory:
            parts.append(
                "<tr>"
                f"<td>{esc(row.get('factor_id'))}</td>"
                f"<td>{esc(row.get('name'))}</td>"
                f"<td><code>{esc(row.get('formula'))}</code></td>"
                f"<td>{esc(row.get('owner'))}</td>"
                f"<td>{esc(row.get('risk_tier'))}</td>"
                f"<td><code>{esc(row.get('parent_formula') or '-')}</code></td>"
                f"<td>{esc(row.get('edit_type') or '-')}</td>"
                f"<td>{'yes' if row.get('economic_rationale_attested') else 'no'}</td>"
                "</tr>"
            )
            parts.append(
                "<tr><td colspan='8'>"
                f"<div><strong>Purpose:</strong> {esc(row.get('purpose'))}</div>"
                f"<div><strong>Limitations:</strong> {esc(row.get('limitations'))}</div>"
                f"<div><strong>Intended use:</strong> {esc(row.get('intended_use'))}</div>"
                "</td></tr>"
            )
        parts.append("</tbody></table>")
    parts.append("</section>")

    sound = data.get("conceptual_soundness") or {}
    parts.extend(
        [
            "<section><h2>Conceptual Soundness</h2>",
            f"<p>{esc(sound.get('summary', ''))}</p>",
            f"<p>Attested: <strong>{esc(sound.get('attested_count', 0))}</strong> · "
            f"Unattested: <strong>{esc(sound.get('unattested_count', 0))}</strong></p>",
        ]
    )
    for row in sound.get("economic_rationales") or []:
        parts.append(f"<h3>Rationale: {esc(row.get('name', row.get('factor_id', '-')))}</h3>")
        if not row.get("attested"):
            parts.append(f'<div class="banner">{esc(UNATTESTED_RATIONALE_BANNER)}</div>')
        parts.append(
            f"<p><strong>Mathematical structure:</strong> {esc(row.get('mathematical_structure', ''))}</p>"
        )
        parts.append(
            f"<p><strong>Financial semantics:</strong> {esc(row.get('financial_semantics', ''))}</p>"
        )
        parts.append(
            f"<p><strong>Market logic:</strong> {esc(row.get('market_logic', ''))}</p>"
        )

    parts.append("<h3>Developmental history (lineage)</h3>")
    lineage = sound.get("lineage") or []
    if not lineage:
        parts.append('<p class="muted">No lineage rows.</p>')
    else:
        parts.append(
            "<table><thead><tr><th>Factor</th><th>Parent</th><th>Edit</th><th>Motif</th>"
            "</tr></thead><tbody>"
        )
        for row in lineage:
            parts.append(
                "<tr>"
                f"<td>{esc(row.get('name'))}</td>"
                f"<td><code>{esc(row.get('parent_formula') or '-')}</code></td>"
                f"<td>{esc(row.get('edit_type') or '-')}</td>"
                f"<td>{esc(row.get('edit_motif') or '-')}</td>"
                "</tr>"
            )
        parts.append("</tbody></table>")
    parts.append("</section>")

    outcomes = data.get("outcomes_analysis") or {}
    parts.extend(
        [
            "<section><h2>Outcomes Analysis</h2>",
            f"<p>{esc(outcomes.get('summary', ''))}</p>",
        ]
    )
    for key in ("significance", "pbo", "cpcv", "causal", "library_metrics"):
        parts.append(f"<h3>{esc(key)}</h3>")
        parts.append(f"<pre>{esc(_brief_json(outcomes.get(key, {})))}</pre>")
    parts.append("</section>")

    monitoring = data.get("ongoing_monitoring") or {}
    parts.extend(
        [
            "<section><h2>Ongoing Monitoring</h2>",
            f"<p>{esc(monitoring.get('summary', ''))}</p>",
            "<h3>decay</h3>",
            f"<pre>{esc(_brief_json(monitoring.get('decay', {})))}</pre>",
            "<h3>lineage coverage</h3>",
            f"<pre>{esc(_brief_json(monitoring.get('lineage_coverage', {})))}</pre>",
            "<h3>Attestation gaps</h3>",
            "<ul>",
        ]
    )
    gaps = monitoring.get("attestation_gaps") or []
    if not gaps:
        parts.append("<li class='muted'>None.</li>")
    else:
        for gap in gaps:
            parts.append(f"<li>{esc(gap.get('name', gap.get('factor_id', '-')))}</li>")
    parts.append("</ul></section>")

    notes = data.get("notes") or []
    if notes:
        parts.append("<section><h2>Notes</h2><ul>")
        for note in notes:
            parts.append(f"<li>{esc(note)}</li>")
        parts.append("</ul></section>")

    parts.append("</body></html>")
    return "\n".join(parts)
