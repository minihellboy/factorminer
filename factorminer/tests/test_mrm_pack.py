"""Tests for the MRM validation pack assembler and report render hooks."""

from __future__ import annotations

import html

from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.provenance import (
    UNATTESTED_RATIONALE_BANNER,
    draft_economic_rationale,
)
from factorminer.evaluation.mrm_pack import (
    MRM_DISCLAIMER,
    build_mrm_pack,
    render_mrm_pack_html,
    render_mrm_pack_markdown,
)
from factorminer.evaluation.report_viewer import generate_report


def _library_with_rationale() -> FactorLibrary:
    lib = FactorLibrary(correlation_threshold=0.7, ic_threshold=0.02)
    rationale = draft_economic_rationale(
        "Neg(CsRank(Delta($close, 5)))",
        factor_name="mom_rev",
        category="Momentum",
    ).to_dict()
    # Inject a script tag to prove HTML escaping in renders.
    rationale["market_logic"] = (
        rationale["market_logic"] + " <script>alert('xss')</script>"
    )
    factor = Factor(
        id=1,
        name="mom_rev",
        formula="Neg(CsRank(Delta($close, 5)))",
        category="Momentum",
        ic_mean=0.05,
        icir=0.9,
        ic_win_rate=0.55,
        max_correlation=0.1,
        batch_number=1,
        ic_paper_mean=0.05,
        ic_paper_icir=0.9,
        provenance={
            "parent_formula": "CsRank(Delta($close, 5))",
            "edit_type": "mutation",
            "edit_motif": "sign_flip",
            "economic_rationale": rationale,
        },
    )
    lib.admit_factor(factor)
    return lib


def test_build_mrm_pack_has_four_examiner_sections() -> None:
    pack = build_mrm_pack(
        _library_with_rationale(),
        significance_summary={"bootstrap_p_value": 0.01, "status": "supplied"},
        pbo_summary={"pbo": 0.22, "passes": True},
        cpcv_summary={"n_splits": 10, "status": "supplied"},
        causal_summary=[{"factor_name": "mom_rev", "passes": True}],
        decay_summary=[{"factor_id": "1", "classification": "stable"}],
        library_source="test_library",
    )
    data = pack.to_dict()
    assert data["inventory"], "inventory must be non-empty"
    inv = data["inventory"][0]
    assert inv["purpose"]
    assert inv["limitations"]
    assert inv["intended_use"]
    assert inv["owner"]
    assert inv["risk_tier"]
    assert inv["parent_formula"] == "CsRank(Delta($close, 5))"
    assert inv["economic_rationale_attested"] is False

    sound = data["conceptual_soundness"]
    assert sound["lineage"]
    assert sound["economic_rationales"]
    assert sound["unattested_count"] == 1

    outcomes = data["outcomes_analysis"]
    assert outcomes["significance"]["status"] == "supplied" or "bootstrap_p_value" in outcomes["significance"]
    assert outcomes["pbo"]["pbo"] == 0.22
    assert outcomes["cpcv"]["n_splits"] == 10
    assert outcomes["causal"]["count"] == 1

    monitoring = data["ongoing_monitoring"]
    assert monitoring["decay"]["count"] == 1
    assert monitoring["attestation_gaps"]
    assert MRM_DISCLAIMER in data["disclaimer"]
    assert "compliant" not in data["disclaimer"].lower() or "not" in data["disclaimer"].lower()


def test_mrm_pack_markdown_and_html_render_sections_and_escape() -> None:
    pack = build_mrm_pack(_library_with_rationale(), library_source="unit")
    md = render_mrm_pack_markdown(pack)
    assert "## Model Inventory" in md
    assert "## Conceptual Soundness" in md
    assert "## Outcomes Analysis" in md
    assert "## Ongoing Monitoring" in md
    assert UNATTESTED_RATIONALE_BANNER in md
    assert "SR 26-2 compliant" not in md.lower()

    html_doc = render_mrm_pack_html(pack)
    assert "Model Inventory" in html_doc
    assert "Conceptual Soundness" in html_doc
    assert "Outcomes Analysis" in html_doc
    assert "Ongoing Monitoring" in html_doc
    assert UNATTESTED_RATIONALE_BANNER in html_doc
    # XSS payload must be escaped, never raw.
    assert "<script>alert('xss')</script>" not in html_doc
    assert html.escape("<script>alert('xss')</script>") in html_doc


def test_report_viewer_mrm_pack_flag_embeds_pack() -> None:
    lib = _library_with_rationale()
    library_payload = {
        "correlation_threshold": 0.7,
        "ic_threshold": 0.02,
        "metric_version": "paper_ic_v2",
        "dependence_metric": "spearman",
        "factors": [f.to_dict() for f in lib.list_factors()],
    }
    report = generate_report(
        library_payload,
        format="html",
        include_mrm_pack=True,
    )
    assert "MRM Validation Pack" in report
    assert "Model Inventory" in report
    assert "Conceptual Soundness" in report
    assert "Outcomes Analysis" in report
    assert "Ongoing Monitoring" in report
    assert UNATTESTED_RATIONALE_BANNER in report
