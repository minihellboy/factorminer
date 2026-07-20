"""Tests for economic-rationale schema, attestation, and HTML escaping."""

from __future__ import annotations

import html

from factorminer.core.provenance import (
    UNATTESTED_RATIONALE_BANNER,
    EconomicRationale,
    attest_economic_rationale,
    build_factor_provenance,
    draft_economic_rationale,
)
from factorminer.evaluation.report_viewer import generate_report, render_html_report


def test_draft_rationale_never_auto_attests() -> None:
    drafted = draft_economic_rationale(
        "Neg(CsRank(Delta($close, 5)))",
        factor_name="mom",
        category="Momentum",
        use_llm=False,
    )
    assert drafted.attested is False
    assert drafted.source == "template"
    assert drafted.mathematical_structure
    assert drafted.financial_semantics
    assert drafted.market_logic

    # from_dict must not honor attested=True from non-human sources.
    sneaky = drafted.to_dict()
    sneaky["attested"] = True
    sneaky["source"] = "llm"
    restored = EconomicRationale.from_dict(sneaky)
    assert restored.attested is False


def test_only_human_attestation_sets_attested_true() -> None:
    drafted = draft_economic_rationale("Mean($close, 5)", factor_name="x").to_dict()
    assert drafted["attested"] is False
    attested = attest_economic_rationale(drafted, attestor="reviewer-a")
    assert attested["attested"] is True
    assert attested["source"] == "human"
    assert attested["attestor"] == "reviewer-a"
    assert "attested_at" in attested


def test_build_factor_provenance_embeds_unattested_rationale() -> None:
    prov = build_factor_provenance(
        run_manifest={"run_id": "r1", "session_id": "s1", "loop_type": "ralph"},
        factor_name="f1",
        formula="Mean($close, 5)",
        factor_category="Momentum",
        factor_id=1,
        iteration=1,
        batch_number=1,
        candidate_rank=1,
        generator_family="FactorGenerator",
        memory_signal={},
        library_state={"recent_admissions": []},
        evaluation={"ic_paper_mean": 0.04},
        admission={"admitted": True},
        parent_formula="",
        edit_type="fresh",
        draft_rationale=True,
    ).to_dict()
    rationale = prov["economic_rationale"]
    assert rationale["attested"] is False
    assert rationale["mathematical_structure"]
    assert prov["edit_type"] == "fresh"


def test_html_report_shows_unattested_banner_and_escapes_llm_text() -> None:
    evil = (
        "Momentum story with <script>alert('xss')</script> payload "
        "and an <img src=x onerror=alert(1)> tag."
    )
    rationale = EconomicRationale(
        mathematical_structure=evil,
        financial_semantics="semantics",
        market_logic="logic",
        attested=False,
        source="llm",
    ).to_dict()
    library_payload = {
        "correlation_threshold": 0.5,
        "ic_threshold": 0.04,
        "metric_version": "paper_ic_v2",
        "dependence_metric": "spearman",
        "factors": [
            {
                "id": 1,
                "name": "evil_factor",
                "formula": "Mean($close, 5)",
                "category": "Momentum",
                "ic_mean": 0.04,
                "ic_paper_mean": 0.04,
                "ic_abs_mean": 0.04,
                "icir": 0.5,
                "ic_paper_icir": 0.5,
                "ic_win_rate": 0.5,
                "max_correlation": 0.1,
                "batch_number": 1,
                "provenance": {
                    "loop_type": "ralph",
                    "parent_formula": "Mean($close, 10)",
                    "edit_type": "mutation",
                    "economic_rationale": rationale,
                },
            }
        ],
    }
    report = generate_report(library_payload, format="html", include_mrm_pack=True)
    assert UNATTESTED_RATIONALE_BANNER in report
    assert "<script>alert('xss')</script>" not in report
    assert html.escape("<script>alert('xss')</script>") in report
    # Banner must be visually present as a dedicated marker, not only buried JSON.
    assert "UNATTESTED" in report

    # Direct render path also escapes.
    from factorminer.evaluation.report_viewer import build_report_payload

    payload = build_report_payload(library_payload, include_mrm_pack=True)
    html_doc = render_html_report(payload)
    assert UNATTESTED_RATIONALE_BANNER in html_doc
    assert "<script>alert('xss')</script>" not in html_doc
