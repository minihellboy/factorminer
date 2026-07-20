"""Tests for Agora-style sealed multi-evaluator joint search.

Acceptance coverage:
1. Three differently-biased evaluators genuinely disagree on synthetic candidates
   (IC-only accept vs robustness reject is mandatory).
2. Sealed boundary: prompt-facing output has only coarse agreement signals.
3. End-to-end batch promotion with disagreement diagnostics.
4. Opt-in significance bridge does not alter default check_significance.
5. Public module does not re-export concrete evaluator weight classes.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from factorminer.architecture.sealed_joint_search import (
    RESEARCH_MODE_CAVEAT,
    AgreementRule,
    CandidateObservation,
    SealedFeedback,
    SealedJointSearchConfig,
    SealedJointSearchEngine,
    assert_feedback_is_sealed,
    build_sealed_engine,
    observation_from_arrays,
)
from factorminer.evaluation.significance import (
    SealedAgreementConfig,
    check_significance,
    summarize_sealed_agreement,
)

# ---------------------------------------------------------------------------
# Synthetic observations engineered to force panel disagreement
# ---------------------------------------------------------------------------


def _high_ic_brittle() -> CandidateObservation:
    """High IC, terrible robustness / high CPCV variance — IC yes, robust no."""
    return CandidateObservation(
        name="high_ic_brittle",
        formula="CsRank(Delta($close, 1))",
        ic_paper_mean=0.08,
        ic_mean=0.08,
        ic_std=0.12,
        icir=0.67,
        ic_win_rate=0.62,
        intervention_robustness=0.15,  # fails robustness hard gate (min 0.45)
        cpcv_ic_std=0.10,  # fails max_cpcv_ic_std 0.06
        cpcv_ic_mean=0.08,
        max_library_dependence=0.25,
        novelty_score=0.75,
    )


def _high_ic_crowded() -> CandidateObservation:
    """High IC but near-duplicate of library — IC yes, novelty no."""
    return CandidateObservation(
        name="high_ic_crowded",
        formula="CsRank(Delta($close, 5))",
        ic_paper_mean=0.07,
        ic_mean=0.07,
        ic_std=0.03,
        icir=2.3,
        ic_win_rate=0.70,
        intervention_robustness=0.80,
        cpcv_ic_std=0.015,
        cpcv_ic_mean=0.07,
        max_library_dependence=0.92,  # fails novelty hard gate (max 0.70)
        novelty_score=0.08,
    )


def _balanced_solid() -> CandidateObservation:
    """Solid across IC, robustness, and novelty — should promote under majority."""
    return CandidateObservation(
        name="balanced_solid",
        formula="Neg(CsZScore(Div(Sub($close, SMA($close, 20)), SMA($close, 20))))",
        ic_paper_mean=0.045,
        ic_mean=0.045,
        ic_std=0.02,
        icir=2.25,
        ic_win_rate=0.60,
        intervention_robustness=0.75,
        cpcv_ic_std=0.018,
        cpcv_ic_mean=0.045,
        max_library_dependence=0.20,
        novelty_score=0.80,
    )


def _weak_all_around() -> CandidateObservation:
    """Weak everywhere — should be rejected unanimously."""
    return CandidateObservation(
        name="weak_noise",
        formula="CsRank($volume)",
        ic_paper_mean=0.005,
        ic_mean=0.005,
        ic_std=0.08,
        icir=0.06,
        ic_win_rate=0.48,
        intervention_robustness=0.20,
        cpcv_ic_std=0.09,
        cpcv_ic_mean=0.005,
        max_library_dependence=0.85,
        novelty_score=0.15,
    )


def _engine_numeric_only() -> SealedJointSearchEngine:
    """Three numeric evaluators only (no LLM judge) for crisp disagreement tests."""
    return SealedJointSearchEngine(
        SealedJointSearchConfig(
            enabled=True,
            agreement_rule=AgreementRule.MAJORITY,
            include_llm_judge=False,
            retain_internal_scores=True,
        )
    )


# ---------------------------------------------------------------------------
# Core disagreement proofs
# ---------------------------------------------------------------------------


def test_ic_accepts_but_robustness_rejects_brittle_candidate() -> None:
    """Concrete proof: IC maximizer passes a brittle high-IC factor; robustness fails it."""
    # Import private panel ONLY inside tests that must inspect internals.
    from factorminer.architecture._sealed_evaluator_panel import (
        ICMaximizingEvaluator,
        RobustnessEmphasizingEvaluator,
    )

    obs = _high_ic_brittle()
    ic_score = ICMaximizingEvaluator().evaluate(obs)
    rob_score = RobustnessEmphasizingEvaluator().evaluate(obs)

    assert ic_score.passed is True, (
        f"IC maximizer should accept high-IC brittle factor, got score={ic_score.score}"
    )
    assert rob_score.passed is False, (
        f"Robustness evaluator should reject brittle factor, got score={rob_score.score}, "
        f"components={rob_score.components}"
    )
    # Genuine score gap, not a fluke around the threshold.
    assert ic_score.score > rob_score.score


def test_ic_accepts_but_novelty_rejects_crowded_candidate() -> None:
    from factorminer.architecture._sealed_evaluator_panel import (
        ICMaximizingEvaluator,
        NoveltyEmphasizingEvaluator,
    )

    obs = _high_ic_crowded()
    ic_score = ICMaximizingEvaluator().evaluate(obs)
    nov_score = NoveltyEmphasizingEvaluator().evaluate(obs)

    assert ic_score.passed is True
    assert nov_score.passed is False
    assert "novelty" in nov_score.components or nov_score.components.get("hard_gate") == 0.0


def test_novelty_evaluator_does_not_double_count_dependence_proxy() -> None:
    """Regression test: when novelty_score is unsupplied (auto-proxied from
    max_library_dependence), the score must not double-count that single
    measurement under two separate weights.

    A prior implementation set ``novelty = 1 - max_library_dependence``
    (the proxy) and then ALSO added ``dependence_penalty_weight *
    independence`` -- numerically the *same* value -- as if it were a
    second, distinct signal. The combined 0.65+0.20 weight then applied
    to one underlying measurement instead of two.
    """
    from factorminer.architecture._sealed_evaluator_panel import NoveltyEmphasizingEvaluator

    ev = NoveltyEmphasizingEvaluator()

    # novelty_score explicitly supplied (not a proxy case): weights sum to
    # 1.0 already, so the formula must be unchanged by the fix.
    explicit = CandidateObservation(
        name="explicit_novelty",
        formula="X",
        ic_paper_mean=0.05,
        max_library_dependence=0.3,
        novelty_score=0.8,
    )
    r_explicit = ev.evaluate(explicit)
    ic_term = min(abs(0.05) / ev.ic_scale, 1.0)
    expected_explicit = (
        ev.novelty_weight * 0.8 + ev.dependence_penalty_weight * (1 - 0.3) + ev.ic_weight * ic_term
    )
    assert r_explicit.score == pytest.approx(expected_explicit, abs=1e-9)

    # novelty_score unsupplied (0.0): triggers the dependence proxy. The
    # fixed score must equal the renormalized two-signal formula, and must
    # NOT equal the old (double-counted) three-term formula.
    proxied = CandidateObservation(
        name="proxied_novelty",
        formula="Y",
        ic_paper_mean=0.02,
        max_library_dependence=0.3,
        novelty_score=0.0,
    )
    r_proxied = ev.evaluate(proxied)
    ic_term2 = min(abs(0.02) / ev.ic_scale, 1.0)
    independence = 1 - 0.3
    old_double_counted = (
        ev.novelty_weight * independence
        + ev.dependence_penalty_weight * independence
        + ev.ic_weight * ic_term2
    )
    fixed = (ev.novelty_weight * independence + ev.ic_weight * ic_term2) / (
        ev.novelty_weight + ev.ic_weight
    )
    assert r_proxied.score == pytest.approx(fixed, abs=1e-9)
    assert r_proxied.score != pytest.approx(old_double_counted, abs=1e-6)
    assert 0.0 <= r_proxied.score <= 1.0


def test_panel_disagrees_on_synthetic_batch() -> None:
    """Batch must produce at least one disagreed candidate (not unanimous panel)."""
    engine = _engine_numeric_only()
    report = engine.evaluate_batch(
        [
            _high_ic_brittle(),
            _high_ic_crowded(),
            _balanced_solid(),
            _weak_all_around(),
        ]
    )
    assert report.n_candidates == 4
    assert report.disagreement_rate > 0.0, (
        "Evaluators always agreed — biases are not genuinely different"
    )
    # Balanced solid should promote under majority of 3 numeric judges.
    assert "balanced_solid" in report.promoted_names()
    # Weak noise should not promote.
    assert "weak_noise" in report.rejected_names()

    brittle = next(d for d in report.decisions if d.observation.name == "high_ic_brittle")
    assert brittle.disagreement is True
    assert "ic" in brittle.feedback.passed_personas
    assert "robustness" in brittle.feedback.failed_personas


def test_unanimous_rule_is_stricter_than_majority() -> None:
    eng_maj = SealedJointSearchEngine(
        SealedJointSearchConfig(
            agreement_rule=AgreementRule.MAJORITY, include_llm_judge=False
        )
    )
    eng_uni = SealedJointSearchEngine(
        SealedJointSearchConfig(
            agreement_rule=AgreementRule.UNANIMOUS, include_llm_judge=False
        )
    )
    # Crowded high-IC: IC+robustness may pass, novelty fails → majority maybe,
    # unanimous never.
    obs = _high_ic_crowded()
    maj = eng_maj.evaluate_one(obs)
    uni = eng_uni.evaluate_one(obs)
    assert uni.promoted is False
    assert uni.n_passed < uni.n_evaluators
    # Majority promotes only if >= 2 of 3 pass.
    if maj.n_passed >= 2:
        assert maj.promoted is True
        assert maj.promoted != uni.promoted


# ---------------------------------------------------------------------------
# Sealed boundary
# ---------------------------------------------------------------------------


def test_sealed_feedback_has_no_raw_score_internals() -> None:
    engine = _engine_numeric_only()
    decision = engine.evaluate_one(_balanced_solid())
    payload = decision.feedback.to_prompt_dict()
    assert_feedback_is_sealed(payload)

    forbidden = {
        "score",
        "scores",
        "components",
        "threshold",
        "weights",
        "ic_weight",
        "raw_scores",
        "evaluator_scores",
        "rationale",
    }
    assert forbidden.isdisjoint(payload.keys())
    # Coarse fields present.
    assert payload["n_passed"] + len(payload["failed_personas"]) >= 1
    assert payload["sealed"] is True
    assert "passed" in payload["summary"].lower() or "held" in payload["summary"].lower()
    # Summary must not embed raw weight dumps.
    assert "ic_weight" not in payload["summary"]
    assert "0.70" not in payload["summary"]


def test_prompt_context_extras_are_sealed() -> None:
    engine = _engine_numeric_only()
    report = engine.evaluate_batch([_high_ic_brittle(), _balanced_solid()])
    extras = engine.prompt_context_extras(report)
    assert "sealed_joint_search" in extras
    blob = extras["sealed_joint_search"]
    assert blob["disagreement_rate"] >= 0.0
    assert RESEARCH_MODE_CAVEAT in blob["caveat"]
    for cand in blob["candidates"]:
        assert_feedback_is_sealed(cand)
        assert "components" not in cand
        assert "score" not in cand


def test_assert_feedback_is_sealed_catches_leaks() -> None:
    dirty = SealedFeedback(
        candidate_name="x",
        n_evaluators=3,
        n_passed=2,
        agreement_fraction=0.66,
        promoted=True,
        agreement_rule="majority",
    ).to_prompt_dict()
    dirty["ic_weight"] = 0.7
    with pytest.raises(AssertionError, match="leaked"):
        assert_feedback_is_sealed(dirty)


def test_public_module_does_not_export_concrete_evaluators() -> None:
    """Structural sealing: generation-facing module must not re-export weight classes."""
    import factorminer.architecture.sealed_joint_search as public

    for name in (
        "ICMaximizingEvaluator",
        "RobustnessEmphasizingEvaluator",
        "NoveltyEmphasizingEvaluator",
        "LLMJudgeEvaluator",
        "EvaluatorScore",
        "build_default_evaluators",
        "run_panel",
    ):
        assert name not in public.__all__
        assert not hasattr(public, name) or name in {
            # CandidateObservation is intentionally public (metrics only).
        }


def test_private_panel_module_name_is_underscored() -> None:
    """Private panel lives under a leading-underscore module name."""
    panel_path = (
        Path(__file__).resolve().parents[1]
        / "architecture"
        / "_sealed_evaluator_panel.py"
    )
    assert panel_path.is_file()
    # Public module must not appear to star-import private evaluator classes.
    public_src = (
        Path(__file__).resolve().parents[1]
        / "architecture"
        / "sealed_joint_search.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(public_src)
    exported = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    exported = {
                        (elt.value if isinstance(elt, ast.Constant) else None)
                        for elt in node.value.elts
                    }
                    exported.discard(None)
    assert "ICMaximizingEvaluator" not in exported
    assert "SealedJointSearchEngine" in exported


# ---------------------------------------------------------------------------
# End-to-end synthetic factor set
# ---------------------------------------------------------------------------


def test_end_to_end_synthetic_factor_set_promotion(capsys: pytest.CaptureFixture[str]) -> None:
    """Build tiny synthetic factors, run sealed promotion, print diagnostics."""
    rng = np.random.RandomState(7)
    M, T = 40, 120
    # Common return driver
    latent = rng.normal(size=(M, T))
    returns = 0.01 * latent + 0.02 * rng.normal(size=(M, T))

    # Factor A: aligned with latent (strong IC), plus noise that will hurt robustness proxy
    signals_a = latent + 0.1 * rng.normal(size=(M, T))
    # Factor B: weak noise
    signals_b = rng.normal(size=(M, T))
    # Factor C: moderate alignment, independent of a library factor
    signals_c = 0.5 * latent + 0.5 * rng.normal(size=(M, T))
    # Library factor nearly collinear with A
    library = [latent + 0.05 * rng.normal(size=(M, T))]

    obs_a = observation_from_arrays(
        "synth_strong",
        signals_a,
        returns,
        formula="aligned_latent",
        library_signals=library,
        intervention_robustness=0.20,  # force brittle
        seed=7,
    )
    obs_b = observation_from_arrays(
        "synth_noise",
        signals_b,
        returns,
        formula="pure_noise",
        library_signals=library,
        intervention_robustness=0.30,
        seed=7,
    )
    obs_c = observation_from_arrays(
        "synth_balanced",
        signals_c,
        returns,
        formula="moderate_latent",
        library_signals=[],  # novel vs empty library slice
        intervention_robustness=0.80,
        seed=7,
    )
    # Ensure C looks novel
    obs_c = CandidateObservation(
        name=obs_c.name,
        formula=obs_c.formula,
        ic_paper_mean=max(obs_c.ic_paper_mean, 0.03),
        ic_mean=obs_c.ic_mean,
        ic_std=min(obs_c.ic_std, 0.03),
        icir=max(obs_c.icir, 1.0),
        ic_win_rate=max(obs_c.ic_win_rate, 0.55),
        intervention_robustness=0.80,
        cpcv_ic_std=min(obs_c.cpcv_ic_std, 0.025),
        cpcv_ic_mean=obs_c.cpcv_ic_mean,
        max_library_dependence=0.15,
        novelty_score=0.85,
    )

    engine = _engine_numeric_only()
    report = engine.evaluate_batch([obs_a, obs_b, obs_c])

    # Print real diagnostics (captured; also useful when run with -s).
    print("=" * 60)
    print("Sealed joint search — synthetic end-to-end")
    print(f"Agreement rule: {report.agreement_rule}")
    print(f"Evaluators: {list(report.evaluator_ids)}")
    print(f"Promoted: {report.promoted_names()}")
    print(f"Rejected: {report.rejected_names()}")
    print(f"Disagreement rate: {report.disagreement_rate:.2f}")
    print(f"Mean agreement fraction: {report.mean_agreement_fraction:.2f}")
    for d in report.decisions:
        fb = d.feedback
        print(
            f"  - {d.observation.name}: promoted={d.promoted} "
            f"passed={d.n_passed}/{d.n_evaluators} "
            f"disagreement={d.disagreement} "
            f"personas_pass={list(fb.passed_personas)} "
            f"personas_fail={list(fb.failed_personas)}"
        )
        if d.internal_scores:
            for s in d.internal_scores:
                print(f"      [{s.evaluator_id}] passed={s.passed} score={s.score:.3f}")
    print(f"Caveat: {report.caveat[:80]}...")
    print("=" * 60)

    out = capsys.readouterr().out
    assert "Disagreement rate" in out
    assert "Promoted" in out
    assert report.n_candidates == 3
    # Noise should not promote under majority of well-separated judges.
    assert "synth_noise" in report.rejected_names()
    # Strong-but-brittle should show disagreement (IC vs robustness).
    strong = next(d for d in report.decisions if d.observation.name == "synth_strong")
    assert strong.disagreement or not strong.promoted
    # Sealed prompt payload stays clean.
    for payload in report.sealed_feedback_batch():
        assert_feedback_is_sealed(payload)


# ---------------------------------------------------------------------------
# Significance bridge (additive, default off)
# ---------------------------------------------------------------------------


def test_significance_bridge_disabled_by_default() -> None:
    result = summarize_sealed_agreement("f", ic_paper_mean=0.05)
    assert result is None


def test_significance_bridge_opt_in_returns_summary() -> None:
    summary = summarize_sealed_agreement(
        "f_brittle",
        ic_paper_mean=0.08,
        icir=0.7,
        ic_win_rate=0.6,
        intervention_robustness=0.15,
        cpcv_ic_std=0.10,
        max_library_dependence=0.2,
        novelty_score=0.8,
        config=SealedAgreementConfig(enabled=True, include_llm_judge=False),
    )
    assert summary is not None
    assert summary.candidate_name == "f_brittle"
    assert summary.n_evaluators >= 3
    assert 0.0 <= summary.agreement_fraction <= 1.0
    d = summary.to_dict()
    assert "ic_weight" not in d
    assert "components" not in d


def test_check_significance_unchanged_by_sealed_extension() -> None:
    """Classical entry point still works and does not require sealed config."""
    rng = np.random.RandomState(0)
    ic = 0.05 + 0.01 * rng.normal(size=80)
    ls = 0.001 + 0.01 * rng.normal(size=80)
    passes, reason, details = check_significance("classic", ic, ls, n_total_trials=10)
    assert isinstance(passes, bool)
    assert isinstance(details, dict)
    assert "sealed" not in details  # default path untouched


# ---------------------------------------------------------------------------
# Config / engine hygiene
# ---------------------------------------------------------------------------


def test_config_defaults_are_opt_in() -> None:
    cfg = SealedJointSearchConfig()
    assert cfg.enabled is False
    assert "research mode" in RESEARCH_MODE_CAVEAT.lower() or "single-seed" in RESEARCH_MODE_CAVEAT


def test_build_sealed_engine_factory() -> None:
    eng = build_sealed_engine(
        SealedJointSearchConfig(enabled=True, include_llm_judge=True)
    )
    assert isinstance(eng, SealedJointSearchEngine)
    # Default panel: 3 numeric + optional llm judge
    assert len(eng.evaluator_ids) >= 3


def test_llm_judge_mock_uses_deterministic_fallback() -> None:
    from factorminer.agent.llm_interface import MockProvider
    from factorminer.architecture._sealed_evaluator_panel import LLMJudgeEvaluator

    judge = LLMJudgeEvaluator(llm_provider=MockProvider(), use_llm=True)
    score = judge.evaluate(_balanced_solid())
    assert score.components.get("mode") == 0.0  # deterministic fallback
    assert 0.0 <= score.score <= 1.0


def test_complements_evaluation_kernel_not_replacement() -> None:
    """Documented relationship: sealed search is alongside the kernel."""
    src = inspect.getsource(
        __import__(
            "factorminer.architecture.sealed_joint_search", fromlist=["*"]
        )
    )
    assert "EvaluationKernel" in src
    assert "not a replacement" in src.lower() or "never a replacement" in src.lower()
    assert "single-seed" in src.lower() or "−0.755" in src or "-0.755" in src
