"""Tests for the AlphaMemo-style edit-motif memory policy.

Covers (a) AST-diff edit-motif extraction against the fixed taxonomy, (b) confidence
gating and the asymmetric process veto driven purely through the public
``form()``/``retrieve()``/``score_action`` contract, and (c) persist/restore
round-tripping, mirroring the conventions in ``test_architecture.py`` and
``test_memory.py``.
"""

from __future__ import annotations

import pytest

from factorminer.architecture.families import infer_family
from factorminer.architecture.memory_policy import (
    EDIT_MOTIF_LABELS,
    EditAwareMemoryPolicy,
    MotifStats,
    ParentContext,
    build_memory_policy,
    extract_edit_motif,
)
from factorminer.architecture.paper_protocol import PaperProtocol
from factorminer.memory.memory_store import ExperienceMemory
from factorminer.utils.config import load_config


@pytest.fixture
def protocol() -> PaperProtocol:
    return PaperProtocol.from_config(load_config())


# ---------------------------------------------------------------------------
# (a) AST-diff edit-motif extraction
# ---------------------------------------------------------------------------

MOTIF_CASES = [
    pytest.param(
        "Greater($close, $open)", "Less($close, $open)", "operator_swap", id="operator_swap"
    ),
    pytest.param("Mean($close, 5)", "Mean($close, 20)", "window_rescale", id="window_rescale"),
    pytest.param(
        "CsRank(Delta($close, 5))",
        "CsRank(IfElse(Greater($volume, 0), Delta($close, 5), 0))",
        "add_conditional",
        id="add_conditional",
    ),
    pytest.param(
        "CsRank(Delta($close, 5))", "CsRank(Delta($open, 5))", "feature_swap", id="feature_swap"
    ),
    pytest.param(
        "CsRank(Delta($close, 5))",
        "EMA(CsRank(Delta($close, 5)), 10)",
        "wrap_smoothing",
        id="wrap_smoothing",
    ),
    pytest.param(
        "CsRank(Delta($close, 5))",
        "Neg(CsRank(Delta($close, 5)))",
        "sign_flip",
        id="sign_flip",
    ),
    pytest.param(
        "Delta($close, 5)",
        "Mul(Delta($close, 5), CsRank($volume))",
        "add_interaction",
        id="add_interaction",
    ),
    pytest.param("TsRank($close, 10)", "CsRank($close)", "rank_swap", id="rank_swap"),
    pytest.param(
        "$close",
        "Add(TsMax($close, 10), TsMin($close, 10))",
        "structural_grow",
        id="structural_grow",
    ),
    pytest.param("Mean($close, 5)", "Mean($close, 5)", "other", id="other_noop"),
]


@pytest.mark.parametrize("parent, child, expected", MOTIF_CASES)
def test_extract_edit_motif_matches_taxonomy(parent: str, child: str, expected: str) -> None:
    assert expected in EDIT_MOTIF_LABELS
    assert extract_edit_motif(parent, child) == expected


def test_edit_motif_taxonomy_is_small_and_fixed() -> None:
    # Mirrors the paper's "nine named edit types plus an 'other' bucket".
    assert len(EDIT_MOTIF_LABELS) == 10
    assert "other" in EDIT_MOTIF_LABELS
    assert len(set(EDIT_MOTIF_LABELS)) == len(EDIT_MOTIF_LABELS)


def test_extract_edit_motif_falls_back_to_other_on_parse_failure() -> None:
    assert extract_edit_motif("not a valid formula (((", "Mean($close, 5)") == "other"
    assert extract_edit_motif("Mean($close, 5)", "not valid (((") == "other"


# ---------------------------------------------------------------------------
# Parent-context bucketing
# ---------------------------------------------------------------------------


def test_parent_context_key_reflects_family_quality_and_depth(protocol: PaperProtocol) -> None:
    policy = EditAwareMemoryPolicy(protocol)
    context = policy._context_for("Mean($close, 5)", 0.03)
    assert isinstance(context, ParentContext)
    assert context.family == infer_family("Mean($close, 5)")
    assert context.quality_bucket == "moderate"
    assert context.depth_bucket == "shallow"
    assert context.key == f"{context.family}|moderate|shallow"


# ---------------------------------------------------------------------------
# (b) confidence gating + asymmetric veto
# ---------------------------------------------------------------------------


def test_motif_stats_confidence_grows_with_volume_and_consistency() -> None:
    sparse = MotifStats()
    sparse.update(-0.05)
    noisy = MotifStats()
    for residual in (-0.2, 0.3, -0.1, 0.25, -0.05, 0.15):
        noisy.update(residual)
    consistent = MotifStats()
    for _ in range(30):
        consistent.update(-0.05)

    kappa = 8.0
    c_sparse = sparse.confidence(kappa)
    c_noisy = noisy.confidence(kappa)
    c_consistent = consistent.confidence(kappa)

    assert c_sparse < 0.3, "a single observation must not carry much confidence"
    assert c_noisy < 0.3, "inconsistent (low SNR) residuals must not carry much confidence"
    assert c_consistent > 0.7, "many consistent observations must saturate confidence"
    assert c_consistent > c_sparse
    assert c_consistent > c_noisy


def test_score_action_with_no_history_returns_base_score_unmodified(
    protocol: PaperProtocol,
) -> None:
    policy = EditAwareMemoryPolicy(protocol)
    action = policy.score_action("Mean($close, 5)", "window_rescale", base_score=0.03)
    assert action.confidence == 0.0
    assert action.residual == 0.0
    assert action.vetoed is False
    assert action.adjusted_score == 0.03


def test_negative_residual_history_penalizes_motif_in_later_retrieval(
    protocol: PaperProtocol,
) -> None:
    """AlphaMemo's asymmetric veto: a (context, motif) with consistently negative
    residuals must be measurably penalized (here: vetoed) relative to a motif with
    no history at all, once observed through a full form() -> retrieve() cycle.
    """
    policy = EditAwareMemoryPolicy(protocol)
    memory = ExperienceMemory()
    memory.success_patterns = []

    parent_formula = "Mean($close, 5)"
    trajectory: list[dict] = []

    # Background edges (feature_swap motif) establish a healthy context baseline.
    for i in range(40):
        trajectory.append(
            {
                "factor_id": f"bg_{i}",
                "formula": "Mean($open, 5)",
                "ic_paper_mean": 0.08,
                "admitted": True,
                "parent_formula": parent_formula,
                "parent_ic_paper_mean": 0.03,
            }
        )

    # Target edges (window_rescale motif) are consistently, substantially worse
    # than the context baseline.
    for i in range(20):
        trajectory.append(
            {
                "factor_id": f"tgt_{i}",
                "formula": "Mean($close, 20)",
                "ic_paper_mean": -0.05,
                "admitted": False,
                "parent_formula": parent_formula,
                "parent_ic_paper_mean": 0.03,
            }
        )

    formed = policy.form(memory, trajectory, iteration=1)

    signal = policy.retrieve(
        formed,
        library_state={
            "library_size": 1,
            "recent_admissions": [{"formula": parent_formula, "ic_paper_mean": 0.03}],
        },
    )

    guidance = signal["edit_motif_guidance"][parent_formula]
    penalized = guidance["window_rescale"]
    no_history = guidance["add_conditional"]

    assert penalized["vetoed"] is True
    assert penalized["residual"] < 0.0
    assert no_history["vetoed"] is False
    assert penalized["score"] < no_history["score"] - 1.0, (
        "a reliably-negative motif must be measurably penalized vs. one with no history"
    )
    assert f"window_rescale on '{parent_formula}'" in "\n".join(
        line for line in signal["prompt_text"].splitlines() if "window_rescale" in line
    )


def test_positive_residual_boost_is_bounded_while_negative_is_not(
    protocol: PaperProtocol,
) -> None:
    """Task requirement 4: positive evidence only ever gives a bounded soft boost,
    while negative evidence can drive an unbounded (short of veto) penalty or a full
    veto -- an explicit asymmetry, not a signed weight applied both ways.
    """
    policy = EditAwareMemoryPolicy(
        protocol, positive_boost_cap=0.05, memory_weight=1.0, confidence_kappa=1.0
    )

    huge_positive = MotifStats()
    for _ in range(50):
        huge_positive.update(10.0)  # large, perfectly consistent -> confidence near 1
    context_key = policy._context_for("AnyParent", 0.0).key
    policy._motif_stats[(context_key, "rank_swap")] = huge_positive

    action = policy.score_action("AnyParent", "rank_swap", base_score=0.0, parent_quality=0.0)
    assert action.vetoed is False
    assert action.confidence > 0.9
    # Bounded: the fused influence never exceeds positive_boost_cap, however large
    # the underlying residual/confidence are.
    assert action.adjusted_score == pytest.approx(policy.positive_boost_cap)

    huge_negative = MotifStats()
    for _ in range(50):
        huge_negative.update(-10.0)
    policy._motif_stats[(context_key, "operator_swap")] = huge_negative
    negative_action = policy.score_action(
        "AnyParent", "operator_swap", base_score=0.0, parent_quality=0.0
    )
    # Vetoed outright, and in any case far below the positive side's bounded cap.
    assert negative_action.vetoed is True
    assert negative_action.adjusted_score < -policy.positive_boost_cap


# ---------------------------------------------------------------------------
# (c) persist / restore round trip
# ---------------------------------------------------------------------------


def test_edit_aware_memory_policy_exposes_schema_and_serialization(
    protocol: PaperProtocol,
) -> None:
    policy = EditAwareMemoryPolicy(protocol)
    schema = policy.schema()
    assert schema["policy"] == "edit_aware"
    assert schema["edit_motif_taxonomy"] == list(EDIT_MOTIF_LABELS)
    assert "sspm" in schema

    payload = policy.serialize(ExperienceMemory())
    assert payload["version"] == 0
    assert "memory_policy" in payload
    assert "edit_motif_memory" in payload


def test_edit_aware_memory_policy_persist_restore_roundtrip(protocol: PaperProtocol) -> None:
    policy = EditAwareMemoryPolicy(protocol)
    memory = ExperienceMemory()
    memory.success_patterns = []

    trajectory = [
        {
            "factor_id": "c1",
            "formula": "Mean($close, 20)",
            "ic_paper_mean": -0.04,
            "admitted": False,
            "parent_formula": "Mean($close, 5)",
            "parent_ic_paper_mean": 0.03,
        }
    ]
    formed = policy.form(memory, trajectory, iteration=1)
    payload = policy.serialize(formed)

    assert payload["edit_motif_memory"]["motif_stats"], "expect at least one recorded edge"

    # Same-instance restore (mirrors KGMemoryPolicy's own test convention).
    restored = policy.restore(payload)
    assert restored.version == formed.version
    assert len(policy._motif_stats) == 1

    # Fresh-instance restore must reproduce identical scoring behavior.
    fresh_policy = EditAwareMemoryPolicy(protocol)
    fresh_restored = fresh_policy.restore(payload)
    assert fresh_restored.version == formed.version

    original_action = policy.score_action(
        "Mean($close, 5)", "window_rescale", base_score=0.03, parent_quality=0.03
    )
    restored_action = fresh_policy.score_action(
        "Mean($close, 5)", "window_rescale", base_score=0.03, parent_quality=0.03
    )
    assert restored_action.residual == pytest.approx(original_action.residual)
    assert restored_action.confidence == pytest.approx(original_action.confidence)
    assert restored_action.vetoed == original_action.vetoed


def test_edit_aware_memory_policy_restore_without_prior_state_is_a_noop(
    protocol: PaperProtocol,
) -> None:
    policy = EditAwareMemoryPolicy(protocol)
    payload = policy.serialize(ExperienceMemory())
    # No edges observed yet -> empty but present motif_stats list.
    assert payload["edit_motif_memory"]["motif_stats"] == []
    restored = policy.restore(payload)
    assert restored.version == 0
    assert policy._motif_stats == {}


# ---------------------------------------------------------------------------
# build_memory_policy dispatch + config allow-list
# ---------------------------------------------------------------------------


def test_build_memory_policy_dispatches_edit_aware(protocol: PaperProtocol) -> None:
    cfg = load_config()
    cfg.memory.policy = "edit_aware"
    policy = build_memory_policy(cfg, protocol)
    assert isinstance(policy, EditAwareMemoryPolicy)
    assert policy.schema()["policy"] == "edit_aware"


def test_memory_config_accepts_edit_aware_policy_name() -> None:
    cfg = load_config()
    cfg.memory.policy = "edit_aware"
    cfg.memory.validate()  # must not raise

    cfg.memory.policy = "not_a_real_policy"
    with pytest.raises(ValueError):
        cfg.memory.validate()
