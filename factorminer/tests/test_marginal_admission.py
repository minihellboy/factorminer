"""Marginal-value admission: value beyond a reference basis and the library."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from factorminer.application.mining_budget import EvaluationResult
from factorminer.application.validation_pipeline import ValidationPipeline
from factorminer.architecture.trial_ledger import TrialLedger
from factorminer.evaluation.marginal import (
    MarginalBasis,
    MarginalConfig,
    passes_marginal_admission,
)
from factorminer.utils.config import AdmissionConfig, load_config

ASSETS, PERIODS = 150, 400


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(11)
    reference = [rng.standard_normal((ASSETS, PERIODS)) for _ in range(6)]
    novel = rng.standard_normal((ASSETS, PERIODS))
    noise = rng.standard_normal((ASSETS, PERIODS))
    returns = 0.004 * (reference[0] + 0.5 * reference[1]) + 0.003 * novel + 0.02 * noise
    returns[rng.random((ASSETS, PERIODS)) < 0.1] = np.nan  # members vary by period
    return SimpleNamespace(reference=reference, novel=novel, returns=returns, rng=rng)


@pytest.fixture()
def basis(market):
    config = MarginalConfig(reference_components=0, min_excess_bps=None)
    return MarginalBasis(market.returns, market.reference, config)


def test_a_restated_reference_factor_has_no_marginal_value(basis, market):
    stats = basis.evaluate(market.reference[0])
    assert stats.residual_share < 0.05
    assert abs(stats.partial_ic) < 0.01
    admitted, reason = passes_marginal_admission(stats, basis.config)
    assert not admitted and "residual share" in reason


def test_new_information_is_admitted_and_is_rank_invariant(basis, market):
    stats = basis.evaluate(market.novel)
    assert stats.partial_ic > 0.05 and stats.partial_ic_t > 5
    assert stats.residual_share > 0.9
    assert passes_marginal_admission(stats, basis.config)[0]
    transformed = basis.evaluate(np.exp(market.novel))  # monotone: same ranks
    assert transformed.partial_ic == pytest.approx(stats.partial_ic)
    flipped = basis.evaluate(-market.novel)
    assert flipped.partial_ic == pytest.approx(-stats.partial_ic)
    assert flipped.partial_ic_t == pytest.approx(stats.partial_ic_t)


def test_library_members_join_the_basis_and_sync_replaces_them(basis, market):
    basis.add("novel", market.novel)
    assert basis.evaluate(market.novel).residual_share < 0.05
    basis.sync({})
    assert basis.library_keys == ()
    assert basis.evaluate(market.novel).residual_share > 0.9


def test_principal_components_compress_the_reference(market):
    basis = MarginalBasis(market.returns, iter(market.reference), MarginalConfig(reference_components=2))
    assert basis.size == 2 and 0.0 < basis.reference_explained < 1.0


def test_topk_dropout_turnover_respects_the_drop_limit(basis, market):
    stats = basis.evaluate(market.novel)
    # The book changes by at most drop_fraction per day, plus universe exits.
    assert 0.0 < stats.top_turnover <= basis.config.drop_fraction + 0.12
    assert stats.top_excess_bps > 0
    assert stats.top_net_bps == pytest.approx(
        stats.top_excess_bps - basis.config.cost_bps * stats.top_turnover, abs=0.5
    )


def test_economic_gates(market):
    strict = MarginalConfig(reference_components=0, min_excess_bps=None, min_net_bps=1e6)
    basis = MarginalBasis(market.returns, market.reference, strict)
    admitted, reason = passes_marginal_admission(basis.evaluate(market.novel), strict)
    assert not admitted and "book net" in reason


def _pipeline(market, library_signals=()):
    from factorminer.core.factor_library import Factor, FactorLibrary

    library = FactorLibrary(correlation_threshold=0.5, ic_threshold=0.02)
    for index, signals in enumerate(library_signals):
        library.admit_factor(Factor(id=0, name=f"lib{index}", formula="Neg($close)",
                                    category="test", ic_mean=0.03, icir=0.5,
                                    ic_win_rate=0.5, max_correlation=0.0,
                                    batch_number=0, signals=signals))
    data = np.zeros((ASSETS, PERIODS, 1))
    pipeline = ValidationPipeline(data_tensor=data, returns=market.returns, library=library,
                                  fast_screen_assets=0)
    pipeline.marginal_basis = MarginalBasis(
        market.returns, market.reference, MarginalConfig(reference_components=0,
                                                         min_excess_bps=None))
    return pipeline


def _result(name: str, signals: np.ndarray) -> EvaluationResult:
    return EvaluationResult(factor_name=name, formula=name, parse_ok=True, signals=signals)


def test_greedy_batch_selection_admits_one_of_two_copies(market):
    pipeline = _pipeline(market)
    noisy = market.novel + 0.05 * market.rng.standard_normal(market.novel.shape)
    results = [pipeline._marginal_admission(_result(n, s), s)
               for n, s in (("a", market.novel), ("b", noisy), ("ref", market.reference[0]))]
    assert [r.admitted for r in results] == [True, True, False]
    selected = pipeline._select_marginal(results)
    assert sum(r.admitted for r in selected) == 1
    rejected = next(r for r in selected[:2] if not r.admitted)
    assert "earlier picks" in rejected.rejection_reason
    assert pipeline.marginal_basis.library_keys == ()  # batch picks never leak


def test_basis_follows_the_admitted_library(market):
    pipeline = _pipeline(market, library_signals=[market.novel])
    pipeline.sync_marginal_basis()
    result = pipeline._marginal_admission(_result("copy", market.novel), market.novel)
    assert not result.admitted
    assert result.marginal["residual_share"] < 0.05


def test_admission_config_defaults_keep_config_digests():
    cfg = load_config()
    assert cfg.admission == AdmissionConfig() and "admission" not in cfg.to_dict()
    marginal = load_config(overrides={"admission": {"policy": "marginal", "min_t": 2.5}})
    assert marginal.to_dict()["admission"]["min_t"] == 2.5
    assert MarginalConfig(**marginal.admission.marginal_settings()).min_t == 2.5
    with pytest.raises(ValueError, match="unknown admission settings"):
        load_config(overrides={"admission": {"min_tt": 2.5}})
    with pytest.raises(ValueError, match="admission.policy"):
        load_config(overrides={"admission": {"policy": "greedy"}})
    with pytest.raises(ValueError, match="admission.reference"):
        load_config(overrides={"admission": {"policy": "marginal", "reference": "alpha999"}})


def test_trial_ledger_records_marginal_statistics(tmp_path, market, basis):
    stats = basis.evaluate(market.novel).to_dict()
    result = SimpleNamespace(parse_ok=True, factor_name="a", formula="Neg($close)",
                             target_stats={}, stage_passed=3, admitted=True,
                             rejection_reason="", marginal=stats)
    ledger = TrialLedger(tmp_path, campaign_id="c")
    (observation,) = ledger.record_batch_results(iteration=1, results=[result],
                                                 dataset_id="d", target_name="paper")
    assert observation.metadata["marginal"]["partial_ic_t"] == stats["partial_ic_t"]


def test_planner_gain_counts_marginal_value_under_marginal_admission(market):
    from factorminer.application.research_actions import ResearchActionService
    from factorminer.core.factor_library import Factor

    pipeline = _pipeline(market)
    pipeline.library.admit_factor(Factor(
        id=0, name="m", formula="Neg($close)", category="test", ic_mean=0.09, icir=0.5,
        ic_win_rate=0.5, max_correlation=0.0, batch_number=0, signals=market.novel,
        ic_paper_mean=0.09, research_metrics={"marginal": {"partial_ic": -0.025}},
    ))
    loop = SimpleNamespace(pipeline=pipeline, library=pipeline.library,
                           settings=SimpleNamespace(ic_threshold=0.02))
    service = ResearchActionService.__new__(ResearchActionService)
    service.loop = loop
    assert service.library_utility() == pytest.approx(0.025 - 0.005)
    pipeline.marginal_basis = None
    assert service.library_utility() == pytest.approx(0.09 - 0.02)


def test_recent_decay_check_refuses_a_factor_that_stopped_working(market):
    decayed = market.novel.copy()
    tail = PERIODS // 4
    decayed[:, -tail:] = market.rng.standard_normal((ASSETS, tail))  # no information lately
    config = MarginalConfig(reference_components=0, min_excess_bps=None, recent_periods=tail,
                            min_recent_t=1.0)
    basis = MarginalBasis(market.returns, market.reference, config)
    stale = basis.evaluate(decayed)
    assert stale.partial_ic_t > 3  # the full-window average still looks strong
    admitted, reason = passes_marginal_admission(stale, config)
    assert not admitted and "recent partial IC" in reason
    fresh = basis.evaluate(market.novel)
    assert fresh.recent_partial_ic > 0 and passes_marginal_admission(fresh, config)[0]


def test_topk_dropout_never_sells_a_name_to_buy_a_worse_one():
    # Qlib TopkDropout sells held names only from the bottom of held + fresh.
    assets, periods = 60, 3
    score = np.tile(np.arange(assets, 0, -1, dtype=np.float64)[:, None], (1, periods))
    returns = np.zeros((assets, periods))
    config = MarginalConfig(reference_components=0, min_excess_bps=None, min_assets=5)
    basis = MarginalBasis(returns, [np.random.default_rng(0).standard_normal((assets, periods))],
                          config)
    # A stable ranking: the top-k book is bought once and never traded again.
    _, turnover, _, _ = basis._bucket(basis._gather(score))
    assert turnover == 0.0


def test_invalid_marginal_settings_fail_config_validation():
    with pytest.raises(ValueError, match="topk_fraction"):
        load_config(overrides={"admission": {"policy": "marginal", "topk_fraction": 0.8}})


def test_top_of_book_statistics_separate_head_from_tail_signal():
    """A candidate that only orders the bottom of the cross-section has partial
    IC but no top-of-book value (at its regression weight it only adds noise to
    the top); one that orders the top has both."""
    rng = np.random.default_rng(5)
    reference = [rng.standard_normal((ASSETS, PERIODS)) for _ in range(3)]
    base = reference[0]
    novel = rng.standard_normal((ASSETS, PERIODS))
    noise = rng.standard_normal((ASSETS, PERIODS))
    rank = base.argsort(axis=0).argsort(axis=0) / (ASSETS - 1)
    head, tail = rank > 2 / 3, rank < 1 / 3
    returns = 0.01 * base + 0.01 * noise
    config = MarginalConfig(reference_components=0, min_excess_bps=None, min_top_t=2.0,
                            min_top_lift_t=2.0)
    head_basis = MarginalBasis(returns + 0.02 * novel * head, reference, config)
    tail_basis = MarginalBasis(returns + 0.02 * novel * tail, reference, config)

    head_stats, tail_stats = head_basis.evaluate(novel), tail_basis.evaluate(novel)
    assert head_stats.partial_ic_t > 5 and tail_stats.partial_ic_t > 5
    assert head_stats.top_partial_ic > 0.1 and head_stats.top_partial_ic_t > 5
    assert head_stats.top_lift_bps > 0 and head_stats.top_lift_t > 3
    assert tail_stats.top_partial_ic_t < 3 and tail_stats.top_lift_bps < 0
    assert passes_marginal_admission(head_stats, config)[0]
    admitted, reason = passes_marginal_admission(tail_stats, config)
    assert not admitted and "top lift" in reason


def test_top_of_book_gates_are_off_by_default(basis, market):
    stats = basis.evaluate(market.novel)
    assert stats.top_lift_bps != 0.0
    assert basis.config.min_top_t is None and basis.config.min_top_lift_t is None
    assert passes_marginal_admission(stats, basis.config)[0]


def test_block_statistic_refuses_a_factor_that_only_worked_in_one_regime(market):
    one_regime = market.novel.copy()
    half = PERIODS // 2
    one_regime[:, half:] = market.rng.standard_normal((ASSETS, PERIODS - half))
    config = MarginalConfig(reference_components=0, min_excess_bps=None, blocks=4, min_block_t=0.5)
    basis = MarginalBasis(market.returns, market.reference, config)
    stale = basis.evaluate(one_regime)
    assert stale.partial_ic_t > 3  # pooled evidence is strong
    assert stale.block_t_min < 0.5 and stale.block_t_mean_minus_sd < stale.block_t_min + 3
    admitted, reason = passes_marginal_admission(stale, config)
    assert not admitted and "block partial IC t min" in reason
    steady = basis.evaluate(market.novel)
    assert steady.block_t_min > 0.5 and passes_marginal_admission(steady, config)[0]


def test_retention_shrinkage_and_regime_weight(market):
    retention = ((0.0, 0.25), (3.0, 0.35), (4.0, 0.45))
    config = MarginalConfig(reference_components=0, min_excess_bps=None, retention=retention,
                            min_shrunk_partial_ic=0.5)  # unreachable bar: every candidate is refused
    assert config.retention_at(2.0) == 0.25 and config.retention_at(3.5) == 0.35
    assert config.retention_at(-9.0) == 0.45 and MarginalConfig().retention_at(9.0) == 1.0
    basis = MarginalBasis(market.returns, market.reference, config)
    stats = basis.evaluate(market.novel)
    assert stats.shrunk_partial_ic == pytest.approx(
        abs(stats.partial_ic) * config.retention_at(stats.partial_ic_t))
    admitted, reason = passes_marginal_admission(stats, config)
    assert not admitted and "shrunk partial IC" in reason

    # Weighting the recent regime moves the mean toward the recent partial IC.
    decayed = market.novel.copy()
    tail = PERIODS // 4
    decayed[:, -tail:] = market.rng.standard_normal((ASSETS, tail))
    plain = MarginalBasis(market.returns, market.reference,
                          MarginalConfig(reference_components=0, min_excess_bps=None, recent_periods=tail))
    weighted = MarginalBasis(market.returns, market.reference,
                             MarginalConfig(reference_components=0, min_excess_bps=None, recent_periods=tail,
                                            regime_periods=tail, regime_weight=3.0))
    a, b = plain.evaluate(decayed), weighted.evaluate(decayed)
    assert b.partial_ic < a.partial_ic and b.partial_ic_t < a.partial_ic_t
    assert b.recent_partial_ic == pytest.approx(a.recent_partial_ic)


def test_out_of_sample_gates_pass_through_config():
    cfg = load_config(overrides={"admission": {
        "policy": "marginal", "blocks": 4, "min_block_t": 1.0, "retention": [[0, 0.25], [4, 0.45]],
        "min_shrunk_partial_ic": 0.002, "regime_periods": 500, "regime_weight": 2.0}})
    marginal = MarginalConfig(**cfg.admission.marginal_settings())
    assert marginal.blocks == 4 and marginal.retention == ((0.0, 0.25), (4.0, 0.45))
    assert marginal.retention_at(4.0) == 0.45 and marginal.regime_weight == 2.0
    with pytest.raises(ValueError, match="blocks"):
        load_config(overrides={"admission": {"policy": "marginal", "blocks": 1}})


def test_zone_score_defines_the_zone_and_joins_the_basis(market):
    config = MarginalConfig(reference_components=0, min_excess_bps=None, zone_fraction=0.4)
    plain = MarginalBasis(market.returns, market.reference, config)
    staged = MarginalBasis(market.returns, market.reference, config, zone_score=market.reference[0])
    assert staged.size == plain.size + 1
    # the zone is the best 40% by the stage-1 score, not by the basis's own prediction
    ranks = staged._ranks(staged._gather(market.reference[0]))
    expected = staged._top_mask(ranks, 0.4)
    assert np.array_equal(staged._zone, expected) and not np.array_equal(plain._zone, expected)
    # the stage-1 score itself has no marginal value once it is in the basis
    assert abs(staged.evaluate(market.reference[0]).partial_ic) < 0.005
    novel = staged.evaluate(market.novel)
    assert novel.partial_ic_t > 3 and novel.top_partial_ic > 0


def test_zone_residual_refits_inside_the_zone(market):
    base = MarginalConfig(reference_components=0, min_excess_bps=None, zone_fraction=0.4)
    basis = MarginalBasis(market.returns, market.reference, base, zone_score=market.reference[0])
    pooled = basis.evaluate(market.novel)
    within = basis.evaluate(market.novel, replace(base, zone_residual=True))
    assert within.partial_ic == pooled.partial_ic  # the whole-panel statistics do not change
    assert within.top_partial_ic != pooled.top_partial_ic and within.top_partial_ic > 0
    assert within.top_partial_ic_t > 2


def test_zone_leaf_passes_through_config_and_resolves_a_panel_leaf():
    from factorminer.application.marginal_admission import zone_panel

    cfg = load_config(overrides={"admission": {"policy": "marginal", "zone_leaf": "s1", "zone_residual": True}})
    settings = cfg.admission.marginal_settings()
    assert "zone_leaf" not in settings and MarginalConfig(**settings).zone_residual
    panel = np.ones((3, 4))
    assert zone_panel(cfg.admission, {"$s1": panel}) is not None
    assert zone_panel(cfg.admission, {"s1": panel}) is not None
    assert zone_panel(AdmissionConfig(), {}) is None
    with pytest.raises(KeyError):
        zone_panel(cfg.admission, {"$close": panel})
