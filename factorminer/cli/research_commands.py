"""Research-mode CLI commands.

These commands expose experimental research workflows.  Keeping them outside
the CLI root prevents their optional scientific dependencies from becoming
part of every command's import path.
"""

from __future__ import annotations

import json

import click
import numpy as np

from factorminer.cli.app import _create_llm_provider
from factorminer.cli.context import main


@main.command("verify-evidence")
@click.pass_context
def verify_evidence(ctx: click.Context) -> None:
    """Verify every content-addressed evidence pack in the output directory."""
    from factorminer.application.evidence_service import EvidenceStore

    result = EvidenceStore(ctx.obj["output_dir"]).verify_all()
    click.echo(json.dumps(result, indent=2, sort_keys=True))
    if not result["ok"]:
        raise click.ClickException("Evidence verification failed")


@main.command("ingest-research")
@click.argument("note_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--mock", is_flag=True, help="Use mock LLM provider (no API calls).")
@click.option(
    "--eligibility-mode",
    type=click.Choice(["ohlcv_only", "alt_enabled"]),
    default="ohlcv_only",
    show_default=True,
    help="A-layer gate: ohlcv_only (default) or alt_enabled (keep fragments "
    "that map onto registered non-OHLCV leaves such as $eps).",
)
@click.pass_context
def ingest_research(
    ctx: click.Context,
    note_path: str,
    mock: bool,
    eligibility_mode: str,
) -> None:
    """Absorb a research report fragment into reusable mechanism cues."""
    from factorminer.application.research_knowledge import ResearchKnowledgeStore
    from factorminer.architecture.research_absorption import (
        ResearchAbsorptionService,
        read_research_note,
    )

    cfg = ctx.obj["config"]
    provider = _create_llm_provider(cfg, mock)
    service = ResearchAbsorptionService(
        llm_provider=provider,
        eligibility_mode=eligibility_mode,
    )
    note = read_research_note(note_path)
    store = ResearchKnowledgeStore(ctx.obj["output_dir"])

    click.echo("FactorMiner -- Research Ingestion (RMA)")
    click.echo("=" * 60)
    click.echo(f"Source: {note.source}")

    source, hypothesis = store.ingest(note, service)
    click.echo(f"A-layer verdict: {'KEEP' if source.eligible else 'DROP'}")
    click.echo(f"Reason:          {source.eligibility_reason}")
    click.echo(f"Source ID:       {source.source_id}")

    if hypothesis is None:
        click.echo("=" * 60)
        click.echo("Fragment recorded as ineligible; it will not enter generation context.")
        return

    click.echo(f"Hypothesis ID:   {hypothesis.hypothesis_id}")
    click.echo(f"Mechanism family: {hypothesis.mechanism_family}")
    click.echo(f"Fine family:      {hypothesis.fine_family}")
    click.echo(f"Archetype name:   {hypothesis.name}")
    click.echo(f"Mechanism role:   {hypothesis.mechanism_role}")
    click.echo("Research paths:")
    for path in hypothesis.research_paths:
        click.echo(f"  - {path}")
    click.echo("=" * 60)


@main.command("sealed-search")
@click.option(
    "--agreement-rule",
    type=click.Choice(["majority", "unanimous", "all_but_one", "threshold"]),
    default="majority",
    show_default=True,
    help="Multi-evaluator agreement rule for promotion eligibility.",
)
@click.option(
    "--min-agree",
    type=int,
    default=2,
    show_default=True,
    help="Minimum evaluator passes when --agreement-rule=threshold.",
)
@click.option(
    "--no-llm-judge",
    is_flag=True,
    help="Disable the optional LLM-as-judge persona (numeric panel only).",
)
@click.option(
    "--demo",
    is_flag=True,
    help="Run the built-in synthetic disagreement demo (no library/data needed).",
)
@click.option("--mock", is_flag=True, help="Use mock LLM provider if LLM judge is enabled.")
@click.pass_context
def sealed_search(
    ctx: click.Context,
    agreement_rule: str,
    min_agree: int,
    no_llm_judge: bool,
    demo: bool,
    mock: bool,
) -> None:
    """Run sealed multi-evaluator promotion in opt-in research mode."""
    from factorminer.architecture.sealed_joint_search import (
        RESEARCH_MODE_CAVEAT,
        AgreementRule,
        CandidateObservation,
        SealedJointSearchConfig,
        SealedJointSearchEngine,
    )

    cfg = ctx.obj["config"]
    provider = None
    if not no_llm_judge:
        provider = _create_llm_provider(cfg, True)

    engine = SealedJointSearchEngine(
        SealedJointSearchConfig(
            enabled=True,
            agreement_rule=AgreementRule(agreement_rule),
            min_agree=min_agree,
            include_llm_judge=not no_llm_judge,
            retain_internal_scores=True,
        ),
        llm_provider=provider,
    )

    click.echo("=" * 60)
    click.echo("FactorMiner -- Sealed Joint Search (research mode)")
    click.echo("=" * 60)
    click.echo(f"Agreement rule: {agreement_rule}")
    click.echo(f"Evaluators:     {', '.join(engine.evaluator_ids)}")
    click.echo(RESEARCH_MODE_CAVEAT)
    click.echo("-" * 60)

    if not demo:
        click.echo("No library input supplied — running built-in synthetic demo.")

    observations = [
        CandidateObservation(
            name="high_ic_brittle",
            formula="CsRank(Delta($close, 1))",
            ic_paper_mean=0.08,
            ic_mean=0.08,
            ic_std=0.12,
            icir=0.67,
            ic_win_rate=0.62,
            intervention_robustness=0.15,
            cpcv_ic_std=0.10,
            cpcv_ic_mean=0.08,
            max_library_dependence=0.25,
            novelty_score=0.75,
        ),
        CandidateObservation(
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
            max_library_dependence=0.92,
            novelty_score=0.08,
        ),
        CandidateObservation(
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
        ),
        CandidateObservation(
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
        ),
    ]

    report = engine.evaluate_batch(observations)
    click.echo(f"Candidates:     {report.n_candidates}")
    click.echo(f"Promoted:       {report.promoted_names()}")
    click.echo(f"Rejected:       {report.rejected_names()}")
    click.echo(f"Disagreement:   {report.disagreement_rate:.2%}")
    click.echo(f"Mean agreement: {report.mean_agreement_fraction:.2%}")
    click.echo("-" * 60)
    for decision in report.decisions:
        fb = decision.feedback
        status = "PROMOTED" if decision.promoted else "held"
        click.echo(
            f"  [{status:8s}] {decision.observation.name:18s} "
            f"passed {decision.n_passed}/{decision.n_evaluators} "
            f"rank={decision.batch_rank} "
            f"disagree={decision.disagreement}"
        )
        if fb is not None:
            click.echo(
                f"             personas+={list(fb.passed_personas)} "
                f"personas-={list(fb.failed_personas)}"
            )
    click.echo("=" * 60)
    click.echo("Prompt-safe sealed feedback (no raw evaluator scores):")
    for payload in report.sealed_feedback_batch():
        click.echo(
            f"  {payload['candidate_name']}: "
            f"{payload['n_passed']}/{payload['n_evaluators']} "
            f"promoted={payload['promoted']}"
        )
    click.echo("=" * 60)


@main.command("model-co-optimize")
@click.option(
    "--model-kind",
    type=click.Choice(["ridge", "lasso", "xgboost", "corr_graphsage"]),
    default="ridge",
    show_default=True,
    help="Downstream model family to fit on the factor library signals.",
)
@click.option(
    "--train-objective",
    type=click.Choice(["mse", "margin_pairwise", "listnet", "bpr"]),
    default="mse",
    show_default=True,
    help="Training objective. Ranking losses re-fit linear models; xgboost uses rank:pairwise.",
)
@click.option("--alpha", type=float, default=1.0, show_default=True, help="L2/L1 strength (linear models).")
@click.option(
    "--train-fraction",
    type=float,
    default=0.7,
    show_default=True,
    help="Fraction of periods used for training (rest held out).",
)
@click.option(
    "--graph-corr-threshold",
    type=float,
    default=0.3,
    show_default=True,
    help="Absolute return-correlation threshold for corr_graphsage edges.",
)
@click.option(
    "--graph-hidden-dim",
    type=int,
    default=8,
    show_default=True,
    help="Hidden width of the corr_graphsage encoder.",
)
@click.option(
    "--permutation-repeats",
    type=int,
    default=10,
    show_default=True,
    help="Repeats for held-out permutation importance.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="RNG seed.")
@click.option("--mock", is_flag=True, help="Run against a built-in synthetic factor panel (no library needed).")
@click.option("--json", "json_output", is_flag=True, help="Emit the co-optimization report as JSON.")
def model_co_optimize_cmd(
    model_kind: str,
    train_objective: str,
    alpha: float,
    train_fraction: float,
    graph_corr_threshold: float,
    graph_hidden_dim: int,
    permutation_repeats: int,
    seed: int,
    mock: bool,
    json_output: bool,
) -> None:
    """Fit a downstream model and rank factor contributions."""
    from factorminer.evaluation.model_zoo import ModelZooConfig, ModelZooEvaluator

    if not mock:
        click.echo("No library input supplied — running built-in synthetic mock panel.")

    rng = np.random.default_rng(seed)
    assets, periods = 20, 80
    strong = rng.standard_normal((assets, periods))
    weak = rng.standard_normal((assets, periods))
    noise_a = rng.standard_normal((assets, periods))
    noise_b = rng.standard_normal((assets, periods))
    returns = 0.75 * strong + 0.15 * weak + 0.4 * rng.standard_normal((assets, periods))
    factor_signals = {1: strong, 2: weak, 3: noise_a, 4: noise_b}
    factor_names = {1: "alpha_strong", 2: "alpha_weak", 3: "noise_a", 4: "noise_b"}

    config = ModelZooConfig(
        model_kind=model_kind,
        train_objective=train_objective,
        alpha=alpha,
        train_fraction=train_fraction,
        graph_corr_threshold=graph_corr_threshold,
        graph_hidden_dim=graph_hidden_dim,
        permutation_repeats=permutation_repeats,
        seed=seed,
        xgb_n_estimators=40,
        xgb_max_depth=3,
    )

    click.echo("=" * 60)
    click.echo("FactorMiner -- Model Co-Optimize (downstream zoo)")
    click.echo("=" * 60)
    click.echo(f"Model kind:       {model_kind}")
    click.echo(f"Train objective:  {train_objective}")
    click.echo(f"Train fraction:   {train_fraction}")
    click.echo(f"Panel:            {assets} assets x {periods} periods (mock)")
    click.echo("-" * 60)

    try:
        report = ModelZooEvaluator().evaluate(
            factor_signals, factor_names, returns, config=config, iteration=0
        )
    except RuntimeError as exc:
        click.echo(f"Error: {exc}")
        raise click.Abort() from exc

    if json_output:
        click.echo(json.dumps(report.to_dict(), indent=2, default=str))
        return

    click.echo(f"Held-out IC:      {report.held_out_ic:.4f}")
    click.echo(f"Held-out R^2:     {report.held_out_r2:.4f}")
    click.echo(f"Held-out Sharpe:  {report.held_out_sharpe:.4f}")
    click.echo(f"Baseline EQ IC:   {report.baseline_equal_weight_ic:.4f}")
    click.echo(f"Train/test n:     {report.n_train_samples}/{report.n_test_samples}")
    if report.neighbor_influence_summary:
        click.echo(f"Neighbors:        {report.neighbor_influence_summary}")
    click.echo("-" * 60)
    click.echo(f"{'Rank':>4s}  {'Factor':<20s}  {'PermImp':>10s}  {'Coef':>10s}  {'dIC':>8s}")
    for contribution in report.contributions:
        coef = (
            f"{contribution.coefficient:.4f}"
            if contribution.coefficient is not None
            else "n/a"
        )
        click.echo(
            f"{contribution.rank:4d}  {contribution.factor_name:<20s}  "
            f"{contribution.permutation_importance_mean:10.4f}  {coef:>10s}  "
            f"{contribution.ensemble_marginal_delta_ic:8.4f}"
        )
    click.echo("=" * 60)
