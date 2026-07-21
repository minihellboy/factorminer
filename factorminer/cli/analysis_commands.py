"""Analysis, portfolio, visualization, and export CLI commands."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np

from factorminer.cli import app as _app
from factorminer.cli.context import main

_analysis_output_path = _app._analysis_output_path
_artifact_map_by_id = _app._artifact_map_by_id
_load_library_from_path = _app._load_library_from_path
_print_recomputed_factor_table = _app._print_recomputed_factor_table
_print_split_summary = _app._print_split_summary
_report_artifact_failures = _app._report_artifact_failures
_select_artifacts_for_ids = _app._select_artifacts_for_ids

logger = logging.getLogger(__name__)


def _load_runtime_dataset_for_analysis(cfg, data_path: str | None, mock: bool):
    """Forward through the compatibility surface used by CLI integrations."""
    return _app._load_runtime_dataset_for_analysis(cfg, data_path, mock)


def _recompute_analysis_artifacts(library, dataset, signal_failure_policy: str):
    """Forward recomputation so existing programmatic patches remain effective."""
    return _app._recompute_analysis_artifacts(library, dataset, signal_failure_policy)

# report
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option(
    "--session-log",
    type=click.Path(exists=True),
    default=None,
    help="Optional session_log.json path.",
)
@click.option(
    "--benchmark",
    "benchmark_paths",
    type=click.Path(exists=True),
    multiple=True,
    help="Optional benchmark JSON path. May be passed multiple times.",
)
@click.option(
    "--format",
    "report_format",
    type=click.Choice(["markdown", "html"]),
    default="markdown",
    show_default=True,
    help="Static report format.",
)
@click.option(
    "--output",
    "-o",
    "report_output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write the report to this path instead of stdout.",
)
@click.option(
    "--mrm-pack/--no-mrm-pack",
    default=False,
    show_default=True,
    help=(
        "Include an MRM validation pack (model inventory, conceptual soundness, "
        "outcomes analysis, ongoing monitoring). Evidence for a qualified reviewer "
        "only — not a compliance determination."
    ),
)
@click.option(
    "--attest-rationale",
    "attest_factor_ids",
    multiple=True,
    type=str,
    help=(
        "Human attestation: mark economic rationale for this factor id/name as "
        "attested. Repeatable. Never set automatically by generation code."
    ),
)
def report(
    library_path: str,
    session_log: str | None,
    benchmark_paths: tuple[str, ...],
    report_format: str,
    report_output: str | None,
    mrm_pack: bool,
    attest_factor_ids: tuple[str, ...],
) -> None:
    """Generate a static report from FactorMiner artifacts."""
    import json
    from pathlib import Path

    from factorminer.core.provenance import attest_economic_rationale
    from factorminer.evaluation.report_viewer import generate_report

    # Optional human attestation mutates a working copy of the library JSON only
    # when writing an output report path alongside --attest-rationale.
    library_source: str | Path | dict = library_path
    if attest_factor_ids:
        path = Path(library_path)
        if path.is_dir():
            path = path / "factor_library.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        wanted = {str(x) for x in attest_factor_ids}
        for factor in payload.get("factors", []):
            fid = str(factor.get("id", ""))
            name = str(factor.get("name", ""))
            if fid not in wanted and name not in wanted:
                continue
            prov = factor.setdefault("provenance", {})
            rationale = prov.get("economic_rationale") or factor.get("economic_rationale") or {}
            prov["economic_rationale"] = attest_economic_rationale(rationale, attestor="cli-human")
        library_source = payload
        if report_output:
            attested_path = Path(report_output).with_suffix(".attested_library.json")
            attested_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            click.echo(f"Attested library snapshot written to: {attested_path}")

    rendered = generate_report(
        library_source,
        session_log_source=session_log,
        benchmark_sources=benchmark_paths,
        format=report_format,
        output_path=report_output,
        include_mrm_pack=mrm_pack,
    )

    if report_output is None:
        click.echo(rendered)
    else:
        click.echo(f"Report written to: {report_output}")



# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for evaluation.")
@click.option("--period", type=click.Choice(["train", "test", "both"]), default="test", help="Evaluation period.")
@click.option("--top-k", type=int, default=None, help="Evaluate only the top-K factors by IC.")
@click.pass_context
def evaluate(
    ctx: click.Context,
    library_path: str,
    data_path: str | None,
    mock: bool,
    period: str,
    top_k: int | None,
) -> None:
    """Evaluate a factor library on historical data."""
    cfg = ctx.obj["config"]
    signal_failure_policy = cfg.evaluation.signal_failure_policy

    click.echo("=" * 60)
    click.echo("FactorMiner -- Factor Evaluation")
    click.echo("=" * 60)

    # Load library
    library = _load_library_from_path(library_path)

    try:
        dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    click.echo(f"  Period: {period} | Backend: {cfg.evaluation.backend}")
    click.echo(
        f"  Data: {len(dataset.asset_ids)} assets x {len(dataset.timestamps)} periods"
    )

    artifacts = _recompute_analysis_artifacts(library, dataset, signal_failure_policy)
    failures = _report_artifact_failures(artifacts, header="Evaluation warnings")

    from factorminer.evaluation.runtime import analysis_split_names, select_top_k

    split_names = analysis_split_names(period)
    selection_split = "train" if period == "both" else split_names[0]
    selected = select_top_k(artifacts, selection_split, top_k)
    if not selected:
        click.echo("No factors successfully recomputed for evaluation.")
        if signal_failure_policy == "reject" and failures:
            raise click.Abort()
        raise click.Abort()

    if top_k is not None and top_k < len([a for a in artifacts if a.succeeded]):
        if period == "both":
            click.echo(
                f"  Evaluating top {top_k} factors by train paper IC for train/test comparison"
            )
        else:
            click.echo(f"  Evaluating top {top_k} factors by {selection_split} paper IC")

    for split_name in split_names:
        click.echo("-" * 60)
        click.echo(f"Split: {split_name}")
        _print_recomputed_factor_table(selected, split_name)
        _print_split_summary(selected, split_name)

    if period == "both" and selected:
        click.echo("-" * 60)
        click.echo("Decay summary (train -> test)")
        click.echo(
            f"{'ID':>4s}  {'Name':<35s}  {'Train Paper IC':>14s}  "
            f"{'Test Paper IC':>13s}  {'Delta':>8s}"
        )
        click.echo("-" * 80)
        for artifact in selected:
            train_ic = artifact.split_stats["train"].get(
                "ic_paper_mean",
                artifact.split_stats["train"]["ic_abs_mean"],
            )
            test_ic = artifact.split_stats["test"].get(
                "ic_paper_mean",
                artifact.split_stats["test"]["ic_abs_mean"],
            )
            click.echo(
                f"{artifact.factor_id:4d}  {artifact.name:<35s}  "
                f"{train_ic:10.4f}  {test_ic:9.4f}  {test_ic - train_ic:8.4f}"
            )

    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for combination.")
@click.option(
    "--fit-period",
    type=click.Choice(["train", "test", "both"]),
    default="train",
    help="Split used for top-k selection and model/weight fitting.",
)
@click.option(
    "--eval-period",
    type=click.Choice(["train", "test", "both"]),
    default="test",
    help="Split used to evaluate the combined signal.",
)
@click.option(
    "--method", "-m",
    type=click.Choice(["equal-weight", "ic-weighted", "orthogonal", "temporal-reweight", "all"]),
    default="all",
    help="Factor combination method.",
)
@click.option(
    "--lookback",
    type=int,
    default=60,
    help="Trailing window size (periods) for --method temporal-reweight.",
)
@click.option(
    "--rebalance-every",
    type=int,
    default=20,
    help="Periods between weight recomputations for --method temporal-reweight.",
)
@click.option(
    "--selection", "-s",
    type=click.Choice(["lasso", "stepwise", "xgboost", "none"]),
    default="none",
    help="Factor selection method to run before combination.",
)
@click.option("--top-k", type=int, default=None, help="Select top-K factors before combining.")
@click.pass_context
def combine(
    ctx: click.Context,
    library_path: str,
    data_path: str | None,
    mock: bool,
    fit_period: str,
    eval_period: str,
    method: str,
    lookback: int,
    rebalance_every: int,
    selection: str,
    top_k: int | None,
) -> None:
    """Run factor combination and selection methods."""
    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]

    click.echo("=" * 60)
    click.echo("FactorMiner -- Factor Combination")
    click.echo("=" * 60)

    # Load library
    library = _load_library_from_path(library_path)

    from factorminer.evaluation.runtime import (
        resolve_split_for_fit_eval,
        select_top_k,
    )

    try:
        dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    artifacts = _recompute_analysis_artifacts(
        library,
        dataset,
        cfg.evaluation.signal_failure_policy,
    )
    failures = _report_artifact_failures(artifacts, header="Combination warnings")

    fit_split = resolve_split_for_fit_eval(fit_period)
    eval_split = resolve_split_for_fit_eval(eval_period)

    selected_artifacts = select_top_k(artifacts, fit_split, top_k)
    if not selected_artifacts:
        click.echo("No factors successfully recomputed for combination.")
        if cfg.evaluation.signal_failure_policy == "reject" and failures:
            raise click.Abort()
        raise click.Abort()

    if top_k is not None and top_k < len([a for a in artifacts if a.succeeded]):
        click.echo(
            f"  Pre-selected top {len(selected_artifacts)} factors by {fit_split} paper IC"
        )

    click.echo(f"  Fit split:  {fit_split}")
    click.echo(f"  Eval split: {eval_split}")
    click.echo(f"  Combining {len(selected_artifacts)} factors")
    click.echo("-" * 60)

    # Run selection if requested
    selected_ids = [artifact.factor_id for artifact in selected_artifacts]
    fit_returns_tn = dataset.get_split(fit_split).returns.T
    fit_factor_signals = {
        artifact.factor_id: artifact.split_signals[fit_split].T
        for artifact in selected_artifacts
    }

    if selection != "none":
        click.echo(f"\n  Running {selection} selection...")
        from factorminer.evaluation.selection import FactorSelector

        selector = FactorSelector()

        try:
            if selection == "lasso":
                results = selector.lasso_selection(fit_factor_signals, fit_returns_tn)
            elif selection == "stepwise":
                results = selector.forward_stepwise(fit_factor_signals, fit_returns_tn)
            elif selection == "xgboost":
                results = selector.xgboost_selection(fit_factor_signals, fit_returns_tn)
            else:
                results = []

            if results:
                selected_ids = [factor_id for factor_id, _ in results]
                click.echo(f"\n  {selection.capitalize()} selection results:")
                click.echo(f"  {'Factor ID':>10s}  {'Score':>10s}")
                click.echo("  " + "-" * 25)
                for fid, score in results[:20]:  # Show top 20
                    click.echo(f"  {fid:10d}  {score:10.4f}")
                click.echo(f"  Total selected: {len(selected_ids)}")
            else:
                click.echo(f"  {selection} selection returned no factors.")
        except ImportError as e:
            click.echo(f"  Selection method '{selection}' requires additional packages: {e}")
        except Exception as e:
            click.echo(f"  Selection error: {e}")
            logger.exception("Selection failed")

    # Run combination methods
    from factorminer.evaluation.combination import FactorCombiner
    from factorminer.evaluation.portfolio import PortfolioBacktester

    combiner = FactorCombiner()
    backtester = PortfolioBacktester()
    artifact_map = _artifact_map_by_id(selected_artifacts)
    eval_factor_signals = {
        factor_id: artifact_map[factor_id].split_signals[eval_split].T
        for factor_id in selected_ids
        if factor_id in artifact_map
    }
    ic_values = {
        factor_id: artifact_map[factor_id].split_stats[fit_split]["ic_mean"]
        for factor_id in eval_factor_signals
    }
    eval_returns_tn = dataset.get_split(eval_split).returns.T

    methods_to_run = []
    if method == "all":
        methods_to_run = ["equal-weight", "ic-weighted", "orthogonal", "temporal-reweight"]
    else:
        methods_to_run = [method]

    for m in methods_to_run:
        click.echo(f"\n  {m.upper()} combination:")
        try:
            if m == "equal-weight":
                composite = combiner.equal_weight(eval_factor_signals)
            elif m == "ic-weighted":
                composite = combiner.ic_weighted(eval_factor_signals, ic_values)
            elif m == "orthogonal":
                composite = combiner.orthogonal(eval_factor_signals)
            elif m == "temporal-reweight":
                composite = combiner.temporal_reweight(
                    eval_factor_signals,
                    eval_returns_tn,
                    lookback=lookback,
                    rebalance_every=rebalance_every,
                    method="ic_weighted",
                )
            else:
                continue

            stats = backtester.quintile_backtest(composite, eval_returns_tn)
            click.echo(f"    IC Mean:      {stats['ic_mean']:.4f}")
            click.echo(f"    Paper IC:     {abs(stats['ic_mean']):.4f}")
            click.echo(f"    ICIR:         {stats['icir']:.4f}")
            click.echo(f"    Long-Short:   {stats['ls_return']:.4f}")
            click.echo(f"    Monotonicity: {stats['monotonicity']:.2f}")
            click.echo(f"    Avg Turnover: {stats['avg_turnover']:.4f}")
        except Exception as e:
            click.echo(f"    Error: {e}")
            logger.exception("Combination method %s failed", m)

    if cfg.research.enabled and cfg.benchmark.mode == "research":
        click.echo("\n  Research model suite:")
        try:
            from factorminer.evaluation.research import run_research_model_suite

            research_reports = run_research_model_suite(
                eval_factor_signals,
                eval_returns_tn,
                cfg.research,
            )
            research_path = output_dir / "research_model_suite.json"
            research_path.write_text(json.dumps(research_reports, indent=2))
            for model_name, report in research_reports.items():
                if not report.get("available", True):
                    click.echo(f"    {model_name}: unavailable ({report.get('error', 'unknown error')})")
                    continue
                click.echo(
                    f"    {model_name}: "
                    f"net IR={report.get('mean_test_net_ir', 0.0):.4f}, "
                    f"ICIR={report.get('mean_test_icir', 0.0):.4f}, "
                    f"stability={report.get('selection_stability', 0.0):.3f}"
                )
            click.echo(f"    Saved: {research_path}")
        except Exception as e:
            click.echo(f"    Research suite error: {e}")
            logger.exception("Research model suite failed")

    click.echo("\n" + "=" * 60)

# ---------------------------------------------------------------------------
# portfolio-construct
# ---------------------------------------------------------------------------

@main.command("portfolio-construct")
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for portfolio construction.")
@click.option(
    "--method", "-m",
    type=click.Choice(["hrp", "risk_parity", "cvar"]),
    default="hrp",
    help="Risk-based portfolio construction method.",
)
@click.option(
    "--top-k", type=int, default=None,
    help="Select top-K factors (by paper IC on the chosen split) before constructing the portfolio.",
)
@click.option(
    "--alpha", type=float, default=0.95,
    help="CVaR confidence level (only used with --method cvar).",
)
@click.option(
    "--period",
    type=click.Choice(["train", "test", "both"]),
    default="test",
    help="Split used to build per-factor return proxies and construct the portfolio.",
)
@click.pass_context
def portfolio_construct(
    ctx: click.Context,
    library_path: str,
    data_path: str | None,
    mock: bool,
    method: str,
    top_k: int | None,
    alpha: float,
    period: str,
) -> None:
    """Construct risk-based portfolio weights over a factor library's strategies.

    Each selected factor's own quintile long-short return series is used as
    an asset-level return proxy; HRP / naive risk parity / CVaR-optimal
    weights are then computed across those proxies (research artifact only,
    not a trade recommendation).
    """
    cfg = ctx.obj["config"]

    click.echo("=" * 60)
    click.echo("FactorMiner -- Risk-Based Portfolio Construction")
    click.echo("=" * 60)

    library = _load_library_from_path(library_path)

    from factorminer.evaluation.runtime import resolve_split_for_fit_eval, select_top_k

    try:
        dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    artifacts = _recompute_analysis_artifacts(
        library,
        dataset,
        cfg.evaluation.signal_failure_policy,
    )
    _report_artifact_failures(artifacts, header="Portfolio construction warnings")

    split = resolve_split_for_fit_eval(period)
    selected_artifacts = select_top_k(artifacts, split, top_k)
    if len(selected_artifacts) < 2:
        click.echo("Need at least 2 successfully recomputed factors for portfolio construction.")
        raise click.Abort()

    click.echo(f"  Split:   {split}")
    click.echo(f"  Method:  {method}")
    click.echo(f"  Assets (factor strategies): {len(selected_artifacts)}")
    click.echo("-" * 60)

    from factorminer.evaluation.portfolio import PortfolioBacktester
    from factorminer.evaluation.risk_portfolio import RiskPortfolioConfig, construct_portfolio

    backtester = PortfolioBacktester()
    split_returns_tn = dataset.get_split(split).returns.T

    asset_ids = []
    return_series = []
    for artifact in selected_artifacts:
        signal_tn = artifact.split_signals[split].T
        stats = backtester.quintile_backtest(signal_tn, split_returns_tn)
        asset_ids.append(artifact.factor_id)
        return_series.append(stats["ls_net_series"])

    returns_matrix = np.column_stack(return_series)
    valid_mask = np.all(np.isfinite(returns_matrix), axis=1)
    returns_matrix = returns_matrix[valid_mask]
    if returns_matrix.shape[0] < 2:
        click.echo("Not enough overlapping valid periods across selected factors.")
        raise click.Abort()

    config = RiskPortfolioConfig(cvar_alpha=alpha)
    try:
        result = construct_portfolio(
            returns_matrix, method=method, asset_ids=asset_ids, config=config
        )
    except Exception as e:
        click.echo(f"Portfolio construction failed: {e}")
        logger.exception("Portfolio construction failed")
        raise click.Abort()

    click.echo(f"  {'Factor ID':>10s}  {'Weight':>8s}")
    click.echo("  " + "-" * 22)
    for factor_id, weight in zip(result.asset_ids, result.weights):
        click.echo(f"  {factor_id:10d}  {weight:8.4f}")

    click.echo("-" * 60)
    click.echo(f"  Method:         {result.method}")
    click.echo(f"  Realized vol:   {result.realized_vol:.6f}")
    click.echo(f"  Realized CVaR:  {result.realized_cvar:.6f}")
    click.echo(f"  Effective N:    {result.effective_n:.2f} (of {len(result.asset_ids)} assets)")
    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# crowding
# ---------------------------------------------------------------------------

@main.command("crowding")
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for crowding diagnostics.")
@click.option(
    "--fixture",
    "fixture_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Offline Ken French-format CSV fixture (default: bundled FF3 fixture).",
)
@click.option(
    "--fetch-consensus/--no-fetch-consensus",
    default=False,
    show_default=True,
    help="Fetch live Ken French panel over HTTPS (fail-closed). Default uses fixture.",
)
@click.option("--top-k", type=int, default=None, help="Score only the top-K factors by IC.")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--window",
    type=int,
    default=63,
    show_default=True,
    help="Rolling window for Lou-Polk CoMetric.",
)
@click.option(
    "--cometric-residual-mode",
    "cometric_residual_mode",
    type=click.Choice(["cross_sectional", "factor_regression"]),
    default="cross_sectional",
    show_default=True,
    help=(
        "CoMetric residualization: 'cross_sectional' (fast, no external data) "
        "or 'factor_regression' (Lou & Polk's actual FF3-regression residuals; "
        "requires the consensus panel to be non-empty, falls back with a "
        "warning otherwise)."
    ),
)
@click.pass_context
def crowding_cmd(
    ctx: click.Context,
    library_path: str,
    data_path: str | None,
    mock: bool,
    fixture_path: str | None,
    fetch_consensus: bool,
    top_k: int | None,
    json_output: bool,
    window: int,
    cometric_residual_mode: str,
) -> None:
    """Score library factors for consensus-overlap / CoMetric crowding risk.

    Research risk annotations only — not a trade timer or mining objective.
    Composes consensus novelty, Lou-Polk CoMetric, and hyperbolic decay
    taxonomy from evaluation/decay.py.
    """
    from factorminer.evaluation.crowding import (
        ConsensusFactorPanel,
        CrowdingConfig,
        score_factor_crowding,
    )

    cfg = ctx.obj["config"]
    signal_failure_policy = cfg.evaluation.signal_failure_policy

    click.echo("=" * 60)
    click.echo("FactorMiner -- Factor Crowding Diagnostics")
    click.echo("=" * 60)

    library = _load_library_from_path(library_path)

    try:
        dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    crowding_cfg = CrowdingConfig(
        cometric_window=window, cometric_residual_mode=cometric_residual_mode
    )
    if fetch_consensus:
        panel = ConsensusFactorPanel.fetch(config=crowding_cfg)
        click.echo(f"  Consensus panel: fetch ({panel.source}) factors={panel.factor_names}")
    else:
        panel = ConsensusFactorPanel.from_fixture(fixture_path, config=crowding_cfg)
        click.echo(f"  Consensus panel: fixture factors={panel.factor_names}")

    if panel.empty:
        click.echo(
            "  WARNING: consensus panel empty (fail-closed). "
            "Overlap scores will be unavailable; CoMetric still runs."
        )

    artifacts = _recompute_analysis_artifacts(library, dataset, signal_failure_policy)
    failures = _report_artifact_failures(artifacts, header="Crowding warnings")
    from factorminer.evaluation.runtime import select_top_k

    selected = select_top_k(artifacts, "test", top_k)
    if not selected:
        click.echo("No factors successfully recomputed for crowding.")
        if signal_failure_policy == "reject" and failures:
            raise click.Abort()
        raise click.Abort()

    ret_full = np.asarray(dataset.returns, dtype=np.float64)

    rows: list[dict] = []
    for artifact in selected:
        signals = artifact.signals_full
        if signals is None:
            signals = artifact.split_signals.get("test") or artifact.split_signals.get("full")
        if signals is None:
            continue

        sig = np.asarray(signals, dtype=np.float64)
        ret = ret_full
        # Align returns time axis to signals when using a split.
        if sig.shape != ret.shape and sig.ndim == 2 and ret.ndim == 2:
            if sig.shape[0] == ret.shape[0] and sig.shape[1] < ret.shape[1]:
                split = dataset.splits.get("test") or dataset.splits.get("full")
                if split is not None:
                    ret = np.asarray(split.returns, dtype=np.float64)
            elif sig.shape == ret.T.shape:
                ret = ret.T

        score = score_factor_crowding(
            signals=sig,
            returns=ret,
            panel=panel,
            formula=artifact.formula or "",
            factor_id=str(artifact.factor_id),
            config=crowding_cfg,
        )
        rows.append(score.to_dict())

    if json_output:
        click.echo(json.dumps({"crowding": rows}, indent=2, default=str))
        return

    click.echo("-" * 60)
    click.echo(
        f"{'ID':>6s}  {'Label':<28s}  {'max|ρ|':>7s}  {'CoMOM':>6s}  "
        f"{'NovMod':>6s}  Detail"
    )
    click.echo("-" * 100)
    for row in rows:
        cons = row.get("consensus") or {}
        com = row.get("cometric") or {}
        max_rho = cons.get("max_abs_rho", 0.0) if cons.get("available") else float("nan")
        comom = com.get("comom", 0.0) if com.get("available") else float("nan")
        click.echo(
            f"{str(row.get('factor_id', '')):>6s}  "
            f"{row.get('composite_label', ''):<28s}  "
            f"{max_rho:7.3f}  {comom:6.3f}  "
            f"{row.get('novelty_modulation', 0.0):6.3f}  "
            f"{(row.get('rationale') or '')[:60]}"
        )
    click.echo("=" * 60)
    click.echo(f"Scored {len(rows)} factor(s). Research risk labels only.")


# ---------------------------------------------------------------------------
# jump-worth (Hypothesis-Redundancy geometric gate)
# ---------------------------------------------------------------------------

@main.command("jump-worth")
@click.argument("library_path", type=click.Path(exists=True))
@click.option(
    "--threshold",
    type=float,
    default=0.45,
    show_default=True,
    help="Recommend LLM jump when jump_worth >= threshold.",
)
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def jump_worth_cmd(
    ctx: click.Context,
    library_path: str,
    threshold: float,
    json_output: bool,
) -> None:
    """Assess whether a non-local LLM jump is worth its cost for this library.

    Implements the Hypothesis-Redundancy geometric gate (arXiv:2606.14386)
    as JumpWorthAssessment: spectral compression × orthogonal escape
    (× residual alignment when a target is available). Advisory only.
    """
    from factorminer.architecture.geometry import (
        assess_llm_jump_worth,
        collect_library_span_matrix,
    )

    library = _load_library_from_path(library_path)
    span = collect_library_span_matrix(library)

    click.echo("=" * 60)
    click.echo("FactorMiner -- LLM Jump-Worth Gate")
    click.echo("=" * 60)
    click.echo(f"  Library size: {library.size}  span shape: {span.shape}")

    if span.size == 0:
        # Probe against empty span.
        probe = np.ones(16, dtype=np.float64)
    else:
        # Default probe: a direction partially outside the span (linear trend).
        probe = np.linspace(-1.0, 1.0, span.shape[0], dtype=np.float64)

    assessment = assess_llm_jump_worth(span, probe, threshold=threshold)
    payload = assessment.to_dict()

    if json_output:
        click.echo(json.dumps(payload, indent=2))
        return

    click.echo(f"  jump_worth:           {assessment.jump_worth:.4f}")
    click.echo(f"  spectral_compression: {assessment.spectral_compression:.4f}")
    click.echo(f"  orthogonal_escape:    {assessment.orthogonal_escape:.4f}")
    click.echo(f"  residual_alignment:   {assessment.residual_alignment:.4f}")
    click.echo(f"  library_rank:         {assessment.library_rank}/{assessment.library_size}")
    click.echo(f"  recommend_llm_jump:   {assessment.recommend_llm_jump}")
    click.echo(f"  rationale:            {assessment.rationale}")
    click.echo("=" * 60)



# ---------------------------------------------------------------------------
# visualize
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for visualization.")
@click.option("--period", type=click.Choice(["train", "test", "both"]), default="test", help="Evaluation split to visualize.")
@click.option("--factor-id", "factor_ids", type=int, multiple=True, help="Specific factor ID(s) to visualize.")
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Top-K factors by split paper IC for set-level plots.",
)
@click.option("--tearsheet", is_flag=True, help="Generate a full factor tear sheet.")
@click.option("--correlation", is_flag=True, help="Plot factor correlation heatmap.")
@click.option("--ic-timeseries", is_flag=True, help="Plot IC time series.")
@click.option("--quintile", is_flag=True, help="Plot quintile returns.")
@click.option("--format", "fmt", type=click.Choice(["png", "pdf", "svg"]), default="png", help="Output format.")
@click.pass_context
def visualize(
    ctx: click.Context,
    library_path: str,
    data_path: str | None,
    mock: bool,
    period: str,
    factor_ids: tuple[int, ...],
    top_k: int | None,
    tearsheet: bool,
    correlation: bool,
    ic_timeseries: bool,
    quintile: bool,
    fmt: str,
) -> None:
    """Generate plots and tear sheets for a factor library."""
    output_dir = ctx.obj["output_dir"]
    cfg = ctx.obj["config"]

    click.echo("=" * 60)
    click.echo("FactorMiner -- Visualization")
    click.echo("=" * 60)

    # Load library
    library = _load_library_from_path(library_path)

    # Determine what to plot
    plot_all = not (tearsheet or correlation or ic_timeseries or quintile)
    if plot_all:
        click.echo("No specific plots requested; generating all available.")
        correlation = True
        ic_timeseries = True
        quintile = True

    output_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"  Output format: {fmt}")
    click.echo(f"  Output dir:    {output_dir}")
    click.echo(f"  Period:        {period}")
    click.echo("-" * 60)

    try:
        dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    artifacts = _recompute_analysis_artifacts(
        library,
        dataset,
        cfg.evaluation.signal_failure_policy,
    )
    failures = _report_artifact_failures(artifacts, header="Visualization warnings")

    from factorminer.evaluation.runtime import (
        analysis_split_names,
        compute_correlation_matrix,
        select_top_k,
    )
    from factorminer.utils.tearsheet import FactorTearSheet
    from factorminer.utils.visualization import (
        plot_correlation_heatmap,
        plot_ic_timeseries,
        plot_quintile_returns,
    )

    split_names = analysis_split_names(period)
    explicit_artifacts = _select_artifacts_for_ids(artifacts, factor_ids)
    if not explicit_artifacts and factor_ids:
        if cfg.evaluation.signal_failure_policy == "reject" and failures:
            raise click.Abort()
        raise click.Abort()

    for split_name in split_names:
        split = dataset.get_split(split_name)
        click.echo(f"  Split: {split_name}")

        if correlation:
            if factor_ids:
                corr_artifacts = explicit_artifacts
            else:
                corr_artifacts = select_top_k(artifacts, split_name, top_k)

            if corr_artifacts:
                click.echo("    Generating correlation heatmap...")
                corr_matrix = compute_correlation_matrix(corr_artifacts, split_name)
                save_path = _analysis_output_path(output_dir, "correlation_heatmap", split_name, fmt)
                plot_correlation_heatmap(
                    corr_matrix,
                    [artifact.name[:20] for artifact in corr_artifacts],
                    title=f"Factor Correlation Heatmap ({split_name})",
                    save_path=save_path,
                )
                click.echo(f"      Saved: {save_path}")
            else:
                click.echo("    Skipped: no successfully recomputed factors for correlation heatmap.")

        factor_artifacts = explicit_artifacts
        if not factor_ids and (ic_timeseries or quintile or tearsheet):
            factor_artifacts = select_top_k(artifacts, split_name, 1)
            if factor_artifacts:
                click.echo(
                    f"    Defaulted to factor #{factor_artifacts[0].factor_id} "
                    f"{factor_artifacts[0].name} for factor-specific plots."
                )

        if ic_timeseries:
            click.echo("    Generating IC time series plot(s)...")
            for artifact in factor_artifacts:
                stats = artifact.split_stats[split_name]
                dates = [str(ts)[:10] for ts in split.timestamps]
                save_path = _analysis_output_path(
                    output_dir,
                    f"ic_timeseries_factor_{artifact.factor_id}",
                    split_name,
                    fmt,
                )
                plot_ic_timeseries(
                    stats["ic_series"],
                    dates,
                    title=f"{artifact.name} IC Time Series ({split_name})",
                    save_path=save_path,
                )
                click.echo(f"      Saved: {save_path}")

        if quintile:
            click.echo("    Generating quintile return plot(s)...")
            for artifact in factor_artifacts:
                stats = artifact.split_stats[split_name]
                save_path = _analysis_output_path(
                    output_dir,
                    f"quintile_returns_factor_{artifact.factor_id}",
                    split_name,
                    fmt,
                )
                plot_quintile_returns(
                    {
                        f"Q{i}": stats[f"Q{i}"] for i in range(1, 6)
                    }
                    | {
                        "long_short": stats["long_short"],
                        "monotonicity": stats["monotonicity"],
                    },
                    title=f"{artifact.name} Quintile Returns ({split_name})",
                    save_path=save_path,
                )
                click.echo(f"      Saved: {save_path}")

        if tearsheet:
            click.echo("    Generating tear sheet(s)...")
            ts = FactorTearSheet()
            dates = [str(ts_)[:10] for ts_ in split.timestamps]
            for artifact in factor_artifacts:
                save_path = _analysis_output_path(
                    output_dir,
                    f"tearsheet_factor_{artifact.factor_id}",
                    split_name,
                    fmt,
                )
                ts.generate(
                    factor_id=artifact.factor_id,
                    factor_name=artifact.name,
                    formula=artifact.formula,
                    signals=artifact.split_signals[split_name],
                    returns=split.returns,
                    dates=dates,
                    save_path=save_path,
                )
                click.echo(f"      Saved: {save_path}")

    click.echo("=" * 60)
    click.echo("Visualization complete.")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@main.command(name="export")
@click.argument("library_path", type=click.Path(exists=True))
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "csv", "formulas", "qlib"]),
    default="json",
    help="Export format.",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.option(
    "--anonymize",
    is_flag=True,
    help="Emit a redacted factor table (formula replaced by a hash) instead of the raw --format export.",
)
@click.pass_context
def export_cmd(
    ctx: click.Context, library_path: str, fmt: str, output: str | None, anonymize: bool,
) -> None:
    """Export a factor library to various formats."""
    output_dir = ctx.obj["output_dir"]

    click.echo("=" * 60)
    click.echo("FactorMiner -- Export")
    click.echo("=" * 60)

    # Load library
    library = _load_library_from_path(library_path)

    # Determine output path
    if output is None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if anonymize:
            anon_ext = "json" if fmt in ("json", "formulas", "qlib") else fmt
            output = str(output_dir / f"library_anonymized.{anon_ext}")
        elif fmt == "formulas":
            output = str(output_dir / "library_formulas.txt")
        elif fmt == "qlib":
            output = str(output_dir / "library_qlib.json")
        else:
            output = str(output_dir / f"library.{fmt}")

    click.echo(f"  Format:  {fmt}{' (anonymized)' if anonymize else ''}")
    click.echo(f"  Output:  {output}")
    click.echo("-" * 60)

    try:
        from factorminer.core.library_io import (
            export_anonymized,
            export_csv,
            export_formulas,
            save_library,
        )
        from factorminer.data.qlib_library import export_formulas_qlib

        if anonymize:
            # The redacted export is a fixed row-table shape; --format only
            # picks its container (csv, or json for the non-tabular formats).
            anon_fmt = "json" if fmt in ("json", "formulas", "qlib") else fmt
            export_anonymized(library, output, fmt=anon_fmt)
            click.echo(f"  Exported {library.size} anonymized factors to {output}")

        elif fmt == "json":
            # save_library expects base path without extension
            out_path = Path(output)
            if out_path.suffix == ".json":
                base = out_path.with_suffix("")
            else:
                base = out_path
            save_library(library, base, save_signals=False)
            click.echo(f"  Exported {library.size} factors to {base}.json")

        elif fmt == "csv":
            export_csv(library, output)
            click.echo(f"  Exported {library.size} factors to {output}")

        elif fmt == "formulas":
            export_formulas(library, output)
            click.echo(f"  Exported {library.size} formulas to {output}")

        elif fmt == "qlib":
            export_formulas_qlib(library, output)
            click.echo(f"  Exported {library.size} Qlib-translated formulas to {output}")

    except Exception as e:
        click.echo(f"Export error: {e}")
        logger.exception("Export failed")
        raise click.Abort()

    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# export-rft-dataset
# ---------------------------------------------------------------------------

@main.command(name="export-rft-dataset")
@click.argument(
    "lifecycle_path",
    type=click.Path(exists=True),
    required=False,
    default=None,
)
@click.option(
    "--output", "-o", "output_path",
    type=click.Path(),
    default=None,
    help="Destination JSONL path (default: <output_dir>/rft_dataset.jsonl).",
)
@click.option(
    "--data", "data_path",
    type=click.Path(exists=True),
    default=None,
    help="Optional market data file used only for regime-aware task bucketing.",
)
@click.option(
    "--mock",
    is_flag=True,
    help="Use mock returns for regime-aware task bucketing (no network).",
)
@click.option(
    "--include-failed-parses",
    is_flag=True,
    help="Keep parse-failed candidates in the exported trajectory.",
)
@click.pass_context
def export_rft_dataset_cmd(
    ctx: click.Context,
    lifecycle_path: str | None,
    output_path: str | None,
    data_path: str | None,
    mock: bool,
    include_failed_parses: bool,
) -> None:
    """Export a reward-annotated offline RFT trajectory dataset as JSONL.

    Exports a reward-annotated training dataset for external reinforcement
    fine-tuning (e.g. GRPO via Verl/vLLM on a GPU host). This command does
    NOT train a model -- policy-weight training requires external GPU
    infrastructure not available in this environment.

    Reads ``factor_lifecycle.jsonl`` from a mining session output directory
    (or a direct path to that file) and writes one JSON object per candidate
    with the documented ``rft_v1`` schema:
    ``(state, action/formula, reward, regime_context)``.
    """
    from factorminer.architecture.rft_export import (
        RFT_EXPORT_HONESTY,
        RFTExportConfig,
        export_rft_dataset,
    )

    session_output = ctx.obj["output_dir"]
    cfg = ctx.obj["config"]

    # Resolve lifecycle source: explicit arg, else the session output dir.
    source = lifecycle_path or str(session_output)
    if output_path is None:
        session_output.mkdir(parents=True, exist_ok=True)
        output_path = str(Path(session_output) / "rft_dataset.jsonl")

    click.echo("=" * 60)
    click.echo("FactorMiner -- Export RFT Dataset (offline only)")
    click.echo("=" * 60)
    click.echo(RFT_EXPORT_HONESTY)
    click.echo("-" * 60)
    click.echo(f"  Lifecycle source: {source}")
    click.echo(f"  Output JSONL:     {output_path}")

    returns = None
    if mock or data_path is not None:
        try:
            dataset = _load_runtime_dataset_for_analysis(cfg, data_path, mock)
            returns = dataset.returns
            click.echo(
                f"  Regime bucketing: enabled "
                f"({len(dataset.asset_ids)} assets x {len(dataset.timestamps)} periods)"
            )
        except Exception as e:
            click.echo(f"  Regime bucketing skipped (data load failed: {e})")
            returns = None
    else:
        click.echo("  Regime bucketing: skipped (pass --mock or --data to enable)")

    export_cfg = RFTExportConfig(include_failed_parses=include_failed_parses)
    try:
        result = export_rft_dataset(
            source,
            output_path,
            returns=returns,
            config=export_cfg,
        )
    except Exception as e:
        click.echo(f"RFT export error: {e}")
        logger.exception("RFT dataset export failed")
        raise click.Abort()

    click.echo("-" * 60)
    click.echo(f"  Records:       {result.n_records}")
    click.echo(f"  Iterations:    {result.n_iterations}")
    click.echo(f"  Schema:        {result.schema_version}")
    click.echo(
        f"  Reward mean/std/min/max: "
        f"{result.reward_mean:.6f} / {result.reward_std:.6f} / "
        f"{result.reward_min:.6f} / {result.reward_max:.6f}"
    )
    if result.regime_task_counts:
        mix = ", ".join(f"{k}={v}" for k, v in sorted(result.regime_task_counts.items()))
        click.echo(f"  Regime tasks:  {mix}")
    if result.manifest_path:
        click.echo(f"  Manifest:      {result.manifest_path}")
    click.echo(f"  Wrote:         {result.path}")
    click.echo("-" * 60)
    click.echo("Reminder: this command does NOT train a model.")
    click.echo("=" * 60)


# ---------------------------------------------------------------------------
