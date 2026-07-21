"""Factor-mining CLI commands and their presentation layer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from factorminer.cli import app as _app
from factorminer.cli.context import main
from factorminer.utils.config import load_config

logger = logging.getLogger(__name__)


@main.command("quickstart")
@click.option(
    "--output-dir",
    "quickstart_output_dir",
    type=click.Path(file_okay=False),
    default="/tmp/factorminer-quickstart",
    show_default=True,
    help="Directory for quickstart artifacts.",
)
@click.option("--iterations", "-n", type=int, default=2, show_default=True)
@click.option("--batch-size", "-b", type=int, default=8, show_default=True)
@click.option("--target", "-t", type=int, default=2, show_default=True)
@click.pass_context
def quickstart(
    ctx: click.Context,
    quickstart_output_dir: str,
    iterations: int,
    batch_size: int,
    target: int,
) -> None:
    """Run a mock end-to-end mining session and generate a static report."""
    output_dir = Path(quickstart_output_dir)
    starter = _app._starter_config()
    starter["output_dir"] = str(output_dir)
    starter["mining"].update(
        target_library_size=target,
        batch_size=batch_size,
        max_iterations=iterations,
    )
    starter["llm"]["batch_candidates"] = batch_size
    cfg = load_config(overrides=starter)
    setattr(cfg, "_raw", starter)

    original_cfg = ctx.obj["config"]
    original_output_dir = ctx.obj["output_dir"]
    ctx.obj.update(config=cfg, output_dir=output_dir)

    click.echo("Running doctor with mock quickstart settings...")
    checks = _app._doctor_checks(cfg, starter, output_dir)
    _app._print_doctor_report(checks)
    if any(item["status"] == "error" for item in checks):
        ctx.obj.update(config=original_cfg, output_dir=original_output_dir)
        raise click.Abort()

    try:
        ctx.invoke(
            mine,
            iterations=iterations,
            batch_size=batch_size,
            target=target,
            resume=None,
            mock=True,
            data_path=None,
        )
    finally:
        ctx.obj.update(config=original_cfg, output_dir=original_output_dir)

    library_path = output_dir / "factor_library.json"
    session_log_path = output_dir / "session_log.json"
    report_path = output_dir / "quickstart_report.html"
    if library_path.exists():
        from factorminer.evaluation.report_viewer import generate_report

        generate_report(
            library_path,
            session_log_source=session_log_path if session_log_path.exists() else None,
            format="html",
            output_path=report_path,
        )
        click.echo(f"Static report written to: {report_path}")
    else:
        click.echo("Quickstart completed without a factor_library.json artifact.")

    click.echo("\nNext real-data commands")
    click.echo("-" * 60)
    click.echo("uv run factorminer validate-data path/to/market_data.csv")
    click.echo("uv run factorminer init-config factorminer.local.yaml")
    click.echo(
        "uv run factorminer -c factorminer.local.yaml -o output-real "
        "mine --data path/to/market_data.csv"
    )


@main.command()
@click.option("--iterations", "-n", type=int, default=None, help="Override max_iterations.")
@click.option("--batch-size", "-b", type=int, default=None, help="Override batch_size.")
@click.option("--target", "-t", type=int, default=None, help="Override target_library_size.")
@click.option("--resume", type=click.Path(exists=True), default=None, help="Resume from a saved library.")
@click.option("--mock", is_flag=True, help="Use mock data and mock LLM provider (for testing).")
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.pass_context
def mine(
    ctx: click.Context,
    iterations: int | None,
    batch_size: int | None,
    target: int | None,
    resume: str | None,
    mock: bool,
    data_path: str | None,
) -> None:
    """Run a factor mining session."""
    cfg = ctx.obj["config"]
    output_dir = ctx.obj["output_dir"]
    if iterations is not None:
        cfg.mining.max_iterations = iterations
    if batch_size is not None:
        cfg.mining.batch_size = batch_size
    if target is not None:
        cfg.mining.target_library_size = target
    try:
        cfg.validate()
    except ValueError as exc:
        click.echo(f"Configuration error: {exc}")
        raise click.Abort() from exc

    click.echo("=" * 60)
    click.echo("FactorMiner -- Mining Session")
    click.echo("=" * 60)
    click.echo(f"  Target library size: {cfg.mining.target_library_size}")
    click.echo(f"  Batch size:          {cfg.mining.batch_size}")
    click.echo(f"  Max iterations:      {cfg.mining.max_iterations}")
    click.echo(f"  IC threshold:        {cfg.mining.ic_threshold}")
    click.echo(f"  Correlation limit:   {cfg.mining.correlation_threshold}")
    click.echo(f"  Output directory:    {output_dir}")
    click.echo("-" * 60)

    try:
        dataset = _app._load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as exc:
        click.echo(f"Error loading data: {exc}")
        raise click.Abort() from exc
    click.echo(
        f"  Data loaded: {len(dataset.asset_ids)} assets x "
        f"{len(dataset.timestamps)} periods"
    )
    click.echo("  Preparing data tensors...")
    provider = _app._create_llm_provider(cfg, mock)
    library = None
    if resume:
        click.echo(f"  Resuming from: {resume}")
        library = _app._load_library_from_path(resume)

    from factorminer.application.runtime_context import build_run_context
    from factorminer.core.ralph_loop import RalphLoop

    run_context = build_run_context(cfg, output_dir=output_dir, dataset=dataset, mock=mock)
    click.echo("-" * 60)
    click.echo("Starting Ralph Loop...")

    def progress(iteration: int, stats: dict) -> None:
        click.echo(
            f"  Iteration {iteration:3d}: Library={stats.get('library_size', 0)}, "
            f"Admitted={stats.get('admitted', 0)}, "
            f"Yield={stats.get('yield_rate', 0) * 100:.1f}%"
        )

    try:
        loop = RalphLoop(
            config=cfg,
            data_tensor=dataset.data_tensor,
            returns=dataset.returns,
            llm_provider=provider,
            library=library,
            run_context=run_context,
        )
        result_library = loop.run(callback=progress)
    except KeyboardInterrupt:
        click.echo("\nMining interrupted by user.")
        return
    except Exception as exc:
        click.echo(f"Mining error: {exc}")
        logger.exception("Mining failed")
        raise click.Abort() from exc

    lib_path = _app._save_result_library(result_library, output_dir)
    click.echo("=" * 60)
    click.echo(f"Mining complete! Library size: {result_library.size}")
    click.echo(f"Library saved to: {lib_path}")
    click.echo("=" * 60)


@main.command()
@click.option("--iterations", "-n", type=int, default=None, help="Override max_iterations.")
@click.option("--batch-size", "-b", type=int, default=None, help="Override batch_size.")
@click.option("--target", "-t", type=int, default=None, help="Override target_library_size.")
@click.option("--resume", type=click.Path(exists=True), default=None, help="Resume from a saved library.")
@click.option("--causal/--no-causal", default=None, help="Enable/disable causal validation.")
@click.option("--regime/--no-regime", default=None, help="Enable/disable regime-conditional evaluation.")
@click.option("--debate/--no-debate", default=None, help="Enable/disable multi-specialist debate generation.")
@click.option("--canonicalize/--no-canonicalize", default=None, help="Enable/disable SymPy canonicalization.")
@click.option("--mock", is_flag=True, help="Use mock data and mock LLM provider (for testing).")
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.pass_context
def helix(
    ctx: click.Context,
    iterations: int | None,
    batch_size: int | None,
    target: int | None,
    resume: str | None,
    causal: bool | None,
    regime: bool | None,
    debate: bool | None,
    canonicalize: bool | None,
    mock: bool,
    data_path: str | None,
) -> None:
    """Run the enhanced Helix Loop with configured evaluation features."""
    cfg = ctx.obj["config"]
    if iterations is not None:
        cfg.mining.max_iterations = iterations
    if batch_size is not None:
        cfg.mining.batch_size = batch_size
    if target is not None:
        cfg.mining.target_library_size = target
    if causal is not None:
        cfg.phase2.causal.enabled = causal
    if regime is not None:
        cfg.phase2.regime.enabled = regime
    if debate is not None:
        cfg.phase2.debate.enabled = debate
    if canonicalize is not None:
        if canonicalize:
            cfg.phase2.helix.enabled = True
        cfg.phase2.helix.enable_canonicalization = canonicalize
    try:
        cfg.validate()
    except ValueError as exc:
        click.echo(f"Configuration error: {exc}")
        raise click.Abort() from exc

    output_dir = ctx.obj["output_dir"]
    enabled_features = _app._active_phase2_features(cfg)
    click.echo("HelixFactor Phase 2 mining engine.")
    click.echo(
        f"  Target: {cfg.mining.target_library_size} | "
        f"Batch: {cfg.mining.batch_size} | "
        f"Max iterations: {cfg.mining.max_iterations}"
    )
    click.echo(f"  Output directory: {output_dir}")
    click.echo(
        f"  Active Phase 2 features: {', '.join(enabled_features)}"
        if enabled_features
        else "  No Phase 2 features enabled. Configure phase2.* to enable features."
    )
    if resume:
        click.echo(f"  Resuming from: {resume}")
    try:
        dataset = _app._load_runtime_dataset_for_analysis(cfg, data_path, mock)
    except Exception as exc:
        click.echo(f"Error loading data: {exc}")
        raise click.Abort() from exc

    click.echo("  Preparing data tensors...")
    provider = _app._create_llm_provider(cfg, mock)
    library = _app._load_library_from_path(resume) if resume else None
    from factorminer.application.runtime_context import build_run_context
    from factorminer.core.helix_loop import HelixLoop

    run_context = build_run_context(cfg, output_dir=output_dir, dataset=dataset, mock=mock)
    phase2_configs = _app._build_phase2_runtime_configs(cfg)
    volume = (
        _app._extract_capacity_volume(dataset.data_tensor)
        if cfg.phase2.capacity.enabled
        else None
    )
    click.echo("-" * 60)
    click.echo("Starting Helix Loop...")

    def progress(iteration: int, stats: dict) -> None:
        message = (
            f"  Iteration {iteration:3d}: Library={stats.get('library_size', 0)}, "
            f"Admitted={stats.get('admitted', 0)}, "
            f"Yield={stats.get('yield_rate', 0) * 100:.1f}%"
        )
        if stats.get("canonical_duplicates_removed", 0):
            message += f", CanonDupes={stats['canonical_duplicates_removed']}"
        if stats.get("phase2_rejections", 0):
            message += f", Phase2Reject={stats['phase2_rejections']}"
        click.echo(message)

    try:
        loop = HelixLoop(
            config=cfg,
            data_tensor=dataset.data_tensor,
            returns=dataset.returns,
            llm_provider=provider,
            library=library,
            run_context=run_context,
            debate_config=phase2_configs["debate_config"],
            enable_knowledge_graph=(
                cfg.phase2.helix.enabled and cfg.phase2.helix.enable_knowledge_graph
            ),
            enable_embeddings=(
                cfg.phase2.helix.enabled and cfg.phase2.helix.enable_embeddings
            ),
            enable_auto_inventor=cfg.phase2.auto_inventor.enabled,
            auto_invention_interval=cfg.phase2.auto_inventor.invention_interval,
            canonicalize=(
                cfg.phase2.helix.enabled and cfg.phase2.helix.enable_canonicalization
            ),
            forgetting_lambda=cfg.phase2.helix.forgetting_lambda,
            causal_config=phase2_configs["causal_config"],
            regime_config=phase2_configs["regime_config"],
            capacity_config=phase2_configs["capacity_config"],
            significance_config=phase2_configs["significance_config"],
            volume=volume,
        )
        result_library = loop.run(callback=progress)
    except KeyboardInterrupt:
        click.echo("\nHelix mining interrupted by user.")
        return
    except Exception as exc:
        click.echo(f"Helix mining error: {exc}")
        logger.exception("Helix loop failed")
        raise click.Abort() from exc

    lib_path = _app._save_result_library(result_library, output_dir)
    click.echo("=" * 60)
    click.echo(f"Helix mining complete! Library size: {result_library.size}")
    click.echo(f"Library saved to: {lib_path}")
    click.echo("=" * 60)


@main.command("retrieval-smoke")
@click.option(
    "--embeddings/--no-embeddings",
    default=False,
    help="Include dense embedder ranks (hash/TF-IDF fallback; no forced download).",
)
@click.option(
    "--rerank/--no-rerank",
    default=False,
    help="Enable lightweight reranking over the fused top pool.",
)
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
def retrieval_smoke(embeddings: bool, rerank: bool, json_output: bool) -> None:
    """Run hybrid retrieval quality checks on a synthetic labeled set."""
    from factorminer.memory.retrieval import (
        HybridRetrievalConfig,
        retrieval_quality_smoke,
    )

    embedder = None
    if embeddings:
        from factorminer.memory.embeddings import FormulaEmbedder

        embedder = FormulaEmbedder(use_faiss=False)
    result = retrieval_quality_smoke(
        embedder=embedder,
        hybrid_config=HybridRetrievalConfig(enabled=True, enable_rerank=rerank),
    )
    if json_output:
        click.echo(json.dumps(result, indent=2, default=str))
        if not result.get("passed"):
            raise click.Abort()
        return
    click.echo("FactorMiner -- Retrieval Quality Smoke")
    click.echo("=" * 60)
    click.echo(f"  Passed:            {result.get('passed')}")
    click.echo(f"  Hybrid ranking:    {result.get('hybrid_ranking')}")
    click.echo(f"  Heuristic ranking: {result.get('heuristic_ranking')}")
    click.echo(f"  Criterion:         {result.get('criterion')}")
    click.echo("=" * 60)
    if not result.get("passed"):
        raise click.ClickException("retrieval quality smoke failed")
