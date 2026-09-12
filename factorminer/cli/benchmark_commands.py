"""Benchmark command group and presentation."""

from __future__ import annotations

import json

import click

from factorminer.cli.context import main


def _print_summary(title: str, payload: dict) -> None:
    """Emit a concise benchmark summary for CLI runs."""
    click.echo("=" * 60)
    click.echo(title)
    click.echo("=" * 60)
    if not payload:
        click.echo("No benchmark results produced.")
        return

    if all(isinstance(value, dict) and "universes" in value for value in payload.values()):
        for baseline, result in payload.items():
            click.echo(f"Baseline: {baseline}")
            click.echo(
                f"  Freeze library: {result.get('freeze_library_size', 0)} "
                f"| Frozen Top-K: {len(result.get('frozen_top_k', []))}"
            )
            for universe, metrics in result.get("universes", {}).items():
                library = metrics.get("library", {})
                click.echo(
                    f"  {universe}: library IC={library.get('ic', 0.0):.4f}, "
                    f"ICIR={library.get('icir', 0.0):.4f}, "
                    f"Avg|rho|={library.get('avg_abs_rho', 0.0):.4f}"
                )
    else:
        click.echo(json.dumps(payload, indent=2))


@main.group()
def benchmark() -> None:
    """Run strict paper/research benchmark workflows."""


@benchmark.command("research-actions")
@click.option("--seeds", default="0,1,2", show_default=True, help="Comma-separated search seeds.")
@click.option("--evaluations", type=click.IntRange(min=2), default=24, show_default=True,
              help="Common evaluation allowance per benchmark episode; not a mining quota.")
@click.pass_context
def research_actions(ctx: click.Context, seeds: str, evaluations: int) -> None:
    """Compare action policies on four frozen controlled worlds."""
    from factorminer.benchmark.research_actions import run_research_action_benchmark

    try:
        selected = tuple(int(seed.strip()) for seed in seeds.split(","))
        payload = run_research_action_benchmark(ctx.obj["output_dir"], seeds=selected,
                                                evaluation_horizon=evaluations)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(payload, indent=2))


@benchmark.command("research-skills")
@click.option("--source-seeds", default="1000,1001,1002,1003,1004,1005,1006,1007", show_default=True)
@click.option("--target-seeds", default="2000,2001,2002,2003,2004,2005,2006,2007,2008,2009", show_default=True)
@click.pass_context
def research_skills(ctx: click.Context, source_seeds: str, target_seeds: str) -> None:
    """Compare frozen procedure transfer on disjoint diagnostic campaign families."""
    from factorminer.benchmark.research_skills import run_skill_transfer_benchmark

    try:
        result = run_skill_transfer_benchmark(ctx.obj["output_dir"],
            source_seeds=tuple(int(s.strip()) for s in source_seeds.split(",")),
            target_seeds=tuple(int(s.strip()) for s in target_seeds.split(",")))
    except (ValueError, FileExistsError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps({k: v for k, v in result.items() if k not in {"runs", "source_runs"}}, indent=2))


def _common_options(fn):
    fn = click.option(
        "--data",
        "data_path",
        type=click.Path(exists=True),
        default=None,
        help="Path to market data file.",
    )(fn)
    fn = click.option(
        "--mock",
        is_flag=True,
        help="Use mock data for benchmark execution.",
    )(fn)
    fn = click.option(
        "--factor-miner-library",
        type=click.Path(exists=True),
        default=None,
        help="Optional saved library for the FactorMiner baseline.",
    )(fn)
    fn = click.option(
        "--factor-miner-no-memory-library",
        type=click.Path(exists=True),
        default=None,
        help="Optional saved library for the FactorMiner No Memory baseline.",
    )(fn)
    return click.pass_context(fn)


@benchmark.command("table1")
@click.option("--baseline", "baselines", multiple=True, help="Restrict to baseline ids.")
@_common_options
def table1(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
    baselines: tuple[str, ...],
) -> None:
    """Run the Top-K freeze benchmark across configured universes."""
    from factorminer.benchmark.runtime import run_table1_benchmark

    payload = run_table1_benchmark(
        ctx.obj["config"],
        ctx.obj["output_dir"],
        data_path=data_path,
        mock=mock,
        baseline_names=list(baselines) if baselines else None,
        factor_miner_library_path=factor_miner_library,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library,
    )
    _print_summary("FactorMiner -- Benchmark Table 1", payload)


@benchmark.command("ablation-memory")
@_common_options
def ablation_memory(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
) -> None:
    """Run the experience-memory ablation benchmark."""
    from factorminer.benchmark.runtime import run_ablation_memory_benchmark

    payload = run_ablation_memory_benchmark(
        ctx.obj["config"],
        ctx.obj["output_dir"],
        data_path=data_path,
        mock=mock,
        factor_miner_library_path=factor_miner_library,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library,
    )
    _print_summary("FactorMiner -- Memory Ablation", payload)


@benchmark.command("ablation-strategy")
@click.option("--baseline", default="factor_miner", help="Runtime baseline id to evaluate.")
@_common_options
def ablation_strategy(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
    baseline: str,
) -> None:
    """Run ablations across memory policy, dependence metric, and backend."""
    from factorminer.benchmark.runtime import run_ablation_strategy_benchmark

    payload = run_ablation_strategy_benchmark(
        ctx.obj["config"],
        ctx.obj["output_dir"],
        baseline=baseline,
        data_path=data_path,
        mock=mock,
    )
    _print_summary("FactorMiner -- Strategy Ablation", payload)


@benchmark.command("cost-pressure")
@click.option("--baseline", default="factor_miner", help="Baseline id to evaluate.")
@_common_options
def cost_pressure(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
    baseline: str,
) -> None:
    """Run transaction-cost pressure testing."""
    from factorminer.benchmark.runtime import run_cost_pressure_benchmark

    payload = run_cost_pressure_benchmark(
        ctx.obj["config"],
        ctx.obj["output_dir"],
        baseline=baseline,
        data_path=data_path,
        mock=mock,
        factor_miner_library_path=factor_miner_library,
    )
    _print_summary("FactorMiner -- Cost Pressure", payload)


@benchmark.command("cpcv")
@click.option("--baseline", default="factor_miner", help="Baseline id to evaluate.")
@_common_options
def cpcv(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
    baseline: str,
) -> None:
    """Run Combinatorial Purged CV and PBO diagnostics."""
    from factorminer.benchmark.runtime import run_cpcv_benchmark

    payload = run_cpcv_benchmark(
        ctx.obj["config"],
        data_path=data_path,
        mock=mock,
        baseline=baseline,
    )
    _print_summary("FactorMiner -- CPCV / PBO", payload)


@benchmark.command("efficiency")
@click.pass_context
def efficiency(ctx: click.Context) -> None:
    """Run operator-level and factor-level efficiency benchmarks."""
    from factorminer.benchmark.runtime import run_efficiency_benchmark

    payload = run_efficiency_benchmark(ctx.obj["config"], ctx.obj["output_dir"])
    _print_summary("FactorMiner -- Efficiency Benchmark", payload)


@benchmark.command("suite")
@_common_options
def suite(
    ctx: click.Context,
    data_path: str | None,
    mock: bool,
    factor_miner_library: str | None,
    factor_miner_no_memory_library: str | None,
) -> None:
    """Run the full benchmark suite."""
    from factorminer.benchmark.runtime import run_benchmark_suite

    payload = run_benchmark_suite(
        ctx.obj["config"],
        ctx.obj["output_dir"],
        data_path=data_path,
        mock=mock,
        factor_miner_library_path=factor_miner_library,
        factor_miner_no_memory_library_path=factor_miner_no_memory_library,
    )
    _print_summary("FactorMiner -- Benchmark Suite", payload)


__all__ = ["benchmark"]
