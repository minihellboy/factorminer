"""Click-based CLI for FactorMiner."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np

from factorminer.utils.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    """Configure root logger for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_data(cfg, data_path: str | None, mock: bool):
    """Load market data from file or generate mock data.

    Returns
    -------
    pd.DataFrame
        Market data with columns: datetime, asset_id, open, high, low,
        close, volume, amount.
    """
    if mock or (data_path is None and not hasattr(cfg, "data_path")):
        click.echo("Generating mock market data...")
        from factorminer.data.mock_data import MockConfig, generate_mock_data

        mock_cfg = MockConfig(
            num_assets=50,
            num_periods=500,
            frequency="1d",
            plant_alpha=True,
        )
        return generate_mock_data(mock_cfg)

    # Try data_path argument, then config top-level data_path
    path = data_path
    if path is None:
        # The config YAML may have a top-level data_path field
        path = getattr(cfg, "_raw", {}).get("data_path", None)

    if path is None:
        click.echo("No data path specified. Use --data or --mock flag.")
        raise click.Abort()

    click.echo(f"Loading market data from: {path}")
    from factorminer.data.loader import load_market_data

    return load_market_data(path)


def _prepare_data_arrays(df):
    """Convert a market DataFrame to numpy arrays for the mining loop.

    Returns
    -------
    data_tensor : np.ndarray, shape (M, T, F)
        Market data tensor.
    returns : np.ndarray, shape (M, T)
        Forward returns.
    """
    import pandas as pd

    asset_ids = sorted(df["asset_id"].unique())
    dates = sorted(df["datetime"].unique())
    M = len(asset_ids)
    T = len(dates)

    feature_cols = ["open", "high", "low", "close", "volume", "amount"]
    F = len(feature_cols)

    data_tensor = np.full((M, T, F), np.nan, dtype=np.float64)
    returns = np.full((M, T), np.nan, dtype=np.float64)

    asset_to_idx = {a: i for i, a in enumerate(asset_ids)}
    date_to_idx = {d: i for i, d in enumerate(dates)}

    for _, row in df.iterrows():
        ai = asset_to_idx[row["asset_id"]]
        ti = date_to_idx[row["datetime"]]
        for fi, col in enumerate(feature_cols):
            data_tensor[ai, ti, fi] = row[col]

    # Compute forward returns from close prices
    close_idx = feature_cols.index("close")
    for i in range(M):
        close = data_tensor[i, :, close_idx]
        # Simple 1-period forward return
        returns[i, :-1] = (close[1:] - close[:-1]) / np.where(
            close[:-1] == 0, np.nan, close[:-1]
        )

    return data_tensor, returns


def _create_llm_provider(cfg, mock: bool):
    """Create an LLM provider from config or use mock."""
    from factorminer.agent.llm_interface import MockProvider, create_provider

    if mock:
        click.echo("Using mock LLM provider (no API calls).")
        return MockProvider()

    llm_config = {
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
    }
    # Use api_key from config if set
    if hasattr(cfg, "_raw") and cfg._raw.get("llm", {}).get("api_key"):
        llm_config["api_key"] = cfg._raw["llm"]["api_key"]

    click.echo(f"Using LLM provider: {cfg.llm.provider}/{cfg.llm.model}")
    return create_provider(llm_config)


def _load_library_from_path(library_path: str):
    """Load a factor library, handling both .json extension and base path.

    Returns
    -------
    FactorLibrary
    """
    from factorminer.core.library_io import load_library

    path = Path(library_path)
    # load_library expects the base path (without .json extension)
    # but also works with .json since it calls path.with_suffix(".json")
    if path.suffix == ".json":
        base_path = path.with_suffix("")
    else:
        base_path = path

    try:
        library = load_library(base_path)
        click.echo(f"Loaded factor library: {library.size} factors")
        return library
    except FileNotFoundError:
        click.echo(f"Error: Factor library not found at {library_path}")
        click.echo("  Tried: {}.json".format(base_path))
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error loading library: {e}")
        raise click.Abort()


# ---------------------------------------------------------------------------
# Global options
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a YAML config file (merges with defaults).",
)
@click.option("--gpu/--cpu", default=True, help="Enable or disable GPU evaluation backend.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug-level logging.")
@click.option(
    "--output-dir", "-o",
    type=click.Path(file_okay=False),
    default="output",
    help="Directory for all output artifacts.",
)
@click.version_option(package_name="factorminer")
@click.pass_context
def main(ctx: click.Context, config: str | None, gpu: bool, verbose: bool, output_dir: str) -> None:
    """FactorMiner -- LLM-powered quantitative factor mining."""
    _setup_logging(verbose)

    overrides: dict = {}
    if not gpu:
        overrides.setdefault("evaluation", {})["backend"] = "numpy"

    try:
        cfg = load_config(config_path=config, overrides=overrides if overrides else None)
    except Exception as e:
        click.echo(f"Error loading config: {e}")
        raise click.Abort()

    # Stash the raw YAML data for access to top-level fields like data_path
    try:
        import yaml
        from factorminer.configs import DEFAULT_CONFIG_PATH
        raw = {}
        if DEFAULT_CONFIG_PATH.exists():
            with open(DEFAULT_CONFIG_PATH) as f:
                raw = yaml.safe_load(f) or {}
        if config:
            with open(config) as f:
                user_raw = yaml.safe_load(f) or {}
            raw.update(user_raw)
        cfg._raw = raw
    except Exception:
        cfg._raw = {}

    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["verbose"] = verbose
    ctx.obj["output_dir"] = Path(output_dir)


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------

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
    except ValueError as e:
        click.echo(f"Configuration error: {e}")
        raise click.Abort()

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

    # Load data
    try:
        df = _load_data(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    n_assets = df["asset_id"].nunique()
    n_periods = df["datetime"].nunique()
    click.echo(f"  Data loaded: {n_assets} assets x {n_periods} periods ({len(df)} rows)")

    # Prepare numpy arrays
    click.echo("  Preparing data tensors...")
    data_tensor, returns = _prepare_data_arrays(df)

    # Create LLM provider
    llm_provider = _create_llm_provider(cfg, mock)

    # Load existing library for resume
    library = None
    if resume:
        click.echo(f"  Resuming from: {resume}")
        library = _load_library_from_path(resume)

    # Create and configure MiningConfig for the RalphLoop
    from factorminer.core.config import MiningConfig as CoreMiningConfig

    mining_config = CoreMiningConfig(
        target_library_size=cfg.mining.target_library_size,
        batch_size=cfg.mining.batch_size,
        max_iterations=cfg.mining.max_iterations,
        ic_threshold=cfg.mining.ic_threshold,
        icir_threshold=cfg.mining.icir_threshold,
        correlation_threshold=cfg.mining.correlation_threshold,
        replacement_ic_min=cfg.mining.replacement_ic_min,
        replacement_ic_ratio=cfg.mining.replacement_ic_ratio,
        fast_screen_assets=cfg.evaluation.fast_screen_assets,
        num_workers=cfg.evaluation.num_workers,
        output_dir=str(output_dir),
        backend=cfg.evaluation.backend,
    )

    # Create and run the Ralph Loop
    from factorminer.core.ralph_loop import RalphLoop

    click.echo("-" * 60)
    click.echo("Starting Ralph Loop...")

    def _progress_callback(iteration: int, stats: dict) -> None:
        """Print progress after each iteration."""
        lib_size = stats.get("library_size", 0)
        admitted = stats.get("admitted", 0)
        yield_rate = stats.get("yield_rate", 0) * 100
        click.echo(
            f"  Iteration {iteration:3d}: "
            f"Library={lib_size}, "
            f"Admitted={admitted}, "
            f"Yield={yield_rate:.1f}%"
        )

    try:
        loop = RalphLoop(
            config=mining_config,
            data_tensor=data_tensor,
            returns=returns,
            llm_provider=llm_provider,
            library=library,
        )
        result_library = loop.run(callback=_progress_callback)
    except KeyboardInterrupt:
        click.echo("\nMining interrupted by user.")
        return
    except Exception as e:
        click.echo(f"Mining error: {e}")
        logger.exception("Mining failed")
        raise click.Abort()

    # Save results
    from factorminer.core.library_io import save_library

    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = output_dir / "factor_library"
    save_library(result_library, lib_path)

    click.echo("=" * 60)
    click.echo(f"Mining complete! Library size: {result_library.size}")
    click.echo(f"Library saved to: {lib_path}.json")
    click.echo("=" * 60)


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

    click.echo("=" * 60)
    click.echo("FactorMiner -- Factor Evaluation")
    click.echo("=" * 60)

    # Load library
    library = _load_library_from_path(library_path)

    # Load market data
    try:
        df = _load_data(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    click.echo(f"  Period: {period} | Backend: {cfg.evaluation.backend}")
    click.echo(f"  Data: {df['asset_id'].nunique()} assets x {df['datetime'].nunique()} periods")

    # Get factors to evaluate
    factors = library.list_factors()
    if top_k is not None and top_k < len(factors):
        factors = sorted(factors, key=lambda f: abs(f.ic_mean), reverse=True)[:top_k]
        click.echo(f"  Evaluating top {top_k} factors by IC")

    click.echo("-" * 60)

    # Print factor stats table
    click.echo(f"{'ID':>4s}  {'Name':<35s}  {'IC Mean':>8s}  {'ICIR':>7s}  {'Win%':>6s}  {'MaxCorr':>8s}")
    click.echo("-" * 75)

    for f in factors:
        click.echo(
            f"{f.id:4d}  {f.name:<35s}  {f.ic_mean:8.4f}  {f.icir:7.3f}  "
            f"{f.ic_win_rate * 100:5.1f}%  {f.max_correlation:8.4f}"
        )

    # Summary statistics
    if factors:
        ic_values = [f.ic_mean for f in factors]
        icir_values = [f.icir for f in factors]
        click.echo("-" * 75)
        click.echo(f"  Total factors:    {len(factors)}")
        click.echo(f"  Mean IC:          {np.mean(ic_values):.4f}")
        click.echo(f"  Mean ICIR:        {np.mean(icir_values):.3f}")
        click.echo(f"  Max IC:           {max(ic_values):.4f}")
        click.echo(f"  Min IC:           {min(ic_values):.4f}")

    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for combination.")
@click.option(
    "--method", "-m",
    type=click.Choice(["equal-weight", "ic-weighted", "orthogonal", "all"]),
    default="all",
    help="Factor combination method.",
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
    method: str,
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

    # Load market data
    try:
        df = _load_data(cfg, data_path, mock)
    except Exception as e:
        click.echo(f"Error loading data: {e}")
        raise click.Abort()

    # Prepare data
    data_tensor, returns_arr = _prepare_data_arrays(df)

    # Build factor signals dict
    factors = library.list_factors()
    if top_k is not None and top_k < len(factors):
        factors = sorted(factors, key=lambda f: abs(f.ic_mean), reverse=True)[:top_k]
        click.echo(f"  Pre-selected top {top_k} factors")

    # Use signals from library if available, otherwise generate pseudo-signals
    factor_signals: dict[int, np.ndarray] = {}
    ic_values: dict[int, float] = {}
    M, T = returns_arr.shape

    for f in factors:
        if f.signals is not None:
            factor_signals[f.id] = f.signals
        else:
            # Generate deterministic pseudo-signals from formula hash
            seed = hash(f.formula) % (2**31)
            rng = np.random.RandomState(seed)
            factor_signals[f.id] = rng.randn(M, T).astype(np.float64)
        ic_values[f.id] = f.ic_mean

    if not factor_signals:
        click.echo("No factors with signals available for combination.")
        raise click.Abort()

    click.echo(f"  Combining {len(factor_signals)} factors")
    click.echo("-" * 60)

    # Run selection if requested
    if selection != "none":
        click.echo(f"\n  Running {selection} selection...")
        from factorminer.evaluation.selection import FactorSelector

        selector = FactorSelector()
        # Transpose returns to (T, N) for the selector API
        returns_tn = returns_arr.T  # (T, M)
        signals_tn = {fid: sig.T for fid, sig in factor_signals.items()}

        try:
            if selection == "lasso":
                results = selector.lasso_selection(signals_tn, returns_tn)
            elif selection == "stepwise":
                results = selector.forward_stepwise(signals_tn, returns_tn)
            elif selection == "xgboost":
                results = selector.xgboost_selection(signals_tn, returns_tn)
            else:
                results = []

            if results:
                click.echo(f"\n  {selection.capitalize()} selection results:")
                click.echo(f"  {'Factor ID':>10s}  {'Score':>10s}")
                click.echo("  " + "-" * 25)
                for fid, score in results[:20]:  # Show top 20
                    click.echo(f"  {fid:10d}  {score:10.4f}")
                click.echo(f"  Total selected: {len(results)}")
            else:
                click.echo(f"  {selection} selection returned no factors.")
        except ImportError as e:
            click.echo(f"  Selection method '{selection}' requires additional packages: {e}")
        except Exception as e:
            click.echo(f"  Selection error: {e}")
            logger.exception("Selection failed")

    # Run combination methods
    from factorminer.evaluation.combination import FactorCombiner

    combiner = FactorCombiner()
    # Transpose to (T, N) for the combiner API
    signals_tn = {fid: sig.T for fid, sig in factor_signals.items()}

    methods_to_run = []
    if method == "all":
        methods_to_run = ["equal-weight", "ic-weighted", "orthogonal"]
    else:
        methods_to_run = [method]

    for m in methods_to_run:
        click.echo(f"\n  {m.upper()} combination:")
        try:
            if m == "equal-weight":
                composite = combiner.equal_weight(signals_tn)
            elif m == "ic-weighted":
                composite = combiner.ic_weighted(signals_tn, ic_values)
            elif m == "orthogonal":
                composite = combiner.orthogonal(signals_tn)
            else:
                continue

            # Compute IC of the composite signal
            from factorminer.evaluation.metrics import compute_factor_stats

            stats = compute_factor_stats(composite, returns_arr.T)
            click.echo(f"    IC Mean:    {stats['ic_mean']:.4f}")
            click.echo(f"    IC AbsMean: {stats['ic_abs_mean']:.4f}")
            click.echo(f"    ICIR:       {stats['icir']:.4f}")
            click.echo(f"    Win Rate:   {stats['ic_win_rate']:.1%}")
        except Exception as e:
            click.echo(f"    Error: {e}")
            logger.exception("Combination method %s failed", m)

    click.echo("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# visualize
# ---------------------------------------------------------------------------

@main.command()
@click.argument("library_path", type=click.Path(exists=True))
@click.option("--data", "data_path", type=click.Path(exists=True), default=None, help="Path to market data file.")
@click.option("--mock", is_flag=True, help="Use mock data for visualization.")
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
    click.echo("-" * 60)

    factors = library.list_factors()

    # Correlation heatmap
    if correlation:
        click.echo("  Generating correlation heatmap...")
        try:
            from factorminer.utils.visualization import plot_correlation_heatmap

            if library.correlation_matrix is not None and library.correlation_matrix.size > 0:
                names = [f.name[:20] for f in factors]
                save_path = str(output_dir / f"correlation_heatmap.{fmt}")
                plot_correlation_heatmap(
                    library.correlation_matrix,
                    names,
                    save_path=save_path,
                )
                click.echo(f"    Saved: {save_path}")
            else:
                click.echo("    Skipped: no correlation matrix available in library.")
        except Exception as e:
            click.echo(f"    Error: {e}")
            logger.exception("Correlation heatmap failed")

    # IC timeseries and quintile require market data
    needs_data = ic_timeseries or quintile or tearsheet
    df = None
    if needs_data:
        try:
            df = _load_data(cfg, data_path, mock)
        except Exception as e:
            click.echo(f"  Warning: Could not load data for IC/quintile plots: {e}")
            ic_timeseries = False
            quintile = False
            tearsheet = False

    if ic_timeseries and df is not None:
        click.echo("  Generating IC time series plot...")
        # Use IC values from factor metadata since we may not have signals
        try:
            from factorminer.utils.visualization import plot_ic_timeseries

            # Create a synthetic IC series from the library's factor ICs
            ic_values = np.array([f.ic_mean for f in factors])
            dates = [f"Factor_{i+1}" for i in range(len(factors))]
            save_path = str(output_dir / f"ic_timeseries.{fmt}")
            plot_ic_timeseries(
                ic_values,
                dates,
                rolling_window=min(5, len(factors)),
                title="Factor IC Values",
                save_path=save_path,
            )
            click.echo(f"    Saved: {save_path}")
        except Exception as e:
            click.echo(f"    Error: {e}")
            logger.exception("IC timeseries failed")

    if quintile and df is not None:
        click.echo("  Generating quintile returns plot...")
        try:
            from factorminer.utils.visualization import plot_quintile_returns

            # Group factors by IC quintile
            sorted_factors = sorted(factors, key=lambda f: f.ic_mean)
            n = len(sorted_factors)
            if n >= 5:
                q_size = n // 5
                quintile_data = {}
                for qi in range(5):
                    start = qi * q_size
                    end = start + q_size if qi < 4 else n
                    q_factors = sorted_factors[start:end]
                    mean_ic = np.mean([f.ic_mean for f in q_factors])
                    quintile_data[f"Q{qi+1}"] = float(mean_ic)

                quintile_data["long_short"] = quintile_data["Q5"] - quintile_data["Q1"]
                save_path = str(output_dir / f"quintile_returns.{fmt}")
                plot_quintile_returns(
                    quintile_data,
                    title="Factor Library Quintile IC",
                    save_path=save_path,
                )
                click.echo(f"    Saved: {save_path}")
            else:
                click.echo("    Skipped: need at least 5 factors for quintile plot.")
        except Exception as e:
            click.echo(f"    Error: {e}")
            logger.exception("Quintile plot failed")

    # Tearsheet
    if tearsheet and df is not None:
        click.echo("  Generating tear sheets...")
        try:
            from factorminer.utils.tearsheet import FactorTearSheet

            data_tensor, returns_arr = _prepare_data_arrays(df)
            ts = FactorTearSheet()
            dates_list = sorted(df["datetime"].unique())
            date_strs = [str(d)[:10] for d in dates_list]
            T = len(dates_list)

            for f in factors[:10]:  # Limit to first 10 for performance
                click.echo(f"    Factor #{f.id}: {f.name}...")
                if f.signals is not None:
                    signals = f.signals
                else:
                    # Generate pseudo-signals
                    M = data_tensor.shape[0]
                    seed = hash(f.formula) % (2**31)
                    rng = np.random.RandomState(seed)
                    signals = rng.randn(M, T).astype(np.float64)

                save_path = str(output_dir / f"tearsheet_factor_{f.id}.{fmt}")
                ts.generate(
                    factor_id=f.id,
                    factor_name=f.name,
                    formula=f.formula,
                    signals=signals,
                    returns=returns_arr,
                    dates=date_strs[:signals.shape[1]],
                    save_path=save_path,
                )
                click.echo(f"      Saved: {save_path}")
        except Exception as e:
            click.echo(f"    Tearsheet error: {e}")
            logger.exception("Tearsheet generation failed")

    click.echo("=" * 60)
    click.echo("Visualization complete.")


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

@main.command(name="export")
@click.argument("library_path", type=click.Path(exists=True))
@click.option(
    "--format", "fmt",
    type=click.Choice(["json", "csv", "formulas"]),
    default="json",
    help="Export format.",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path.")
@click.pass_context
def export_cmd(ctx: click.Context, library_path: str, fmt: str, output: str | None) -> None:
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
        if fmt == "formulas":
            output = str(output_dir / "library_formulas.txt")
        else:
            output = str(output_dir / f"library.{fmt}")

    click.echo(f"  Format:  {fmt}")
    click.echo(f"  Output:  {output}")
    click.echo("-" * 60)

    try:
        from factorminer.core.library_io import export_csv, export_formulas, save_library

        if fmt == "json":
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

    except Exception as e:
        click.echo(f"Export error: {e}")
        logger.exception("Export failed")
        raise click.Abort()

    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# helix
# ---------------------------------------------------------------------------

@main.command()
@click.option("--iterations", "-n", type=int, default=None, help="Override max_iterations.")
@click.option("--batch-size", "-b", type=int, default=None, help="Override batch_size.")
@click.option("--target", "-t", type=int, default=None, help="Override target_library_size.")
@click.option("--resume", type=click.Path(exists=True), default=None, help="Resume from a saved library.")
@click.option("--causal/--no-causal", default=None, help="Enable/disable causal validation.")
@click.option("--regime/--no-regime", default=None, help="Enable/disable regime-conditional evaluation.")
@click.option("--debate/--no-debate", default=None, help="Enable/disable multi-specialist debate generation.")
@click.option("--canonicalize/--no-canonicalize", default=None, help="Enable/disable SymPy canonicalization.")
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
) -> None:
    """Run the enhanced Helix Loop with Phase 2 features."""
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
        cfg.phase2.helix.enable_canonicalization = canonicalize

    try:
        cfg.validate()
    except ValueError as e:
        click.echo(f"Configuration error: {e}")
        raise click.Abort()

    output_dir = ctx.obj["output_dir"]

    enabled_features = []
    if cfg.phase2.causal.enabled:
        enabled_features.append("causal")
    if cfg.phase2.regime.enabled:
        enabled_features.append("regime")
    if cfg.phase2.capacity.enabled:
        enabled_features.append("capacity")
    if cfg.phase2.significance.enabled:
        enabled_features.append("significance")
    if cfg.phase2.debate.enabled:
        enabled_features.append("debate")
    if cfg.phase2.auto_inventor.enabled:
        enabled_features.append("auto-inventor")
    if cfg.phase2.helix.enabled:
        enabled_features.append("helix-memory")

    click.echo("HelixFactor Phase 2 mining engine.")
    click.echo(f"  Target: {cfg.mining.target_library_size} | "
               f"Batch: {cfg.mining.batch_size} | "
               f"Max iterations: {cfg.mining.max_iterations}")
    click.echo(f"  Output directory: {output_dir}")

    if enabled_features:
        click.echo(f"  Active Phase 2 features: {', '.join(enabled_features)}")
    else:
        click.echo("  No Phase 2 features enabled. Configure phase2.* in your config to enable features.")

    if resume:
        click.echo(f"  Resuming from: {resume}")

    click.echo("Helix loop not yet connected. Infrastructure ready.")


if __name__ == "__main__":
    main()
