"""MCP and external financial-data integration commands."""

from __future__ import annotations

import json
from pathlib import Path

import click
import yaml

from factorminer.cli.context import main


@main.command("mcp-serve")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"], case_sensitive=False),
    default="stdio",
    show_default=True,
    help="MCP transport; HTTP requires a bearer token.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Bind host for HTTP transport.",
)
@click.option("--port", type=int, default=8765, show_default=True)
@click.option(
    "--auth-token-env",
    default="FACTORMINER_MCP_TOKEN",
    show_default=True,
    help="Environment variable containing the HTTP bearer token.",
)
def mcp_serve(transport: str, host: str, port: int, auth_token_env: str) -> None:
    """Run the FactorMiner MCP server."""
    try:
        from factorminer.mcp.server import run_server
    except ModuleNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    try:
        run_server(
            transport=transport.lower(),  # type: ignore[arg-type]
            host=host,
            port=port,
            auth_token_env=auth_token_env,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("mcp-connectors")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
def mcp_connectors(json_output: bool) -> None:
    """List bundled financial-services MCP connector endpoints."""
    from factorminer.data.mcp_source import known_mcp_connectors

    connectors = known_mcp_connectors()
    if json_output:
        click.echo(json.dumps({"connectors": connectors}, indent=2, sort_keys=True))
        return

    click.echo("FactorMiner -- FSI MCP Connectors")
    click.echo("=" * 60)
    for connector in connectors:
        click.echo(f"  {connector['name']:<12s} {connector['url']}")
        click.echo(f"  {'':<12s} {connector['best_for']}")
    click.echo("=" * 60)
    click.echo("Use these endpoints in plugin .mcp.json or an MCP-source YAML config.")


@main.command("fetch-data")
@click.option(
    "--mcp-config",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="MCP-source YAML configuration.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Destination market-data file.",
)
def fetch_data(mcp_config: str, output_path: str) -> None:
    """Fetch market data from an external MCP connector."""
    from factorminer.data.mcp_source import fetch_to_file, load_mcp_source_config

    click.echo("FactorMiner -- MCP Data Fetch")
    click.echo("=" * 60)
    click.echo(f"  Config: {mcp_config}")
    try:
        config = load_mcp_source_config(mcp_config)
        click.echo(f"  Connector: {config.transport} -> tool '{config.tool}'")
        written = fetch_to_file(config, output_path)
    except ModuleNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Fetch error: {exc}")
        raise click.Abort() from exc

    click.echo(f"  Wrote: {written}")
    click.echo("=" * 60)
    click.echo(f"Next: uv run factorminer validate-data {written}")


@main.command("attach-edgar")
@click.option(
    "--data",
    "data_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="OHLCV market-data file.",
)
@click.option(
    "--cik-map",
    "cik_map_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="YAML/JSON mapping of asset_id to CIK.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
)
@click.option("--cache-dir", default=None, type=click.Path(file_okay=False))
@click.option("--user-agent", default=None, help="SEC-compliant User-Agent.")
@click.option(
    "--fixture",
    "fixture_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Offline companyfacts JSON fixture.",
)
def attach_edgar(
    data_path: str,
    cik_map_path: str,
    output_path: str,
    cache_dir: str | None,
    user_agent: str | None,
    fixture_path: str | None,
) -> None:
    """Join point-in-time SEC EDGAR fundamentals onto an OHLCV panel."""
    from factorminer.data.edgar_source import (
        DEFAULT_USER_AGENT,
        EdgarConfig,
        attach_edgar_to_panel,
        register_edgar_features,
    )
    from factorminer.data.loader import load_market_data

    click.echo("FactorMiner -- Attach EDGAR Fundamentals")
    click.echo("=" * 60)
    panel = load_market_data(data_path)
    raw_map = Path(cik_map_path).read_text(encoding="utf-8")
    cik_map = yaml.safe_load(raw_map) if cik_map_path.endswith((".yaml", ".yml")) else json.loads(raw_map)
    if not isinstance(cik_map, dict) or not cik_map:
        raise click.ClickException("cik-map must be a non-empty asset_id -> CIK mapping")

    offline = None
    if fixture_path is not None:
        fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
        if isinstance(fixture, dict) and "facts" in fixture:
            offline = {str(asset): fixture for asset in cik_map}
        elif isinstance(fixture, dict):
            offline = fixture
        else:
            raise click.ClickException("fixture must be a companyfacts object or asset-keyed map")

    register_edgar_features()
    joined = attach_edgar_to_panel(
        panel,
        {str(key): value for key, value in cik_map.items()},
        config=EdgarConfig(user_agent=user_agent or DEFAULT_USER_AGENT, cache_dir=cache_dir),
        offline_payloads=offline,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".parquet", ".pq"}:
        joined.to_parquet(out, index=False)
    elif out.suffix.lower() in {".h5", ".hdf5"}:
        joined.to_hdf(out, key="data", mode="w")
    else:
        joined.to_csv(out, index=False)

    click.echo(f"  Assets: {joined['asset_id'].nunique()}")
    click.echo(f"  Rows:   {len(joined)}")
    for column in ("eps", "revenue", "book_equity", "shares_out"):
        if column in joined.columns:
            click.echo(f"  {column}: {int(joined[column].notna().sum())} non-null values")
    click.echo(f"  Wrote:  {out}")
    click.echo("=" * 60)


@main.command("build-futures")
@click.option(
    "--data",
    "data_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
)
@click.option("--mock", is_flag=True, help="Generate a deterministic synthetic panel.")
@click.option("--multiplier", default=1.0, show_default=True, type=float)
def build_futures(
    data_path: str | None,
    output_path: str,
    mock: bool,
    multiplier: float,
) -> None:
    """Build a roll-adjusted continuous futures panel with basis leaves."""
    from factorminer.data.futures_source import (
        FuturesConfig,
        build_continuous_futures_panel,
        generate_mock_futures_panel,
        register_futures_features,
    )
    from factorminer.data.loader import load_market_data

    click.echo("FactorMiner -- Build Continuous Futures Panel")
    click.echo("=" * 60)
    config = FuturesConfig(contract_multiplier=float(multiplier))
    register_futures_features()
    if mock or data_path is None:
        click.echo("  Generating mock continuous futures panel...")
        panel = generate_mock_futures_panel(config=config)
    else:
        panel = build_continuous_futures_panel(load_market_data(data_path), config=config)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".parquet", ".pq"}:
        panel.to_parquet(out, index=False)
    elif out.suffix.lower() in {".h5", ".hdf5"}:
        panel.to_hdf(out, key="data", mode="w")
    else:
        panel.to_csv(out, index=False)

    click.echo(f"  Assets: {panel['asset_id'].nunique()}")
    click.echo(f"  Rows:   {len(panel)}")
    for column in ("basis", "spot", "premium", "roll_yield", "oi"):
        if column in panel.columns:
            click.echo(f"  leaf ${column}: present")
    click.echo(f"  Wrote:  {out}")
    click.echo("=" * 60)


__all__ = ["attach_edgar", "build_futures", "fetch_data", "mcp_connectors", "mcp_serve"]
