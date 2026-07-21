"""CLI bootstrap, configuration loading, and shared execution context."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import click
import yaml

from factorminer.configs import load_default_yaml
from factorminer.utils.config import load_config


def setup_logging(verbose: bool) -> None:
    """Configure command-line logging without affecting package imports."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not verbose:
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings(
            "ignore",
            message="Degrees of freedom <= 0 for slice.",
            category=RuntimeWarning,
        )


def deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge two plain dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_raw_config_data(config_path: str | None) -> dict:
    """Load raw YAML including top-level fields not mapped to dataclasses."""
    raw = load_default_yaml()
    if config_path:
        with open(config_path) as stream:
            user_raw = yaml.safe_load(stream) or {}
        if isinstance(user_raw, dict):
            raw = deep_merge_dict(raw, user_raw)
    return raw


def load_market_frame(cfg, data_path: str | None, mock: bool):
    """Load a market frame from the configured source or mock generator."""
    raw_cfg = getattr(cfg, "_raw", {})
    configured_path = raw_cfg.get("data_path")

    if mock:
        click.echo("Generating mock market data...")
        from factorminer.data.mock_data import MockConfig, generate_mock_data

        return generate_mock_data(
            MockConfig(
                num_assets=50,
                num_periods=500,
                frequency="1d",
                plant_alpha=True,
            )
        )

    path = data_path or configured_path
    if path is None:
        click.echo("No data path specified. Use --data or --mock flag.")
        raise click.Abort()

    click.echo(f"Loading market data from: {path}")
    from factorminer.data.loader import load_market_data

    return load_market_data(path)


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a YAML config file (merges with defaults).",
)
@click.option(
    "--gpu/--cpu",
    default=None,
    help="Override evaluation backend. Omit to use the configured backend.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug-level logging.")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False),
    default="output",
    help="Directory for all output artifacts.",
)
@click.version_option(package_name="factorminer")
@click.pass_context
def main(
    ctx: click.Context,
    config: str | None,
    gpu: bool | None,
    verbose: bool,
    output_dir: str,
) -> None:
    """FactorMiner -- LLM-powered quantitative factor mining."""
    setup_logging(verbose)

    overrides: dict = {}
    if gpu is True:
        overrides.setdefault("evaluation", {})["backend"] = "gpu"
    elif gpu is False:
        overrides.setdefault("evaluation", {})["backend"] = "numpy"

    try:
        cfg = load_config(config_path=config, overrides=overrides or None)
    except Exception as exc:
        click.echo(f"Error loading config: {exc}")
        raise click.Abort() from exc

    raw_config = load_raw_config_data(config)
    setattr(cfg, "_raw", raw_config)
    if output_dir == "output":
        output_dir = raw_config.get("output_dir", output_dir)

    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["verbose"] = verbose
    ctx.obj["output_dir"] = Path(output_dir)


__all__ = [
    "deep_merge_dict",
    "load_market_frame",
    "load_raw_config_data",
    "main",
    "setup_logging",
]
