"""Configuration, session-inspection, and market-data CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click
import yaml

from factorminer.cli.app import (
    _doctor_checks,
    _inspect_session_dir,
    _json_safe,
    _load_lifecycle_telemetry,
    _print_doctor_report,
    _print_session_inspection,
    _print_telemetry_summary,
    _render_validation_next_steps,
    _run_session_sensitivity,
    _starter_config,
)
from factorminer.cli.context import load_raw_config_data, main
from factorminer.utils.config import load_config


@main.command("doctor")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--config",
    "doctor_config",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a YAML config file.",
)
@click.pass_context
def doctor(ctx: click.Context, json_output: bool, doctor_config: str | None) -> None:
    """Check local install, config, optional dependencies, and output paths."""
    cfg = ctx.obj["config"]
    raw = getattr(cfg, "_raw", {})
    if doctor_config:
        cfg = load_config(config_path=doctor_config)
        raw = load_raw_config_data(doctor_config)
    output_dir = ctx.obj["output_dir"]
    checks = _doctor_checks(cfg, raw, output_dir)
    payload = {
        "ok": not any(item["status"] == "error" for item in checks),
        "checks": checks,
    }
    if json_output:
        click.echo(json.dumps(_json_safe(payload), indent=2))
    else:
        _print_doctor_report(checks)
    if not payload["ok"]:
        ctx.exit(1)


@main.command("init-config")
@click.argument(
    "path",
    required=False,
    type=click.Path(dir_okay=False),
    default="factorminer.local.yaml",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config file.")
def init_config(path: str, force: bool) -> None:
    """Write a CPU-safe starter YAML config."""
    output_path = Path(path)
    if output_path.exists() and not force:
        raise click.ClickException(f"{output_path} already exists. Pass --force to overwrite.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file:
        yaml.safe_dump(_starter_config(), file, sort_keys=False)
    click.echo(f"Wrote starter config to {output_path}")


@main.group("session")
def session_group() -> None:
    """Inspect FactorMiner session artifacts."""


@session_group.command("inspect")
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON.")
@click.option(
    "--telemetry",
    is_flag=True,
    help=(
        "Include per-round mining telemetry (parse errors, duplicates, "
        "rejection reasons) parsed from factor_lifecycle.jsonl."
    ),
)
@click.option(
    "--sensitivity",
    "sensitivity_factor_id",
    default=None,
    help=(
        "Run formula AST sensitivity/ablation for a factor id or name from the "
        "session library (uses mock panel data; offline/reproducible)."
    ),
)
def session_inspect(
    output_dir: str,
    json_output: bool,
    telemetry: bool,
    sensitivity_factor_id: str | None,
) -> None:
    """Summarize run artifacts in an output directory."""
    output_path = Path(output_dir)
    payload = _inspect_session_dir(output_path)
    telemetry_payload = _load_lifecycle_telemetry(output_path) if telemetry else None
    if telemetry_payload is not None:
        payload["telemetry"] = telemetry_payload

    sensitivity_payload = None
    if sensitivity_factor_id:
        sensitivity_payload = _run_session_sensitivity(output_path, sensitivity_factor_id)
        payload["sensitivity"] = sensitivity_payload

    if json_output:
        click.echo(json.dumps(_json_safe(payload), indent=2))
    else:
        _print_session_inspection(payload)
        if telemetry_payload is not None:
            _print_telemetry_summary(telemetry_payload)
        if sensitivity_payload is not None:
            from factorminer.utils.tearsheet import format_sensitivity_panel

            click.echo("")
            click.echo(format_sensitivity_panel(sensitivity_payload))


@main.command("validate-data")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--strict", is_flag=True, help="Treat warnings as failures.")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit machine-readable JSON instead of text.",
)
@click.option(
    "--hdf-key",
    default="data",
    show_default=True,
    help="HDF5 key to read when validating .h5/.hdf5 files.",
)
@click.pass_context
def validate_data(
    ctx: click.Context,
    path: str,
    strict: bool,
    json_output: bool,
    hdf_key: str,
) -> None:
    """Validate a market-data file before mining."""
    from factorminer.data.validation import render_validation_report, validate_market_data

    try:
        report = validate_market_data(path, hdf_key=hdf_key)
    except Exception as exc:  # noqa: BLE001 - surfaced to CLI
        click.echo(f"Validation error: {exc}")
        raise click.Abort() from exc

    if json_output:
        click.echo(json.dumps(report.to_dict(strict=strict), indent=2, sort_keys=True))
    else:
        click.echo(render_validation_report(report, strict=strict))
        click.echo(_render_validation_next_steps(report, ctx.obj["config"], path, hdf_key))

    code = report.exit_code(strict=strict)
    if code != 0:
        ctx.exit(code)


@main.command("resample-data")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(dir_okay=False))
@click.option("--rule", default="10min", show_default=True, help="Pandas resample rule.")
@click.option(
    "--hdf-key",
    default="data",
    show_default=True,
    help="HDF5 key to read/write for .h5/.hdf5 files.",
)
def resample_data(input_path: str, output_path: str, rule: str, hdf_key: str) -> None:
    """Resample canonical OHLCV market data."""
    from factorminer.data.loader import load_market_data, resample_market_data

    source = load_market_data(input_path, hdf_key=hdf_key)
    resampled = resample_market_data(source, rule=rule)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = output.suffix.lower()
    if suffix == ".csv":
        resampled.to_csv(output, index=False)
    elif suffix in {".parquet", ".pq"}:
        resampled.to_parquet(output, index=False)
    elif suffix in {".h5", ".hdf5"}:
        resampled.to_hdf(output, key=hdf_key, index=False)
    else:
        raise click.ClickException(
            "Could not infer output format. Use .csv, .parquet, .pq, .h5, or .hdf5."
        )

    click.echo("FactorMiner -- Data Resample")
    click.echo("=" * 60)
    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output}")
    click.echo(f"Rule:   {rule}")
    click.echo(
        f"Rows:   {len(source)} -> {len(resampled)} | "
        f"Assets: {source['asset_id'].nunique()} -> {resampled['asset_id'].nunique()}"
    )
