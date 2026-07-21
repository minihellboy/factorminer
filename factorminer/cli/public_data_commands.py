"""Commands for checksum-locked public evidence datasets."""

from __future__ import annotations

from pathlib import Path

import click

from factorminer.cli.context import main


@main.group("public-data")
def public_data() -> None:
    """Lock, prepare, and verify public evidence inputs."""


@public_data.command("lock")
@click.argument("spec", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def lock_dataset(spec: Path, output: Path) -> None:
    """Resolve an editable YAML source specification into a pinned JSON lock."""
    from factorminer.data.public_archive import lock_public_dataset_spec

    result = lock_public_dataset_spec(spec, output)
    click.echo(str(result))


@public_data.command("prepare")
@click.argument("lock", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option(
    "--offline", is_flag=True, help="Require every archive to exist in the verified cache."
)
def prepare_dataset(lock: Path, output_dir: Path, cache_dir: Path | None, offline: bool) -> None:
    """Download and normalize every checksum-pinned archive."""
    from factorminer.data.public_archive import prepare_public_dataset

    result = prepare_public_dataset(lock, output_dir, cache_dir=cache_dir, offline=offline)
    click.echo(str(result))


@public_data.command("verify")
@click.argument("dataset_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
def verify_dataset(dataset_dir: Path) -> None:
    """Offline-verify a prepared public dataset against its lock and manifest."""
    from factorminer.data.public_archive import verify_prepared_public_dataset

    result = verify_prepared_public_dataset(dataset_dir)
    for mismatch in result.mismatches:
        click.echo(mismatch, err=True)
    if not result.passed:
        raise SystemExit(1)
    click.echo(
        f"Public dataset verification passed: {result.dataset_id} "
        f"({result.rows} rows, {result.assets} assets)"
    )


__all__ = ["public_data"]
