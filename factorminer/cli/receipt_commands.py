"""Offline verification for published ExternalResearchReceipt artifacts."""

from __future__ import annotations

from pathlib import Path

import click

from factorminer.cli.context import main


@main.command("verify-receipt")
@click.argument("release_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--commitment-key",
    default=None,
    help="Withheld hex key for verifying a private-partner HMAC dataset commitment.",
)
@click.option(
    "--commitment-input",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Local raw dataset file to re-hash against the stored commitment.",
)
def verify_receipt(
    release_dir: Path, commitment_key: str | None, commitment_input: Path | None
) -> None:
    """Offline-verify every artifact hash and the release_id in a receipt directory."""
    from factorminer.benchmark.receipt import verify_research_receipt

    result = verify_research_receipt(
        release_dir, commitment_key=commitment_key, commitment_input=commitment_input
    )
    for mismatch in result.mismatches:
        click.echo(mismatch, err=True)
    if result.passed:
        click.echo("Receipt verification passed")
    else:
        click.echo(f"Receipt verification failed: {len(result.mismatches)} mismatch(es)", err=True)
        raise SystemExit(1)


__all__ = ["verify_receipt"]
