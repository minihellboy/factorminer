"""Explicit external-review commands for private partner receipts."""

from __future__ import annotations

import json
from pathlib import Path

import click

from factorminer.cli.context import main


@main.group("partner-review")
def partner_review() -> None:
    """Prepare, sign, and verify bounded partner acknowledgments."""


@partner_review.command("prepare")
@click.argument("release_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--partner-pseudonym", required=True)
@click.option("--assertion", "assertions", multiple=True, required=True)
@click.option("--valid-days", type=click.IntRange(1, 365), default=30, show_default=True)
def prepare_review(
    release_dir: Path,
    output: Path,
    partner_pseudonym: str,
    assertions: tuple[str, ...],
    valid_days: int,
) -> None:
    """Create a request bound to one immutable private receipt."""
    from factorminer.benchmark.partner_review import (
        prepare_partner_review,
        write_partner_review_artifact,
    )

    request = prepare_partner_review(
        release_dir,
        partner_pseudonym=partner_pseudonym,
        requested_assertions=assertions,
        valid_days=valid_days,
    )
    click.echo(str(write_partner_review_artifact(request.to_dict(), output)))


@partner_review.command("acknowledge")
@click.argument("request_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--reviewer-pseudonym", required=True)
@click.option("--assertion", "assertions", multiple=True, required=True)
@click.option(
    "--publication-consent",
    type=click.Choice(["private", "anonymous", "public"]),
    default="private",
    show_default=True,
)
@click.option(
    "--key-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Partner-controlled 32-byte hex key. The key is never written to the acknowledgment.",
)
@click.option(
    "--feedback-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
def acknowledge_review(
    request_path: Path,
    output: Path,
    reviewer_pseudonym: str,
    assertions: tuple[str, ...],
    publication_consent: str,
    key_file: Path,
    feedback_json: Path | None,
) -> None:
    """Sign the subset of requested assertions personally observed by the partner."""
    from factorminer.benchmark.partner_review import (
        acknowledge_partner_review,
        parse_partner_review_request,
        write_partner_review_artifact,
    )

    request = parse_partner_review_request(json.loads(request_path.read_text()))
    feedback = json.loads(feedback_json.read_text()) if feedback_json else {}
    acknowledgment = acknowledge_partner_review(
        request,
        reviewer_pseudonym=reviewer_pseudonym,
        assertions=assertions,
        publication_consent=publication_consent,
        key_hex=key_file.read_text().strip(),
        structured_feedback=feedback,
    )
    click.echo(str(write_partner_review_artifact(acknowledgment.to_dict(), output)))


@partner_review.command("verify")
@click.argument("request_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("acknowledgment_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("release_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--key-file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def verify_review(
    request_path: Path,
    acknowledgment_path: Path,
    release_dir: Path,
    key_file: Path,
) -> None:
    """Verify request, receipt digest, selected assertions, and partner signature."""
    from factorminer.benchmark.partner_review import (
        parse_partner_acknowledgment,
        parse_partner_review_request,
        verify_partner_acknowledgment,
    )

    request = parse_partner_review_request(json.loads(request_path.read_text()))
    acknowledgment = parse_partner_acknowledgment(json.loads(acknowledgment_path.read_text()))
    passed, mismatches = verify_partner_acknowledgment(
        request,
        acknowledgment,
        release_dir=release_dir,
        key_hex=key_file.read_text().strip(),
    )
    for mismatch in mismatches:
        click.echo(mismatch, err=True)
    if not passed:
        raise SystemExit(1)
    click.echo("Partner acknowledgment verification passed")


__all__ = ["partner_review"]
