"""CLI tests for the Helix command."""

from __future__ import annotations

import json

from click.testing import CliRunner

from factorminer.cli import main


def test_helix_cli_runs_with_mock_data(tmp_path):
    """The helix command should execute end-to-end and save a library."""
    output_dir = tmp_path / "helix-output"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "--cpu",
            "--output-dir",
            str(output_dir),
            "helix",
            "--mock",
            "-n",
            "1",
            "-b",
            "5",
            "-t",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Starting Helix Loop..." in result.output
    assert "Helix mining complete!" in result.output

    library_path = output_dir / "factor_library.json"
    assert library_path.exists()

    payload = json.loads(library_path.read_text())
    assert "factors" in payload


def test_helix_cli_reports_enabled_features(tmp_path):
    """Explicit feature flags should be reflected in the CLI output."""
    output_dir = tmp_path / "helix-flags"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "--cpu",
            "--output-dir",
            str(output_dir),
            "helix",
            "--mock",
            "--debate",
            "--canonicalize",
            "-n",
            "1",
            "-b",
            "4",
            "-t",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Active Phase 2 features: debate, canonicalization" in result.output
