"""Tests for the static report viewer/exporter."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from factorminer.cli import main as cli_main
from factorminer.evaluation.report_viewer import (
    build_report_payload,
    generate_report,
    main,
    render_html_report,
    render_markdown_report,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "report_viewer"


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_build_report_payload_summarizes_library_session_and_benchmarks() -> None:
    payload = build_report_payload(
        FIXTURE_DIR / "factor_library.json",
        session_log_source=FIXTURE_DIR / "session_log.json",
        benchmark_sources=[FIXTURE_DIR / "benchmark_suite.json"],
    )

    assert payload["library"]["count"] == 2
    assert payload["library"]["family_counts"][0] == ("mean_reversion", 1)
    assert payload["session"]["counts"]["admitted"] == 3
    assert payload["session"]["counts"]["rejection_reasons"][0] == ("IC below threshold", 1)
    assert len(payload["benchmarks"]) == 2
    assert payload["benchmarks"][0]["label"] in {"table1", "ablation_memory"}
    assert payload["benchmarks"][0]["universe_rows"]


def test_render_markdown_report_contains_expected_sections() -> None:
    payload = build_report_payload(
        _load_fixture("factor_library.json"),
        session_log_source=_load_fixture("session_log.json"),
        benchmark_sources=[_load_fixture("benchmark_suite.json")],
    )

    markdown = render_markdown_report(payload)

    assert "# FactorMiner Static Report" in markdown
    assert "## Library Summary" in markdown
    assert "mean_reversion" in markdown
    assert "## Lifecycle and Admission" in markdown
    assert "IC below threshold" in markdown
    assert "## Benchmarks" in markdown
    assert "CSI500" in markdown
    assert "factor_miner_no_memory" in markdown


def test_render_html_report_contains_tables_and_sections() -> None:
    payload = build_report_payload(
        _load_fixture("factor_library.json"),
        session_log_source=_load_fixture("session_log.json"),
        benchmark_sources=[_load_fixture("benchmark_suite.json")],
    )

    html = render_html_report(payload)

    assert "<!doctype html>" in html.lower()
    assert "<h2>Library Summary</h2>" in html
    assert "<table>" in html
    assert "mean_reversion" in html
    assert "CSI500" in html


def test_generate_report_writes_output_file(tmp_path) -> None:
    output_path = tmp_path / "report.html"
    report = generate_report(
        FIXTURE_DIR / "factor_library.json",
        session_log_source=FIXTURE_DIR / "session_log.json",
        benchmark_sources=[FIXTURE_DIR / "benchmark_suite.json"],
        format="html",
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == report
    assert "FactorMiner Static Report" in report


def test_module_entrypoint_writes_markdown_report(tmp_path) -> None:
    output_path = tmp_path / "report.md"
    code = main(
        [
            str(FIXTURE_DIR / "factor_library.json"),
            "--session-log",
            str(FIXTURE_DIR / "session_log.json"),
            "--benchmark",
            str(FIXTURE_DIR / "benchmark_suite.json"),
            "--output",
            str(output_path),
        ]
    )

    assert code == 0
    assert output_path.exists()
    text = output_path.read_text(encoding="utf-8")
    assert "FactorMiner Static Report" in text
    assert "CSI500" in text


def test_cli_report_command_writes_html_report(tmp_path) -> None:
    output_path = tmp_path / "report.html"
    runner = CliRunner()

    result = runner.invoke(
        cli_main,
        [
            "--cpu",
            "report",
            str(FIXTURE_DIR / "factor_library.json"),
            "--session-log",
            str(FIXTURE_DIR / "session_log.json"),
            "--benchmark",
            str(FIXTURE_DIR / "benchmark_suite.json"),
            "--format",
            "html",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Report written to:" in result.output
    assert output_path.exists()
    assert "<h2>Library Summary</h2>" in output_path.read_text(encoding="utf-8")
