"""Tests for first-run CLI helper commands."""

from __future__ import annotations

import json

from click.testing import CliRunner

import factorminer.cli as cli_module
from factorminer.cli import main


def test_doctor_json_succeeds_with_default_cpu_backend(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["--output-dir", str(tmp_path / "doctor-output"), "doctor", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    statuses = {check["name"]: check["status"] for check in payload["checks"]}
    assert statuses["packaged_config"] == "ok"
    assert statuses["effective_backend"] == "ok"
    assert statuses["llm"] in {"ok", "warning"}


def test_init_config_writes_and_refuses_overwrite(tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "factorminer.local.yaml"

    first = runner.invoke(main, ["init-config", str(config_path)])
    assert first.exit_code == 0, first.output
    text = config_path.read_text(encoding="utf-8")
    assert "backend: numpy" in text
    assert "provider: mock" in text

    second = runner.invoke(main, ["init-config", str(config_path)])
    assert second.exit_code != 0
    assert "already exists" in second.output

    third = runner.invoke(main, ["init-config", str(config_path), "--force"])
    assert third.exit_code == 0, third.output


def test_session_inspect_handles_partial_and_inconsistent_artifacts(tmp_path):
    output_dir = tmp_path / "session"
    output_dir.mkdir()
    (output_dir / "factor_library.json").write_text(
        json.dumps({"factors": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}),
        encoding="utf-8",
    )
    (output_dir / "session.json").write_text(
        json.dumps({"status": "completed", "total_iterations": 1, "last_library_size": 1}),
        encoding="utf-8",
    )
    (output_dir / "session_log.json").write_text(
        json.dumps(
            {
                "iterations": [{"iteration": 1, "library_size": 1}],
                "summary": {
                    "total_iterations": 1,
                    "total_candidates": 4,
                    "total_admitted": 2,
                    "overall_yield_rate": 0.5,
                    "final_library_size": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(main, ["session", "inspect", str(output_dir), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["library_size"] == 2
    assert payload["status"] == "completed"
    assert payload["yield_rate"] == 0.5
    assert any("last_library_size=1" in warning for warning in payload["warnings"])
    assert any("Missing run_manifest" in warning for warning in payload["warnings"])


def test_quickstart_runs_mine_and_writes_report(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def _fake_mine_callback(**kwargs):
        import click

        captured["callback"] = "mine"
        captured["kwargs"] = kwargs
        output_dir = click.get_current_context().obj["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "factor_library.json").write_text(
            json.dumps(
                {
                    "correlation_threshold": 1.0,
                    "ic_threshold": 0.0001,
                    "dependence_metric": "spearman",
                    "metric_version": "paper_ic_v2",
                    "factors": [
                        {
                            "id": 1,
                            "name": "quickstart_factor",
                            "formula": "Neg($close)",
                            "category": "test",
                            "ic_mean": -0.05,
                            "ic_paper_mean": 0.05,
                            "ic_abs_mean": 0.05,
                            "icir": -0.8,
                            "ic_paper_icir": 0.8,
                            "ic_win_rate": 0.4,
                            "max_correlation": 0.0,
                            "batch_number": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(cli_module.mine, "callback", _fake_mine_callback)

    output_dir = tmp_path / "quickstart"
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "quickstart",
            "--output-dir",
            str(output_dir),
            "--iterations",
            "1",
            "--batch-size",
            "2",
            "--target",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["callback"] == "mine"
    assert captured["kwargs"]["mock"] is True
    assert (output_dir / "quickstart_report.html").exists()
    assert "Next real-data commands" in result.output
    assert "uv run factorminer validate-data path/to/market_data.csv" in result.output
