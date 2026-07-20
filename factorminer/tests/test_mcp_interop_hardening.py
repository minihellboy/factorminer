"""MCP interop hardening: tool docs, debate visibility, HTTP auth posture."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")  # noqa: F841 — optional extra

from factorminer.mcp import server as mcp_server  # noqa: E402 -- after importorskip
from factorminer.mcp.server import (  # noqa: E402 -- after importorskip
    DEFAULT_AUTH_TOKEN_ENV,
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PORT,
    StaticBearerTokenVerifier,
    configure_http_auth,
    doctor,
    get_factor_library,
    inspect_debate,
    list_fsi_connectors,
    run_server,
)

GUARDRAIL = "Returns a research artifact only -- never executes trades or size positions."

TOOL_FUNCS = [
    mcp_server.doctor,
    mcp_server.validate_data,
    mcp_server.fetch_data,
    mcp_server.list_fsi_connectors,
    mcp_server.resample_data,
    mcp_server.mine_factors,
    mcp_server.helix_mine,
    mcp_server.ingest_research_note,
    mcp_server.evaluate_library,
    mcp_server.screen_factors,
    mcp_server.combine_factors,
    mcp_server.run_benchmark,
    mcp_server.generate_report,
    mcp_server.export_library,
    mcp_server.inspect_session,
    mcp_server.get_factor_library,
    mcp_server.inspect_debate,
]


def test_every_tool_docstring_has_guardrail_and_shapes() -> None:
    for fn in TOOL_FUNCS:
        doc = fn.__doc__ or ""
        assert GUARDRAIL in doc, f"{fn.__name__} missing guardrail line"
        assert "Returns:" in doc or "Returns a research artifact" in doc, fn.__name__
        assert "Args:" in doc, f"{fn.__name__} missing Args section"


def test_inspect_debate_reads_debate_log(tmp_path: Path) -> None:
    payload = {
        "ok": True,
        "research_artifact_only": True,
        "n_rounds": 1,
        "rounds": [
            {
                "iteration": 1,
                "specialist_proposals": {
                    "momentum": ["rank(close/delay(close,5))"],
                    "meanrev": ["-rank(close/ts_mean(close,20))"],
                },
                "critic_scores": [
                    {
                        "factor_name": "mom5",
                        "formula": "rank(close/delay(close,5))",
                        "source_specialist": "momentum",
                        "scores": {"novelty": 0.8},
                        "composite_score": 0.7,
                        "keep": True,
                        "critique": "clean momentum",
                    },
                    {
                        "factor_name": "mr20",
                        "formula": "-rank(close/ts_mean(close,20))",
                        "source_specialist": "meanrev",
                        "scores": {"novelty": 0.2},
                        "composite_score": 0.2,
                        "keep": False,
                        "critique": "too correlated",
                    },
                ],
                "after_critic": ["rank(close/delay(close,5))"],
                "after_dedup": [
                    "rank(close/delay(close,5))",
                    "-rank(close/ts_mean(close,20))",
                ],
            }
        ],
    }
    (tmp_path / "debate_log.json").write_text(json.dumps(payload), encoding="utf-8")

    result = inspect_debate(str(tmp_path))
    assert result["ok"] is True
    assert result["research_artifact_only"] is True
    assert result["n_rounds"] == 1
    assert "momentum" in result["specialist_proposals"]
    assert len(result["critic_scores"]) == 2
    assert any(s.get("factor_name") == "mom5" for s in result["shortlist"])
    assert any(r.get("factor_name") == "mr20" for r in result["rejections"])
    assert GUARDRAIL.split(" -- ")[0].lower() in (inspect_debate.__doc__ or "").lower() or GUARDRAIL in (
        inspect_debate.__doc__ or ""
    )


def test_inspect_debate_missing_artifacts(tmp_path: Path) -> None:
    result = inspect_debate(str(tmp_path))
    assert result["ok"] is False
    assert "debate" in result["error"].lower()
    assert result["n_rounds"] == 0
    assert result["specialist_proposals"] == {}


def test_static_bearer_token_verifier() -> None:
    verifier = StaticBearerTokenVerifier("s3cret")

    async def _run() -> None:
        ok = await verifier.verify_token("s3cret")
        bad = await verifier.verify_token("nope")
        assert ok is not None
        assert ok.client_id == "factorminer-mcp"
        assert "factorminer" in ok.scopes
        assert bad is None

    asyncio.run(_run())


def test_configure_http_auth_sets_loopback_and_verifier() -> None:
    # Snapshot and restore module MCP auth state so other tests stay clean.
    prev_auth = mcp_server.mcp.settings.auth
    prev_host = mcp_server.mcp.settings.host
    prev_port = mcp_server.mcp.settings.port
    prev_verifier = getattr(mcp_server.mcp, "_token_verifier", None)
    try:
        configure_http_auth(
            host=DEFAULT_HTTP_HOST,
            port=DEFAULT_HTTP_PORT,
            auth_token="unit-test-token",
        )
        assert mcp_server.mcp.settings.host == "127.0.0.1"
        assert mcp_server.mcp.settings.port == 8765
        assert mcp_server.mcp.settings.auth is not None
        assert mcp_server.mcp._token_verifier is not None  # noqa: SLF001
        assert DEFAULT_HTTP_HOST != "0.0.0.0"
    finally:
        mcp_server.mcp.settings.auth = prev_auth
        mcp_server.mcp.settings.host = prev_host
        mcp_server.mcp.settings.port = prev_port
        mcp_server.mcp._token_verifier = prev_verifier  # noqa: SLF001


def test_run_server_http_refuses_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEFAULT_AUTH_TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="Refusing to start HTTP"):
        run_server(transport="http", auth_token_env=DEFAULT_AUTH_TOKEN_ENV)


def test_run_server_http_refuses_empty_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEFAULT_AUTH_TOKEN_ENV, "")
    with pytest.raises(RuntimeError, match="Refusing to start HTTP"):
        run_server(transport="http")


def test_run_server_sse_refuses_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test: 'sse' is a real network transport too and must not
    bind unauthenticated. A prior implementation gated 'streamable-http'
    but not 'sse' -- run_server(transport='sse') bound host:port with no
    token check and no configure_http_auth call. Not reachable via the
    CLI (click.Choice only offers stdio|http), but run_server() is a
    public function any direct caller could invoke with transport='sse'.
    """
    monkeypatch.delenv(DEFAULT_AUTH_TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match="Refusing to start HTTP"):
        run_server(transport="sse", auth_token_env=DEFAULT_AUTH_TOKEN_ENV)


def test_run_server_sse_refuses_empty_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEFAULT_AUTH_TOKEN_ENV, "")
    with pytest.raises(RuntimeError, match="Refusing to start HTTP"):
        run_server(transport="sse")


def test_doctor_redacts_package_install_path() -> None:
    """The packaged_config check's absolute server install path must never
    reach an MCP client -- unlike library_path/output_dir/debate_log_path
    (all echoes of a caller-supplied path), this one is genuine server
    filesystem layout the caller never supplied.
    """
    from factorminer.mcp.server import _PACKAGE_ROOT

    result = doctor()
    assert result["ok"] is True

    assert _PACKAGE_ROOT not in result["stdout"]
    assert _PACKAGE_ROOT not in (result["stderr"] or "")
    assert "<factorminer-install>" in result["stdout"]

    checks = result["data"]["checks"]
    packaged = next(c for c in checks if c["name"] == "packaged_config")
    assert packaged["status"] == "ok"
    assert _PACKAGE_ROOT not in packaged["detail"]
    assert "<factorminer-install>" in packaged["detail"]
    assert packaged["detail"].endswith("default.yaml")

    # Every other check must survive redaction untouched (this must be a
    # targeted redaction, not a destructive one).
    other_names = {c["name"] for c in checks if c["name"] != "packaged_config"}
    assert "effective_backend" in other_names


def test_list_fsi_connectors_shape() -> None:
    result = list_fsi_connectors()
    assert result["ok"] is True
    assert "connectors" in result
    assert "datetime" in result["required_schema"]
    assert "amount" in result["required_schema"]


def test_get_factor_library_missing(tmp_path: Path) -> None:
    result = get_factor_library(str(tmp_path / "nope.json"))
    assert result["ok"] is False
    assert "not found" in result["error"].lower()


def test_cli_mcp_serve_help_lists_transport_options() -> None:
    from click.testing import CliRunner

    from factorminer.cli import main

    runner = CliRunner()
    result = runner.invoke(main, ["mcp-serve", "--help"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "--transport" in out
    assert "stdio" in out
    assert "http" in out
    assert "--host" in out
    assert "--port" in out
    assert "--auth-token-env" in out
    assert "FACTORMINER_MCP_TOKEN" in out
    assert "127.0.0.1" in out
