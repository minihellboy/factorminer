"""FactorMiner MCP server.

Exposes the FactorMiner CLI as Model Context Protocol tools. Each tool is a thin
subprocess wrapper over ``python -m factorminer.cli`` -- the engine stays the
single source of truth, and the MCP layer carries only orchestration. That
boundary is deliberate: skill/agent prose can change freely without ever
depending on FactorMiner internals.

Launch with ``factorminer mcp-serve`` (stdio by default). Opt into authenticated
streamable-HTTP with ``factorminer mcp-serve --transport http``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

try:
    from mcp.server.auth.provider import AccessToken, TokenVerifier
    from mcp.server.auth.settings import AuthSettings
    from mcp.server.fastmcp import FastMCP
    from pydantic import AnyHttpUrl
except ModuleNotFoundError as exc:  # pragma: no cover - surfaced to the operator
    raise ModuleNotFoundError(
        "The FactorMiner MCP server requires the 'mcp' package. "
        "Install it with:  pip install 'factorminer[mcp]'"
    ) from exc

# Repo root is two levels up from factorminer/mcp/server.py. Used only to locate
# the docs/ tree for the documentation resource; tool paths are always explicit.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

# The installed factorminer/ package directory (one level up from mcp/server.py
# itself). Used only to redact this server's own filesystem install location
# from doctor's diagnostic text before it's relayed to an MCP client -- see
# `_redact_package_paths`.
_PACKAGE_ROOT = str(Path(__file__).resolve().parents[1])

# Mining and benchmarks can run for a long time, so the default ceiling is wide.
DEFAULT_TIMEOUT = 1800
QUICK_TIMEOUT = 300

# Default bind for the opt-in HTTP transport. Never 0.0.0.0.
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8765
DEFAULT_AUTH_TOKEN_ENV = "FACTORMINER_MCP_TOKEN"

mcp = FastMCP(
    "factorminer",
    instructions=(
        "FactorMiner mines, evaluates, and backtests quantitative alpha factors. "
        "Every output is a research artifact staged for review by a qualified "
        "professional -- these tools do not recommend trades or execute anything. "
        "Typical flow: validate_data -> mine_factors (or helix_mine) -> "
        "evaluate_library -> screen_factors -> generate_report."
    ),
)


class StaticBearerTokenVerifier:
    """Simple shared-secret bearer verifier for local streamable-HTTP.

    Implements :class:`mcp.server.auth.provider.TokenVerifier`. Tokens are read
    from an environment variable at process start; there is no OAuth dance.
    """

    def __init__(self, expected_token: str, *, client_id: str = "factorminer-mcp") -> None:
        if not expected_token:
            raise ValueError("expected_token must be a non-empty string")
        self._expected_token = expected_token
        self._client_id = client_id

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != self._expected_token:
            return None
        return AccessToken(
            token=token,
            client_id=self._client_id,
            scopes=["factorminer"],
        )


# Structural check against the TokenVerifier protocol.
_: TokenVerifier = StaticBearerTokenVerifier("unused")  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------

def _run_cli(args: list[str], timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Invoke the FactorMiner CLI in a subprocess and capture its result.

    The same interpreter that runs the server runs the CLI, so the server's
    virtualenv (and its installed ``factorminer``) is always used.
    """
    cmd = [sys.executable, "-m", "factorminer.cli", *args]
    try:
        proc = subprocess.run(  # noqa: S603 - args are constructed, not user shell
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"factorminer timed out after {timeout}s",
            "command": " ".join(args),
        }
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "command": " ".join(args),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _run_cli_json(args: list[str], timeout: int = QUICK_TIMEOUT) -> dict[str, Any]:
    """Run a CLI command that supports ``--json`` and parse its payload."""
    result = _run_cli(args, timeout=timeout)
    if result["ok"]:
        try:
            result["data"] = json.loads(result["stdout"])
        except json.JSONDecodeError:
            result["data"] = None
    return result


def _session_summary(output_dir: str) -> dict[str, Any] | None:
    """Best-effort structured summary of a finished mining session."""
    result = _run_cli_json(["session", "inspect", output_dir, "--json"])
    return result.get("data")


def _redact_package_paths(text: str | None) -> str | None:
    """Redact this server's own installation directory from text.

    ``doctor``'s ``packaged_config`` check reports the packaged
    ``default.yaml``'s absolute path -- unlike other MCP tool path fields
    (``library_path``, ``output_dir``, ``debate_log_path``, which all echo
    a path the CALLER supplied), this one is never caller-supplied, so it
    is real server-filesystem-layout disclosure, not an echo. Redacted
    only for the MCP relay; the CLI's own ``--json``/text output for local
    terminal users is left fully informative -- there is no information
    asymmetry there, since a local user already has full filesystem
    access to their own machine.
    """
    if not text:
        return text
    return text.replace(_PACKAGE_ROOT, "<factorminer-install>")


def _redact_doctor_result(result: dict[str, Any]) -> dict[str, Any]:
    """Apply :func:`_redact_package_paths` across a doctor CLI result."""
    result["stdout"] = _redact_package_paths(result.get("stdout", "")) or ""
    result["stderr"] = _redact_package_paths(result.get("stderr", "")) or ""
    data = result.get("data")
    if isinstance(data, dict):
        for check in data.get("checks") or []:
            if isinstance(check, dict) and isinstance(check.get("detail"), str):
                check["detail"] = _redact_package_paths(check["detail"])
    return result



# ---------------------------------------------------------------------------
# Diagnostics & data
# ---------------------------------------------------------------------------

@mcp.tool()
def doctor() -> dict[str, Any]:
    """Check the FactorMiner install and runtime health.

    Args:
        (none)

    Returns:
        dict with keys ok (bool), returncode (int), command (str), stdout (str),
        stderr (str), plus data (dict | None): parsed doctor --json payload when
        ok (packaged config, optional deps, evaluation backend, LLM credentials,
        output-dir writability). The packaged_config check's absolute
        install path is redacted to "<factorminer-install>" -- it is the
        one field in this tool that discloses server filesystem layout the
        caller did not itself supply.

    Returns a research artifact only -- never executes trades or size positions.
    """
    return _redact_doctor_result(_run_cli_json(["doctor", "--json"], timeout=120))


@mcp.tool()
def validate_data(path: str, strict: bool = False, hdf_key: str = "data") -> dict[str, Any]:
    """Validate a market-data file against the FactorMiner OHLCV schema.

    Accepts CSV, Parquet, and HDF5. Reports column aliasing, missing/derived
    fields, and train/test split coverage. Always run this before mining.

    Args:
        path: Path to the market-data file (str).
        strict: Treat warnings as failures (bool, default False).
        hdf_key: HDF5 key for .h5/.hdf5 files (str, default "data").

    Returns:
        dict with keys ok, returncode, command, stdout, stderr, and data
        (parsed validate-data --json payload when ok).

    Returns a research artifact only -- never executes trades or size positions.
    """
    args = ["validate-data", path, "--json", "--hdf-key", hdf_key]
    if strict:
        args.append("--strict")
    return _run_cli_json(args)


@mcp.tool()
def fetch_data(mcp_config: str, output: str) -> dict[str, Any]:
    """Pull market data from an external FSI MCP connector into a local file.

    Reads an MCP-source YAML config (server URL, tool name, field mapping),
    calls the connector, and writes a canonical OHLCV file usable by mine_factors.

    Args:
        mcp_config: Path to the MCP-source YAML config (str).
        output: Destination file path ending in .csv / .parquet / .h5 (str).

    Returns:
        dict with keys ok (bool), returncode (int), command (str), stdout (str),
        stderr (str). On success stdout describes the written file.

    Returns a research artifact only -- never executes trades or size positions.
    """
    return _run_cli(["fetch-data", "--mcp-config", mcp_config, "--output", output])


@mcp.tool()
def list_fsi_connectors() -> dict[str, Any]:
    """List bundled financial-services MCP connector endpoints.

    Use this before drafting an MCP-source config. Only endpoints that return a
    full OHLCV + amount panel can feed FactorMiner directly; fundamentals,
    transcripts, and documents should be treated as context for other agents.

    Args:
        (none)

    Returns:
        dict with keys:
          - ok (bool): always True on success
          - connectors (list[dict]): known connector endpoint descriptors
          - required_schema (list[str]): canonical OHLCV + amount column names

    Returns a research artifact only -- never executes trades or size positions.
    """
    from factorminer.data.mcp_source import known_mcp_connectors

    return {
        "ok": True,
        "connectors": known_mcp_connectors(),
        "required_schema": [
            "datetime",
            "asset_id",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ],
    }


@mcp.tool()
def resample_data(input_path: str, output_path: str, rule: str = "10min") -> dict[str, Any]:
    """Resample canonical OHLCV data to a coarser frequency (e.g. 5m -> 10min).

    Args:
        input_path: Source market-data file (str).
        output_path: Destination file (.csv / .parquet / .h5) (str).
        rule: Pandas resample rule, e.g. "10min", "1h", "1d" (str, default "10min").

    Returns:
        dict with keys ok, returncode, command, stdout, stderr.

    Returns a research artifact only -- never executes trades or size positions.
    """
    return _run_cli(["resample-data", input_path, output_path, "--rule", rule])


# ---------------------------------------------------------------------------
# Research engine
# ---------------------------------------------------------------------------

@mcp.tool()
def mine_factors(
    output_dir: str,
    data_path: str | None = None,
    mock: bool = False,
    iterations: int | None = None,
    batch_size: int | None = None,
    target: int | None = None,
    config: str | None = None,
    resume: str | None = None,
) -> dict[str, Any]:
    """Run a factor-mining session (the paper-faithful Ralph loop).

    The loop retrieves memory priors, proposes candidate factors with an LLM,
    evaluates them (IC / correlation / replacement), and admits the survivors to
    a factor library.

    Args:
        output_dir: Directory for all run artifacts (str).
        data_path: Market-data file path (str | None). Omit only when mock=True.
        mock: Use synthetic data and a mock LLM (bool, default False).
        iterations: Override max mining iterations (int | None).
        batch_size: Override candidates generated per iteration (int | None).
        target: Override target library size (int | None).
        config: Optional YAML config path (str | None).
        resume: Resume from a previously saved factor library path (str | None).

    Returns:
        dict with keys ok, returncode, command, stdout, stderr; on success also
        session (dict | None from session inspect) and library_path (str).

    Returns a research artifact only -- never executes trades or size positions.
    """
    args: list[str] = []
    if config:
        args += ["-c", config]
    args += ["-o", output_dir, "mine"]
    if data_path:
        args += ["--data", data_path]
    if mock:
        args.append("--mock")
    if iterations is not None:
        args += ["--iterations", str(iterations)]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]
    if target is not None:
        args += ["--target", str(target)]
    if resume:
        args += ["--resume", resume]

    result = _run_cli(args)
    if result["ok"]:
        result["session"] = _session_summary(output_dir)
        result["library_path"] = str(Path(output_dir) / "factor_library.json")
    return result


@mcp.tool()
def helix_mine(
    output_dir: str,
    data_path: str | None = None,
    mock: bool = False,
    iterations: int | None = None,
    batch_size: int | None = None,
    target: int | None = None,
    causal: bool | None = None,
    regime: bool | None = None,
    debate: bool | None = None,
    canonicalize: bool | None = None,
    config: str | None = None,
) -> dict[str, Any]:
    """Run the enhanced Helix loop with optional Phase 2 research features.

    Helix extends the Ralph loop with causal validation, regime-conditional
    evaluation, multi-specialist debate generation, and SymPy canonicalization.
    Each feature is opt-in; leaving a flag as None keeps the config default.
    When debate=True, specialist/critic artifacts are written to
    ``{output_dir}/debate_log.json`` and can be read via ``inspect_debate``.

    Args:
        output_dir: Directory for all run artifacts (str).
        data_path: Market-data file path (str | None). Omit only when mock=True.
        mock: Use synthetic data and a mock LLM (bool, default False).
        iterations: Override max mining iterations (int | None).
        batch_size: Override candidates per iteration (int | None).
        target: Override target library size (int | None).
        causal: Enable/disable do-calculus causal validation (bool | None).
        regime: Enable/disable regime-conditional evaluation (bool | None).
        debate: Enable/disable multi-specialist debate generation (bool | None).
        canonicalize: Enable/disable SymPy duplicate elimination (bool | None).
        config: Optional YAML config path (str | None).

    Returns:
        dict with keys ok, returncode, command, stdout, stderr; on success also
        session (dict | None) and library_path (str).

    Returns a research artifact only -- never executes trades or size positions.
    """
    args: list[str] = []
    if config:
        args += ["-c", config]
    args += ["-o", output_dir, "helix"]
    if data_path:
        args += ["--data", data_path]
    if mock:
        args.append("--mock")
    if iterations is not None:
        args += ["--iterations", str(iterations)]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]
    if target is not None:
        args += ["--target", str(target)]
    for flag, value in (
        ("causal", causal),
        ("regime", regime),
        ("debate", debate),
        ("canonicalize", canonicalize),
    ):
        if value is True:
            args.append(f"--{flag}")
        elif value is False:
            args.append(f"--no-{flag}")

    result = _run_cli(args)
    if result["ok"]:
        result["session"] = _session_summary(output_dir)
        result["library_path"] = str(Path(output_dir) / "factor_library.json")
    return result


@mcp.tool()
def ingest_research_note(note_path: str, mock: bool = False) -> dict[str, Any]:
    """Absorb a research report fragment via Report-to-Memory Absorption (RMA).

    Screens the fragment for OHLCV-representability (A-layer); if KEPT,
    classifies it into a broad mechanism family and reusable research-path
    hypothesis cues (B/C-layer).

    Args:
        note_path: Path to a plain-text research report/paper fragment (str).
        mock: Use the mock LLM provider; deterministic, for testing (bool).

    Returns:
        dict with keys ok, returncode, command, stdout, stderr.

    Returns a research artifact only -- never executes trades or size positions.
    """
    args = ["ingest-research", note_path]
    if mock:
        args.append("--mock")
    return _run_cli(args, timeout=QUICK_TIMEOUT)


# ---------------------------------------------------------------------------
# Evaluation, screening, backtesting
# ---------------------------------------------------------------------------

@mcp.tool()
def evaluate_library(
    library_path: str,
    data_path: str | None = None,
    mock: bool = False,
    period: str = "test",
    top_k: int | None = None,
) -> dict[str, Any]:
    """Recompute a factor library's metrics (IC, ICIR, win rate, turnover).

    Args:
        library_path: Path to a factor_library.json (str).
        data_path: Market-data file path (str | None). Omit only when mock=True.
        mock: Evaluate on synthetic data (bool, default False).
        period: One of "train", "test", "both" (str, default "test").
        top_k: Restrict to the top-K factors by IC (int | None).

    Returns:
        dict with keys ok, returncode, command, stdout, stderr. stdout holds the
        rendered metric table text when ok.

    Returns a research artifact only -- never executes trades or size positions.
    """
    args = ["evaluate", library_path, "--period", period]
    if data_path:
        args += ["--data", data_path]
    if mock:
        args.append("--mock")
    if top_k is not None:
        args += ["--top-k", str(top_k)]
    return _run_cli(args)


@mcp.tool()
def screen_factors(
    library_path: str,
    data_path: str | None = None,
    mock: bool = False,
    top_k: int = 10,
    period: str = "test",
) -> dict[str, Any]:
    """Rank a factor library and return the strongest signals.

    This is the bridge to research-idea workflows: it surfaces the top-K factors
    by out-of-sample IC, each with its formula and stats -- a quantitative signal
    shortlist a research agent can fold into an investment thesis.

    Args:
        library_path: Path to a factor_library.json (str).
        data_path: Market-data file path (str | None). Omit only when mock=True.
        mock: Screen on synthetic data (bool, default False).
        top_k: Number of top factors to return (int, default 10).
        period: Evaluation split to rank on -- "train" or "test" (str).

    Returns:
        dict with keys ok, returncode, command, stdout, stderr. stdout holds the
        ranked shortlist table when ok.

    Returns a research artifact only -- never executes trades or size positions.
    """
    args = ["evaluate", library_path, "--period", period, "--top-k", str(top_k)]
    if data_path:
        args += ["--data", data_path]
    if mock:
        args.append("--mock")
    return _run_cli(args)


@mcp.tool()
def combine_factors(
    library_path: str,
    data_path: str | None = None,
    mock: bool = False,
    method: str = "all",
    selection: str = "none",
    top_k: int | None = None,
    fit_period: str = "train",
    eval_period: str = "test",
) -> dict[str, Any]:
    """Combine library factors into a composite signal and quintile-backtest it.

    Reports composite IC, ICIR, long-short return, monotonicity, and turnover --
    the portfolio-level view that single-factor evaluation does not give.

    Args:
        library_path: Path to a factor_library.json (str).
        data_path: Market-data file path (str | None). Omit only when mock=True.
        mock: Run on synthetic data (bool, default False).
        method: "equal-weight", "ic-weighted", "orthogonal", or "all" (str).
        selection: Pre-selection -- "lasso", "stepwise", "xgboost", "none" (str).
        top_k: Pre-select the top-K factors before combining (int | None).
        fit_period: Split used for selection / weight fitting (str, default "train").
        eval_period: Split used to evaluate the composite (str, default "test").

    Returns:
        dict with keys ok, returncode, command, stdout, stderr.

    Returns a research artifact only -- never executes trades or size positions.
    """
    args = [
        "combine",
        library_path,
        "--method",
        method,
        "--selection",
        selection,
        "--fit-period",
        fit_period,
        "--eval-period",
        eval_period,
    ]
    if data_path:
        args += ["--data", data_path]
    if mock:
        args.append("--mock")
    if top_k is not None:
        args += ["--top-k", str(top_k)]
    return _run_cli(args)


# ---------------------------------------------------------------------------
# Benchmark & reporting
# ---------------------------------------------------------------------------

_BENCHMARK_MODES = {
    "table1",
    "ablation-memory",
    "ablation-strategy",
    "cost-pressure",
    "efficiency",
    "suite",
}


@mcp.tool()
def run_benchmark(
    mode: str,
    output_dir: str = "output",
    data_path: str | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    """Run a FactorMiner benchmark workflow.

    Args:
        mode: One of table1, ablation-memory, ablation-strategy, cost-pressure,
            efficiency, suite (str).
        output_dir: Directory for benchmark artifacts (str, default "output").
        data_path: Market-data file path (str | None). Omit only when mock=True.
        mock: Run the benchmark on synthetic data (bool, default False).

    Returns:
        dict with keys ok, returncode, command, stdout, stderr; or
        {ok: False, error: str} when mode is unknown.

    Returns a research artifact only -- never executes trades or size positions.
    """
    if mode not in _BENCHMARK_MODES:
        return {
            "ok": False,
            "error": f"Unknown benchmark mode '{mode}'. Choose from: "
            + ", ".join(sorted(_BENCHMARK_MODES)),
        }
    args = ["-o", output_dir, "benchmark", mode]
    if mode != "efficiency":
        if data_path:
            args += ["--data", data_path]
        if mock:
            args.append("--mock")
    return _run_cli(args)


@mcp.tool()
def generate_report(
    library_path: str,
    output: str,
    session_log: str | None = None,
    benchmark: list[str] | None = None,
    report_format: str = "markdown",
) -> dict[str, Any]:
    """Generate a static report (markdown or HTML) from FactorMiner artifacts.

    Args:
        library_path: Path to a factor_library.json (str).
        output: Destination file for the report (str).
        session_log: Optional session_log.json path (str | None).
        benchmark: Optional list of benchmark JSON paths (list[str] | None).
        report_format: "markdown" or "html" (str, default "markdown").

    Returns:
        dict with keys ok, returncode, command, stdout, stderr.

    Returns a research artifact only -- never executes trades or size positions.
    """
    args = ["report", library_path, "--format", report_format, "--output", output]
    if session_log:
        args += ["--session-log", session_log]
    for path in benchmark or []:
        args += ["--benchmark", path]
    return _run_cli(args)


@mcp.tool()
def export_library(
    library_path: str,
    output: str,
    export_format: str = "json",
) -> dict[str, Any]:
    """Export a factor library to json, csv, or a plain formulas list.

    Args:
        library_path: Path to a factor_library.json (str).
        output: Destination file (str).
        export_format: "json", "csv", or "formulas" (str, default "json").

    Returns:
        dict with keys ok, returncode, command, stdout, stderr.

    Returns a research artifact only -- never executes trades or size positions.
    """
    return _run_cli(
        ["export", library_path, "--format", export_format, "--output", output]
    )


@mcp.tool()
def inspect_session(output_dir: str) -> dict[str, Any]:
    """Summarize a FactorMiner run directory.

    Args:
        output_dir: Path to a mining/helix output directory (str).

    Returns:
        dict with keys ok, returncode, command, stdout, stderr, and data
        (parsed session inspect JSON: status, library size, iterations,
        yield rate, artifact-consistency warnings) when ok.

    Returns a research artifact only -- never executes trades or size positions.
    """
    return _run_cli_json(["session", "inspect", output_dir, "--json"])


@mcp.tool()
def get_factor_library(library_path: str) -> dict[str, Any]:
    """Return the raw contents of a factor_library.json.

    Args:
        library_path: Path to a factor_library.json (str).

    Returns:
        On success: {ok: True, library: dict} where library is the parsed JSON
        (factors list plus metadata).
        On failure: {ok: False, error: str}.

    Returns a research artifact only -- never executes trades or size positions.
    """
    path = Path(library_path)
    if not path.is_file():
        return {"ok": False, "error": f"Factor library not found: {library_path}"}
    try:
        return {"ok": True, "library": json.loads(path.read_text(encoding="utf-8"))}
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Invalid JSON in {library_path}: {exc}"}


@mcp.tool()
def inspect_debate(output_dir: str) -> dict[str, Any]:
    """Inspect multi-specialist debate / critic artifacts from a Helix session.

    Reads ``{output_dir}/debate_log.json`` written when ``helix_mine(..., debate=True)``
    runs. Surfaces specialist proposals, critic scores, shortlist/rejection
    reasoning, and optional specialist leaderboard stats.

    Args:
        output_dir: Path to a Helix run directory that may contain debate_log.json (str).

    Returns:
        dict with keys:
          - ok (bool)
          - research_artifact_only (bool): always True when ok
          - output_dir (str)
          - debate_log_path (str | None)
          - n_rounds (int)
          - rounds (list[dict]): each round may include specialist_proposals,
            critic_scores (factor_name, formula, source_specialist, scores,
            composite_score, keep, critique), after_dedup, after_critic,
            all_proposals, specialist_success_rates, debate_stats,
            specialist_leaderboard
          - specialist_proposals (dict[str, list]): merged across rounds
          - critic_scores (list[dict]): flattened across rounds
          - shortlist (list): factors kept by the critic (keep=True) or after_critic
          - rejections (list[dict]): critic rows with keep=False
          - error (str): present only when ok is False

    Returns a research artifact only -- never executes trades or size positions.
    """
    root = Path(output_dir)
    path = root / "debate_log.json"
    if not path.is_file():
        # Fall back to sparse signals in session artifacts if debate_log is absent.
        fallback = _debate_fallback_from_session(root)
        if fallback is not None:
            return fallback
        return {
            "ok": False,
            "error": (
                f"No debate artifacts in {output_dir}. "
                "Re-run helix_mine with debate=True to persist debate_log.json."
            ),
            "output_dir": str(root),
            "debate_log_path": None,
            "n_rounds": 0,
            "rounds": [],
            "specialist_proposals": {},
            "critic_scores": [],
            "shortlist": [],
            "rejections": [],
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "error": f"Invalid JSON in {path}: {exc}",
            "output_dir": str(root),
            "debate_log_path": str(path),
        }

    rounds = list(payload.get("rounds") or [])
    specialist_proposals: dict[str, list[Any]] = {}
    critic_scores: list[dict[str, Any]] = []
    shortlist: list[Any] = []
    rejections: list[dict[str, Any]] = []

    for rnd in rounds:
        if not isinstance(rnd, dict):
            continue
        for name, formulas in (rnd.get("specialist_proposals") or {}).items():
            bucket = specialist_proposals.setdefault(str(name), [])
            if isinstance(formulas, list):
                bucket.extend(formulas)
            else:
                bucket.append(formulas)
        for score in rnd.get("critic_scores") or []:
            if not isinstance(score, dict):
                continue
            critic_scores.append(score)
            if score.get("keep"):
                shortlist.append(
                    {
                        "factor_name": score.get("factor_name"),
                        "formula": score.get("formula"),
                        "source_specialist": score.get("source_specialist"),
                        "composite_score": score.get("composite_score"),
                        "critique": score.get("critique"),
                        "reason": "critic_keep",
                    }
                )
            else:
                rejections.append(
                    {
                        "factor_name": score.get("factor_name"),
                        "formula": score.get("formula"),
                        "source_specialist": score.get("source_specialist"),
                        "composite_score": score.get("composite_score"),
                        "critique": score.get("critique"),
                        "reason": "critic_reject",
                    }
                )
        after_critic = rnd.get("after_critic") or []
        if after_critic and not any(s.get("keep") for s in (rnd.get("critic_scores") or []) if isinstance(s, dict)):
            for item in after_critic:
                shortlist.append({"candidate": item, "reason": "after_critic"})

    return {
        "ok": True,
        "research_artifact_only": True,
        "output_dir": str(root),
        "debate_log_path": str(path),
        "n_rounds": int(payload.get("n_rounds") or len(rounds)),
        "rounds": rounds,
        "specialist_proposals": specialist_proposals,
        "critic_scores": critic_scores,
        "shortlist": shortlist,
        "rejections": rejections,
    }


def _debate_fallback_from_session(output_dir: Path) -> dict[str, Any] | None:
    """Best-effort parse of session_log / lifecycle when debate_log is missing."""
    session_path = output_dir / "session_log.json"
    lifecycle_path = output_dir / "factor_lifecycle.jsonl"
    specialist_proposals: dict[str, list[Any]] = {}
    critic_scores: list[dict[str, Any]] = []
    found = False

    if session_path.is_file():
        try:
            session = json.loads(session_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            session = None
        if isinstance(session, dict):
            for key in ("debate_rounds", "debate", "specialist_proposals"):
                if key in session:
                    found = True
            raw_props = session.get("specialist_proposals")
            if isinstance(raw_props, dict):
                specialist_proposals = {str(k): list(v) if isinstance(v, list) else [v] for k, v in raw_props.items()}
            raw_scores = session.get("critic_scores")
            if isinstance(raw_scores, list):
                critic_scores = [s for s in raw_scores if isinstance(s, dict)]

    if lifecycle_path.is_file():
        try:
            lines = lifecycle_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if any(k in row for k in ("specialist", "source_specialist", "critic_score", "debate")):
                found = True
            specialist = row.get("source_specialist") or row.get("specialist")
            formula = row.get("formula")
            if specialist and formula is not None:
                specialist_proposals.setdefault(str(specialist), []).append(formula)
            if "critic_score" in row or "composite_score" in row:
                critic_scores.append(row)

    if not found:
        return None

    shortlist = [s for s in critic_scores if s.get("keep")]
    rejections = [s for s in critic_scores if "keep" in s and not s.get("keep")]
    return {
        "ok": True,
        "research_artifact_only": True,
        "output_dir": str(output_dir),
        "debate_log_path": None,
        "n_rounds": 0,
        "rounds": [],
        "specialist_proposals": specialist_proposals,
        "critic_scores": critic_scores,
        "shortlist": shortlist,
        "rejections": rejections,
        "note": "Reconstructed from session artifacts; debate_log.json was absent.",
    }


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("factorminer://docs/{topic}")
def read_doc(topic: str) -> str:
    """Return a FactorMiner documentation file.

    Current topics are architecture, reproducibility, and security. Available
    topics are discovered from Markdown files under ``docs/`` at runtime.
    """
    safe = topic.replace("..", "").strip("/")
    path = DOCS_DIR / f"{safe}.md"
    if not path.is_file():
        if DOCS_DIR.is_dir():
            available = ", ".join(sorted(p.stem for p in DOCS_DIR.glob("*.md")))
        else:
            available = "none (docs/ directory not found)"
        return f"Doc '{topic}' not found. Available topics: {available}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Transport entry
# ---------------------------------------------------------------------------

def configure_http_auth(
    *,
    host: str = DEFAULT_HTTP_HOST,
    port: int = DEFAULT_HTTP_PORT,
    auth_token: str,
) -> None:
    """Attach bearer-token auth and bind settings for HTTP-family transports.

    Mutates the module-level ``mcp`` instance in place so tool registrations
    stay intact. Call only when launching ``transport in
    ('streamable-http', 'sse')`` -- never for ``stdio``, which is
    process-local and needs no auth.
    """
    if not auth_token:
        raise ValueError("auth_token must be a non-empty bearer secret")
    mcp.settings.host = host
    mcp.settings.port = port
    base = f"http://{host}:{port}"
    mcp.settings.auth = AuthSettings(
        issuer_url=AnyHttpUrl(base),
        resource_server_url=AnyHttpUrl(f"{base}/mcp"),
        required_scopes=["factorminer"],
    )
    mcp._token_verifier = StaticBearerTokenVerifier(auth_token)  # noqa: SLF001


def run_server(
    transport: Literal["stdio", "http", "streamable-http", "sse"] = "stdio",
    *,
    host: str = DEFAULT_HTTP_HOST,
    port: int = DEFAULT_HTTP_PORT,
    auth_token_env: str = DEFAULT_AUTH_TOKEN_ENV,
) -> None:
    """Run the MCP server on the requested transport.

    ``stdio`` (default) needs no auth. ``http`` / ``streamable-http`` bind to
    ``host:port`` (default 127.0.0.1:8765) and require a non-empty bearer token
    in the environment variable named by ``auth_token_env``.
    """
    normalized: Literal["stdio", "sse", "streamable-http"]
    if transport == "http":
        normalized = "streamable-http"
    elif transport in ("stdio", "sse", "streamable-http"):
        normalized = transport  # type: ignore[assignment]
    else:  # pragma: no cover - guarded by CLI choices
        raise ValueError(f"Unsupported transport: {transport}")

    if normalized in ("streamable-http", "sse"):
        token = os.environ.get(auth_token_env, "")
        if not token:
            raise RuntimeError(
                f"Refusing to start HTTP MCP transport without auth. "
                f"Set a non-empty bearer token in ${auth_token_env}, e.g. "
                f"export {auth_token_env}=$(openssl rand -hex 32)"
            )
        configure_http_auth(host=host, port=port, auth_token=token)
        mcp.run(transport=normalized)
        return

    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    run_server()
