"""FactorMiner MCP server.

Exposes the FactorMiner CLI as Model Context Protocol tools. Each tool is a thin
subprocess wrapper over ``python -m factorminer.cli`` -- the engine stays the
single source of truth, and the MCP layer carries only orchestration. That
boundary is deliberate: skill/agent prose can change freely without ever
depending on FactorMiner internals.

Launch with ``factorminer mcp-serve`` (stdio transport).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - surfaced to the operator
    raise ModuleNotFoundError(
        "The FactorMiner MCP server requires the 'mcp' package. "
        "Install it with:  pip install 'factorminer[mcp]'"
    ) from exc

# Repo root is two levels up from factorminer/mcp/server.py. Used only to locate
# the docs/ tree for the documentation resource; tool paths are always explicit.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

# Mining and benchmarks can run for a long time, so the default ceiling is wide.
DEFAULT_TIMEOUT = 1800
QUICK_TIMEOUT = 300

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


# ---------------------------------------------------------------------------
# Diagnostics & data
# ---------------------------------------------------------------------------

@mcp.tool()
def doctor() -> dict[str, Any]:
    """Check the FactorMiner install: packaged config, optional dependencies,
    evaluation backend, LLM credentials, and output-directory writability.

    Run this first when a session misbehaves -- it is the fastest way to see
    whether the environment is healthy.
    """
    return _run_cli_json(["doctor", "--json"], timeout=120)


@mcp.tool()
def validate_data(path: str, strict: bool = False, hdf_key: str = "data") -> dict[str, Any]:
    """Validate a market-data file against the FactorMiner OHLCV schema.

    Accepts CSV, Parquet, and HDF5. Reports column aliasing, missing/derived
    fields, and train/test split coverage. Always run this before mining.

    Args:
        path: Path to the market-data file.
        strict: Treat warnings as failures.
        hdf_key: HDF5 key to read for .h5/.hdf5 files.
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
        mcp_config: Path to the MCP-source YAML config.
        output: Destination file (.csv / .parquet / .h5).
    """
    return _run_cli(["fetch-data", "--mcp-config", mcp_config, "--output", output])


@mcp.tool()
def list_fsi_connectors() -> dict[str, Any]:
    """List bundled financial-services MCP connector endpoints.

    Use this before drafting an MCP-source config. Only endpoints that return a
    full OHLCV + amount panel can feed FactorMiner directly; fundamentals,
    transcripts, and documents should be treated as context for other agents.
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
        input_path: Source market-data file.
        output_path: Destination file (.csv / .parquet / .h5).
        rule: Pandas resample rule, e.g. "10min", "1h", "1d".
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
    a factor library. Returns the CLI log plus a structured session summary and
    the path to the saved factor_library.json.

    Args:
        output_dir: Directory for all run artifacts.
        data_path: Market-data file. Omit only when mock=True.
        mock: Use synthetic data and a mock LLM (no API calls) -- for smoke tests.
        iterations: Override max mining iterations.
        batch_size: Override candidates generated per iteration.
        target: Override target library size.
        config: Optional YAML config path.
        resume: Resume from a previously saved factor library.
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

    Args:
        output_dir: Directory for all run artifacts.
        data_path: Market-data file. Omit only when mock=True.
        mock: Use synthetic data and a mock LLM.
        iterations: Override max mining iterations.
        batch_size: Override candidates per iteration.
        target: Override target library size.
        causal: Enable/disable do-calculus causal validation.
        regime: Enable/disable regime-conditional evaluation.
        debate: Enable/disable multi-specialist debate generation.
        canonicalize: Enable/disable SymPy duplicate elimination.
        config: Optional YAML config path.
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

    Returns the rendered metric table as text. Use period="both" to see
    train -> test decay.

    Args:
        library_path: Path to a factor_library.json.
        data_path: Market-data file. Omit only when mock=True.
        mock: Evaluate on synthetic data.
        period: One of "train", "test", "both".
        top_k: Restrict to the top-K factors by IC.
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
        library_path: Path to a factor_library.json.
        data_path: Market-data file. Omit only when mock=True.
        mock: Screen on synthetic data.
        top_k: Number of top factors to return.
        period: Evaluation split to rank on ("train" or "test").
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
        library_path: Path to a factor_library.json.
        data_path: Market-data file. Omit only when mock=True.
        mock: Run on synthetic data.
        method: "equal-weight", "ic-weighted", "orthogonal", or "all".
        selection: Pre-selection method -- "lasso", "stepwise", "xgboost", "none".
        top_k: Pre-select the top-K factors before combining.
        fit_period: Split used for selection / weight fitting.
        eval_period: Split used to evaluate the composite.
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
            efficiency, suite.
        output_dir: Directory for benchmark artifacts.
        data_path: Market-data file. Omit only when mock=True.
        mock: Run the benchmark on synthetic data.
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
        library_path: Path to a factor_library.json.
        output: Destination file for the report.
        session_log: Optional session_log.json to include run metadata.
        benchmark: Optional list of benchmark JSON paths to include.
        report_format: "markdown" or "html".
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
        library_path: Path to a factor_library.json.
        output: Destination file.
        export_format: "json", "csv", or "formulas".
    """
    return _run_cli(
        ["export", library_path, "--format", export_format, "--output", output]
    )


@mcp.tool()
def inspect_session(output_dir: str) -> dict[str, Any]:
    """Summarize a FactorMiner run directory: status, library size, iterations,
    yield rate, and any artifact-consistency warnings.
    """
    return _run_cli_json(["session", "inspect", output_dir, "--json"])


@mcp.tool()
def get_factor_library(library_path: str) -> dict[str, Any]:
    """Return the raw contents of a factor_library.json (factors and metadata)."""
    path = Path(library_path)
    if not path.is_file():
        return {"ok": False, "error": f"Factor library not found: {library_path}"}
    try:
        return {"ok": True, "library": json.loads(path.read_text(encoding="utf-8"))}
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"Invalid JSON in {library_path}: {exc}"}


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

@mcp.resource("factorminer://docs/{topic}")
def read_doc(topic: str) -> str:
    """Return a FactorMiner documentation file.

    Topics include architecture, metrics, reproducibility, baselines, faq,
    paper-claims, repo-audit, and binance-reproduction.
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


if __name__ == "__main__":  # pragma: no cover
    mcp.run()
