#!/usr/bin/env python3
"""HelixFactor Phase 2 Comprehensive Benchmark Runner

Generates a complete publication-quality benchmarking report comparing
HelixFactor (Phase 2) against FactorMiner (Ralph Loop) and all baselines.

Usage:
    python run_phase2_benchmark.py --mock                  # quick mock data run
    python run_phase2_benchmark.py --mock --n-factors 40   # custom factor count
    python run_phase2_benchmark.py --mock --full-ablation  # include all ablations
    python run_phase2_benchmark.py --data path/to/data.csv # real data

Outputs (in results/phase2_benchmark/):
    benchmark_report.html      — full interactive HTML report
    benchmark_report.md        — GitHub-ready Markdown table
    benchmark_report_full.md    — narrative markdown report
    latex_table.tex            — publication LaTeX Table 1
    ablation_table.tex         — ablation study LaTeX table
    statistical_tests.json     — all statistical test results
    phase2_manifest.json       — machine-readable artifact/provenance manifest
    library_metrics.csv        — per-method library metrics
    combination_metrics.csv    — per-method combination metrics
    selection_metrics.csv      — per-method selection metrics
    turnover_metrics.csv       — runtime turnover metrics
    cost_pressure_metrics.csv  — runtime cost-adjusted metrics
    runtime_topk.csv           — runtime top-k summary
    comparison_plot.png        — bar chart comparison figure
    ablation_contributions.csv — component contribution summary
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Insert the repo root so the legacy Phase 2 report runner can import
# `helix_benchmark.py` directly without depending on the canonical runtime
# benchmark package surface.
_REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT))


def _load_module_direct(module_name: str, file_path: Path):
    """Load a Python module directly from a file path, bypassing package init."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so lazy imports inside the module work
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HelixFactor Phase 2 Comprehensive Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use synthetic mock data (no API keys needed)",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to real market data CSV",
    )
    parser.add_argument(
        "--n-factors", type=int, default=40,
        help="Target library size per method",
    )
    parser.add_argument(
        "--n-assets", type=int, default=100,
        help="Number of assets in mock data",
    )
    parser.add_argument(
        "--n-periods", type=int, default=600,
        help="Number of time periods in mock data",
    )
    parser.add_argument(
        "--output", type=str, default="results/phase2_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--methods", nargs="*", default=None,
        help="Methods to benchmark (default: all 5)",
    )
    parser.add_argument(
        "--full-ablation", action="store_true",
        help="Run full ablation study (slower)",
    )
    parser.add_argument(
        "--skip-ablation", action="store_true",
        help="Skip ablation study entirely",
    )
    parser.add_argument(
        "--log-level", type=str, default="WARNING",
        help="Logging level",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


def _subsection(title: str) -> None:
    print(f"\n  --- {title} ---")


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _json_safe(value: Any) -> Any:
    """Recursively convert a structure into JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path) as fp:
            payload = json.load(fp)
    except Exception as exc:  # pragma: no cover - defensive provenance capture
        return {"path": str(path), "load_error": str(exc)}
    if isinstance(payload, dict):
        return payload
    return {"path": str(path), "payload_type": type(payload).__name__}


def _collect_runtime_manifest_refs(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []

    refs: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("*_manifest.json")):
        if manifest_path.name == "phase2_manifest.json":
            continue
        payload = _load_json(manifest_path)
        if payload is None:
            continue

        refs.append(
            {
                "path": str(manifest_path),
                "sha256": _file_sha256(manifest_path),
                "benchmark_name": payload.get("benchmark_name"),
                "baseline": payload.get("baseline"),
                "mode": payload.get("mode"),
                "artifact_paths": payload.get("artifact_paths", {}),
                "baseline_provenance": payload.get("baseline_provenance", {}),
            }
        )
    return refs


def _build_phase2_manifest(
    *,
    output_dir: Path,
    methods: list[str],
    seed: int,
    n_factors: int,
    mock: bool,
    data_path: str | None,
    full_ablation: bool,
    skip_ablation: bool,
    artifact_paths: dict[str, str],
    statistical_tests: dict[str, Any],
    ablation_configs: list[str] | None = None,
    runtime_manifest_root: Path | None = None,
) -> dict[str, Any]:
    runtime_refs = _collect_runtime_manifest_refs(
        runtime_manifest_root or output_dir
    )
    return {
        "benchmark_name": "phase2",
        "output_dir": str(output_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_parameters": {
            "methods": methods,
            "seed": seed,
            "n_factors": n_factors,
            "mock": mock,
            "data_path": data_path,
            "full_ablation": full_ablation,
            "skip_ablation": skip_ablation,
        },
        "artifact_paths": artifact_paths,
        "statistical_tests": _json_safe(statistical_tests),
        "ablation": {
            "configs": ablation_configs or [],
        },
        "runtime_manifest_root": str(runtime_manifest_root or output_dir),
        "runtime_manifest_refs": runtime_refs,
    }


def _derive_split_periods(raw_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Derive contiguous train/test periods from the loaded market data."""
    timestamps = pd.to_datetime(raw_df["datetime"]).sort_values().unique()
    if len(timestamps) < 2:
        raise ValueError("Need at least two timestamps to derive train/test splits")

    split_idx = max(int(len(timestamps) * 0.7), 1)
    split_idx = min(split_idx, len(timestamps) - 1)
    train_start = pd.Timestamp(timestamps[0]).isoformat()
    train_end = pd.Timestamp(timestamps[split_idx - 1]).isoformat()
    test_start = pd.Timestamp(timestamps[split_idx]).isoformat()
    test_end = pd.Timestamp(timestamps[-1]).isoformat()
    return [train_start, train_end], [test_start, test_end]


def _runtime_topk_markdown(runtime_artifacts: dict[str, Any]) -> str:
    frame = _runtime_topk_frame(runtime_artifacts)
    if frame.empty:
        return ""
    return frame.to_markdown(index=False, floatfmt=".4f")


def _runtime_topk_frame(runtime_artifacts: dict[str, Any]) -> pd.DataFrame:
    payloads = runtime_artifacts.get("runtime_payloads", {})
    rows = []
    for method, runs in payloads.items():
        if not runs:
            continue
        topk = runs[0].get("frozen_top_k", [])
        for rank, item in enumerate(topk[:10], 1):
            rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "name": item.get("name", ""),
                    "train_ic": item.get("train_ic", 0.0),
                    "train_icir": item.get("train_icir", 0.0),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _print_improvement_table(bench_result) -> None:
    """Print clear table showing HelixFactor improvement over FactorMiner."""
    lib = bench_result.factor_library_metrics
    comb = bench_result.combination_metrics
    sel = bench_result.selection_metrics

    helix_lib = lib[lib["method"] == "helix_phase2"]
    ralph_lib = lib[lib["method"] == "ralph_loop"]
    helix_comb = comb[comb["method"] == "helix_phase2"]
    ralph_comb = comb[comb["method"] == "ralph_loop"]
    helix_sel = sel[sel["method"] == "helix_phase2"]
    ralph_sel = sel[sel["method"] == "ralph_loop"]

    if helix_lib.empty or ralph_lib.empty:
        print("  (Could not compute improvement — method results missing)")
        return

    def _get(df, col, default=0.0):
        if df.empty or col not in df.columns:
            return default
        v = df.iloc[0][col]
        return float(v) if v == v else default  # NaN check

    h_ic = _get(helix_lib, "ic_pct")
    r_ic = _get(ralph_lib, "ic_pct")
    h_icir = _get(helix_lib, "icir")
    r_icir = _get(ralph_lib, "icir")
    h_ew = _get(helix_comb, "ew_ic_pct")
    r_ew = _get(ralph_comb, "ew_ic_pct")
    h_icw = _get(helix_comb, "icw_ic_pct")
    r_icw = _get(ralph_comb, "icw_ic_pct")
    h_las = _get(helix_sel, "lasso_ic_pct")
    r_las = _get(ralph_sel, "lasso_ic_pct")
    h_xgb = _get(helix_sel, "xgb_ic_pct")
    r_xgb = _get(ralph_sel, "xgb_ic_pct")

    def _delta(h, r):
        if r < 1e-8:
            return "N/A"
        return f"+{(h - r) / r * 100:.1f}%"

    print(f"\n  {'Metric':<28} {'FactorMiner':>12} {'HelixFactor':>12} {'Improvement':>12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12}")
    metrics = [
        ("Library IC (%)", r_ic, h_ic),
        ("Library ICIR", r_icir, h_icir),
        ("EW Combo IC (%)", r_ew, h_ew),
        ("ICW Combo IC (%)", r_icw, h_icw),
        ("LASSO Sel IC (%)", r_las, h_las),
        ("XGBoost Sel IC (%)", r_xgb, h_xgb),
    ]
    for name, r_val, h_val in metrics:
        print(
            f"  {name:<28} {r_val:>12.4f} {h_val:>12.4f} {_delta(h_val, r_val):>12}"
        )


def _fmt_stat(v, fmt=".4f") -> str:
    """Format a stat value, showing N/A for NaN."""
    if v is None:
        return "N/A"
    try:
        f = float(v)
        if f != f:  # NaN
            return "N/A"
        return format(f, fmt)
    except (TypeError, ValueError):
        return str(v)


def _print_stat_tests(stat_tests: dict) -> None:
    dm = stat_tests.get("diebold_mariano", {})
    boot = stat_tests.get("bootstrap_ci_95", {})
    tt = stat_tests.get("paired_t_test", {})
    wil = stat_tests.get("wilcoxon", {})
    mean_diff = stat_tests.get("mean_ic_difference", 0.0)

    print(f"  Mean IC difference (Helix - Ralph): {_fmt_stat(mean_diff, '+.4f')}")
    print(f"  Helix outperforms: {stat_tests.get('helix_outperforms', '?')}")
    print()

    dm_p = dm.get("p_value", float("nan"))
    dm_stat_val = dm.get("dm_stat", float("nan"))
    try:
        sig_dm = "  *" if float(dm_p) < 0.05 else ""
    except (TypeError, ValueError):
        sig_dm = ""
    print("  Diebold-Mariano test:")
    print(f"    DM statistic = {_fmt_stat(dm_stat_val)}{sig_dm}")
    print(f"    p-value      = {_fmt_stat(dm_p)}")
    print(f"    Direction    = {dm.get('direction', '?')}")
    print()

    tt_p = tt.get("p_value", float("nan"))
    try:
        sig_tt = "  *" if float(tt_p) < 0.05 else ""
    except (TypeError, ValueError):
        sig_tt = ""
    print("  Paired t-test:")
    print(f"    t-stat  = {_fmt_stat(tt.get('t_stat', float('nan')))}{sig_tt}")
    print(f"    p-value = {_fmt_stat(tt_p)}")
    print(f"    n       = {tt.get('n', 0)}")
    print()

    lo = boot.get("lower", 0.0)
    hi = boot.get("upper", 0.0)
    print("  Block-bootstrap 95% CI on IC difference:")
    print(f"    [{_fmt_stat(lo)}, {_fmt_stat(hi)}]  "
          f"{'(excludes zero **)' if boot.get('excludes_zero') else ''}")
    print()

    wil_p = wil.get("p_value", float("nan"))
    try:
        sig_wil = "  *" if float(wil_p) < 0.05 else ""
    except (TypeError, ValueError):
        sig_wil = ""
    print("  Wilcoxon signed-rank:")
    print(f"    stat    = {_fmt_stat(wil.get('statistic', 0.0), '.1f')}{sig_wil}")
    print(f"    p-value = {_fmt_stat(wil_p)}")


def _generate_markdown_report(bench_result, ablation_result, output_dir: Path) -> str:
    """Build and write a comprehensive narrative Markdown report."""
    md = ["# HelixFactor Phase 2 Benchmark Report\n"]
    md.append(f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    md.append("\n## Table 1: Factor Library Metrics\n")
    md.append(bench_result.factor_library_metrics.to_markdown(index=False, floatfmt=".4f"))

    md.append("\n\n## Table 2: Factor Combination Metrics\n")
    md.append(bench_result.combination_metrics.to_markdown(index=False, floatfmt=".4f"))

    md.append("\n\n## Table 3: Factor Selection Metrics\n")
    md.append(bench_result.selection_metrics.to_markdown(index=False, floatfmt=".4f"))

    md.append("\n\n## Table 4: Speed Benchmarks\n")
    md.append(bench_result.speed_metrics.to_markdown(index=False, floatfmt=".3f"))

    if not getattr(bench_result, "turnover_metrics", pd.DataFrame()).empty:
        md.append("\n\n## Table 5: Turnover Metrics\n")
        md.append(bench_result.turnover_metrics.to_markdown(index=False, floatfmt=".4f"))

    if not getattr(bench_result, "cost_pressure_metrics", pd.DataFrame()).empty:
        md.append("\n\n## Table 6: Cost Pressure Metrics\n")
        md.append(
            bench_result.cost_pressure_metrics.to_markdown(index=False, floatfmt=".4f")
        )

    runtime_topk = _runtime_topk_markdown(getattr(bench_result, "runtime_artifacts", {}))
    if runtime_topk:
        md.append("\n\n## Runtime Top-K\n")
        md.append(runtime_topk)

    # Statistical tests
    stat = bench_result.statistical_tests
    if stat:
        md.append("\n\n## Statistical Tests (HelixFactor vs FactorMiner)\n")
        dm = stat.get("diebold_mariano", {})
        boot = stat.get("bootstrap_ci_95", {})
        tt = stat.get("paired_t_test", {})
        md.append("| Test | Statistic | p-value | Significant |\n|---|---|---|---|\n")
        md.append(f"| Diebold-Mariano | {dm.get('dm_stat', 0):.4f} | {dm.get('p_value', 1):.4f} | {dm.get('significant', False)} |\n")
        md.append(f"| Paired t-test | {tt.get('t_stat', 0):.4f} | {tt.get('p_value', 1):.4f} | {tt.get('p_value', 1) < 0.05} |\n")
        md.append(f"| Bootstrap CI (95%) | [{boot.get('lower', 0):.4f}, {boot.get('upper', 0):.4f}] | — | {boot.get('excludes_zero', False)} |\n")

    if ablation_result is not None and ablation_result.contributions is not None:
        md.append("\n\n## Ablation Study: Component Contributions\n")
        md.append(ablation_result.contributions.to_markdown(index=False, floatfmt=".4f"))

    content = "\n".join(md)
    path = output_dir / "benchmark_report_full.md"
    with open(path, "w") as f:
        f.write(content)
    return str(path)


def _write_markdown_table(bench_result, output_dir: Path) -> str:
    """Write the concise GitHub-ready markdown table artifact."""
    content = bench_result.to_markdown_table()
    path = output_dir / "benchmark_report.md"
    with open(path, "w") as f:
        f.write(content)
    # Keep the historical filename as a compatibility alias.
    with open(output_dir / "readme_table.md", "w") as f:
        f.write(content)
    return str(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_t0 = time.perf_counter()

    # ================================================================
    # STEP 1: Data
    # ================================================================
    _section("STEP 1: Prepare Data")

    # Load benchmark modules directly to avoid triggering the package __init__
    _hb = _load_module_direct(
        "factorminer.benchmark.helix_benchmark",
        _REPO_ROOT / "factorminer" / "benchmark" / "helix_benchmark.py",
    )
    _json_safe = _hb._json_safe
    HelixBenchmark = _hb.HelixBenchmark
    from factorminer.utils.config import load_config

    cfg = load_config()

    if args.mock or args.data is None:
        print(f"  Using mock data: {args.n_assets} assets x {args.n_periods} periods")
        t0 = time.perf_counter()
        from factorminer.data.mock_data import MockConfig, generate_mock_data

        raw_df = generate_mock_data(
            MockConfig(
                num_assets=args.n_assets,
                num_periods=args.n_periods,
                frequency="10min",
                universe=cfg.data.universe,
                plant_alpha=True,
                seed=args.seed,
            )
        )
        print(f"  Generated in {time.perf_counter()-t0:.1f}s")
    else:
        print(f"  Loading real data from: {args.data}")
        t0 = time.perf_counter()
        from factorminer.data.loader import load_market_data

        raw_df = load_market_data(args.data, universe=cfg.data.universe)
        print(f"  Loaded in {time.perf_counter()-t0:.1f}s")

    train_period, test_period = _derive_split_periods(raw_df)
    cfg_runtime = copy.deepcopy(cfg)
    cfg_runtime.data.train_period = train_period
    cfg_runtime.data.test_period = test_period
    cfg_runtime.mining.target_library_size = args.n_factors
    cfg_runtime.mining.max_iterations = max(20, args.n_factors * 5)
    cfg_runtime.benchmark.seed = args.seed
    cfg_runtime.evaluation.backend = "numpy"
    cfg_runtime.evaluation.num_workers = min(max(int(cfg_runtime.evaluation.num_workers), 1), 8)
    if args.mock:
        cfg_runtime.mining.ic_threshold = 0.0
        cfg_runtime.mining.icir_threshold = -1.0
        cfg_runtime.mining.correlation_threshold = 1.1

    print(f"  Shape: M={raw_df['asset_id'].nunique()}, T={raw_df.groupby('asset_id').size().min()}")
    print(f"  Train: [{train_period[0]}, {train_period[1]}]  Test: [{test_period[0]}, {test_period[1]}]")

    # ================================================================
    # STEP 2: Main Comparison Benchmark
    # ================================================================
    _section("STEP 2: Main Method Comparison")

    # HelixBenchmark already loaded via _load_module_direct above

    methods = args.methods or [
        "random_exploration",
        "alpha101_classic",
        "alpha101_adapted",
        "ralph_loop",
        "helix_phase2",
    ]
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Target library size: {args.n_factors}")
    print()

    bench = HelixBenchmark(seed=args.seed)
    t0 = time.perf_counter()
    bench_result, runtime_artifacts = bench.run_runtime_comparison(
        cfg_runtime,
        output_dir,
        raw_df=raw_df,
        mock=args.mock,
        baseline_methods=methods,
        n_target_factors=args.n_factors,
        n_runs=1,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Print method-by-method summary
    _subsection("Factor Library Metrics")
    print(bench_result.factor_library_metrics.to_string(index=False, float_format="{:.4f}".format))

    _subsection("Factor Combination Metrics")
    print(bench_result.combination_metrics.to_string(index=False, float_format="{:.4f}".format))

    _subsection("Factor Selection Metrics")
    print(bench_result.selection_metrics.to_string(index=False, float_format="{:.4f}".format))

    # ================================================================
    # STEP 3: HelixFactor vs FactorMiner Improvement Table
    # ================================================================
    _section("STEP 3: HelixFactor vs FactorMiner — Improvement Summary")
    _print_improvement_table(bench_result)

    # ================================================================
    # STEP 4: Statistical Tests
    # ================================================================
    _section("STEP 4: Statistical Significance Tests")
    if bench_result.statistical_tests:
        _print_stat_tests(bench_result.statistical_tests)
    else:
        print("  (No statistical tests available — need both helix_phase2 and ralph_loop methods)")

    # ================================================================
    # STEP 5: Speed Benchmark
    # ================================================================
    _section("STEP 5: Computational Speed Benchmark")
    print(bench_result.speed_metrics.to_string(index=False, float_format="{:.3f}".format))

    # ================================================================
    # STEP 6: Ablation Study
    # ================================================================
    ablation_result = None
    if not args.skip_ablation:
        _section("STEP 6: Ablation Study")

        if args.full_ablation:
            configs_to_run = [
                "full",
                "no_debate",
                "no_causal",
                "no_canonicalize",
                "no_regime",
                "no_capacity",
                "no_significance",
                "no_memory",
            ]
        else:
            configs_to_run = [
                "full",
                "no_debate",
                "no_regime",
                "no_capacity",
                "no_significance",
                "no_memory",
            ]

        print(f"  Configurations: {', '.join(configs_to_run)}")
        t0 = time.perf_counter()
        ablation_result = bench.run_runtime_ablation_study(
            cfg_runtime,
            output_dir,
            raw_df=raw_df,
            mock=args.mock,
            configs_to_run=configs_to_run,
            n_target_factors=args.n_factors,
            n_runs=1,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Completed in {elapsed:.1f}s")

        # Attach ablation result to bench_result
        bench_result.ablation_result = ablation_result
    else:
        _section("STEP 6: Ablation Study")
        print("  (Skipped via --skip-ablation)")

    # ================================================================
    # STEP 7: Save All Outputs
    # ================================================================
    _section("STEP 7: Save Outputs")

    # CSV tables
    bench_result.factor_library_metrics.to_csv(output_dir / "library_metrics.csv", index=False)
    bench_result.combination_metrics.to_csv(output_dir / "combination_metrics.csv", index=False)
    bench_result.selection_metrics.to_csv(output_dir / "selection_metrics.csv", index=False)
    bench_result.speed_metrics.to_csv(output_dir / "speed_metrics.csv", index=False)
    if not bench_result.turnover_metrics.empty:
        bench_result.turnover_metrics.to_csv(output_dir / "turnover_metrics.csv", index=False)
    if not bench_result.cost_pressure_metrics.empty:
        bench_result.cost_pressure_metrics.to_csv(output_dir / "cost_pressure_metrics.csv", index=False)
    runtime_topk = _runtime_topk_frame(runtime_artifacts)
    if not runtime_topk.empty:
        runtime_topk.to_csv(output_dir / "runtime_topk.csv", index=False)

    # Statistical tests JSON
    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(_json_safe(bench_result.statistical_tests), f, indent=2, allow_nan=False)

    # LaTeX table (Table 1 style)
    with open(output_dir / "latex_table.tex", "w") as f:
        f.write(bench_result.to_latex_table())

    # Markdown table
    table_path = _write_markdown_table(bench_result, output_dir)

    # HTML report
    bench_result.generate_full_report(str(output_dir / "benchmark_report.html"))

    # Comprehensive Markdown report
    md_path = _generate_markdown_report(bench_result, ablation_result, output_dir)

    # Ablation outputs
    if ablation_result is not None:
        if ablation_result.contributions is not None:
            ablation_result.contributions.to_csv(
                output_dir / "ablation_contributions.csv", index=False
            )
        with open(output_dir / "ablation_table.tex", "w") as f:
            f.write(ablation_result.contributions.to_latex(index=False) if ablation_result.contributions is not None else "% No ablation data available")

    # Bar chart comparison
    try:
        bench_result.plot_comparison(str(output_dir / "comparison_plot.png"))
        print("  comparison_plot.png saved")
    except Exception as exc:
        print(f"  (Plot skipped: {exc})")

    phase2_artifact_paths = {
        "html_report": str((output_dir / "benchmark_report.html").resolve()),
        "markdown_table": str((output_dir / "benchmark_report.md").resolve()),
        "narrative_markdown": str((output_dir / "benchmark_report_full.md").resolve()),
        "latex_table": str((output_dir / "latex_table.tex").resolve()),
        "manifest": str((output_dir / "phase2_manifest.json").resolve()),
        "statistical_tests": str((output_dir / "statistical_tests.json").resolve()),
        "library_metrics": str((output_dir / "library_metrics.csv").resolve()),
        "combination_metrics": str((output_dir / "combination_metrics.csv").resolve()),
        "selection_metrics": str((output_dir / "selection_metrics.csv").resolve()),
        "speed_metrics": str((output_dir / "speed_metrics.csv").resolve()),
    }
    if (output_dir / "turnover_metrics.csv").exists():
        phase2_artifact_paths["turnover_metrics"] = str(
            (output_dir / "turnover_metrics.csv").resolve()
        )
    if (output_dir / "cost_pressure_metrics.csv").exists():
        phase2_artifact_paths["cost_pressure_metrics"] = str(
            (output_dir / "cost_pressure_metrics.csv").resolve()
        )
    if (output_dir / "runtime_topk.csv").exists():
        phase2_artifact_paths["runtime_topk"] = str(
            (output_dir / "runtime_topk.csv").resolve()
        )
    if (output_dir / "comparison_plot.png").exists():
        phase2_artifact_paths["comparison_plot"] = str(
            (output_dir / "comparison_plot.png").resolve()
        )
    if ablation_result is not None and ablation_result.contributions is not None:
        phase2_artifact_paths["ablation_contributions"] = str(
            (output_dir / "ablation_contributions.csv").resolve()
        )
    if ablation_result is not None:
        phase2_artifact_paths["ablation_table"] = str(
            (output_dir / "ablation_table.tex").resolve()
        )

    phase2_manifest = _build_phase2_manifest(
        output_dir=output_dir.resolve(),
        methods=methods,
        seed=args.seed,
        n_factors=args.n_factors,
        mock=args.mock,
        data_path=args.data,
        full_ablation=args.full_ablation,
        skip_ablation=args.skip_ablation,
        artifact_paths=phase2_artifact_paths,
        statistical_tests=bench_result.statistical_tests,
        ablation_configs=getattr(ablation_result, "configs", None),
        runtime_manifest_root=output_dir,
    )
    with open(output_dir / "phase2_manifest.json", "w") as f:
        json.dump(_json_safe(phase2_manifest), f, indent=2, allow_nan=False)

    print(f"\n  Output files saved to: {output_dir.resolve()}")
    for fpath in sorted(output_dir.glob("*")):
        size = fpath.stat().st_size
        print(f"    {fpath.name:<40} {size:>8,} bytes")

    # ================================================================
    # SUMMARY
    # ================================================================
    _section("BENCHMARK COMPLETE")

    total_elapsed = time.perf_counter() - total_t0
    print(f"  Total runtime: {total_elapsed:.1f}s")
    print()
    print(f"  Methods benchmarked: {len(methods)}")
    print(f"  Factors per method: {args.n_factors}")
    print(f"  Runtime manifests discovered: {len(runtime_artifacts.get('runtime_payloads', {}))}")

    if ablation_result is not None:
        print(f"  Ablation configs: {len(ablation_result.configs)}")

    if bench_result.statistical_tests.get("helix_outperforms"):
        print()
        print("  *** HelixFactor OUTPERFORMS FactorMiner ***")
        dm = bench_result.statistical_tests.get("diebold_mariano", {})
        if dm.get("significant"):
            print(f"  *** DM test significant: p={dm.get('p_value', 1):.4f} ***")
    print()
    print(f"  Full report: {output_dir.resolve() / 'benchmark_report.html'}")
    print(f"  Markdown table: {table_path}")
    print(f"  Narrative markdown: {md_path}")
    print()


if __name__ == "__main__":
    main()
