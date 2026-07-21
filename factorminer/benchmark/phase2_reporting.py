"""Phase-2 benchmark console and publication-report rendering."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
                "metric_version": payload.get("metric_version"),
                "dataset_hashes": payload.get("dataset_hashes", {}),
                "runtime_contract": payload.get("runtime_contract", {}),
                "walk_forward_contract": payload.get("walk_forward_contract", {}),
                "stress_contract": payload.get("stress_contract", {}),
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
    runtime_refs = _collect_runtime_manifest_refs(runtime_manifest_root or output_dir)
    return {
        "benchmark_name": "phase2",
        "output_dir": str(output_dir),
        "generated_at": datetime.now(UTC).isoformat(),
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
    print(f"  {'-' * 28} {'-' * 12} {'-' * 12} {'-' * 12}")
    metrics = [
        ("Library IC (%)", r_ic, h_ic),
        ("Library ICIR", r_icir, h_icir),
        ("EW Combo IC (%)", r_ew, h_ew),
        ("ICW Combo IC (%)", r_icw, h_icw),
        ("LASSO Sel IC (%)", r_las, h_las),
        ("XGBoost Sel IC (%)", r_xgb, h_xgb),
    ]
    for name, r_val, h_val in metrics:
        print(f"  {name:<28} {r_val:>12.4f} {h_val:>12.4f} {_delta(h_val, r_val):>12}")


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
    paired_runs = stat_tests.get("paired_tests_by_run", [])
    if paired_runs:
        print(f"  Paired tests by seed ({len(paired_runs)} runs):")
        for entry in paired_runs:
            tests = entry.get("tests", {})
            dm = tests.get("diebold_mariano", {})
            boot = tests.get("bootstrap_ci_95", {})
            print(
                f"    run {entry.get('run_id', '?')} / seed {entry.get('seed', '?')}: "
                f"mean_diff={_fmt_stat(tests.get('mean_ic_difference'), '+.4f')}, "
                f"DM p={_fmt_stat(dm.get('p_value'))}, "
                f"bootstrap=[{_fmt_stat(boot.get('lower'))}, "
                f"{_fmt_stat(boot.get('upper'))}], "
                f"helix_outperforms={tests.get('helix_outperforms', '?')}"
            )
        return

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
    print(
        f"    [{_fmt_stat(lo)}, {_fmt_stat(hi)}]  "
        f"{'(excludes zero **)' if boot.get('excludes_zero') else ''}"
    )
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
    md.append(
        f"**Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )

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
        md.append(bench_result.cost_pressure_metrics.to_markdown(index=False, floatfmt=".4f"))

    runtime_topk = _runtime_topk_markdown(getattr(bench_result, "runtime_artifacts", {}))
    if runtime_topk:
        md.append("\n\n## Runtime Top-K\n")
        md.append(runtime_topk)

    # Statistical tests
    stat = bench_result.statistical_tests
    if stat:
        md.append("\n\n## Statistical Tests (HelixFactor vs FactorMiner)\n")
        paired_runs = stat.get("paired_tests_by_run", [])
        if paired_runs:
            md.append(
                "| Run | Seed | Mean IC diff | DM stat | DM p-value | "
                "Paired t p-value | Bootstrap 95% CI | Helix outperforms |\n"
                "|---:|---:|---:|---:|---:|---:|---|---|\n"
            )
            for entry in paired_runs:
                tests = entry.get("tests", {})
                dm = tests.get("diebold_mariano", {})
                boot = tests.get("bootstrap_ci_95", {})
                tt = tests.get("paired_t_test", {})
                md.append(
                    f"| {entry.get('run_id', '?')} | {entry.get('seed', '?')} | "
                    f"{_fmt_stat(tests.get('mean_ic_difference'))} | "
                    f"{_fmt_stat(dm.get('dm_stat'))} | "
                    f"{_fmt_stat(dm.get('p_value'))} | "
                    f"{_fmt_stat(tt.get('p_value'))} | "
                    f"[{_fmt_stat(boot.get('lower'))}, {_fmt_stat(boot.get('upper'))}] | "
                    f"{tests.get('helix_outperforms', '?')} |\n"
                )
        else:
            dm = stat.get("diebold_mariano", {})
            boot = stat.get("bootstrap_ci_95", {})
            tt = stat.get("paired_t_test", {})
            md.append("| Test | Statistic | p-value | Significant |\n|---|---|---|---|\n")
            md.append(
                f"| Diebold-Mariano | {dm.get('dm_stat', 0):.4f} | "
                f"{dm.get('p_value', 1):.4f} | {dm.get('significant', False)} |\n"
            )
            md.append(
                f"| Paired t-test | {tt.get('t_stat', 0):.4f} | "
                f"{tt.get('p_value', 1):.4f} | {tt.get('p_value', 1) < 0.05} |\n"
            )
            md.append(
                f"| Bootstrap CI (95%) | [{boot.get('lower', 0):.4f}, "
                f"{boot.get('upper', 0):.4f}] | — | "
                f"{boot.get('excludes_zero', False)} |\n"
            )

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
