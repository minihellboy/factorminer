#!/usr/bin/env python3
"""HelixFactor Phase 2 Comprehensive Benchmark Runner

Generates a complete publication-quality benchmarking report comparing
HelixFactor (Phase 2) against FactorMiner (Ralph Loop) and all baselines.

Usage:
    python scripts/run_phase2_benchmark.py --mock                  # quick mock data run
    python scripts/run_phase2_benchmark.py --mock --n-factors 40   # custom factor count
    python scripts/run_phase2_benchmark.py --mock --full-ablation  # include all ablations
    python scripts/run_phase2_benchmark.py --data path/to/data.csv # real data

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
    industry_evidence_summary.csv — Tier-0 IC/HAC/turnover/DSR/gate summary
    runtime_topk.csv           — runtime top-k summary
    comparison_plot.png        — bar chart comparison figure
    ablation_contributions.csv — component contribution summary
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
import warnings
from pathlib import Path

from factorminer.benchmark.phase2_reporting import (
    _build_phase2_manifest,
    _derive_split_periods,
    _generate_markdown_report,
    _industry_evidence_frame,
    _json_safe,
    _print_improvement_table,
    _print_stat_tests,
    _runtime_topk_frame,
    _section,
    _subsection,
    _write_markdown_table,
)
from factorminer.benchmark.runtime import (
    run_phase2_ablation_study,
    run_phase2_comparison,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HelixFactor Phase 2 Comprehensive Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic mock data (no API keys needed)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to real market data CSV",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=40,
        help="Target library size per method",
    )
    parser.add_argument(
        "--n-assets",
        type=int,
        default=100,
        help="Number of assets in mock data",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=600,
        help="Number of time periods in mock data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/phase2_benchmark",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--runs",
        type=_positive_int,
        default=1,
        help="Independent runs per method/variant with derived seeds "
        "(seed + run_id); frames report across-seed means",
    )
    parser.add_argument(
        "--evidence-tier",
        type=str,
        default=None,
        choices=["simulated", "public_reproducible", "private_partner_observed", "unverified"],
        help="Evidence tier for the published receipt. Defaults to 'simulated' when "
        "--mock is set, else 'unverified' (must be raised explicitly for a real claim).",
    )
    parser.add_argument(
        "--data-license-class",
        type=str,
        default=None,
        choices=[
            "proprietary_licensed",
            "public_domain",
            "publicly_retrievable",
            "redistributable_with_attribution",
            "synthetic",
            "unknown",
            "vendor_redistributable_sample",
        ],
        help="License class for the exact benchmark input.",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=str,
        default=None,
        help="Verified dataset_manifest.json produced by `factorminer public-data prepare`.",
    )
    parser.add_argument(
        "--commitment-key-file",
        type=str,
        default=None,
        help="File containing the withheld hex HMAC key required for private evidence.",
    )
    parser.add_argument(
        "--supersedes-release-id",
        type=str,
        default=None,
        help="release_id this run's receipt supersedes, if any.",
    )
    parser.add_argument(
        "--portable-release",
        action="store_true",
        help="Copy declared artifacts into a relocatable content-addressed release bundle.",
    )
    parser.add_argument(
        "--bundle-public-data",
        action="store_true",
        help="Include the committed input inside a portable public release. Requires an explicitly redistributable public license class.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Methods to benchmark (default: all 5)",
    )
    parser.add_argument(
        "--full-ablation",
        action="store_true",
        help="Run full ablation study (slower)",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip ablation study entirely",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="Logging level",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    actual_mock = bool(args.mock or args.data is None)
    evidence_tier_value = args.evidence_tier or ("simulated" if actual_mock else "unverified")
    data_license_class = args.data_license_class or ("synthetic" if actual_mock else "unknown")

    prepared_dataset_manifest = None
    if args.dataset_manifest:
        from factorminer.data.public_archive import verify_prepared_public_dataset

        dataset_manifest_path = Path(args.dataset_manifest).resolve()
        canonical_manifest_path = dataset_manifest_path.parent / "dataset_manifest.json"
        if dataset_manifest_path != canonical_manifest_path:
            raise ValueError("--dataset-manifest must name dataset_manifest.json")
        prepared = verify_prepared_public_dataset(dataset_manifest_path.parent)
        if not prepared.passed:
            raise ValueError(
                "prepared public dataset verification failed: " + "; ".join(prepared.mismatches)
            )
        prepared_dataset_manifest = json.loads(dataset_manifest_path.read_text())
        expected_data = (
            dataset_manifest_path.parent / prepared_dataset_manifest["data_path"]
        ).resolve()
        if args.data is None or Path(args.data).resolve() != expected_data:
            raise ValueError("--data must reference the file bound by --dataset-manifest")
        manifest_license = str(
            prepared_dataset_manifest.get("license", {}).get("data_license_class", "unknown")
        )
        if args.data_license_class and args.data_license_class != manifest_license:
            raise ValueError("--data-license-class conflicts with the verified dataset manifest")
        data_license_class = manifest_license

    if evidence_tier_value == "public_reproducible":
        if not args.portable_release:
            raise ValueError("public_reproducible evidence requires --portable-release")
        if prepared_dataset_manifest is None:
            raise ValueError(
                "public_reproducible evidence requires a verified --dataset-manifest"
            )

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

    from factorminer.utils.config import load_config

    cfg = load_config()
    if prepared_dataset_manifest is not None:
        dataset_identity = str(prepared_dataset_manifest["dataset_id"])
        dataset_asset_class = str(prepared_dataset_manifest["asset_class"])
        cfg.data.market = dataset_asset_class
        cfg.data.asset_class = dataset_asset_class
        cfg.data.universe = dataset_identity
        cfg.data.frequency = str(prepared_dataset_manifest["frequency"])
        cfg.data.periods_per_year = float(prepared_dataset_manifest["periods_per_year"])
        cfg.data.targets = [dict(prepared_dataset_manifest["target"])]
        cfg.data.default_target = str(prepared_dataset_manifest["target"]["name"])
        cfg.phase2.capacity.enabled = bool(prepared_dataset_manifest["liquidity_evidence"])
        cfg.benchmark.freeze_universe = dataset_identity
        cfg.benchmark.report_universes = [dataset_identity]

    if actual_mock:
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
        print(f"  Generated in {time.perf_counter() - t0:.1f}s")
    else:
        print(f"  Loading real data from: {args.data}")
        t0 = time.perf_counter()
        from factorminer.data.loader import load_market_data

        raw_df = load_market_data(
            args.data,
            universe=None if prepared_dataset_manifest is not None else cfg.data.universe,
        )
        print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    train_period, validation_period, test_period = _derive_split_periods(raw_df)
    cfg_runtime = copy.deepcopy(cfg)
    cfg_runtime.data.train_period = train_period
    cfg_runtime.data.validation_period = validation_period
    cfg_runtime.data.test_period = test_period
    cfg_runtime.data.purge_bars = 1
    cfg_runtime.data.embargo_bars = 0
    cfg_runtime.mining.target_library_size = args.n_factors
    cfg_runtime.mining.max_iterations = max(20, args.n_factors * 5)
    cfg_runtime.benchmark.seed = args.seed
    cfg_runtime.evaluation.backend = "numpy"
    cfg_runtime.evaluation.num_workers = min(max(int(cfg_runtime.evaluation.num_workers), 1), 8)
    if actual_mock:
        cfg_runtime.mining.ic_threshold = 0.0
        cfg_runtime.mining.icir_threshold = -1.0
        cfg_runtime.mining.correlation_threshold = 1.1

    print(f"  Shape: M={raw_df['asset_id'].nunique()}, T={raw_df.groupby('asset_id').size().min()}")
    print(
        f"  Train: [{train_period[0]}, {train_period[1]}]  "
        f"Validation: [{validation_period[0]}, {validation_period[1]}]  "
        f"Test: [{test_period[0]}, {test_period[1]}]  (purge=1, embargo=0 bars)"
    )

    # ================================================================
    # STEP 2: Main Comparison Benchmark
    # ================================================================
    _section("STEP 2: Main Method Comparison")

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

    t0 = time.perf_counter()
    bench_result, runtime_artifacts = run_phase2_comparison(
        cfg_runtime,
        output_dir,
        raw_df=raw_df,
        mock=actual_mock,
        baseline_methods=methods,
        n_target_factors=args.n_factors,
        n_runs=args.runs,
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
    if {"helix_phase2", "ralph_loop"}.issubset(methods):
        _section("STEP 3: HelixFactor vs FactorMiner — Improvement Summary")
        _print_improvement_table(bench_result)
    else:
        _section("STEP 3: Method Scope")
        print(
            "  Deterministic formula baselines only; no Helix-vs-Ralph improvement "
            "claim is computed."
        )

    # ================================================================
    # STEP 4: Statistical Tests
    # ================================================================
    _section("STEP 4: Statistical Significance Tests")
    has_paired_method_test = bool(
        bench_result.statistical_tests.get("paired_tests_by_run")
        or "diebold_mariano" in bench_result.statistical_tests
    )
    if has_paired_method_test:
        _print_stat_tests(bench_result.statistical_tests)
    else:
        print(
            "  (No Helix-vs-Ralph paired test: both methods are required. "
            "Across-seed dispersion remains recorded for every selected method.)"
        )

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
        ablation_result = run_phase2_ablation_study(
            cfg_runtime,
            output_dir,
            raw_df=raw_df,
            mock=actual_mock,
            configs_to_run=configs_to_run,
            n_target_factors=args.n_factors,
            n_runs=args.runs,
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
        bench_result.cost_pressure_metrics.to_csv(
            output_dir / "cost_pressure_metrics.csv", index=False
        )
    runtime_topk = _runtime_topk_frame(runtime_artifacts)
    if not runtime_topk.empty:
        runtime_topk.to_csv(output_dir / "runtime_topk.csv", index=False)
    industry_evidence = _industry_evidence_frame(runtime_artifacts)
    if not industry_evidence.empty:
        industry_evidence.to_csv(output_dir / "industry_evidence_summary.csv", index=False)

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
            f.write(
                ablation_result.contributions.to_latex(index=False)
                if ablation_result.contributions is not None
                else "% No ablation data available"
            )

    # Bar chart comparison
    try:
        bench_result.plot_comparison(str(output_dir / "comparison_plot.png"))
        print("  comparison_plot.png saved")
    except Exception as exc:
        print(f"  (Plot skipped: {exc})")

    effective_config_path = output_dir / "effective_config.json"
    with open(effective_config_path, "w") as file:
        json.dump(_json_safe(cfg_runtime.to_dict()), file, indent=2, allow_nan=False)

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
        "effective_config": str(effective_config_path.resolve()),
    }
    if prepared_dataset_manifest is not None:
        phase2_artifact_paths["dataset_manifest"] = str(dataset_manifest_path)
        phase2_artifact_paths["dataset_lock"] = str(
            (dataset_manifest_path.parent / str(prepared_dataset_manifest["lock_path"])).resolve()
        )
    if (output_dir / "turnover_metrics.csv").exists():
        phase2_artifact_paths["turnover_metrics"] = str(
            (output_dir / "turnover_metrics.csv").resolve()
        )
    if (output_dir / "cost_pressure_metrics.csv").exists():
        phase2_artifact_paths["cost_pressure_metrics"] = str(
            (output_dir / "cost_pressure_metrics.csv").resolve()
        )
    if (output_dir / "runtime_topk.csv").exists():
        phase2_artifact_paths["runtime_topk"] = str((output_dir / "runtime_topk.csv").resolve())
    if (output_dir / "industry_evidence_summary.csv").exists():
        phase2_artifact_paths["industry_evidence_summary"] = str(
            (output_dir / "industry_evidence_summary.csv").resolve()
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
        phase2_artifact_paths["ablation_table"] = str((output_dir / "ablation_table.tex").resolve())

    phase2_manifest = _build_phase2_manifest(
        output_dir=output_dir.resolve(),
        methods=methods,
        seed=args.seed,
        n_factors=args.n_factors,
        mock=actual_mock,
        data_path=Path(args.data).name if args.data else None,
        full_ablation=args.full_ablation,
        skip_ablation=args.skip_ablation,
        artifact_paths=phase2_artifact_paths,
        statistical_tests=bench_result.statistical_tests,
        ablation_configs=getattr(ablation_result, "configs", None),
        runtime_manifest_root=output_dir,
    )
    with open(output_dir / "phase2_manifest.json", "w") as f:
        json.dump(_json_safe(phase2_manifest), f, indent=2, allow_nan=False)

    from factorminer.architecture.memory_policy import build_memory_policy
    from factorminer.architecture.paper_protocol import PaperProtocol
    from factorminer.architecture.research_receipt import EvidenceTier, RunStatus
    from factorminer.benchmark.receipt import (
        build_research_receipt,
        publish_portable_bundle,
        write_receipt,
    )
    from factorminer.evaluation.metrics import METRIC_VERSION

    protocol = PaperProtocol.from_config(cfg_runtime)
    memory_policy = build_memory_policy(cfg_runtime, protocol)
    runtime_refs = list(phase2_manifest.get("runtime_manifest_refs", []))
    walk_forward_contract = {
        str(ref.get("baseline") or index): dict(ref.get("walk_forward_contract") or {})
        for index, ref in enumerate(runtime_refs)
    }
    stress_contract = {
        str(ref.get("baseline") or index): dict(ref.get("stress_contract") or {})
        for index, ref in enumerate(runtime_refs)
    }
    if prepared_dataset_manifest is not None:
        dataset_descriptor = {
            "kind": "checksum_locked_public_archive",
            "identity": prepared_dataset_manifest["dataset_id"],
            "format": Path(args.data).suffix.lower().lstrip("."),
            "data_sha256": prepared_dataset_manifest["data_sha256"],
            "source_lock_sha256": prepared_dataset_manifest["lock_sha256"],
            "source_archives": prepared_dataset_manifest["source_archives"],
            "provider": prepared_dataset_manifest["provider"],
            "license": prepared_dataset_manifest["license"],
            "availability_lag": prepared_dataset_manifest["availability_lag"],
            "point_in_time_limitations": prepared_dataset_manifest["point_in_time_limitations"],
            "train_period": train_period,
            "validation_period": validation_period,
            "test_period": test_period,
            "purge_bars": 1,
            "embargo_bars": 0,
            "asset_class": cfg_runtime.data.asset_class,
            "universe": prepared_dataset_manifest["universe"],
            "frequency": prepared_dataset_manifest["frequency"],
        }
    else:
        dataset_descriptor = {
            "kind": "synthetic" if actual_mock else "file",
            "identity": "factorminer-mock" if actual_mock else Path(args.data).name,
            "format": "generated" if actual_mock else Path(args.data).suffix.lower().lstrip("."),
            "train_period": train_period,
            "validation_period": validation_period,
            "test_period": test_period,
            "purge_bars": 1,
            "embargo_bars": 0,
            "asset_class": cfg_runtime.data.asset_class,
            "universe": cfg_runtime.data.universe,
            "frequency": cfg_runtime.data.frequency,
        }
    commitment_key = None
    if evidence_tier_value == "private_partner_observed":
        if not args.commitment_key_file:
            raise ValueError(
                "--commitment-key-file is required for private_partner_observed evidence"
            )
        commitment_key = Path(args.commitment_key_file).read_text().strip()
    receipt = build_research_receipt(
        phase2_manifest=phase2_manifest,
        phase2_manifest_path=(output_dir / "phase2_manifest.json").resolve(),
        evidence_tier=EvidenceTier(evidence_tier_value),
        run_status=(
            RunStatus.PARTIAL
            if (args.skip_ablation or ablation_result is None)
            else RunStatus.COMPLETED
        ),
        seed=args.seed,
        config_path=effective_config_path,
        data_path=Path(args.data).resolve() if args.data else None,
        dataset_descriptor=dataset_descriptor,
        protocol_admission_contract=protocol.admission_contract(),
        protocol_replacement_contract=protocol.replacement_contract(),
        memory_policy_schema=memory_policy.schema() if memory_policy is not None else None,
        ic_metric=protocol.ic_metric,
        metric_version=METRIC_VERSION,
        walk_forward_contract=walk_forward_contract,
        stress_contract=stress_contract,
        data_license_class=data_license_class,
        commitment_key=commitment_key,
        supersedes_release_id=args.supersedes_release_id,
    )
    if args.portable_release:
        receipt_path = publish_portable_bundle(
            receipt,
            phase2_manifest=phase2_manifest,
            releases_root=output_dir / "releases",
            commitment_input=Path(args.data).resolve() if args.data else None,
            include_commitment_input=args.bundle_public_data,
        )
    else:
        if args.bundle_public_data:
            raise ValueError("--bundle-public-data requires --portable-release")
        receipt_path = write_receipt(receipt, releases_root=output_dir / "releases")
    published_release_id = receipt_path.parent.name
    print(f"  Receipt: {receipt_path} (release_id={published_release_id})")

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

    paired_tests = bench_result.statistical_tests.get("paired_tests_by_run", [])
    if paired_tests:
        outperform_count = sum(
            bool(entry.get("tests", {}).get("helix_outperforms"))
            for entry in paired_tests
        )
        significant_count = sum(
            bool(
                entry.get("tests", {})
                .get("diebold_mariano", {})
                .get("significant")
            )
            for entry in paired_tests
        )
        print()
        print(
            f"  HelixFactor outperforms in {outperform_count}/{len(paired_tests)} "
            "seed runs"
        )
        print(
            f"  DM test significant in {significant_count}/{len(paired_tests)} seed runs"
        )
    elif bench_result.statistical_tests.get("helix_outperforms"):
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
