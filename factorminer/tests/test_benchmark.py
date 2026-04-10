"""Benchmark-runtime and CLI coverage."""

from __future__ import annotations

import json
from types import SimpleNamespace

from click.testing import CliRunner
import numpy as np

from factorminer.benchmark.runtime import (
    build_benchmark_library,
    run_ablation_strategy_benchmark,
    run_table1_benchmark,
    select_frozen_top_k,
)
from factorminer.benchmark.helix_benchmark import StatisticalComparisonTests, _json_safe
from factorminer.cli import main
from factorminer.core.factor_library import Factor, FactorLibrary
from factorminer.core.library_io import save_library
from factorminer.core.session import MiningSession
from factorminer.evaluation.runtime import FactorEvaluationArtifact
from factorminer.utils.config import load_config
from run_phase2_benchmark import (
    _build_phase2_manifest,
    _collect_runtime_manifest_refs,
    _generate_markdown_report,
    _write_markdown_table,
)


def _artifact(
    factor_id: int,
    formula: str,
    train_ic: float,
    train_icir: float,
    signal_scale: float,
) -> FactorEvaluationArtifact:
    signal = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [0.5, 0.3, 0.1],
        ],
        dtype=np.float64,
    ) * signal_scale
    return FactorEvaluationArtifact(
        factor_id=factor_id,
        name=f"factor_{factor_id}",
        formula=formula,
        category="test",
        parse_ok=True,
        signals_full=signal,
        split_signals={"train": signal, "test": signal, "full": signal},
        split_stats={
            "train": {
                "ic_mean": train_ic,
                "ic_abs_mean": abs(train_ic),
                "icir": train_icir,
                "ic_win_rate": 0.6,
            },
            "test": {
                "ic_mean": train_ic / 2.0,
                "ic_abs_mean": abs(train_ic / 2.0),
                "icir": train_icir / 2.0,
                "ic_win_rate": 0.5,
            },
            "full": {
                "ic_mean": train_ic,
                "ic_abs_mean": abs(train_ic),
                "icir": train_icir,
                "ic_win_rate": 0.6,
            },
        },
    )


def test_select_frozen_top_k_prefers_thresholded_admitted_then_fills():
    cfg = load_config()
    artifacts = [
        _artifact(1, "Neg($close)", 0.07, 0.8, 1.0),
        _artifact(2, "Neg($open)", 0.06, 0.7, 0.7),
        _artifact(3, "Neg($high)", 0.049, 0.9, 0.2),
    ]
    library, _ = build_benchmark_library(artifacts, cfg, split_name="train")

    frozen = select_frozen_top_k(
        artifacts,
        library,
        top_k=3,
        split_name="train",
        min_ic=0.05,
        min_icir=0.5,
    )

    assert [artifact.formula for artifact in frozen[:2]] == ["Neg($close)", "Neg($open)"]
    assert frozen[2].formula == "Neg($high)"


def test_build_benchmark_library_rejects_low_ic_candidates():
    cfg = load_config()
    artifacts = [
        _artifact(1, "Neg($close)", 0.07, 0.8, 1.0),
        _artifact(2, "Neg($open)", 0.01, 0.6, 0.9),
    ]

    library, stats = build_benchmark_library(artifacts, cfg, split_name="train")

    assert library.size == 1
    assert stats["threshold_rejections"] == 1
    assert stats["admitted"] == 1


def test_benchmark_table1_cli_invokes_runtime(monkeypatch, tmp_path):
    captured = {}

    def _fake_run(*args, **kwargs):
        captured["called"] = True
        return {
            "factor_miner": {
                "freeze_library_size": 12,
                "frozen_top_k": [{"name": "f1"}],
                "universes": {
                    "CSI500": {
                        "library": {"ic": 0.08, "icir": 0.9, "avg_abs_rho": 0.2}
                    }
                },
            }
        }

    monkeypatch.setattr("factorminer.benchmark.runtime.run_table1_benchmark", _fake_run)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--cpu",
            "--output-dir",
            str(tmp_path / "out"),
            "benchmark",
            "table1",
            "--mock",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured.get("called") is True
    assert "Benchmark Table 1" in result.output
    assert "Baseline: factor_miner" in result.output
    assert "CSI500: library IC=0.0800" in result.output


def test_table1_manifest_includes_saved_library_provenance(monkeypatch, tmp_path):
    saved_root = tmp_path / "saved"
    library_base = saved_root / "factor_miner_library"

    library = FactorLibrary()
    factor = Factor(
        id=0,
        name="saved_factor",
        formula="Neg($close)",
        category="test",
        ic_mean=0.07,
        icir=0.8,
        ic_win_rate=0.6,
        max_correlation=0.1,
        batch_number=1,
        signals=np.array(
            [
                [1.0, 2.0, 3.0],
                [0.5, 0.4, 0.3],
                [0.2, 0.3, 0.4],
            ],
            dtype=np.float64,
        ),
    )
    library.admit_factor(factor)
    save_library(library, library_base)

    session = MiningSession(
        session_id="session-001",
        output_dir=str(saved_root),
        library_path=str(library_base),
    )
    session.record_iteration({"candidates": 3, "admitted": 1, "replaced": 0, "library_size": 1})
    session.record_iteration({"candidates": 2, "admitted": 1, "replaced": 0, "library_size": 1})
    session.finalize()
    session.save(saved_root / "session.json")
    with open(saved_root / "session_log.json", "w") as fp:
        json.dump({"summary": session.get_summary(), "iterations": session.iterations}, fp)

    cfg = load_config()
    output_dir = tmp_path / "results"
    artifact = _artifact(1, "Neg($close)", 0.07, 0.8, 1.0)

    monkeypatch.setattr(
        "factorminer.benchmark.runtime.load_benchmark_dataset",
        lambda *args, **kwargs: (SimpleNamespace(), "freeze-hash"),
    )
    monkeypatch.setattr(
        "factorminer.benchmark.runtime.evaluate_factors",
        lambda *args, **kwargs: [artifact],
    )
    monkeypatch.setattr(
        "factorminer.benchmark.runtime.evaluate_frozen_set",
        lambda frozen, dataset, **kwargs: {
            "factor_count": len(frozen),
            "library": {"ic": 0.1, "icir": 1.0, "avg_abs_rho": 0.2},
            "combinations": {},
            "selections": {},
        },
    )

    run_table1_benchmark(
        cfg,
        output_dir,
        baseline_names=["factor_miner"],
        factor_miner_library_path=str(library_base),
    )

    result_path = output_dir / "benchmark" / "table1" / "factor_miner.json"
    manifest_path = output_dir / "benchmark" / "table1" / "factor_miner_manifest.json"
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    provenance = manifest["baseline_provenance"]["factor_miner"]
    assert provenance["kind"] == "saved_library"
    assert provenance["library_summary"]["factor_count"] == 1
    assert provenance["session_summary"]["total_iterations"] == 2
    assert provenance["source_files"]["library_json"]["path"].endswith("factor_miner_library.json")
    assert provenance["source_files"]["signal_cache"]["path"].endswith("factor_miner_library_signals.npz")
    assert manifest["artifact_paths"]["result"] == str(result_path)
    assert manifest["artifact_paths"]["manifest"] == str(manifest_path)
    assert result["provenance"]["kind"] == "saved_library"


def test_phase2_manifest_references_runtime_manifest_and_sanitizes_stats(tmp_path):
    runtime_root = tmp_path / "runtime"
    manifest_path = runtime_root / "benchmark" / "table1" / "factor_miner_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "benchmark_name": "table1",
                "baseline": "factor_miner",
                "mode": "paper",
                "artifact_paths": {"result": "result.json", "manifest": str(manifest_path)},
                "baseline_provenance": {
                    "factor_miner": {
                        "kind": "saved_library",
                        "source": "factor_miner",
                    }
                },
            }
        )
    )

    refs = _collect_runtime_manifest_refs(runtime_root)
    assert len(refs) == 1
    assert refs[0]["path"] == str(manifest_path)
    assert refs[0]["baseline_provenance"]["factor_miner"]["kind"] == "saved_library"

    phase2_manifest = _build_phase2_manifest(
        output_dir=tmp_path / "phase2",
        methods=["ralph_loop", "helix_phase2"],
        seed=7,
        n_factors=40,
        mock=True,
        data_path=None,
        full_ablation=False,
        skip_ablation=True,
        artifact_paths={"html_report": str(tmp_path / "phase2" / "benchmark_report.html")},
        statistical_tests={
            "diebold_mariano": {"dm_stat": np.nan, "p_value": np.inf},
            "bootstrap_ci_95": {"lower": -np.inf, "upper": np.nan},
        },
        ablation_configs=["full"],
        runtime_manifest_root=runtime_root,
    )

    assert phase2_manifest["runtime_manifest_refs"][0]["path"] == str(manifest_path)
    assert phase2_manifest["statistical_tests"]["diebold_mariano"]["dm_stat"] is None
    assert phase2_manifest["statistical_tests"]["diebold_mariano"]["p_value"] is None
    assert phase2_manifest["statistical_tests"]["bootstrap_ci_95"]["lower"] is None
    dumped = json.dumps(_json_safe(phase2_manifest), allow_nan=False)
    assert "NaN" not in dumped


def test_diebold_mariano_handles_identical_series_without_nan_direction():
    tests = StatisticalComparisonTests(seed=42)
    series = np.array([0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float64)

    result = tests.diebold_mariano_test(series, series.copy())

    assert result.direction == "no_difference"
    assert result.p_value == 1.0
    assert np.isfinite(result.dm_statistic)


def test_json_safe_removes_non_finite_values():
    payload = {
        "finite": 1.5,
        "nan": float("nan"),
        "nested": [np.float64(np.inf), {"value": -np.inf}],
    }

    cleaned = _json_safe(payload)

    assert cleaned == {"finite": 1.5, "nan": None, "nested": [None, {"value": None}]}
    dumped = json.dumps(cleaned, allow_nan=False)
    assert "NaN" not in dumped


def test_markdown_artifacts_use_expected_paths(tmp_path):
    table_stub = SimpleNamespace(
        to_markdown=lambda **kwargs: "| a | b |\n|---|---|\n| 1 | 2 |\n"
    )
    bench_result = SimpleNamespace(
        factor_library_metrics=table_stub,
        combination_metrics=table_stub,
        selection_metrics=table_stub,
        speed_metrics=table_stub,
        statistical_tests={"diebold_mariano": {"dm_stat": 0.0, "p_value": 1.0}},
        to_markdown_table=lambda: "| a | b |\n|---|---|\n| 1 | 2 |\n",
    )

    table_path = _write_markdown_table(bench_result, tmp_path)
    report_path = _generate_markdown_report(bench_result, None, tmp_path)

    assert table_path.endswith("benchmark_report.md")
    assert report_path.endswith("benchmark_report_full.md")
    assert (tmp_path / "benchmark_report.md").exists()
    assert (tmp_path / "benchmark_report_full.md").exists()


def _runtime_dataset_stub():
    data_tensor = np.ones((2, 6, 8), dtype=np.float64)
    returns = np.array(
        [
            [0.01, 0.02, 0.01, 0.03, 0.02, 0.01],
            [0.02, 0.01, 0.03, 0.02, 0.01, 0.02],
        ],
        dtype=np.float64,
    )
    splits = {
        "train": SimpleNamespace(
            indices=np.array([0, 1, 2]),
            returns=returns[:, :3],
            timestamps=np.arange(3),
        ),
        "test": SimpleNamespace(
            indices=np.array([3, 4, 5]),
            returns=returns[:, 3:],
            timestamps=np.arange(3, 6),
        ),
        "full": SimpleNamespace(
            indices=np.arange(6),
            returns=returns,
            timestamps=np.arange(6),
        ),
    }

    return SimpleNamespace(
        data_tensor=data_tensor,
        returns=returns,
        data_dict={
            "$open": data_tensor[:, :, 0],
            "$high": data_tensor[:, :, 1],
            "$low": data_tensor[:, :, 2],
            "$close": data_tensor[:, :, 3],
            "$volume": data_tensor[:, :, 4],
            "$amt": data_tensor[:, :, 5],
            "$vwap": data_tensor[:, :, 6],
            "$returns": data_tensor[:, :, 7],
        },
        target_panels={"paper": returns},
        target_specs={"paper": SimpleNamespace(holding_bars=1)},
        get_split=lambda name: splits[name],
    )


def _single_factor_library():
    library = FactorLibrary()
    library.admit_factor(
        Factor(
            id=0,
            name="runtime_factor",
            formula="Neg($close)",
            category="test",
            ic_mean=0.08,
            icir=0.9,
            ic_win_rate=0.6,
            max_correlation=0.0,
            batch_number=1,
            signals=np.ones((2, 3), dtype=np.float64),
        )
    )
    return library


def test_table1_runtime_methods_instantiate_live_loops(monkeypatch, tmp_path):
    cfg = load_config()
    calls = []

    monkeypatch.setattr(
        "factorminer.benchmark.runtime.load_benchmark_dataset",
        lambda *args, **kwargs: (_runtime_dataset_stub(), "dataset-hash"),
    )
    monkeypatch.setattr(
        "factorminer.benchmark.runtime._get_baseline_entries",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("catalog fallback used")),
    )
    monkeypatch.setattr(
        "factorminer.benchmark.runtime.evaluate_factors",
        lambda *args, **kwargs: [_artifact(1, "Neg($close)", 0.08, 0.9, 1.0)],
    )
    monkeypatch.setattr(
        "factorminer.benchmark.runtime.evaluate_frozen_set",
        lambda frozen, dataset, **kwargs: {
            "factor_count": len(frozen),
            "library": {"ic": 0.1, "icir": 1.0, "avg_abs_rho": 0.2},
            "combinations": {
                "equal_weight": {"ic": 0.12, "icir": 1.1, "turnover": 0.3},
                "ic_weighted": {"ic": 0.13, "icir": 1.2, "turnover": 0.25},
            },
            "selections": {"lasso": {"ic": 0.09, "icir": 0.8}},
        },
    )

    def _fake_ralph_run(self, *args, **kwargs):
        calls.append("ralph")
        return _single_factor_library()

    def _fake_helix_run(self, *args, **kwargs):
        calls.append("helix")
        return _single_factor_library()

    monkeypatch.setattr("factorminer.core.ralph_loop.RalphLoop.run", _fake_ralph_run)
    monkeypatch.setattr("factorminer.core.helix_loop.HelixLoop.run", _fake_helix_run)

    payload = run_table1_benchmark(
        cfg,
        tmp_path,
        mock=True,
        baseline_names=["ralph_loop", "helix_phase2"],
        use_runtime_loops=True,
    )

    assert calls == ["ralph", "helix"]
    assert payload["ralph_loop"]["provenance"]["kind"] == "runtime_loop"
    assert payload["helix_phase2"]["provenance"]["kind"] == "runtime_loop"


def test_table1_runtime_methods_fail_loudly_without_catalog_fallback(monkeypatch, tmp_path):
    cfg = load_config()
    fallback_called = {"value": False}

    monkeypatch.setattr(
        "factorminer.benchmark.runtime.load_benchmark_dataset",
        lambda *args, **kwargs: (_runtime_dataset_stub(), "dataset-hash"),
    )

    def _forbidden_catalog(*args, **kwargs):
        fallback_called["value"] = True
        raise AssertionError("catalog fallback used")

    monkeypatch.setattr("factorminer.benchmark.runtime._get_baseline_entries", _forbidden_catalog)
    monkeypatch.setattr(
        "factorminer.benchmark.runtime._run_runtime_mining_loop",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("runtime loop failed")),
    )

    try:
        run_table1_benchmark(
            cfg,
            tmp_path,
            mock=True,
            baseline_names=["ralph_loop"],
            use_runtime_loops=True,
        )
        assert False, "expected runtime loop failure"
    except RuntimeError as exc:
        assert "runtime loop failed" in str(exc)

    assert fallback_called["value"] is False


def test_strategy_ablation_benchmark_expands_runtime_grid(monkeypatch, tmp_path):
    cfg = load_config()
    calls = []

    def _fake_run_table1(*args, **kwargs):
        runtime_manifest = kwargs["runtime_manifests"]["factor_miner"]
        calls.append(runtime_manifest)
        memory_policy = runtime_manifest["memory_policy"]
        dependence_metric = runtime_manifest["redundancy_metric"]
        backend = runtime_manifest["backend"]
        score = {
            ("paper", "spearman", "numpy"): 0.05,
            ("paper", "pearson", "numpy"): 0.06,
            ("kg", "spearman", "numpy"): 0.07,
            ("kg", "pearson", "numpy"): 0.09,
        }[(memory_policy, dependence_metric, backend)]
        return {
            "factor_miner": {
                "freeze_library_size": 5,
                "freeze_stats": {
                    "succeeded": 10,
                    "admitted": 4,
                    "correlation_rejections": 2,
                    "replaced": 1,
                },
                "universes": {
                    "CSI500": {
                        "library": {"ic": score, "icir": score * 10.0, "avg_abs_rho": 0.2}
                    }
                },
                "provenance": {"requested_runtime_manifest": dict(runtime_manifest)},
            }
        }

    monkeypatch.setattr("factorminer.benchmark.runtime.run_table1_benchmark", _fake_run_table1)

    payload = run_ablation_strategy_benchmark(
        cfg,
        tmp_path,
        mock=True,
        baseline="factor_miner",
        memory_policies=["paper", "kg"],
        dependence_metrics=["spearman", "pearson"],
        backends=["numpy"],
    )

    assert len(calls) == 4
    assert payload["best"]["memory_policy"] == "kg"
    assert payload["best"]["dependence_metric"] == "pearson"
    assert payload["leaderboard"][0]["mean_test_library_ic"] == 0.09
    assert (tmp_path / "benchmark" / "ablation" / "strategy_grid.json").exists()
