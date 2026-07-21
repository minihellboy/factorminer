"""Across-seed Phase-2 benchmark behavior (multi-run restore).

``run_phase2_comparison`` and ``run_phase2_ablation_study`` accept
``n_runs`` and must execute one full runtime benchmark per derived seed
(``benchmark.seed + run_id``), report across-seed means in the frames,
keep every per-run result in ``raw_method_results``, and summarize
dispersion under ``statistical_tests["seed_distribution"]``. The
``n_runs=1`` path must keep the legacy single-run shape exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from factorminer.benchmark.phase2_reporting import (
    _generate_markdown_report,
    _print_stat_tests,
)
from factorminer.benchmark.runtime import (
    run_phase2_ablation_study,
    run_phase2_comparison,
)
from factorminer.benchmark.statistics import (
    MethodResult,
    aggregate_method_results,
    method_result_dispersion,
)
from factorminer.utils.config import load_config


def _library_ic(seed: int, method: str) -> float:
    # Seed- and method-dependent so across-seed deltas vary per run.
    return 0.05 + 0.01 * (seed % 10) * (1.0 + 0.1 * (len(method) % 7))


def _fake_payload(seed: int, method: str) -> dict[str, Any]:
    base = _library_ic(seed, method)
    ic_series = [
        base + 0.001 * np.sin((index + 1) * (1 + len(method) % 5))
        for index in range(30)
    ]
    return {
        "freeze_stats": {"succeeded": 10, "admitted": 5},
        "frozen_top_k": [{"name": f"{method}_f1"}],
        "provenance": {"kind": "test"},
        "universes": {
            "CSI500": {
                "factor_count": 4,
                "library": {"ic": base, "icir": 1.0, "avg_abs_rho": 0.2},
                "combinations": {
                    "equal_weight": {
                        "ic": base,
                        "icir": 0.9,
                        "turnover": 0.1,
                        "ic_series": ic_series,
                        "cost_pressure": {
                            "1": {
                                "ic": base,
                                "icir": 0.8,
                                "turnover": 0.1,
                                "long_short": 0.01,
                                "monotonicity": 0.5,
                            }
                        },
                    },
                    "ic_weighted": {"ic": base, "icir": 0.9, "turnover": 0.1},
                },
                "selections": {
                    "lasso": {"ic": base, "icir": 0.7},
                    "xgboost": {"ic": base + 0.01, "icir": 0.8},
                },
            }
        },
    }


@pytest.fixture
def patched_runtime(monkeypatch):
    calls: list[dict[str, Any]] = []

    def _fake_table1(cfg, output_dir, **kwargs):
        names = list(kwargs.get("baseline_names", []))
        calls.append(
            {
                "seed": int(cfg.benchmark.seed),
                "output_dir": Path(output_dir),
                "baselines": names,
            }
        )
        return {name: _fake_payload(int(cfg.benchmark.seed), name) for name in names}

    monkeypatch.setattr("factorminer.benchmark.runtime.run_table1_benchmark", _fake_table1)
    monkeypatch.setattr(
        "factorminer.benchmark.runtime.run_efficiency_benchmark",
        lambda cfg, output_dir: {"operator_level_ms": {}},
    )
    return calls


def test_multi_seed_runs_use_derived_seeds_and_subdirs(patched_runtime, tmp_path):
    cfg = load_config()
    base_seed = int(cfg.benchmark.seed)
    methods = ["ralph_loop", "helix_phase2"]

    result, artifacts = run_phase2_comparison(
        cfg,
        tmp_path,
        mock=True,
        baseline_methods=methods,
        n_target_factors=5,
        n_runs=2,
    )

    assert [call["seed"] for call in patched_runtime] == [base_seed, base_seed + 1]
    assert [call["output_dir"].name for call in patched_runtime] == ["run_0", "run_1"]
    assert artifacts["n_runs"] == 2
    assert artifacts["seeds"] == [base_seed, base_seed + 1]
    assert len(artifacts["runtime_roots"]) == 2

    for method in methods:
        per_run = result.raw_method_results[method]
        assert [item.run_id for item in per_run] == [0, 1]
        expected = [_library_ic(base_seed, method), _library_ic(base_seed + 1, method)]
        assert [item.library_ic for item in per_run] == pytest.approx(expected)
        row = result.factor_library_metrics[
            result.factor_library_metrics["method"] == method
        ].iloc[0]
        assert row["ic_pct"] == pytest.approx(float(np.mean(expected)) * 100.0)
        assert row["n_runs"] == 2

    distribution = result.statistical_tests["seed_distribution"]
    assert distribution["n_runs"] == 2
    assert distribution["seeds"] == [base_seed, base_seed + 1]
    helix = distribution["methods"]["helix_phase2"]["library_ic"]
    expected = [_library_ic(base_seed, "helix_phase2"), _library_ic(base_seed + 1, "helix_phase2")]
    assert helix["values"] == pytest.approx(expected)
    assert helix["std"] == pytest.approx(float(np.std(expected, ddof=1)))

    paired = result.statistical_tests["paired_tests_by_run"]
    assert [item["run_id"] for item in paired] == [0, 1]
    assert [item["seed"] for item in paired] == [base_seed, base_seed + 1]

    assert sorted(result.turnover_metrics["run_id"].unique().tolist()) == [0, 1]
    assert sorted(result.cost_pressure_metrics["run_id"].unique().tolist()) == [0, 1]
    assert len(result.runtime_artifacts["runtime_payloads"]["ralph_loop"]) == 2


def test_multi_seed_statistical_reports_render_per_run(
    patched_runtime, tmp_path, capsys
):
    cfg = load_config()
    result, _ = run_phase2_comparison(
        cfg,
        tmp_path,
        mock=True,
        baseline_methods=["ralph_loop", "helix_phase2"],
        n_target_factors=5,
        n_runs=2,
    )

    _print_stat_tests(result.statistical_tests)
    output = capsys.readouterr().out
    assert "Paired tests by seed (2 runs)" in output
    assert f"seed {cfg.benchmark.seed}" in output
    assert f"seed {cfg.benchmark.seed + 1}" in output

    report_path = _generate_markdown_report(result, None, tmp_path)
    report = Path(report_path).read_text()
    assert "| Run | Seed | Mean IC diff |" in report
    assert f"| 0 | {cfg.benchmark.seed} |" in report
    assert f"| 1 | {cfg.benchmark.seed + 1} |" in report


def test_single_run_keeps_legacy_shape(patched_runtime, tmp_path):
    cfg = load_config()
    methods = ["ralph_loop", "helix_phase2"]

    result, artifacts = run_phase2_comparison(
        cfg,
        tmp_path,
        mock=True,
        baseline_methods=methods,
        n_target_factors=5,
        n_runs=1,
    )

    assert len(patched_runtime) == 1
    assert patched_runtime[0]["output_dir"] == tmp_path
    assert artifacts["runtime_root"] == str((tmp_path / "benchmark").resolve())
    assert "runtime_roots" not in artifacts
    assert "n_runs" not in artifacts
    assert "seeds" not in artifacts
    assert "seed_distribution" not in result.statistical_tests
    assert "n_runs" not in result.factor_library_metrics.columns
    for method in methods:
        assert len(result.raw_method_results[method]) == 1
        assert result.raw_method_results[method][0].run_id == 0


def test_ablation_multi_seed_reports_mean_and_std_deltas(patched_runtime, tmp_path):
    cfg = load_config()
    base_seed = int(cfg.benchmark.seed)

    ablation = run_phase2_ablation_study(
        cfg,
        tmp_path,
        mock=True,
        configs_to_run=["full", "no_debate"],
        n_target_factors=5,
        n_runs=2,
    )

    # two configs x two runs, each in its own run_<id> subdirectory
    assert len(patched_runtime) == 4
    assert {call["output_dir"].name for call in patched_runtime} == {"run_0", "run_1"}

    deltas = [
        _library_ic(base_seed + run_id, "helix_no_debate")
        - _library_ic(base_seed + run_id, "helix_phase2")
        for run_id in (0, 1)
    ]
    row = ablation.contributions[ablation.contributions["config"] == "no_debate"].iloc[0]
    assert row["delta_library_ic"] == pytest.approx(float(np.mean(deltas)))
    assert row["delta_library_ic_std"] == pytest.approx(float(np.std(deltas, ddof=1)))
    assert ablation.results["no_debate"].run_id == -1


def test_aggregate_method_results_mean_and_identity():
    single = MethodResult(method="m", library_ic=0.07)
    assert aggregate_method_results([single]) is single

    pair = [
        MethodResult(method="m", library_ic=0.04, n_factors=10, run_id=0),
        MethodResult(method="m", library_ic=0.08, n_factors=11, run_id=1),
    ]
    aggregate = aggregate_method_results(pair)
    assert aggregate.library_ic == pytest.approx(0.06)
    assert aggregate.n_factors == 10  # round(10.5) banker's rounding
    assert aggregate.run_id == -1

    dispersion = method_result_dispersion(pair)
    assert dispersion["library_ic"]["mean"] == pytest.approx(0.06)
    assert dispersion["library_ic"]["std"] == pytest.approx(float(np.std([0.04, 0.08], ddof=1)))
    assert dispersion["library_ic"]["values"] == pytest.approx([0.04, 0.08])

    with pytest.raises(ValueError):
        aggregate_method_results([])
    with pytest.raises(ValueError):
        method_result_dispersion([])
    with pytest.raises(ValueError, match="one method"):
        aggregate_method_results(
            [MethodResult(method="a"), MethodResult(method="b")]
        )
    with pytest.raises(ValueError, match="one method"):
        method_result_dispersion(
            [MethodResult(method="a"), MethodResult(method="b")]
        )


@pytest.mark.parametrize("n_runs", [0, -1])
def test_non_positive_run_counts_are_rejected(
    patched_runtime, tmp_path, n_runs
):
    cfg = load_config()

    with pytest.raises(ValueError, match="at least 1"):
        run_phase2_comparison(cfg, tmp_path, mock=True, n_runs=n_runs)
    with pytest.raises(ValueError, match="at least 1"):
        run_phase2_ablation_study(cfg, tmp_path, mock=True, n_runs=n_runs)
    assert patched_runtime == []


def test_seed_zero_is_preserved(patched_runtime, tmp_path):
    cfg = load_config()
    cfg.benchmark.seed = 0

    run_phase2_comparison(
        cfg,
        tmp_path,
        mock=True,
        baseline_methods=["ralph_loop"],
        n_runs=2,
    )

    assert [call["seed"] for call in patched_runtime] == [0, 1]

    patched_runtime.clear()
    run_phase2_ablation_study(
        cfg,
        tmp_path,
        mock=True,
        configs_to_run=["full"],
        n_runs=2,
    )
    assert [call["seed"] for call in patched_runtime] == [0, 1]
