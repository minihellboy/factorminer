"""Benchmark-runtime and CLI coverage."""

from __future__ import annotations

from click.testing import CliRunner
import numpy as np

from factorminer.benchmark.runtime import build_benchmark_library, select_frozen_top_k
from factorminer.cli import main
from factorminer.evaluation.runtime import FactorEvaluationArtifact
from factorminer.utils.config import load_config


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
