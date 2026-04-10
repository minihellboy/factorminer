"""Golden-path CLI mining and downstream analysis test."""

from __future__ import annotations

import pandas as pd
from click.testing import CliRunner

from factorminer.cli import main
from factorminer.evaluation.runtime import load_runtime_dataset
from factorminer.utils.config import load_config


def _tiny_raw_panel() -> pd.DataFrame:
    rows = []
    for asset_id, base in [("A", 10.0), ("B", 20.0)]:
        for idx, ts in enumerate(
            [
                "2024-01-02 09:30:00",
                "2024-01-02 09:40:00",
                "2024-01-02 09:50:00",
                "2025-01-02 09:30:00",
                "2025-01-02 09:40:00",
                "2025-01-02 09:50:00",
            ]
        ):
            open_px = base + idx * 0.1
            close_px = open_px + (0.05 if asset_id == "A" else -0.03) + idx * 0.01
            high_px = max(open_px, close_px) + 0.1
            low_px = min(open_px, close_px) - 0.1
            volume = 1000 + idx * 10
            amount = volume * close_px
            rows.append(
                {
                    "datetime": ts,
                    "asset_id": asset_id,
                    "open": open_px,
                    "high": high_px,
                    "low": low_px,
                    "close": close_px,
                    "volume": volume,
                    "amount": amount,
                    "vwap": close_px,
                }
            )
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def test_cli_mine_then_evaluate_golden_path(tmp_path, monkeypatch):
    cfg = load_config()
    dataset = load_runtime_dataset(_tiny_raw_panel(), cfg)
    original_build_core_mining_config = __import__(
        "factorminer.cli", fromlist=["_build_core_mining_config"]
    )._build_core_mining_config

    monkeypatch.setattr(
        "factorminer.cli._load_runtime_dataset_for_analysis",
        lambda cfg_arg, data_path, mock: dataset,
    )
    monkeypatch.setattr(
        "factorminer.cli._build_core_mining_config",
        lambda cfg_arg, output_dir, mock=False: _relaxed_config(
            original_build_core_mining_config(cfg_arg, output_dir, mock=mock)
        ),
    )

    output_dir = tmp_path / "golden-output"
    runner = CliRunner()

    mine_result = runner.invoke(
        main,
        [
            "--cpu",
            "--output-dir",
            str(output_dir),
            "mine",
            "--mock",
            "-n",
            "1",
            "-b",
            "4",
            "-t",
            "2",
        ],
    )
    assert mine_result.exit_code == 0, mine_result.output

    library_path = output_dir / "factor_library.json"
    assert library_path.exists()
    assert (output_dir / "run_manifest.json").exists()
    assert (output_dir / "factor_lifecycle.jsonl").exists()

    eval_result = runner.invoke(
        main,
        [
            "--cpu",
            "--output-dir",
            str(output_dir),
            "evaluate",
            str(library_path),
            "--mock",
            "--period",
            "both",
            "--top-k",
            "1",
        ],
    )
    assert eval_result.exit_code == 0, eval_result.output
    assert "Split: train" in eval_result.output
    assert "Split: test" in eval_result.output


def _relaxed_config(mining_config):
    mining_config.ic_threshold = 0.0
    mining_config.icir_threshold = -1.0
    mining_config.correlation_threshold = 1.1
    return mining_config
