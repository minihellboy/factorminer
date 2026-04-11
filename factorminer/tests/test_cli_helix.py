"""CLI tests for the Helix command."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from click.testing import CliRunner

from factorminer.cli import _build_core_mining_config, _prepare_data_arrays, main
from factorminer.utils.config import load_config


def test_helix_cli_runs_with_mock_data(tmp_path):
    """The helix command should execute end-to-end and save a library."""
    output_dir = tmp_path / "helix-output"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "--cpu",
            "--output-dir",
            str(output_dir),
            "helix",
            "--mock",
            "-n",
            "1",
            "-b",
            "5",
            "-t",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Starting Helix Loop..." in result.output
    assert "Helix mining complete!" in result.output

    library_path = output_dir / "factor_library.json"
    assert library_path.exists()

    payload = json.loads(library_path.read_text())
    assert "factors" in payload


def test_helix_cli_reports_enabled_features(tmp_path):
    """Explicit feature flags should be reflected in the CLI output."""
    output_dir = tmp_path / "helix-flags"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "--cpu",
            "--output-dir",
            str(output_dir),
            "helix",
            "--mock",
            "--debate",
            "--canonicalize",
            "-n",
            "1",
            "-b",
            "4",
            "-t",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Active Phase 2 features: debate, canonicalization" in result.output


def test_prepare_data_arrays_builds_full_factor_feature_surface():
    """The CLI tensor builder should expose the paper's canonical features."""
    df = pd.DataFrame(
        [
            {
                "datetime": "2025-01-01 09:30:00",
                "asset_id": "A",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "volume": 2.0,
                "amount": 20.0,
            },
            {
                "datetime": "2025-01-01 09:40:00",
                "asset_id": "A",
                "open": 10.0,
                "high": 12.0,
                "low": 9.5,
                "close": 11.0,
                "volume": 2.0,
                "amount": 22.0,
            },
            {
                "datetime": "2025-01-01 09:30:00",
                "asset_id": "B",
                "open": 20.0,
                "high": 21.0,
                "low": 19.0,
                "close": 20.0,
                "volume": 4.0,
                "amount": 80.0,
            },
            {
                "datetime": "2025-01-01 09:40:00",
                "asset_id": "B",
                "open": 20.0,
                "high": 22.0,
                "low": 19.5,
                "close": 18.0,
                "volume": 4.0,
                "amount": 72.0,
            },
        ]
    )
    df["datetime"] = pd.to_datetime(df["datetime"])

    data_tensor, forward_returns = _prepare_data_arrays(df)

    assert data_tensor.shape == (2, 2, 8)
    np.testing.assert_allclose(data_tensor[:, :, 6], np.array([[10.0, 11.0], [20.0, 18.0]]))
    assert np.isnan(data_tensor[0, 0, 7])
    np.testing.assert_allclose(data_tensor[:, 1, 7], np.array([0.1, -0.1]))
    assert np.isnan(forward_returns[0, 1])
    np.testing.assert_allclose(forward_returns[:, 0], np.array([0.1, -0.1]))


def test_mock_mining_config_uses_synthetic_signal_failures(tmp_path):
    """Mock mining flows should bypass strict benchmark recomputation defaults."""
    cfg = load_config()

    normal_config = _build_core_mining_config(cfg, tmp_path / "normal", mock=False)
    mock_config = _build_core_mining_config(cfg, tmp_path / "mock", mock=True)

    assert normal_config.signal_failure_policy == "reject"
    assert mock_config.signal_failure_policy == "synthetic"
