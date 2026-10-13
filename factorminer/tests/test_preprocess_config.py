"""Configurable preprocessing and point-in-time universe masking."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from factorminer.architecture.dataset_contract import DatasetContract
from factorminer.data.loader import load_market_data
from factorminer.data.preprocessor import PreprocessConfig, fill_missing, preprocess, winsorise
from factorminer.evaluation.runtime import load_runtime_dataset
from factorminer.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "binance_crypto_5m.csv"
SAMPLE_CONFIG = ROOT / "factorminer" / "configs" / "binance_sample.yaml"
RAW_PRICES = {
    "standardise": False,
    "winsor_lower": 0.0,
    "winsor_upper": 100.0,
    "cross_fill_method": "none",
}


def _sample():
    return load_market_data(SAMPLE)


def test_default_config_keeps_historical_preprocessing_identity():
    cfg = load_config(SAMPLE_CONFIG)
    assert cfg.preprocess == PreprocessConfig()
    dataset = load_runtime_dataset(_sample(), cfg)
    implicit = load_runtime_dataset(_sample(), SimpleNamespace(data=cfg.data))
    assert dataset.preprocessing == implicit.preprocessing
    assert "universe_mask" not in dataset.preprocessing


def test_yaml_preprocess_section_reaches_the_runtime_panel():
    cfg = load_config(SAMPLE_CONFIG, overrides={"preprocess": RAW_PRICES})
    raw = _sample()
    dataset = load_runtime_dataset(raw, cfg)
    assert dataset.preprocessing["config"]["standardise"] is False

    first = raw.sort_values(["datetime", "asset_id"]).iloc[0]
    asset = list(dataset.asset_ids).index(first["asset_id"])
    assert dataset.data_dict["$close"][asset, 0] == pytest.approx(first["close"])

    default = DatasetContract.from_runtime_dataset(
        load_config(SAMPLE_CONFIG), load_runtime_dataset(_sample(), load_config(SAMPLE_CONFIG))
    )
    assert DatasetContract.from_runtime_dataset(cfg, dataset).replay_mismatches(
        default.replay_identity()
    ) == ["panel_digest", "preprocessing_digest"]


def test_runtime_accepts_mapping_and_rejects_unknown_or_invalid_settings():
    cfg = load_config(SAMPLE_CONFIG)
    duck = SimpleNamespace(data=cfg.data, preprocess={"standardise": False})
    assert load_runtime_dataset(_sample(), duck).preprocessing["config"]["standardise"] is False

    with pytest.raises(ValueError, match="unknown preprocess settings"):
        load_runtime_dataset(_sample(), SimpleNamespace(data=cfg.data, preprocess={"zscore": 1}))
    with pytest.raises(TypeError, match="PreprocessConfig or mapping"):
        load_runtime_dataset(_sample(), SimpleNamespace(data=cfg.data, preprocess="raw"))
    with pytest.raises(ValueError, match="winsor bounds"):
        load_config(overrides={"preprocess": {"winsor_lower": 99.0, "winsor_upper": 1.0}})


def test_no_cross_fill_leaves_unlisted_assets_missing():
    frame = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
        "asset_id": ["A", "B", "A", "B"],
        "close": [10.0, np.nan, 11.0, 50.0],
    })
    assert fill_missing(frame, cross_fill_method="median")["close"].isna().sum() == 0
    assert fill_missing(frame, cross_fill_method="none")["close"].isna().sum() == 1


def test_full_winsor_bounds_are_identity():
    frame = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-02"] * 5),
        "close": [1.0, 2.0, 3.0, 4.0, 1000.0],
    })
    pd.testing.assert_frame_equal(winsorise(frame, ["close"], 0.0, 100.0), frame)
    with pytest.raises(ValueError, match="cross_fill_method"):
        preprocess(frame, PreprocessConfig(cross_fill_method="zero"))


def test_universe_mask_hides_targets_but_keeps_prices():
    cfg = load_config(SAMPLE_CONFIG)
    raw = _sample()
    excluded = sorted(raw["asset_id"].unique())[0]
    raw["in_universe"] = (raw["asset_id"] != excluded).astype(int)
    dataset = load_runtime_dataset(raw, cfg)
    row = list(dataset.asset_ids).index(excluded)

    assert np.isnan(dataset.returns[row]).all()
    assert np.isfinite(dataset.data_dict["$close"][row]).any()
    assert np.isfinite(np.delete(dataset.returns, row, axis=0)).any()
    assert dataset.preprocessing["universe_mask"] == "in_universe"

    raw["in_universe"] = 2
    with pytest.raises(ValueError, match="in_universe"):
        load_runtime_dataset(raw, cfg)


def test_default_preprocess_is_omitted_from_serialised_config_identity():
    assert "preprocess" not in load_config().to_dict()
    declared = load_config(overrides={"preprocess": {"standardise": False}}).to_dict()
    assert declared["preprocess"]["standardise"] is False
    assert load_config(overrides=declared).preprocess.standardise is False



def test_mining_screen_prefers_coverage_and_keeps_dense_draw():
    from factorminer.application.validation_pipeline import ValidationPipeline
    from factorminer.evaluation.pipeline import coverage_ranked_subset

    dense = np.zeros((50, 20))
    assert coverage_ranked_subset(dense, 10) is None
    pipeline = ValidationPipeline(data_tensor=np.zeros((50, 20, 8)), returns=dense,
                                  library=None, fast_screen_assets=10)
    expected = np.sort(np.random.RandomState(0).choice(50, 10, replace=False))
    np.testing.assert_array_equal(pipeline._fast_indices, expected)

    masked = np.full((50, 20), np.nan)
    members = np.arange(5, 50, 5)  # nine always-members among 50 ever-members
    masked[members] = 0.0
    masked[0, :3] = 0.0
    pipeline = ValidationPipeline(data_tensor=np.zeros((50, 20, 8)), returns=masked,
                                  library=None, fast_screen_assets=10)
    assert set(members) <= set(pipeline._fast_indices) and 0 in pipeline._fast_indices


def test_evaluation_pipeline_screen_keeps_seeded_draw_on_dense_panels():
    from factorminer.evaluation.pipeline import PipelineConfig, ValidationPipeline

    dense = np.zeros((50, 20))
    pipeline = ValidationPipeline(returns=dense, library=None,
                                  config=PipelineConfig(fast_screen_assets=10))
    expected = np.random.default_rng(42).choice(50, size=10, replace=False)
    np.testing.assert_array_equal(pipeline._fast_idx, expected)


def test_targets_follow_dates_kept_by_the_quality_check():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                            "2025-01-02", "2025-01-03"])
    rows = []
    for asset, base in (("A", 10.0), ("B", 20.0), ("C", 30.0)):
        for step, date in enumerate(dates):
            if date == dates[1] and asset != "A":
                continue  # only 1/3 of assets trade: the date fails min_nonnan_ratio=0.5
            price = base + step
            rows.append({"datetime": date, "asset_id": asset, "open": price, "high": price,
                         "low": price, "close": price, "volume": 1.0, "amount": price})
    cfg = load_config(overrides={
        "preprocess": RAW_PRICES | {"min_nonnan_ratio": 0.5},
        "data": {"train_period": ["2024-01-01", "2024-12-31"],
                 "test_period": ["2025-01-01", "2025-12-31"],
                 "targets": [{"name": "paper", "entry_delay_bars": 0, "holding_bars": 1,
                              "price_pair": "close_to_close"}]},
    })
    dataset = load_runtime_dataset(pd.DataFrame(rows), cfg)
    assert dates[1] not in set(pd.to_datetime(dataset.timestamps))
    a = list(dataset.asset_ids).index("A")
    # One bar after 2024-01-02 on the kept grid is 2024-01-04 (close 12), not 01-03 (11).
    assert dataset.returns[a, 0] == pytest.approx(12.0 / 10.0 - 1.0)


def test_preprocess_typos_are_rejected_and_discovery_keeps_provenance():
    from factorminer.application.research_actions import discovery_dataset

    with pytest.raises(ValueError, match="unknown preprocess settings"):
        load_config(overrides={"preprocess": {"standardize": False}})
    cfg = load_config(SAMPLE_CONFIG, overrides={"preprocess": RAW_PRICES})
    dataset = load_runtime_dataset(_sample(), cfg)
    assert discovery_dataset(dataset).preprocessing == dataset.preprocessing
