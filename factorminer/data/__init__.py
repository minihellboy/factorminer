"""FactorMiner data pipeline: loading, preprocessing, and tensor construction."""

from factorminer.data.loader import (
    OHLCV_COLUMNS,
    REQUIRED_COLUMNS,
    load_market_data,
    load_multiple,
    to_numpy,
)
from factorminer.data.mock_data import (
    MockConfig,
    generate_mock_data,
    generate_with_halts,
)
from factorminer.data.validation import (
    DataValidationReport,
    ValidationIssue,
    render_validation_report,
    validate_market_data,
)
from factorminer.data.preprocessor import (
    PreprocessConfig,
    compute_derived_features,
    compute_returns,
    compute_vwap,
    cross_sectional_standardise,
    fill_missing,
    flag_halts,
    mask_halts,
    preprocess,
    quality_check,
    winsorise,
)
from factorminer.data.tensor_builder import (
    DEFAULT_FEATURES,
    TargetSpec,
    TensorConfig,
    TensorDataset,
    build_pipeline,
    build_tensor,
    compute_target,
    compute_targets,
    sample_assets,
    temporal_split,
)

__all__ = [
    # loader
    "OHLCV_COLUMNS",
    "REQUIRED_COLUMNS",
    "load_market_data",
    "load_multiple",
    "to_numpy",
    # mock_data
    "MockConfig",
    "generate_mock_data",
    "generate_with_halts",
    # validation
    "DataValidationReport",
    "ValidationIssue",
    "render_validation_report",
    "validate_market_data",
    # preprocessor
    "PreprocessConfig",
    "compute_derived_features",
    "compute_returns",
    "compute_vwap",
    "cross_sectional_standardise",
    "fill_missing",
    "flag_halts",
    "mask_halts",
    "preprocess",
    "quality_check",
    "winsorise",
    # tensor_builder
    "DEFAULT_FEATURES",
    "TargetSpec",
    "TensorConfig",
    "TensorDataset",
    "build_pipeline",
    "build_tensor",
    "compute_target",
    "compute_targets",
    "sample_assets",
    "temporal_split",
]
