"""FactorMiner data pipeline: loading, preprocessing, and tensor construction."""

from factorminer.data.edgar_source import (
    EDGAR_FEATURE_LEAVES,
    EdgarClient,
    EdgarConfig,
    attach_edgar_to_panel,
    load_edgar_fundamentals,
    register_edgar_features,
)
from factorminer.data.futures_source import (
    FUTURES_FEATURE_LEAVES,
    FuturesConfig,
    build_continuous_futures_panel,
    generate_mock_futures_panel,
    register_futures_features,
)
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
from factorminer.data.validation import (
    DataValidationReport,
    ValidationIssue,
    render_validation_report,
    validate_market_data,
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
    # edgar / futures alt-data lanes
    "EDGAR_FEATURE_LEAVES",
    "EdgarClient",
    "EdgarConfig",
    "attach_edgar_to_panel",
    "load_edgar_fundamentals",
    "register_edgar_features",
    "FUTURES_FEATURE_LEAVES",
    "FuturesConfig",
    "build_continuous_futures_panel",
    "generate_mock_futures_panel",
    "register_futures_features",
]
