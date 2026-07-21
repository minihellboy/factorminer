"""Data ingestion and tensor APIs exposed through lazy package attributes."""

from __future__ import annotations

from factorminer._lazy_exports import public_dir, resolve_export

_EXPORTS = {
    **{
        name: "factorminer.data.loader"
        for name in (
            "OHLCV_COLUMNS",
            "REQUIRED_COLUMNS",
            "load_market_data",
            "load_multiple",
            "to_numpy",
        )
    },
    **{
        name: "factorminer.data.mock_data"
        for name in ("MockConfig", "generate_mock_data", "generate_with_halts")
    },
    **{
        name: "factorminer.data.validation"
        for name in (
            "DataValidationReport",
            "ValidationIssue",
            "render_validation_report",
            "validate_market_data",
        )
    },
    **{
        name: "factorminer.data.preprocessor"
        for name in (
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
        )
    },
    **{
        name: "factorminer.data.tensor_builder"
        for name in (
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
        )
    },
    **{
        name: "factorminer.data.edgar_source"
        for name in (
            "EDGAR_FEATURE_LEAVES",
            "EdgarClient",
            "EdgarConfig",
            "attach_edgar_to_panel",
            "load_edgar_fundamentals",
            "register_edgar_features",
        )
    },
    **{
        name: "factorminer.data.futures_source"
        for name in (
            "FUTURES_FEATURE_LEAVES",
            "FuturesConfig",
            "build_continuous_futures_panel",
            "generate_mock_futures_panel",
            "register_futures_features",
        )
    },
    **{
        name: "factorminer.data.public_archive"
        for name in (
            "PublicArchive",
            "PublicDatasetLock",
            "PublicDatasetVerification",
            "load_public_dataset_lock",
            "lock_public_dataset_spec",
            "prepare_public_dataset",
            "verify_prepared_public_dataset",
        )
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), name, _EXPORTS)


def __dir__() -> list[str]:
    return public_dir(globals(), _EXPORTS)
