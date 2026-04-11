"""File-level market data validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from factorminer.data.loader import COLUMN_ALIASES, REQUIRED_COLUMNS

FileFormat = Literal["csv", "parquet", "hdf5"]

LEAKY_NAME_PATTERNS = (
    "future",
    "label",
    "target",
    "return_next",
)


@dataclass(slots=True)
class ValidationIssue:
    """One validation message emitted by the validator."""

    severity: Literal["info", "warning", "error"]
    code: str
    message: str


@dataclass(slots=True)
class DataValidationReport:
    """Structured validation result for a market-data file."""

    path: str
    fmt: str
    readable: bool
    raw_columns: list[str] = field(default_factory=list)
    canonical_mapping: dict[str, str] = field(default_factory=dict)
    missing_required_columns: list[str] = field(default_factory=list)
    row_count: int = 0
    asset_count: int = 0
    timestamp_count: int = 0
    observed_pair_count: int = 0
    duplicate_key_count: int = 0
    duplicate_key_groups: int = 0
    coverage_ratio: float = 0.0
    non_monotonic_assets: int = 0
    missingness_by_column: dict[str, float] = field(default_factory=dict)
    ohlc_violations: dict[str, int] = field(default_factory=dict)
    negative_volume_count: int = 0
    negative_amount_count: int = 0
    leakage_warnings: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid_schema(self) -> bool:
        return self.readable and not self.missing_required_columns

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == "warning" for issue in self.issues)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues) or not self.valid_schema

    def exit_code(self, strict: bool = False) -> int:
        if not self.valid_schema:
            return 1
        if strict and self.has_warnings:
            return 1
        return 0

    def to_dict(self, strict: bool = False) -> dict:
        return {
            "path": self.path,
            "format": self.fmt,
            "readable": self.readable,
            "valid_schema": self.valid_schema,
            "strict": strict,
            "status": _status_label(self, strict=strict),
            "raw_columns": list(self.raw_columns),
            "canonical_mapping": dict(self.canonical_mapping),
            "missing_required_columns": list(self.missing_required_columns),
            "row_count": int(self.row_count),
            "asset_count": int(self.asset_count),
            "timestamp_count": int(self.timestamp_count),
            "observed_pair_count": int(self.observed_pair_count),
            "duplicate_key_count": int(self.duplicate_key_count),
            "duplicate_key_groups": int(self.duplicate_key_groups),
            "coverage_ratio": float(self.coverage_ratio),
            "non_monotonic_assets": int(self.non_monotonic_assets),
            "missingness_by_column": {
                key: float(value) for key, value in self.missingness_by_column.items()
            },
            "ohlc_violations": {key: int(value) for key, value in self.ohlc_violations.items()},
            "negative_volume_count": int(self.negative_volume_count),
            "negative_amount_count": int(self.negative_amount_count),
            "leakage_warnings": list(self.leakage_warnings),
            "issues": [asdict(issue) for issue in self.issues],
            "warning_count": sum(issue.severity == "warning" for issue in self.issues),
            "error_count": max(
                sum(issue.severity == "error" for issue in self.issues),
                0 if self.valid_schema else 1,
            ),
        }


def validate_market_data(
    path: str | Path,
    fmt: Optional[FileFormat] = None,
    hdf_key: str = "data",
) -> DataValidationReport:
    """Validate a market-data file against the canonical loader contract."""
    file_path = Path(path)
    resolved_fmt = fmt or _infer_format(file_path)
    report = DataValidationReport(
        path=str(file_path),
        fmt=resolved_fmt,
        readable=False,
    )

    try:
        raw_df = _read_raw_frame(file_path, resolved_fmt, hdf_key=hdf_key)
    except Exception as exc:  # noqa: BLE001 - surfaced to CLI as an error
        report.issues.append(
            ValidationIssue(
                severity="error",
                code="unreadable-file",
                message=f"Could not read {file_path}: {exc}",
            )
        )
        return report

    report.readable = True
    report.raw_columns = [str(col) for col in raw_df.columns]

    mapping, ambiguous, missing = _build_canonical_mapping(raw_df.columns)
    report.canonical_mapping = mapping
    report.missing_required_columns = missing

    for canonical, sources in ambiguous.items():
        report.issues.append(
            ValidationIssue(
                severity="warning",
                code="ambiguous-column-mapping",
                message=(
                    f"Multiple source columns match '{canonical}': {sources}. "
                    f"Using '{mapping[canonical]}'"
                ),
            )
        )

    canonical_df = raw_df.rename(columns={source: canonical for canonical, source in mapping.items()})
    report.row_count = int(len(canonical_df))

    if "datetime" in canonical_df.columns:
        datetime_values = pd.to_datetime(canonical_df["datetime"], errors="coerce")
        report.missingness_by_column["datetime"] = float(datetime_values.isna().mean())
        canonical_df["datetime"] = datetime_values
    if "asset_id" in canonical_df.columns:
        asset_values = canonical_df["asset_id"].astype("string").str.strip()
        report.missingness_by_column["asset_id"] = float(
            (asset_values.isna() | asset_values.eq("")).mean()
        )
        canonical_df["asset_id"] = asset_values

    for col in REQUIRED_COLUMNS:
        if col in {"datetime", "asset_id"}:
            continue
        if col in canonical_df.columns:
            numeric_values = pd.to_numeric(canonical_df[col], errors="coerce")
            report.missingness_by_column[col] = float(numeric_values.isna().mean())
            canonical_df[col] = numeric_values
        else:
            report.missingness_by_column[col] = 1.0

    if report.valid_schema:
        valid_keys = canonical_df.dropna(subset=["asset_id", "datetime"]).copy()
        valid_keys["asset_id"] = valid_keys["asset_id"].astype(str)
        valid_keys["datetime"] = pd.to_datetime(valid_keys["datetime"], errors="coerce")
        valid_keys = valid_keys.dropna(subset=["asset_id", "datetime"])

        report.asset_count = int(valid_keys["asset_id"].nunique())
        report.timestamp_count = int(valid_keys["datetime"].nunique())
        report.observed_pair_count = int(
            valid_keys.drop_duplicates(subset=["asset_id", "datetime"]).shape[0]
        )
        report.duplicate_key_count = int(
            valid_keys.duplicated(subset=["asset_id", "datetime"]).sum()
        )
        duplicated_keys = valid_keys.loc[
            valid_keys.duplicated(subset=["asset_id", "datetime"], keep=False),
            ["asset_id", "datetime"],
        ]
        report.duplicate_key_groups = int(
            duplicated_keys.drop_duplicates(subset=["asset_id", "datetime"]).shape[0]
        )
        if report.asset_count > 0 and report.timestamp_count > 0:
            report.coverage_ratio = float(
                report.observed_pair_count / (report.asset_count * report.timestamp_count)
            )

        report.non_monotonic_assets = _count_non_monotonic_assets(canonical_df)
        report.ohlc_violations = _count_ohlc_violations(canonical_df)
        report.negative_volume_count = _count_negative_values(canonical_df, "volume")
        report.negative_amount_count = _count_negative_values(canonical_df, "amount")

        if report.duplicate_key_count > 0:
            report.issues.append(
                ValidationIssue(
                    severity="warning",
                    code="duplicate-asset-timestamp",
                    message=(
                        f"Found {report.duplicate_key_count} duplicate asset/timestamp rows "
                        f"across {report.duplicate_key_groups} duplicated key groups."
                    ),
                )
            )
        if report.non_monotonic_assets > 0:
            report.issues.append(
                ValidationIssue(
                    severity="warning",
                    code="time-order",
                    message=(
                        f"{report.non_monotonic_assets} asset(s) have non-monotonic "
                        "timestamp order in source row order."
                    ),
                )
            )

        for name, count in report.ohlc_violations.items():
            if count > 0:
                report.issues.append(
                    ValidationIssue(
                        severity="warning",
                        code=f"ohlc-{name}",
                        message=f"{count} row(s) violated OHLC sanity check: {name}.",
                    )
                )

        if report.negative_volume_count > 0:
            report.issues.append(
                ValidationIssue(
                    severity="warning",
                    code="negative-volume",
                    message=f"{report.negative_volume_count} row(s) have negative volume.",
                )
            )
        if report.negative_amount_count > 0:
            report.issues.append(
                ValidationIssue(
                    severity="warning",
                    code="negative-amount",
                    message=f"{report.negative_amount_count} row(s) have negative amount.",
                )
            )

    future_like = _find_leaky_columns(report.raw_columns)
    if future_like:
        report.leakage_warnings = future_like
        report.issues.append(
            ValidationIssue(
                severity="warning",
                code="leakage-risk",
                message=(
                    "Potential leakage-risk column names detected: "
                    f"{', '.join(future_like)}"
                ),
            )
        )

    if not report.raw_columns:
        report.issues.append(
            ValidationIssue(
                severity="warning",
                code="empty-file",
                message="The file loaded successfully but contains no columns.",
            )
        )
    if report.row_count == 0:
        report.issues.append(
            ValidationIssue(
                severity="warning",
                code="empty-file",
                message="The file contains no rows.",
            )
        )

    return report


def render_validation_report(report: DataValidationReport, strict: bool = False) -> str:
    """Render a human-readable validation summary."""
    lines: list[str] = []
    status = _status_label(report, strict=strict)

    lines.append("FactorMiner -- Data Validation")
    lines.append("=" * 60)
    lines.append(f"Status: {status}")
    lines.append(f"Path:   {report.path}")
    lines.append(f"Format: {report.fmt}")
    lines.append(
        "Rows:   {rows} | Assets: {assets} | Timestamps: {timestamps} | "
        "Observed pairs: {pairs} | Coverage: {coverage:.2%}".format(
            rows=report.row_count,
            assets=report.asset_count,
            timestamps=report.timestamp_count,
            pairs=report.observed_pair_count,
            coverage=report.coverage_ratio,
        )
    )
    lines.append(f"Duplicate asset/timestamp rows: {report.duplicate_key_count}")
    lines.append(f"Non-monotonic asset timelines:   {report.non_monotonic_assets}")
    lines.append("")

    lines.append("Schema")
    lines.append("-" * 60)
    if report.canonical_mapping:
        lines.append("Canonical mapping:")
        for canonical in REQUIRED_COLUMNS:
            source = report.canonical_mapping.get(canonical)
            if source is None:
                continue
            if source == canonical:
                lines.append(f"  {canonical} <- {source}")
            else:
                lines.append(f"  {canonical} <- {source} (alias)")
    if report.missing_required_columns:
        lines.append(
            "Missing required columns: "
            + ", ".join(report.missing_required_columns)
        )
    else:
        lines.append("Missing required columns: none")
    lines.append("")

    lines.append("Missingness")
    lines.append("-" * 60)
    if report.missingness_by_column:
        for col in REQUIRED_COLUMNS:
            if col in report.missingness_by_column:
                lines.append(
                    f"  {col}: {report.missingness_by_column[col] * 100:.2f}% missing"
                )
    else:
        lines.append("  No missingness metrics available.")
    lines.append("")

    lines.append("Sanity Checks")
    lines.append("-" * 60)
    if report.ohlc_violations:
        for name, count in report.ohlc_violations.items():
            lines.append(f"  {name}: {count}")
    else:
        lines.append("  No OHLC violations detected.")
    lines.append(f"  Negative volume rows: {report.negative_volume_count}")
    lines.append(f"  Negative amount rows:  {report.negative_amount_count}")
    lines.append("")

    lines.append("Leakage Risk")
    lines.append("-" * 60)
    if report.leakage_warnings:
        for item in report.leakage_warnings:
            lines.append(f"  {item}")
    else:
        lines.append("  No leakage-risk column names detected.")
    lines.append("")

    warnings = [issue for issue in report.issues if issue.severity == "warning"]
    errors = [issue for issue in report.issues if issue.severity == "error"]

    lines.append("Issues")
    lines.append("-" * 60)
    if errors:
        for issue in errors:
            lines.append(f"  ERROR [{issue.code}]: {issue.message}")
    if warnings:
        for issue in warnings:
            lines.append(f"  WARN  [{issue.code}]: {issue.message}")
    if not warnings and not errors:
        lines.append("  No issues detected.")

    return "\n".join(lines)


def _infer_format(path: Path) -> FileFormat:
    suffix = path.suffix.lower()
    mapping: dict[str, FileFormat] = {
        ".csv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
    }
    if suffix not in mapping:
        raise ValueError(
            f"Cannot infer file format from extension '{suffix}'. "
            "Supported formats: csv, parquet, hdf5."
        )
    return mapping[suffix]


def _read_raw_frame(path: Path, fmt: FileFormat, hdf_key: str = "data") -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "hdf5":
        return pd.read_hdf(path, key=hdf_key)
    raise ValueError(f"Unsupported format: {fmt}")


def _normalize(name: str) -> str:
    return str(name).strip().lower()


def _build_canonical_mapping(
    columns: pd.Index,
) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    source_columns = [str(col) for col in columns]
    normalized_sources: dict[str, list[str]] = {}
    for source in source_columns:
        normalized_sources.setdefault(_normalize(source), []).append(source)

    mapping: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    missing: list[str] = []

    for canonical in REQUIRED_COLUMNS:
        candidates = [canonical, *COLUMN_ALIASES.get(canonical, [])]
        matched: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            for source in normalized_sources.get(_normalize(candidate), []):
                if source not in seen:
                    matched.append(source)
                    seen.add(source)
        if matched:
            mapping[canonical] = matched[0]
            if len(matched) > 1:
                ambiguous[canonical] = matched
        else:
            missing.append(canonical)

    return mapping, ambiguous, missing


def _count_non_monotonic_assets(df: pd.DataFrame) -> int:
    if "asset_id" not in df.columns or "datetime" not in df.columns:
        return 0

    non_monotonic = 0
    for _, group in df.groupby("asset_id", dropna=False):
        ts = pd.to_datetime(group["datetime"], errors="coerce")
        ts = ts.dropna()
        if len(ts) <= 1:
            continue
        if not ts.is_monotonic_increasing:
            non_monotonic += 1
    return non_monotonic


def _count_ohlc_violations(df: pd.DataFrame) -> dict[str, int]:
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        return {}

    open_px = pd.to_numeric(df["open"], errors="coerce")
    high_px = pd.to_numeric(df["high"], errors="coerce")
    low_px = pd.to_numeric(df["low"], errors="coerce")
    close_px = pd.to_numeric(df["close"], errors="coerce")

    return {
        "low_gt_high": int((low_px > high_px).sum()),
        "open_outside_range": int(((open_px < low_px) | (open_px > high_px)).sum()),
        "close_outside_range": int(((close_px < low_px) | (close_px > high_px)).sum()),
        "negative_price": int(
            ((open_px < 0) | (high_px < 0) | (low_px < 0) | (close_px < 0)).sum()
        ),
    }


def _count_negative_values(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    values = pd.to_numeric(df[column], errors="coerce")
    return int((values < 0).sum())


def _find_leaky_columns(columns: list[str]) -> list[str]:
    leaky = []
    for name in columns:
        lowered = _normalize(name)
        if any(pattern in lowered for pattern in LEAKY_NAME_PATTERNS):
            leaky.append(name)
    return leaky


def _status_label(report: DataValidationReport, strict: bool = False) -> str:
    if not report.valid_schema:
        return "invalid"
    if strict and report.has_warnings:
        return "invalid (strict)"
    if report.has_warnings:
        return "warning"
    return "valid"
