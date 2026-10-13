"""Shared runtime evaluation helpers for strict factor recomputation."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from factorminer.core.expression_plan import (
    OPERATOR_SEMANTICS_VERSION,
    compile_batch,
    compile_tree,
)
from factorminer.core.factor_library import Factor
from factorminer.core.parser import try_parse
from factorminer.data.tensor_builder import TargetSpec, compute_targets
from factorminer.domain.signal_ref import SignalKey, SignalRef, SplitSignalView
from factorminer.evaluation.metrics import (
    compute_factor_stats,
    compute_pairwise_correlation,
)
from factorminer.evaluation.signal_store import SplitSignalStore, split_selectors

logger = logging.getLogger(__name__)

# Static paper defaults kept for backwards-compatible imports. Prefer the
# helpers below so extra registered leaves (fundamentals, futures, ...) map too.
FEATURE_TO_COLUMN = {
    "$open": "open",
    "$high": "high",
    "$low": "low",
    "$close": "close",
    "$volume": "volume",
    "$amt": "amount",
    "$vwap": "vwap",
    "$returns": "returns",
}

COLUMN_TO_FEATURE = {value: key for key, value in FEATURE_TO_COLUMN.items()}


def _feature_to_column(name: str) -> str:
    """Map a DSL leaf (or bare name) onto a panel DataFrame column."""
    from factorminer.core.types import feature_to_column

    return feature_to_column(name)


def _column_to_feature(column: str) -> str:
    """Map a panel column onto a DSL leaf name."""
    from factorminer.core.types import column_to_feature

    if column in COLUMN_TO_FEATURE:
        return COLUMN_TO_FEATURE[column]
    return column_to_feature(column)


class SignalComputationError(RuntimeError):
    """Raised when a factor cannot be recomputed under strict policies."""


@dataclass
class DatasetSplit:
    """One temporal view into the evaluation dataset."""

    name: str
    indices: np.ndarray
    timestamps: np.ndarray
    returns: np.ndarray
    target_returns: dict[str, np.ndarray] = field(default_factory=dict)
    default_target: str = "target"

    @property
    def size(self) -> int:
        return int(len(self.indices))

    def get_target(self, name: str | None = None) -> np.ndarray:
        target_name = name or self.default_target
        if target_name in self.target_returns:
            return self.target_returns[target_name]
        return self.returns


@dataclass
class EvaluationDataset:
    """Canonical dataset used for analysis commands."""

    data_dict: dict[str, np.ndarray]
    data_tensor: np.ndarray
    returns: np.ndarray
    timestamps: np.ndarray
    asset_ids: np.ndarray
    splits: dict[str, DatasetSplit]
    processed_df: pd.DataFrame = field(repr=False)
    target_panels: dict[str, np.ndarray] = field(default_factory=dict)
    target_specs: dict[str, TargetSpec] = field(default_factory=dict)
    default_target: str = "target"
    preprocessing: dict = field(default_factory=dict)

    def get_split(self, name: str) -> DatasetSplit:
        if name not in self.splits:
            raise KeyError(f"Unknown split: {name}")
        return self.splits[name]

    def get_target(self, name: str | None = None) -> np.ndarray:
        target_name = name or self.default_target
        if target_name in self.target_panels:
            return self.target_panels[target_name]
        return self.returns


def build_runtime_dataset_from_arrays(
    data: Mapping[str, np.ndarray],
    returns: np.ndarray,
    *,
    feature_order: Sequence[str] | None = None,
    target_panels: Mapping[str, np.ndarray] | None = None,
    target_specs: Mapping[str, TargetSpec] | None = None,
    default_target: str = "target",
    split_indices: Mapping[str, Sequence[int] | np.ndarray] | None = None,
    timestamps: Sequence | np.ndarray | None = None,
    asset_ids: Sequence | np.ndarray | None = None,
) -> EvaluationDataset:
    """Build the canonical runtime dataset from aligned numerical panels.

    This is the array-native counterpart to :func:`load_runtime_dataset` and
    is used by synthetic benchmarks and programmatic callers.  All feature and
    target panels must share the canonical ``(assets, periods)`` orientation.
    """
    returns_array = np.asarray(returns, dtype=np.float64)
    if returns_array.ndim != 2:
        raise ValueError(
            f"returns must have shape (assets, periods); got {returns_array.shape}"
        )
    asset_count, period_count = returns_array.shape

    ordered_features = list(feature_order or data.keys())
    if not ordered_features:
        raise ValueError("at least one feature panel is required")
    missing = [name for name in ordered_features if name not in data]
    if missing:
        raise ValueError(f"feature_order contains missing panels: {missing}")

    data_dict: dict[str, np.ndarray] = {}
    for name in ordered_features:
        panel = np.asarray(data[name], dtype=np.float64)
        if panel.shape != returns_array.shape:
            raise ValueError(
                f"feature {name!r} has shape {panel.shape}; expected {returns_array.shape}"
            )
        data_dict[str(name)] = panel
    data_tensor = np.stack([data_dict[name] for name in ordered_features], axis=2)

    targets = {
        str(name): np.asarray(panel, dtype=np.float64)
        for name, panel in (target_panels or {default_target: returns_array}).items()
    }
    if default_target not in targets:
        raise ValueError(f"default target {default_target!r} is not present in target_panels")
    for name, panel in targets.items():
        if panel.shape != returns_array.shape:
            raise ValueError(
                f"target {name!r} has shape {panel.shape}; expected {returns_array.shape}"
            )

    timestamp_array = (
        np.arange(period_count) if timestamps is None else np.asarray(timestamps)
    )
    asset_array = np.arange(asset_count) if asset_ids is None else np.asarray(asset_ids)
    if timestamp_array.shape != (period_count,):
        raise ValueError(
            f"timestamps must have length {period_count}; got {timestamp_array.shape}"
        )
    if asset_array.shape != (asset_count,):
        raise ValueError(f"asset_ids must have length {asset_count}; got {asset_array.shape}")

    partitions = split_indices or {"full": np.arange(period_count)}
    splits: dict[str, DatasetSplit] = {}
    for name, raw_indices in partitions.items():
        indices = np.asarray(raw_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError(f"split {name!r} indices must be one-dimensional")
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= period_count):
            raise ValueError(f"split {name!r} contains out-of-range period indices")
        splits[str(name)] = DatasetSplit(
            name=str(name),
            indices=indices,
            timestamps=timestamp_array[indices],
            returns=returns_array[:, indices],
            target_returns={target: panel[:, indices] for target, panel in targets.items()},
            default_target=default_target,
        )

    return EvaluationDataset(
        data_dict=data_dict,
        data_tensor=data_tensor,
        returns=returns_array,
        timestamps=timestamp_array,
        asset_ids=asset_array,
        splits=splits,
        processed_df=pd.DataFrame(),
        target_panels=targets,
        target_specs=dict(target_specs or {}),
        default_target=default_target,
    )


@dataclass
class FactorEvaluationArtifact:
    """Recomputed signals and metrics for one factor."""

    factor_id: int
    name: str
    formula: str
    category: str
    parse_ok: bool
    signals_full: np.ndarray | None = None
    split_signals: Mapping[str, np.ndarray] = field(default_factory=dict)
    split_stats: dict[str, dict] = field(default_factory=dict)
    target_stats: dict[str, dict[str, dict]] = field(default_factory=dict)
    score_vector: dict | None = None
    research_metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""
    signals_computed: bool = False
    signal_ref: SignalRef | None = None

    @property
    def succeeded(self) -> bool:
        return self.parse_ok and (self.signals_computed or self.signals_full is not None) and not self.error

    def release_signals(self) -> None:
        """Free signal panels after their metrics and benchmark selection are recorded.

        Scores and ``signal_ref.key`` (the signal's provenance) are retained.
        """
        self.signals_computed = self.signals_computed or self.signals_full is not None
        self.signals_full = None
        if isinstance(self.split_signals, SplitSignalView):
            self.split_signals.clear()
            self.split_signals = {}
        else:
            self.split_signals.clear()  # type: ignore[attr-defined]


UNIVERSE_MASK_COLUMN = "in_universe"


def _resolve_preprocess_config(value: Any) -> Any:
    """Return the configured ``PreprocessConfig``; defaults when none is declared."""
    from dataclasses import fields

    from factorminer.data.preprocessor import PreprocessConfig

    if value is None:
        return PreprocessConfig()
    if isinstance(value, PreprocessConfig):
        return value
    if isinstance(value, Mapping):
        known = {item.name for item in fields(PreprocessConfig)}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown preprocess settings: {unknown}")
        return PreprocessConfig(**dict(value))
    raise TypeError(
        f"preprocess config must be a PreprocessConfig or mapping, got {type(value).__name__}"
    )


def _universe_mask(raw_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.Series | None:
    """Align an optional point-in-time membership column to *target_df* rows.

    Rows outside the universe keep their prices, so time-series operators see
    full histories, but their targets are masked out of every metric.
    """
    if UNIVERSE_MASK_COLUMN not in raw_df.columns:
        return None
    membership = raw_df[["datetime", "asset_id", UNIVERSE_MASK_COLUMN]].copy()
    flags = pd.to_numeric(membership[UNIVERSE_MASK_COLUMN], errors="coerce")
    if flags.isna().any() or not flags.isin([0, 1]).all():
        raise ValueError(f"{UNIVERSE_MASK_COLUMN} must contain only 0/1 or boolean values")
    membership[UNIVERSE_MASK_COLUMN] = flags.astype(bool)
    aligned = target_df[["datetime", "asset_id"]].merge(
        membership, on=["datetime", "asset_id"], how="left", validate="one_to_one"
    )
    return pd.Series(
        aligned[UNIVERSE_MASK_COLUMN].fillna(False).to_numpy(dtype=bool), index=target_df.index
    )


def load_runtime_dataset(
    raw_df: pd.DataFrame,
    cfg,
) -> EvaluationDataset:
    """Load raw market data into a canonical evaluation dataset."""
    from dataclasses import asdict

    from factorminer.data.preprocessor import preprocess
    from factorminer.data.tensor_builder import TensorConfig, build_tensor

    if not pd.api.types.is_datetime64_any_dtype(raw_df["datetime"]):
        raw_df = raw_df.copy()
        raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])

    target_specs = _resolve_target_specs(cfg)
    preprocess_config = _resolve_preprocess_config(getattr(cfg, "preprocess", None))
    processed_df = preprocess(raw_df, preprocess_config)
    # Targets count bars on the grid formulas see: timestamps removed by the
    # quality check must not be skipped over by a forward-return offset.
    kept = raw_df["datetime"].isin(processed_df["datetime"].unique())
    target_input = raw_df if kept.all() else raw_df[kept]
    target_df = compute_targets(target_input, target_specs)
    target_columns = [spec.column_name for spec in target_specs]
    merge_columns = ["datetime", "asset_id", *target_columns]
    universe_mask = _universe_mask(target_input, target_df)
    if universe_mask is not None:
        target_df.loc[~universe_mask, target_columns] = np.nan
    processed_df = processed_df.merge(
        target_df[merge_columns],
        on=["datetime", "asset_id"],
        how="left",
    )
    processed_df = processed_df.sort_values(["datetime", "asset_id"]).reset_index(drop=True)

    feature_columns = _resolve_feature_columns(getattr(cfg.data, "features", []))
    tensor_cfg = TensorConfig(
        features=feature_columns,
        backend="numpy",
        dtype="float64",
        target_columns=target_columns,
        default_target=_target_column_for_name(cfg.data.default_target, target_specs),
    )
    dataset = build_tensor(processed_df, tensor_cfg)

    data_tensor = np.asarray(dataset.data, dtype=np.float64)
    returns = np.asarray(dataset.target, dtype=np.float64)
    target_panels = {
        spec.name: np.asarray(dataset.targets[spec.column_name], dtype=np.float64)
        for spec in target_specs
        if spec.column_name in dataset.targets
    }
    timestamps = pd.to_datetime(dataset.timestamps).to_numpy()
    asset_ids = np.asarray(dataset.asset_ids)

    if returns.ndim != 2:
        raise ValueError("Runtime dataset target must be a 2-D (M, T) array")

    preprocessing = {
        "pipeline": "factorminer.data.preprocessor.preprocess",
        "config": asdict(preprocess_config),
        "feature_columns": list(feature_columns),
        "tensor": {"backend": "numpy", "dtype": "float64"},
    }
    if universe_mask is not None:
        # Recorded only when present, so panels without a mask keep their digest.
        preprocessing["universe_mask"] = UNIVERSE_MASK_COLUMN

    data_dict = {
        _column_to_feature(column): data_tensor[:, :, idx]
        for idx, column in enumerate(dataset.feature_names)
    }

    splits = {
        "train": _build_named_split(
            "train",
            timestamps,
            returns,
            target_panels,
            cfg.data.default_target,
            start=cfg.data.train_period[0],
            end=cfg.data.train_period[1],
        ),
        "test": _build_named_split(
            "test",
            timestamps,
            returns,
            target_panels,
            cfg.data.default_target,
            start=cfg.data.test_period[0],
            end=cfg.data.test_period[1],
        ),
        "full": DatasetSplit(
            name="full",
            indices=np.arange(len(timestamps)),
            timestamps=timestamps,
            returns=returns,
            target_returns=target_panels,
            default_target=cfg.data.default_target,
        ),
    }

    validation_period = list(getattr(cfg.data, "validation_period", []))
    if validation_period:
        splits["validation"] = _build_named_split(
            "validation",
            timestamps,
            returns,
            target_panels,
            cfg.data.default_target,
            start=validation_period[0],
            end=validation_period[1],
        )

    for split_name in ("train", "validation", "test"):
        if split_name not in splits:
            continue
        if splits[split_name].size == 0:
            raise ValueError(
                f"{split_name} split is empty for configured period "
                f"{getattr(cfg.data, f'{split_name}_period')}"
            )

    return EvaluationDataset(
        data_dict=data_dict,
        data_tensor=data_tensor,
        returns=returns,
        timestamps=timestamps,
        asset_ids=asset_ids,
        splits=splits,
        processed_df=processed_df,
        target_panels=target_panels,
        target_specs={spec.name: spec for spec in target_specs},
        default_target=cfg.data.default_target,
        preprocessing=preprocessing,
    )


def evaluate_factors(
    factors: Sequence[Factor],
    dataset: EvaluationDataset,
    signal_failure_policy: str = "reject",
    target_name: str | None = None,
    *,
    retain_splits: Sequence[str] | None = None,
    signal_dtype: str = "float64",
    plan_batch_size: int = 16,
    signal_store: SplitSignalStore | None = None,
) -> list[FactorEvaluationArtifact]:
    """Recompute metrics on float64 signals, retaining only requested panels.

    Formulas are compiled in chunks of ``plan_batch_size`` so subexpressions
    shared within a chunk are evaluated once.

    With ``retain_splits=None`` the full panel and split views are retained for
    general analysis. Benchmark callers can request only the splits needed for
    admission or reporting, then release them once selection is complete.

    With ``signal_store`` the retained splits are the store's, and artifacts
    hold :class:`SignalRef` handles instead of arrays; the store's byte budget
    then bounds resident signal memory regardless of candidate count.
    """
    artifacts: list[FactorEvaluationArtifact] = []
    active_target_name = target_name or dataset.default_target
    active_returns = dataset.get_target(active_target_name)
    dtype = np.dtype(signal_dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("signal_dtype must be float32 or float64")
    requested = None if retain_splits is None else set(retain_splits)
    if signal_store is not None:
        if requested is not None and requested != set(signal_store.retain_splits):
            raise ValueError("retain_splits must match the signal store's retained splits")
        requested = set(signal_store.retain_splits)
    if requested is not None and requested - dataset.splits.keys():
        raise ValueError(f"Unknown retained splits: {sorted(requested - dataset.splits.keys())}")

    selectors = split_selectors(dataset.splits)

    for start in range(0, len(factors), max(int(plan_batch_size), 1)):
        chunk = factors[start : start + max(int(plan_batch_size), 1)]
        chunk_artifacts: list[FactorEvaluationArtifact] = []
        parsed: list[tuple[FactorEvaluationArtifact, Any]] = []
        for factor in chunk:
            artifact = FactorEvaluationArtifact(
                factor_id=factor.id,
                name=factor.name,
                formula=factor.formula,
                category=factor.category,
                parse_ok=False,
            )
            chunk_artifacts.append(artifact)
            tree = try_parse(factor.formula)
            if tree is None:
                artifact.error = "Parse failure"
                continue
            artifact.parse_ok = True
            parsed.append((artifact, tree))

        outcomes = compute_batch_signals(
            [tree for _, tree in parsed],
            dataset.data_dict,
            active_returns.shape,
            signal_failure_policy=signal_failure_policy,
        )
        for index, signals, error in outcomes:
            artifact, tree = parsed[index]
            if error is not None or signals is None:
                artifact.error = str(error)
                continue
            if signal_store is not None:
                key = SignalKey(
                    dataset_digest=signal_store.dataset_digest,
                    formula_digest=compile_tree(tree).digest,
                    operator_version=OPERATOR_SEMANTICS_VERSION,
                    backend="numpy",
                    dtype=signal_store.dtype.name,
                )
                signal_store.put(key, signals)
                artifact.signal_ref = SignalRef(key, signal_store, signal_store.retain_splits)
                artifact.split_signals = SplitSignalView(artifact.signal_ref)
                requested_for_artifact: set[str] | None = set()
            else:
                requested_for_artifact = requested
            _record_split_signals(
                artifact,
                signals,
                dataset,
                selectors=selectors,
                requested=requested_for_artifact,
                dtype=dtype,
                active_target_name=active_target_name,
            )
        artifacts.extend(chunk_artifacts)

    return artifacts


def _record_split_signals(
    artifact: FactorEvaluationArtifact,
    signals: np.ndarray,
    dataset: EvaluationDataset,
    *,
    selectors: Mapping[str, slice | np.ndarray],
    requested: set[str] | None,
    dtype: np.dtype,
    active_target_name: str,
) -> None:
    artifact.signals_computed = True
    if requested is None:
        artifact.signals_full = signals
    retained = cast(dict[str, np.ndarray], artifact.split_signals)

    for split_name, split in dataset.splits.items():
        split_signals = signals[:, selectors[split_name]]
        if requested is None:
            retained[split_name] = split_signals
        elif split_name in requested:
            retained[split_name] = np.array(
                split_signals, dtype=dtype, copy=True, order="C"
            )
        active_split_target = split.get_target(active_target_name)
        active_stats = compute_factor_stats(split_signals, active_split_target)
        artifact.split_stats[split_name] = active_stats
        artifact.target_stats[split_name] = {}
        for available_target_name, split_target in split.target_returns.items():
            artifact.target_stats[split_name][available_target_name] = (
                active_stats
                if available_target_name == active_target_name
                else compute_factor_stats(split_signals, split_target)
            )


def compute_batch_signals(
    trees: Sequence,
    data_dict: Mapping[str, np.ndarray],
    returns_shape: tuple[int, int],
    signal_failure_policy: str = "reject",
) -> Iterator[tuple[int, np.ndarray | None, Exception | None]]:
    """Evaluate trees through one shared plan, yielding per-formula outcomes.

    Each item is ``(index, signals, None)`` or ``(index, None, error)`` where
    ``error`` is what :func:`compute_tree_signals` would have raised for that
    tree alone. Shared subexpressions are evaluated once and released after
    their last consumer.
    """
    if not trees:
        return
    batch = compile_batch(trees)
    for index, value in batch.iter_outputs(data_dict):
        try:
            signals = _finalize_signals(
                trees[index].to_string(), value, returns_shape, signal_failure_policy
            )
        except Exception as exc:  # noqa: BLE001 - reported per formula
            yield index, None, exc
            continue
        yield index, signals, None


def compute_tree_signals(
    tree,
    data_dict: dict[str, np.ndarray],
    returns_shape: tuple[int, int],
    signal_failure_policy: str = "reject",
) -> np.ndarray:
    """Evaluate an expression tree under an explicit failure policy."""
    try:
        value: np.ndarray | Exception = compile_tree(tree).evaluate(data_dict)
    except Exception as exc:  # noqa: BLE001 - resolved by the failure policy
        value = exc
    return _finalize_signals(tree.to_string(), value, returns_shape, signal_failure_policy)


def _finalize_signals(
    formula_str: str,
    value: np.ndarray | BaseException | None,
    returns_shape: tuple[int, int],
    signal_failure_policy: str,
) -> np.ndarray:
    if isinstance(value, BaseException):
        if not isinstance(value, Exception):
            raise value
        return _handle_signal_failure(
            formula_str=formula_str,
            returns_shape=returns_shape,
            signal_failure_policy=signal_failure_policy,
            cause=value,
        )

    if value is None or np.all(np.isnan(value)):
        return _handle_signal_failure(
            formula_str=formula_str,
            returns_shape=returns_shape,
            signal_failure_policy=signal_failure_policy,
            cause=SignalComputationError("Signal computation produced only NaN values"),
        )

    return np.asarray(value, dtype=np.float64)


def compute_correlation_matrix(
    artifacts: Sequence[FactorEvaluationArtifact],
    split_name: str,
) -> np.ndarray:
    """Compute a true pairwise factor correlation matrix on one split."""
    selected = [a for a in artifacts if a.succeeded]
    n = len(selected)
    matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            corr = compute_pairwise_correlation(
                selected[i].split_signals[split_name],
                selected[j].split_signals[split_name],
            )
            matrix[i, j] = corr
            matrix[j, i] = corr

    return matrix


def select_top_k(
    artifacts: Sequence[FactorEvaluationArtifact],
    split_name: str,
    top_k: int | None = None,
) -> list[FactorEvaluationArtifact]:
    """Sort succeeded artifacts by split paper IC and return the top-k subset."""
    succeeded = [a for a in artifacts if a.succeeded]
    succeeded.sort(
        key=lambda artifact: abs(
            artifact.split_stats[split_name].get(
                "ic_paper_mean",
                artifact.split_stats[split_name].get("ic_abs_mean", 0.0),
            )
        ),
        reverse=True,
    )
    if top_k is None or top_k >= len(succeeded):
        return succeeded
    return succeeded[:top_k]


def summarize_failures(
    artifacts: Sequence[FactorEvaluationArtifact],
) -> list[str]:
    """Return human-readable failure summaries."""
    return [
        f"{artifact.name or artifact.factor_id}: {artifact.error}"
        for artifact in artifacts
        if not artifact.succeeded
    ]


def resolve_split_for_fit_eval(period: str) -> str:
    """Map fit/eval CLI period values to runtime split names."""
    return "full" if period == "both" else period


def analysis_split_names(period: str) -> list[str]:
    """Map analysis CLI period values to one or two runtime split names."""
    if period == "both":
        return ["train", "test"]
    return [period]


def _resolve_feature_columns(config_features: Sequence[str]) -> list[str]:
    if not config_features:
        return list(COLUMN_TO_FEATURE.values())

    resolved: list[str] = []
    seen: set[str] = set()
    for feature in config_features:
        col = _feature_to_column(str(feature))
        if col not in seen:
            seen.add(col)
            resolved.append(col)
    return resolved


def _build_named_split(
    name: str,
    timestamps: np.ndarray,
    returns: np.ndarray,
    target_panels: dict[str, np.ndarray],
    default_target: str,
    start: str,
    end: str,
) -> DatasetSplit:
    ts = pd.to_datetime(timestamps)
    mask = (ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))
    indices = np.where(mask)[0]
    return DatasetSplit(
        name=name,
        indices=indices,
        timestamps=timestamps[indices],
        returns=returns[:, indices],
        target_returns={
            target_name: panel[:, indices]
            for target_name, panel in target_panels.items()
        },
        default_target=default_target,
    )


def _resolve_target_specs(cfg) -> list[TargetSpec]:
    raw_targets = getattr(cfg.data, "targets", None) or [
        {
            "name": "paper",
            "entry_delay_bars": 1,
            "holding_bars": 1,
            "price_pair": "open_to_close",
            "return_transform": "simple",
        }
    ]
    return [
        TargetSpec(
            name=str(target["name"]),
            entry_delay_bars=int(target.get("entry_delay_bars", 0)),
            holding_bars=int(target.get("holding_bars", 1)),
            price_pair=str(target.get("price_pair", "open_to_close")),
            return_transform=str(target.get("return_transform", "simple")),
        )
        for target in raw_targets
    ]


def _target_column_for_name(target_name: str, specs: Sequence[TargetSpec]) -> str:
    for spec in specs:
        if spec.name == target_name:
            return spec.column_name
    return "target"


def _handle_signal_failure(
    formula_str: str,
    returns_shape: tuple[int, int],
    signal_failure_policy: str,
    cause: Exception,
) -> np.ndarray:
    if signal_failure_policy == "raise":
        raise cause

    if signal_failure_policy == "reject":
        raise SignalComputationError(
            f"Expression evaluation failed for '{formula_str}': {cause}"
        ) from cause

    if signal_failure_policy != "synthetic":
        raise ValueError(
            "signal_failure_policy must be one of: reject, synthetic, raise"
        )

    logger.warning(
        "Expression evaluation failed for '%s': %s — falling back to synthetic signals",
        formula_str,
        cause,
    )
    return generate_synthetic_signals(formula_str, returns_shape)


def generate_synthetic_signals(
    formula_str: str,
    returns_shape: tuple[int, int],
) -> np.ndarray:
    """Deterministic pseudo-signals for demo/mock workflows."""
    m, t = returns_shape
    digest = hashlib.sha256(formula_str.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**31)
    rng = np.random.RandomState(seed)
    signals = rng.randn(m, t).astype(np.float64)
    nan_mask = rng.random((m, t)) < 0.02
    signals[nan_mask] = np.nan
    return signals
