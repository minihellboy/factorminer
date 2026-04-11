"""Shared runtime evaluation helpers for strict factor recomputation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factorminer.core.factor_library import Factor
from factorminer.core.parser import try_parse
from factorminer.data.tensor_builder import TargetSpec, compute_targets
from factorminer.evaluation.metrics import (
    compute_factor_stats,
    compute_pairwise_correlation,
)

logger = logging.getLogger(__name__)

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

    def get_split(self, name: str) -> DatasetSplit:
        if name not in self.splits:
            raise KeyError(f"Unknown split: {name}")
        return self.splits[name]

    def get_target(self, name: str | None = None) -> np.ndarray:
        target_name = name or self.default_target
        if target_name in self.target_panels:
            return self.target_panels[target_name]
        return self.returns


@dataclass
class FactorEvaluationArtifact:
    """Recomputed signals and metrics for one factor."""

    factor_id: int
    name: str
    formula: str
    category: str
    parse_ok: bool
    signals_full: np.ndarray | None = None
    split_signals: dict[str, np.ndarray] = field(default_factory=dict)
    split_stats: dict[str, dict] = field(default_factory=dict)
    target_stats: dict[str, dict[str, dict]] = field(default_factory=dict)
    score_vector: dict | None = None
    research_metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.parse_ok and self.signals_full is not None and not self.error


def load_runtime_dataset(
    raw_df: pd.DataFrame,
    cfg,
) -> EvaluationDataset:
    """Load raw market data into a canonical evaluation dataset."""
    from factorminer.data.preprocessor import preprocess
    from factorminer.data.tensor_builder import TensorConfig, build_tensor

    raw_df = raw_df.copy()
    raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])

    target_specs = _resolve_target_specs(cfg)
    target_df = compute_targets(raw_df, target_specs)
    target_columns = [spec.column_name for spec in target_specs]
    merge_columns = ["datetime", "asset_id", *target_columns]
    processed_df = preprocess(raw_df)
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

    data_dict = {
        COLUMN_TO_FEATURE[column]: data_tensor[:, :, idx]
        for idx, column in enumerate(dataset.feature_names)
        if column in COLUMN_TO_FEATURE
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

    for split_name in ("train", "test"):
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
    )


def evaluate_factors(
    factors: Sequence[Factor],
    dataset: EvaluationDataset,
    signal_failure_policy: str = "reject",
    target_name: str | None = None,
) -> list[FactorEvaluationArtifact]:
    """Recompute factor signals and metrics across all dataset splits."""
    artifacts: list[FactorEvaluationArtifact] = []
    active_target_name = target_name or dataset.default_target
    active_returns = dataset.get_target(active_target_name)

    for factor in factors:
        artifact = FactorEvaluationArtifact(
            factor_id=factor.id,
            name=factor.name,
            formula=factor.formula,
            category=factor.category,
            parse_ok=False,
        )

        tree = try_parse(factor.formula)
        if tree is None:
            artifact.error = "Parse failure"
            artifacts.append(artifact)
            continue

        artifact.parse_ok = True

        try:
            signals = compute_tree_signals(
                tree,
                dataset.data_dict,
                active_returns.shape,
                signal_failure_policy=signal_failure_policy,
            )
        except Exception as exc:
            artifact.error = str(exc)
            artifacts.append(artifact)
            continue

        if signals is None or np.all(np.isnan(signals)):
            artifact.error = "Signal computation produced only NaN values"
            artifacts.append(artifact)
            continue

        artifact.signals_full = np.asarray(signals, dtype=np.float64)

        for split_name, split in dataset.splits.items():
            split_signals = artifact.signals_full[:, split.indices]
            artifact.split_signals[split_name] = split_signals
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

        artifacts.append(artifact)

    return artifacts


def compute_tree_signals(
    tree,
    data_dict: dict[str, np.ndarray],
    returns_shape: tuple[int, int],
    signal_failure_policy: str = "reject",
) -> np.ndarray:
    """Evaluate an expression tree under an explicit failure policy."""
    formula_str = tree.to_string()

    try:
        signals = tree.evaluate(data_dict)
    except Exception as exc:
        return _handle_signal_failure(
            formula_str=formula_str,
            returns_shape=returns_shape,
            signal_failure_policy=signal_failure_policy,
            cause=exc,
        )

    if signals is None or np.all(np.isnan(signals)):
        return _handle_signal_failure(
            formula_str=formula_str,
            returns_shape=returns_shape,
            signal_failure_policy=signal_failure_policy,
            cause=SignalComputationError("Signal computation produced only NaN values"),
        )

    return np.asarray(signals, dtype=np.float64)


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
    """Sort succeeded artifacts by split abs-IC and return the top-k subset."""
    succeeded = [a for a in artifacts if a.succeeded]
    succeeded.sort(
        key=lambda artifact: abs(
            artifact.split_stats[split_name].get("ic_abs_mean", 0.0)
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
        return list(COLUMN_TO_FEATURE.keys())

    resolved: list[str] = []
    for feature in config_features:
        if feature in FEATURE_TO_COLUMN:
            resolved.append(FEATURE_TO_COLUMN[feature])
            continue
        stripped = feature.lstrip("$")
        if stripped == "amt":
            stripped = "amount"
        resolved.append(stripped)
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
    seed = hash(formula_str) % (2**31)
    rng = np.random.RandomState(seed)
    signals = rng.randn(m, t).astype(np.float64)
    nan_mask = rng.random((m, t)) < 0.02
    signals[nan_mask] = np.nan
    return signals
