"""Paper-faithful benchmark and mining protocol contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TargetDefinition:
    """One named return target in the paper/runtime contract."""

    name: str
    entry_delay_bars: int
    holding_bars: int
    price_pair: str
    return_transform: str


@dataclass(frozen=True)
class ArtifactSchema:
    """Artifact inventory expected from one mining or benchmark run."""

    run_manifest: str = "run_manifest.json"
    factor_library: str = "factor_library.json"
    session: str = "session.json"
    session_log: str = "session_log.json"
    checkpoint_dir: str = "checkpoint"
    lifecycle_log: str = "factor_lifecycle.jsonl"
    batch_log: str = "mining_batches.jsonl"


@dataclass(frozen=True)
class TrainTestProtocol:
    """Temporal protocol and Top-K freeze contract."""

    train_period: list[str]
    test_period: list[str]
    freeze_top_k: int
    freeze_universe: str
    report_universes: list[str]


@dataclass(frozen=True)
class PaperProtocol:
    """Single object describing the paper-faithful runtime contract."""

    target_definitions: dict[str, TargetDefinition]
    default_target: str
    ic_metric: str
    redundancy_metric: str
    ic_threshold: float
    icir_threshold: float
    correlation_threshold: float
    replacement_ic_min: float
    replacement_ic_ratio: float
    benchmark_mode: str
    evaluation_backend: str
    signal_failure_policy: str
    train_test_protocol: TrainTestProtocol
    artifact_schema: ArtifactSchema = field(default_factory=ArtifactSchema)

    @classmethod
    def from_config(cls, cfg: Any) -> PaperProtocol:
        raw_targets = list(getattr(getattr(cfg, "data", None), "targets", []) or [])
        if not raw_targets:
            raw_targets = [
                {
                    "name": "paper",
                    "entry_delay_bars": 1,
                    "holding_bars": 1,
                    "price_pair": "open_to_close",
                    "return_transform": "simple",
                }
            ]

        targets = {
            str(target["name"]): TargetDefinition(
                name=str(target["name"]),
                entry_delay_bars=int(target.get("entry_delay_bars", 0)),
                holding_bars=int(target.get("holding_bars", 1)),
                price_pair=str(target.get("price_pair", "open_to_close")),
                return_transform=str(target.get("return_transform", "simple")),
            )
            for target in raw_targets
        }

        benchmark_cfg = getattr(cfg, "benchmark", None)
        data_cfg = getattr(cfg, "data", None)
        eval_cfg = getattr(cfg, "evaluation", None)
        mining_cfg = getattr(cfg, "mining", cfg)

        return cls(
            target_definitions=targets,
            default_target=str(getattr(data_cfg, "default_target", "paper")),
            ic_metric="mean_absolute_spearman_ic",
            redundancy_metric=str(
                getattr(eval_cfg, "redundancy_metric", getattr(cfg, "redundancy_metric", "spearman"))
            ),
            ic_threshold=float(getattr(mining_cfg, "ic_threshold", 0.04)),
            icir_threshold=float(getattr(mining_cfg, "icir_threshold", 0.5)),
            correlation_threshold=float(getattr(mining_cfg, "correlation_threshold", 0.5)),
            replacement_ic_min=float(getattr(mining_cfg, "replacement_ic_min", 0.10)),
            replacement_ic_ratio=float(getattr(mining_cfg, "replacement_ic_ratio", 1.3)),
            benchmark_mode=str(
                getattr(benchmark_cfg, "mode", getattr(cfg, "benchmark_mode", "paper"))
            ),
            evaluation_backend=str(getattr(eval_cfg, "backend", getattr(cfg, "backend", "numpy"))),
            signal_failure_policy=str(
                getattr(
                    eval_cfg,
                    "signal_failure_policy",
                    getattr(cfg, "signal_failure_policy", "reject"),
                )
            ),
            train_test_protocol=TrainTestProtocol(
                train_period=list(getattr(data_cfg, "train_period", ["2024-01-01", "2024-12-31"])),
                test_period=list(getattr(data_cfg, "test_period", ["2025-01-01", "2025-12-31"])),
                freeze_top_k=int(getattr(benchmark_cfg, "freeze_top_k", 40)),
                freeze_universe=str(getattr(benchmark_cfg, "freeze_universe", "CSI500")),
                report_universes=list(getattr(benchmark_cfg, "report_universes", ["CSI500"])),
            ),
        )

    @property
    def target_stack(self) -> list[str]:
        return list(self.target_definitions.keys())

    def admission_contract(self) -> dict[str, float]:
        return {
            "ic_threshold": self.ic_threshold,
            "icir_threshold": self.icir_threshold,
            "correlation_threshold": self.correlation_threshold,
        }

    def replacement_contract(self) -> dict[str, float]:
        return {
            "replacement_ic_min": self.replacement_ic_min,
            "replacement_ic_ratio": self.replacement_ic_ratio,
        }

    def runtime_contract(self) -> dict[str, Any]:
        return {
            "default_target": self.default_target,
            "target_stack": self.target_stack,
            "targets": {name: asdict(target) for name, target in self.target_definitions.items()},
            "metrics": {
                "ic": self.ic_metric,
                "redundancy": self.redundancy_metric,
            },
            "admission": self.admission_contract(),
            "replacement": self.replacement_contract(),
            "train_test_protocol": asdict(self.train_test_protocol),
            "artifact_schema": asdict(self.artifact_schema),
            "benchmark_mode": self.benchmark_mode,
            "evaluation_backend": self.evaluation_backend,
            "signal_failure_policy": self.signal_failure_policy,
        }
