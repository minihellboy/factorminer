"""Typed runtime context over the single hierarchical configuration model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MiningRunContext:
    """Per-run state that is not configuration.

    Output locations and materialized target panels belong to an execution, not
    to the reusable YAML configuration.  Optional policy fields are explicit
    runtime overrides used by mock and benchmark runs.
    """

    output_dir: Path
    target_panels: Mapping[str, Any] | None = None
    target_horizons: Mapping[str, int] | None = None
    signal_failure_policy: str | None = None
    benchmark_mode: str | None = None
    model_co_optimize: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.signal_failure_policy not in {None, "reject", "synthetic", "raise"}:
            raise ValueError(
                "signal_failure_policy must be one of: reject, synthetic, raise"
            )
        if self.benchmark_mode not in {None, "paper", "research"}:
            raise ValueError("benchmark_mode must be one of: paper, research")
        if self.target_horizons is not None and any(
            int(value) < 1 for value in self.target_horizons.values()
        ):
            raise ValueError("target horizons must be positive integers")


class MiningSettings:
    """Read-only mining view over canonical config plus per-run context.

    This is deliberately not another configuration model.  Values are resolved
    from the owning section of the hierarchical ``Config`` and only fall back
    to flat attributes for the temporary compatibility path used by older
    programmatic callers.
    """

    __slots__ = ("config", "run_context")

    def __init__(self, config: Any, run_context: MiningRunContext | None = None) -> None:
        self.config = config
        self.run_context = run_context or build_run_context(config)

    def _value(self, section: str, name: str, default: Any) -> Any:
        owner = getattr(self.config, section, None)
        if owner is not None and hasattr(owner, name):
            return getattr(owner, name)
        return getattr(self.config, name, default)

    @property
    def target_library_size(self) -> int:
        return int(self._value("mining", "target_library_size", 110))

    @property
    def batch_size(self) -> int:
        return int(self._value("mining", "batch_size", 40))

    @property
    def max_iterations(self) -> int:
        return int(self._value("mining", "max_iterations", 200))

    @property
    def ic_threshold(self) -> float:
        return float(self._value("mining", "ic_threshold", 0.04))

    @property
    def icir_threshold(self) -> float:
        return float(self._value("mining", "icir_threshold", 0.5))

    @property
    def correlation_threshold(self) -> float:
        return float(self._value("mining", "correlation_threshold", 0.5))

    @property
    def replacement_ic_min(self) -> float:
        return float(self._value("mining", "replacement_ic_min", 0.10))

    @property
    def replacement_ic_ratio(self) -> float:
        return float(self._value("mining", "replacement_ic_ratio", 1.3))

    @property
    def fast_screen_assets(self) -> int:
        return int(self._value("evaluation", "fast_screen_assets", 100))

    @property
    def num_workers(self) -> int:
        return int(self._value("evaluation", "num_workers", 1))

    @property
    def backend(self) -> str:
        return str(self._value("evaluation", "backend", "numpy"))

    @property
    def gpu_device(self) -> str:
        return str(self._value("evaluation", "gpu_device", "cuda:0"))

    @property
    def redundancy_metric(self) -> str:
        return str(self._value("evaluation", "redundancy_metric", "spearman"))

    @property
    def signal_failure_policy(self) -> str:
        override = self.run_context.signal_failure_policy
        if override is not None:
            return override
        return str(self._value("evaluation", "signal_failure_policy", "reject"))

    @property
    def memory_policy(self) -> str:
        return str(self._value("memory", "policy", "paper"))

    @property
    def memory_regime_lookback_window(self) -> int:
        return int(self._value("memory", "regime_lookback_window", 60))

    @property
    def output_dir(self) -> str:
        return str(self.run_context.output_dir)

    @property
    def target_panels(self) -> Mapping[str, Any] | None:
        if self.run_context.target_panels is not None:
            return self.run_context.target_panels
        return getattr(self.config, "target_panels", None)

    @property
    def target_horizons(self) -> Mapping[str, int] | None:
        if self.run_context.target_horizons is not None:
            return self.run_context.target_horizons
        return getattr(self.config, "target_horizons", None)

    @property
    def research(self) -> Any:
        return getattr(self.config, "research", None)

    @property
    def research_knowledge_retrieval_limit(self) -> int:
        return int(self._value("research", "knowledge_retrieval_limit", 4))

    @property
    def benchmark_mode(self) -> str:
        override = self.run_context.benchmark_mode
        if override is not None:
            return override
        return str(self._value("benchmark", "mode", "paper"))

    @property
    def model_co_optimize(self) -> Mapping[str, Any] | None:
        if self.run_context.model_co_optimize is not None:
            return self.run_context.model_co_optimize
        direct = getattr(self.config, "model_co_optimize", None)
        if direct is not None:
            return direct
        raw = getattr(self.config, "_raw", {})
        value = raw.get("model_co_optimize") if isinstance(raw, Mapping) else None
        return value if isinstance(value, Mapping) else None


def build_run_context(
    config: Any,
    *,
    output_dir: str | Path | None = None,
    dataset: Any | None = None,
    mock: bool = False,
    signal_failure_policy: str | None = None,
    benchmark_mode: str | None = None,
) -> MiningRunContext:
    """Build canonical per-run state from a config and optional dataset."""
    resolved_output = Path(
        output_dir if output_dir is not None else getattr(config, "output_dir", "./output")
    )
    target_panels = None
    target_horizons = None
    if dataset is not None:
        target_panels = dict(getattr(dataset, "target_panels", {}) or {})
        target_horizons = {
            str(name): max(int(getattr(spec, "holding_bars", 1)), 1)
            for name, spec in (getattr(dataset, "target_specs", {}) or {}).items()
        }

    raw = getattr(config, "_raw", {})
    model_co_optimize = raw.get("model_co_optimize") if isinstance(raw, Mapping) else None
    if not isinstance(model_co_optimize, Mapping):
        model_co_optimize = None

    return MiningRunContext(
        output_dir=resolved_output,
        target_panels=target_panels,
        target_horizons=target_horizons,
        signal_failure_policy=(
            signal_failure_policy
            if signal_failure_policy is not None
            else ("synthetic" if mock else None)
        ),
        benchmark_mode=benchmark_mode,
        model_co_optimize=model_co_optimize,
    )
