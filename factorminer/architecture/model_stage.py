"""RD-Agent(Q)-style optional factor+model co-optimization loop stage.

Implements the "co-optimization" half of RD-Agent(Q) (Microsoft Research +
HKUST, NeurIPS 2025) as an opt-in, side-effect-free diagnostic: periodically
(every `every_n_admissions` cumulative library admissions) fit a downstream
Ridge/Lasso/XGBoost model (`factorminer.evaluation.model_zoo`) on the current
factor library's signals against realized returns, evaluate held-out IC and
Sharpe, and rank each admitted factor's marginal contribution to that model.

This stage NEVER mutates factor admission decisions -- it only writes a
JSON-serializable `co_optimization_report` into
`IterationPayload.stage_metrics`, which `LoopExecutionService.build_stats`
already merges into the loop's per-iteration stats/manifest output
(`core/loop_services.py::LoopExecutionService.build_stats`). It defaults to
disabled, per this repo's architecture contributor rule 4 (prefer a new
service/stage over growing Ralph/Helix; new optional behavior defaults off).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from factorminer.architecture.stages import IterationPayload, LoopStage
from factorminer.evaluation.model_zoo import ModelZooConfig, ModelZooEvaluator

logger = logging.getLogger(__name__)

_STAGE_CONFIG_KEYS = ("enabled", "every_n_admissions", "min_factors")


@dataclass(frozen=True)
class ModelCoOptimizeConfig:
    """Config gate for the optional periodic factor+model co-optimization pass.

    Attributes
    ----------
    enabled : bool
        Master opt-in switch. Defaults to False (off) -- the pass never runs
        and admission decisions are never affected unless a caller flips
        this explicitly.
    every_n_admissions : int
        Run the co-optimization pass once at least this many *new* factor
        admissions have accumulated since the last pass.
    min_factors : int
        Minimum library size before a pass is attempted.
    model_zoo_config : ModelZooConfig
        Downstream-model configuration forwarded to `ModelZooEvaluator`.
    """

    enabled: bool = False
    every_n_admissions: int = 5
    min_factors: int = 2
    model_zoo_config: ModelZooConfig = field(default_factory=ModelZooConfig)


class ModelCoOptimizeStage(LoopStage):
    """Optional loop stage: periodically co-evaluate the library with a downstream model.

    Subclasses `LoopStage` directly rather than wrapping one of the thin
    single-field stage adapters in `architecture/stages.py`
    (`RetrieveStage`/`GenerateStage`/`EvaluateStage`/`LibraryUpdateStage`/
    `DistillStage`): each of those binds an injected callable's return value
    onto exactly one dedicated `IterationPayload` field (`library_state`,
    `candidates`, `results`, `admitted_results`). This stage has no
    dedicated field of its own -- it conditionally writes a report dict into
    the generic `stage_metrics` mapping and carries cross-iteration state
    (admissions accumulated since the last pass, the last report produced).
    That shape matches `LoopStage`'s bare `name` + `run(loop, payload)`
    contract directly, so `architecture/stages.py` needed no changes.
    """

    name = "model_co_optimize"

    def __init__(
        self,
        config: ModelCoOptimizeConfig | None = None,
        evaluator: ModelZooEvaluator | None = None,
    ) -> None:
        self.config = config or ModelCoOptimizeConfig()
        self._evaluator = evaluator or ModelZooEvaluator()
        self._admissions_since_last_run = 0
        self.last_report: Any | None = None

    @classmethod
    def from_mining_config(cls, mining_config: Any) -> ModelCoOptimizeStage:
        """Build a stage from an optional `model_co_optimize` dict on a mining config.

        Looks for a plain-mapping attribute (e.g. attached from raw YAML via
        ``mining_config.model_co_optimize = {"enabled": True, ...}``, mirroring
        this repo's ``mining_cfg.research = getattr(cfg, "research", None)``
        passthrough convention in `cli.py`), such as::

            model_co_optimize:
              enabled: true
              every_n_admissions: 5
              model_kind: ridge

        Absent, `None`, or falsy input builds a disabled (no-op) stage,
        matching the opt-in/default-off contract.
        """
        raw = getattr(mining_config, "model_co_optimize", None) or {}
        if not isinstance(raw, Mapping):
            logger.warning("model_co_optimize config must be a mapping; ignoring %r", raw)
            raw = {}
        zoo_field_names = {f.name for f in fields(ModelZooConfig)}
        zoo_kwargs = {k: v for k, v in raw.items() if k in zoo_field_names}
        stage_kwargs = {k: v for k, v in raw.items() if k in _STAGE_CONFIG_KEYS}
        zoo_config = ModelZooConfig(**zoo_kwargs)
        return cls(ModelCoOptimizeConfig(model_zoo_config=zoo_config, **stage_kwargs))

    def run(self, loop: Any, payload: IterationPayload) -> None:
        if not self.config.enabled:
            return

        newly_admitted = getattr(payload, "admitted_results", None) or []
        self._admissions_since_last_run += len(newly_admitted)

        library = getattr(loop, "library", None)
        factors = getattr(library, "factors", None) or {}
        if len(factors) < self.config.min_factors:
            return
        if self._admissions_since_last_run < self.config.every_n_admissions:
            return

        returns = getattr(loop, "returns", None)
        if returns is None:
            return

        factor_signals = {
            fid: factor.signals
            for fid, factor in factors.items()
            if getattr(factor, "signals", None) is not None
        }
        factor_names = {fid: getattr(factor, "name", str(fid)) for fid, factor in factors.items()}
        if len(factor_signals) < self.config.min_factors:
            return

        try:
            report = self._evaluator.evaluate(
                factor_signals,
                factor_names,
                returns,
                config=self.config.model_zoo_config,
                iteration=payload.iteration,
            )
        except Exception:  # pragma: no cover - diagnostic path must never break the loop
            logger.exception("Model co-optimization pass failed; skipping this iteration")
            return

        self._admissions_since_last_run = 0
        self.last_report = report
        payload.stage_metrics["co_optimization_report"] = report.to_dict()
