"""Configuration loading, validation, and management for FactorMiner."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from factorminer.configs import DEFAULT_CONFIG_PATH


@dataclass
class MiningConfig:
    """Parameters controlling the factor mining loop."""

    target_library_size: int = 110
    batch_size: int = 40
    max_iterations: int = 200
    ic_threshold: float = 0.04
    icir_threshold: float = 0.5
    correlation_threshold: float = 0.5
    replacement_ic_min: float = 0.10
    replacement_ic_ratio: float = 1.3

    def validate(self) -> None:
        if self.target_library_size < 1:
            raise ValueError("target_library_size must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not (0.0 < self.ic_threshold < 1.0):
            raise ValueError("ic_threshold must be in (0, 1)")
        if not (0.0 < self.icir_threshold < 10.0):
            raise ValueError("icir_threshold must be in (0, 10)")
        if not (0.0 < self.correlation_threshold <= 1.0):
            raise ValueError("correlation_threshold must be in (0, 1]")
        if self.replacement_ic_min <= self.ic_threshold:
            raise ValueError("replacement_ic_min must be > ic_threshold")
        if self.replacement_ic_ratio < 1.0:
            raise ValueError("replacement_ic_ratio must be >= 1.0")


@dataclass
class EvaluationConfig:
    """Parameters for factor evaluation."""

    num_workers: int = 40
    fast_screen_assets: int = 100
    gpu_device: str = "cuda:0"
    backend: str = "gpu"
    signal_failure_policy: str = "reject"

    def validate(self) -> None:
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if self.fast_screen_assets < 1:
            raise ValueError("fast_screen_assets must be >= 1")
        if self.backend not in ("gpu", "numpy", "c"):
            raise ValueError(f"backend must be one of: gpu, numpy, c (got '{self.backend}')")
        if self.signal_failure_policy not in ("reject", "synthetic", "raise"):
            raise ValueError(
                "signal_failure_policy must be one of: reject, synthetic, raise"
            )


@dataclass
class DataConfig:
    """Parameters for data loading and universes."""

    market: str = "a_shares"
    universe: str = "CSI500"
    frequency: str = "10min"
    features: list[str] = field(
        default_factory=lambda: [
            "$open", "$high", "$low", "$close",
            "$volume", "$amt", "$vwap", "$returns",
        ]
    )
    train_period: list[str] = field(
        default_factory=lambda: ["2024-01-01", "2024-12-31"]
    )
    test_period: list[str] = field(
        default_factory=lambda: ["2025-01-01", "2025-12-31"]
    )

    def validate(self) -> None:
        if len(self.train_period) != 2:
            raise ValueError("train_period must be a list of [start, end]")
        if len(self.test_period) != 2:
            raise ValueError("test_period must be a list of [start, end]")
        if self.train_period[0] >= self.train_period[1]:
            raise ValueError("train_period start must be before end")
        if self.test_period[0] >= self.test_period[1]:
            raise ValueError("test_period start must be before end")
        if not self.features:
            raise ValueError("features must not be empty")


@dataclass
class LLMConfig:
    """Parameters for LLM-based factor generation."""

    provider: str = "google"
    model: str = "gemini-2.0-flash"
    temperature: float = 0.8
    max_tokens: int = 4096
    batch_candidates: int = 40

    def validate(self) -> None:
        if self.provider not in ("google", "openai", "anthropic", "mock"):
            raise ValueError(
                f"provider must be one of: google, openai, anthropic, mock "
                f"(got '{self.provider}')"
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be in [0, 2]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.batch_candidates < 1:
            raise ValueError("batch_candidates must be >= 1")


@dataclass
class MemoryConfig:
    """Parameters for the experience memory system."""

    max_success_patterns: int = 50
    max_failure_patterns: int = 100
    max_insights: int = 30
    consolidation_interval: int = 10

    def validate(self) -> None:
        if self.max_success_patterns < 1:
            raise ValueError("max_success_patterns must be >= 1")
        if self.max_failure_patterns < 1:
            raise ValueError("max_failure_patterns must be >= 1")
        if self.max_insights < 1:
            raise ValueError("max_insights must be >= 1")
        if self.consolidation_interval < 1:
            raise ValueError("consolidation_interval must be >= 1")


@dataclass
class CausalConfig:
    """Parameters for causal validation (Granger + intervention tests)."""

    enabled: bool = False
    granger_max_lag: int = 5
    granger_significance: float = 0.05
    n_interventions: int = 3
    intervention_magnitude: float = 2.0
    intervention_ic_threshold: float = 0.5
    robustness_threshold: float = 0.4
    granger_weight: float = 0.4
    intervention_weight: float = 0.6

    def validate(self) -> None:
        if self.granger_max_lag < 1:
            raise ValueError("granger_max_lag must be >= 1")
        if not (0.0 < self.granger_significance < 1.0):
            raise ValueError("granger_significance must be in (0, 1)")
        if self.n_interventions < 1:
            raise ValueError("n_interventions must be >= 1")
        if self.intervention_magnitude <= 0.0:
            raise ValueError("intervention_magnitude must be > 0")
        if not (0.0 <= self.intervention_ic_threshold <= 1.0):
            raise ValueError("intervention_ic_threshold must be in [0, 1]")
        if not (0.0 <= self.robustness_threshold <= 1.0):
            raise ValueError("robustness_threshold must be in [0, 1]")
        if not (0.0 <= self.granger_weight <= 1.0):
            raise ValueError("granger_weight must be in [0, 1]")
        if not (0.0 <= self.intervention_weight <= 1.0):
            raise ValueError("intervention_weight must be in [0, 1]")
        if abs(self.granger_weight + self.intervention_weight - 1.0) > 1e-6:
            raise ValueError("granger_weight + intervention_weight must equal 1.0")


@dataclass
class RegimeConfig:
    """Parameters for regime-conditional factor evaluation."""

    enabled: bool = False
    lookback_window: int = 60
    bull_return_threshold: float = 0.0
    bear_return_threshold: float = 0.0
    volatility_percentile: float = 0.7
    min_regime_ic: float = 0.03
    min_regimes_passing: int = 2

    def validate(self) -> None:
        if self.lookback_window < 5:
            raise ValueError("lookback_window must be >= 5")
        if not (0.0 < self.volatility_percentile < 1.0):
            raise ValueError("volatility_percentile must be in (0, 1)")
        if self.min_regime_ic < 0.0:
            raise ValueError("min_regime_ic must be >= 0")
        if not (1 <= self.min_regimes_passing <= 4):
            raise ValueError("min_regimes_passing must be in [1, 4]")


@dataclass
class CapacityConfig:
    """Parameters for strategy capacity estimation."""

    enabled: bool = False
    base_capital_usd: float = 1e8
    ic_degradation_limit: float = 0.20
    net_icir_threshold: float = 0.3
    sigma_annual: float = 0.25

    def validate(self) -> None:
        if self.base_capital_usd <= 0.0:
            raise ValueError("base_capital_usd must be > 0")
        if not (0.0 < self.ic_degradation_limit < 1.0):
            raise ValueError("ic_degradation_limit must be in (0, 1)")
        if self.net_icir_threshold < 0.0:
            raise ValueError("net_icir_threshold must be >= 0")
        if self.sigma_annual <= 0.0:
            raise ValueError("sigma_annual must be > 0")


@dataclass
class SignificanceConfig:
    """Parameters for statistical significance testing."""

    enabled: bool = False
    bootstrap_n_samples: int = 1000
    bootstrap_block_size: int = 20
    fdr_level: float = 0.05
    deflated_sharpe_enabled: bool = True
    min_deflated_sharpe: float = 0.0

    def validate(self) -> None:
        if self.bootstrap_n_samples < 100:
            raise ValueError("bootstrap_n_samples must be >= 100")
        if self.bootstrap_block_size < 1:
            raise ValueError("bootstrap_block_size must be >= 1")
        if not (0.0 < self.fdr_level < 1.0):
            raise ValueError("fdr_level must be in (0, 1)")


@dataclass
class DebateConfig:
    """Parameters for multi-specialist debate-based generation."""

    enabled: bool = False
    num_specialists: int = 3
    candidates_per_specialist: int = 15
    enable_critic: bool = True
    top_k_after_critic: int = 40
    critic_temperature: float = 0.3

    def validate(self) -> None:
        if self.num_specialists < 1:
            raise ValueError("num_specialists must be >= 1")
        if self.candidates_per_specialist < 1:
            raise ValueError("candidates_per_specialist must be >= 1")
        if self.top_k_after_critic < 1:
            raise ValueError("top_k_after_critic must be >= 1")
        if not (0.0 <= self.critic_temperature <= 2.0):
            raise ValueError("critic_temperature must be in [0, 2]")


@dataclass
class AutoInventorConfig:
    """Parameters for automatic operator invention."""

    enabled: bool = False
    invention_interval: int = 10
    max_proposals_per_round: int = 5
    min_ic_contribution: float = 0.03
    store_dir: str = "./output/custom_operators"

    def validate(self) -> None:
        if self.invention_interval < 1:
            raise ValueError("invention_interval must be >= 1")
        if self.max_proposals_per_round < 1:
            raise ValueError("max_proposals_per_round must be >= 1")
        if self.min_ic_contribution < 0.0:
            raise ValueError("min_ic_contribution must be >= 0")


@dataclass
class HelixConfig:
    """Parameters for the Helix knowledge and memory system."""

    enabled: bool = False
    enable_knowledge_graph: bool = False
    enable_embeddings: bool = False
    enable_canonicalization: bool = True
    forgetting_lambda: float = 0.95
    forgetting_demotion_threshold: int = 20

    def validate(self) -> None:
        if not (0.0 < self.forgetting_lambda <= 1.0):
            raise ValueError("forgetting_lambda must be in (0, 1]")
        if self.forgetting_demotion_threshold < 1:
            raise ValueError("forgetting_demotion_threshold must be >= 1")


@dataclass
class Phase2Config:
    """Aggregated configuration for all Phase 2 subsystems."""

    causal: CausalConfig = field(default_factory=CausalConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    significance: SignificanceConfig = field(default_factory=SignificanceConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    auto_inventor: AutoInventorConfig = field(default_factory=AutoInventorConfig)
    helix: HelixConfig = field(default_factory=HelixConfig)

    def validate(self) -> None:
        for sub in [
            self.causal,
            self.regime,
            self.capacity,
            self.significance,
            self.debate,
            self.auto_inventor,
            self.helix,
        ]:
            sub.validate()


@dataclass
class BenchmarkConfig:
    """Parameters for paper/research benchmark execution."""

    mode: str = "paper"
    seed: int = 42
    freeze_top_k: int = 40
    freeze_universe: str = "CSI500"
    report_universes: list[str] = field(
        default_factory=lambda: ["CSI500", "CSI1000", "HS300", "Binance"]
    )
    baselines: list[str] = field(
        default_factory=lambda: [
            "alpha101_classic",
            "alpha101_adapted",
            "random_exploration",
            "gplearn",
            "alphaforge_style",
            "alphaagent_style",
            "factor_miner",
            "factor_miner_no_memory",
        ]
    )
    cost_bps: list[float] = field(default_factory=lambda: [1.0, 4.0, 7.0, 10.0, 11.0])
    efficiency_panel_shape: list[int] = field(default_factory=lambda: [12610, 500])

    def validate(self) -> None:
        if self.mode not in ("paper", "research"):
            raise ValueError("benchmark.mode must be one of: paper, research")
        if self.freeze_top_k < 1:
            raise ValueError("benchmark.freeze_top_k must be >= 1")
        if not self.freeze_universe:
            raise ValueError("benchmark.freeze_universe must not be empty")
        if not self.report_universes:
            raise ValueError("benchmark.report_universes must not be empty")
        if any(not universe for universe in self.report_universes):
            raise ValueError("benchmark.report_universes must not contain empty entries")
        if not self.baselines:
            raise ValueError("benchmark.baselines must not be empty")
        if any(cost < 0 for cost in self.cost_bps):
            raise ValueError("benchmark.cost_bps must be non-negative")
        if len(self.efficiency_panel_shape) != 2:
            raise ValueError("benchmark.efficiency_panel_shape must be [periods, assets]")
        if any(dim < 1 for dim in self.efficiency_panel_shape):
            raise ValueError("benchmark.efficiency_panel_shape values must be >= 1")


@dataclass
class Config:
    """Top-level configuration aggregating all sub-configs."""

    mining: MiningConfig = field(default_factory=MiningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def validate(self) -> None:
        """Validate all sub-configurations."""
        self.mining.validate()
        self.evaluation.validate()
        self.data.validate()
        self.llm.validate()
        self.memory.validate()
        self.phase2.validate()
        self.benchmark.validate()

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


_SECTION_MAP: dict[str, type] = {
    "mining": MiningConfig,
    "evaluation": EvaluationConfig,
    "data": DataConfig,
    "llm": LLMConfig,
    "memory": MemoryConfig,
    "phase2": Phase2Config,
    "benchmark": BenchmarkConfig,
}

_PHASE2_SECTION_MAP: dict[str, type] = {
    "causal": CausalConfig,
    "regime": RegimeConfig,
    "capacity": CapacityConfig,
    "significance": SignificanceConfig,
    "debate": DebateConfig,
    "auto_inventor": AutoInventorConfig,
    "helix": HelixConfig,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level")
    return data


def _build_section(section_cls: type, raw: dict[str, Any]) -> Any:
    """Instantiate a config dataclass, ignoring unknown keys."""
    valid_fields = {f.name for f in section_cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    return section_cls(**filtered)


def _build_phase2(raw: dict[str, Any]) -> Phase2Config:
    """Build Phase2Config with nested sub-config dataclasses."""
    subs = {}
    for sub_name, sub_cls in _PHASE2_SECTION_MAP.items():
        sub_raw = raw.get(sub_name, {})
        if isinstance(sub_raw, dict):
            subs[sub_name] = _build_section(sub_cls, sub_raw)
        else:
            subs[sub_name] = sub_cls()
    return Phase2Config(**subs)


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration from YAML with defaults and optional overrides.

    Resolution order:
        1. Built-in defaults (default.yaml shipped with the package)
        2. User-provided config file (if given)
        3. Programmatic overrides dict (if given)

    Args:
        config_path: Path to a user YAML config file. If None, only defaults are used.
        overrides: Dict of overrides keyed by section, e.g.
            {"mining": {"batch_size": 20}, "llm": {"model": "gpt-4"}}.

    Returns:
        A fully validated Config instance.
    """
    # 1. Load package defaults
    defaults = _load_yaml(DEFAULT_CONFIG_PATH)

    # 2. Merge user config
    if config_path is not None:
        user_cfg = _load_yaml(Path(config_path))
        merged = _deep_merge(defaults, user_cfg)
    else:
        merged = defaults

    # 3. Merge programmatic overrides
    if overrides:
        merged = _deep_merge(merged, overrides)

    # 4. Build typed config objects
    sections = {}
    for section_name, section_cls in _SECTION_MAP.items():
        raw = merged.get(section_name, {})
        if section_name == "phase2":
            sections[section_name] = _build_phase2(raw)
        else:
            sections[section_name] = _build_section(section_cls, raw)

    config = Config(**sections)
    config.validate()
    return config
