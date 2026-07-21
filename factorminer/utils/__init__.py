"""Utility APIs exposed without importing reporting or plotting stacks."""

from __future__ import annotations

from factorminer._lazy_exports import public_dir, resolve_export

_CONFIG_MODULE = "factorminer.utils.config"
_EXPORTS = {
    **{
        name: _CONFIG_MODULE
        for name in (
            "AutoInventorConfig",
            "CapacityConfig",
            "CausalConfig",
            "Config",
            "DebateConfig",
            "HelixConfig",
            "MiningConfig",
            "Phase2Config",
            "RegimeConfig",
            "SignificanceConfig",
            "load_config",
        )
    },
    "MiningReporter": "factorminer.utils.reporting",
    "FactorTearSheet": "factorminer.utils.tearsheet",
    **{
        name: "factorminer.utils.visualization"
        for name in (
            "plot_ablation_comparison",
            "plot_correlation_heatmap",
            "plot_cost_pressure",
            "plot_efficiency_benchmark",
            "plot_ic_timeseries",
            "plot_mining_funnel",
            "plot_quintile_returns",
        )
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    return resolve_export(globals(), name, _EXPORTS)


def __dir__() -> list[str]:
    return public_dir(globals(), _EXPORTS)
