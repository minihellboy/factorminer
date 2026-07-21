"""FactorMiner public package surface.

Importing the package is intentionally side-effect free.  Configuration and
runtime components are loaded only when a caller accesses the corresponding
public attribute.
"""

from __future__ import annotations

from factorminer._lazy_exports import public_dir, resolve_export

__version__ = "0.1.0"
__author__ = "FactorMiner Team"

_CONFIG_MODULE = "factorminer.utils.config"
_EXPORTS = {
    name: _CONFIG_MODULE
    for name in (
        "AutoInventorConfig",
        "CapacityConfig",
        "CausalConfig",
        "Config",
        "DataConfig",
        "DebateConfig",
        "EvaluationConfig",
        "HelixConfig",
        "LLMConfig",
        "MemoryConfig",
        "MiningConfig",
        "Phase2Config",
        "RegimeConfig",
        "SignificanceConfig",
        "load_config",
    )
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str):
    return resolve_export(globals(), name, _EXPORTS)


def __dir__() -> list[str]:
    return public_dir(globals(), _EXPORTS)
