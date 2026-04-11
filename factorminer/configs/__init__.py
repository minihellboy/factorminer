"""Configuration defaults and schemas for FactorMiner."""

from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.yaml"


def load_default_yaml() -> dict[str, Any]:
    """Load the default YAML configuration shipped with the package.

    Returns
    -------
    dict
        Parsed YAML contents as a nested dictionary.  Returns an empty
        dict if the default file is missing or empty.
    """
    if not DEFAULT_CONFIG_PATH.exists():
        return {}
    with open(DEFAULT_CONFIG_PATH) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}
