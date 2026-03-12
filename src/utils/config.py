"""Configuration loader for Project SIGNAL."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file and return as nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_nested(config: dict, *keys: str, default: Any = None) -> Any:
    """Safely retrieve nested config values.

    Usage:
        get_nested(cfg, "training", "lr_actor", default=3e-4)
    """
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
