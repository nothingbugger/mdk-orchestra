"""Sensitivity config loader for deterministic_tools.

Reads ``sensitivity.yaml`` once at import time and exposes typed profile
dicts for each flagger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_YAML_PATH = Path(__file__).parent / "sensitivity.yaml"


def _load_profiles() -> dict[str, Any]:
    with _YAML_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_PROFILES: dict[str, Any] = _load_profiles()

VALID_SENSITIVITIES = ("low", "medium", "high")


def get_profile(sensitivity: str) -> dict[str, Any]:
    """Return the sensitivity profile dict for a named level.

    Args:
        sensitivity: one of 'low', 'medium', 'high'.

    Returns:
        Nested dict with per-flagger config keys.

    Raises:
        ValueError: if sensitivity is not a known level.
    """
    if sensitivity not in VALID_SENSITIVITIES:
        raise ValueError(
            f"Unknown sensitivity '{sensitivity}'. "
            f"Choose from: {VALID_SENSITIVITIES}"
        )
    return _PROFILES[sensitivity]


def rule_engine_cfg(sensitivity: str) -> dict[str, Any]:
    """Shortcut: rule-engine sub-section of a sensitivity profile."""
    return get_profile(sensitivity)["rule_engine"]


def if_cfg(sensitivity: str) -> dict[str, Any]:
    """Shortcut: isolation-forest sub-section."""
    return get_profile(sensitivity)["isolation_forest"]


def xgb_cfg(sensitivity: str) -> dict[str, Any]:
    """Shortcut: xgboost sub-section."""
    return get_profile(sensitivity)["xgboost"]
