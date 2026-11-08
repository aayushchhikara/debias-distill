"""Load YAML experiment configs."""

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with `override` merged into `base`; nested dicts are merged, other values replaced."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and deep-merge `overrides` on top.

    A config may set `base: <path>` (relative to its own directory) to inherit from
    another config; keys in the child file take precedence.
    """
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"{path}: top level must be a mapping, got {type(cfg).__name__}")

    base_ref = cfg.pop("base", None)
    if base_ref is not None:
        cfg = _deep_merge(load_config(path.parent / base_ref), cfg)

    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg
