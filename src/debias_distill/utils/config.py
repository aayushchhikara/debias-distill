"""Load YAML experiment configs."""

from pathlib import Path
from typing import Any


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config and deep-merge `overrides` on top."""
    raise NotImplementedError
