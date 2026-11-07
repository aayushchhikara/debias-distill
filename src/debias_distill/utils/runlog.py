"""Per-run logging under experiments/<run_id>/."""

from pathlib import Path
from typing import Any


class RunLogger:
    """Writes resolved config, metadata and metrics for a single run."""

    def __init__(self, run_id: str, config: dict[str, Any], root: str | Path = "experiments"):
        raise NotImplementedError

    def log(self, step: int, **metrics: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
