"""Per-run logging under experiments/<run_id>/.

Each run directory contains:
  config.yaml    resolved config used for the run
  meta.json      git commit, seed, device, library versions, start/end time
  metrics.jsonl  one JSON object per `log()` call
"""

import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import torch
import transformers
import yaml


def _git_state() -> dict[str, Any]:
    """Current commit hash and whether the working tree has uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunLogger:
    """Writes resolved config, metadata and metrics for a single run."""

    def __init__(self, run_id: str, config: dict[str, Any], root: str | Path = "experiments"):
        self.dir = Path(root) / run_id
        if self.dir.exists():
            raise FileExistsError(f"run directory already exists: {self.dir}")
        self.dir.mkdir(parents=True)

        with (self.dir / "config.yaml").open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        self.meta: dict[str, Any] = {
            "run_id": run_id,
            "started": _now(),
            "finished": None,
            "git": _git_state(),
            "seed": config.get("seed"),
            "device": config.get("device"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
        }
        self._write_meta()
        self._metrics = (self.dir / "metrics.jsonl").open("a")
        self._t0 = time.monotonic()

    def _write_meta(self) -> None:
        with (self.dir / "meta.json").open("w") as f:
            json.dump(self.meta, f, indent=2)

    def log(self, step: int, **metrics: float) -> None:
        """Append one metrics record and flush it to disk immediately."""
        record = {"step": step, "elapsed_s": round(time.monotonic() - self._t0, 3), **metrics}
        self._metrics.write(json.dumps(record) + "\n")
        self._metrics.flush()

    def close(self) -> None:
        if self._metrics.closed:
            return
        self._metrics.close()
        self.meta["finished"] = _now()
        self._write_meta()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
