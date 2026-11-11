"""Provenance records for generated artifacts.

Each generated folder gets a `manifest.json` naming the inputs that produced it
(by SHA-256), the config hash and the git commit, so a later stage can tell
whether an artifact is stale.
"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_FILE = "manifest.json"


def git_state() -> dict[str, Any]:
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
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash of a config that does not depend on key order."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def hash_inputs(inputs: dict[str, str | Path]) -> dict[str, str]:
    """SHA-256 of each named input file."""
    return {name: file_sha256(path) for name, path in sorted(inputs.items())}


def write_manifest(
    out_dir: str | Path,
    inputs: dict[str, str | Path],
    cfg: dict[str, Any],
    outputs: list[str] | None = None,
) -> dict[str, Any]:
    """Write `manifest.json` in `out_dir` and return its contents."""
    out_dir = Path(out_dir)
    manifest = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "config_sha256": config_hash(cfg),
        "inputs": hash_inputs(inputs),
        "outputs": {name: file_sha256(out_dir / name) for name in sorted(outputs or [])},
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest(out_dir: str | Path) -> dict[str, Any] | None:
    path = Path(out_dir) / MANIFEST_FILE
    return json.loads(path.read_text()) if path.exists() else None


def inputs_match(out_dir: str | Path, inputs: dict[str, str | Path]) -> bool:
    """True if `out_dir` has a manifest whose recorded inputs equal the current files."""
    manifest = read_manifest(out_dir)
    return manifest is not None and manifest["inputs"] == hash_inputs(inputs)
