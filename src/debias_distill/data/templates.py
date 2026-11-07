"""Evaluation templates, split into train / dev / final."""

from pathlib import Path


def load_templates(split: str, root: str | Path = "data/templates") -> list[str]:
    """Load templates for a split."""
    raise NotImplementedError


def fill(template: str, subject: str, occupation: str | None = None) -> str:
    """Fill template slots."""
    raise NotImplementedError
