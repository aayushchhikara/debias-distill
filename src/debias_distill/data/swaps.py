"""Word-pair substitution lists."""

from pathlib import Path


def load_swap_pairs(swap_dir: str | Path = "data/swaps") -> dict[str, str]:
    """Load word pairs into a bidirectional mapping."""
    raise NotImplementedError


def counterfactual(text: str, pairs: dict[str, str]) -> tuple[str, list[str]]:
    """Return (substituted_text, unhandled_cases)."""
    raise NotImplementedError
