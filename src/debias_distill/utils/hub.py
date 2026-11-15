"""Load models and tokenizers at the revisions pinned in the config."""

from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def load_tokenizer(cfg: dict[str, Any], role: str = "student"):
    """Tokenizer for `cfg[role]` at `cfg[f"{role}_revision"]` (teacher and student share one)."""
    return AutoTokenizer.from_pretrained(cfg[role], revision=cfg.get(f"{role}_revision"))


def load_model(cfg: dict[str, Any], role: str, dtype=None):
    """Causal LM for `cfg[role]` at its pinned revision."""
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        cfg[role], revision=cfg.get(f"{role}_revision"), dtype=dtype
    )


def load_checkpoint(cfg: dict[str, Any], checkpoint: str, dtype=None):
    """A model by role ("teacher" or "student") or from a local directory saved by training."""
    if checkpoint in ("teacher", "student"):
        return load_model(cfg, checkpoint, dtype=dtype)
    if not Path(checkpoint).is_dir():
        raise FileNotFoundError(f"no saved model at {checkpoint}")
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(checkpoint, dtype=dtype)
