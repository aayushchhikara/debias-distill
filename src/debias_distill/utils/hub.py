"""Load models and tokenizers at the revisions pinned in the config."""

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
