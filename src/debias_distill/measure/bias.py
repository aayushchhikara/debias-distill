"""Bias metrics."""


def probability_ratio_score(model, tokenizer, split: str = "dev") -> dict:
    """Template-based bias score with confidence interval."""
    raise NotImplementedError


def template_kl(model, tokenizer, split: str = "dev") -> dict:
    """Divergence between output distributions of paired templates."""
    raise NotImplementedError
