"""Capability metrics."""


def perplexity(model, tokenizer, cfg: dict) -> float:
    """Perplexity on a held-out text slice."""
    raise NotImplementedError


def lm_eval_tasks(model_path: str, tasks: list[str], limit: int) -> dict:
    """Run lm-eval tasks with a sample limit."""
    raise NotImplementedError
