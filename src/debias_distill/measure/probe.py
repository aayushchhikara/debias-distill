"""Linear probes on hidden states."""


def probe_accuracy(model, tokenizer, layer: int | str = "all", split: str = "dev") -> dict:
    """Per-layer probe accuracy and chance baseline."""
    raise NotImplementedError
