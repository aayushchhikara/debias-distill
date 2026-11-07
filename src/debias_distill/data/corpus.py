"""Text corpus sampling and preprocessing."""


def sample_corpus(cfg: dict) -> None:
    """Collect and tokenize the configured corpus slice."""
    raise NotImplementedError


def build_counterfactuals(cfg: dict) -> None:
    """Prepare substituted variants of corpus sequences."""
    raise NotImplementedError
