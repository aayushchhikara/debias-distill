"""Capability metric: perplexity on the held-out slice of the corpus."""

import math
from pathlib import Path

import numpy as np

from debias_distill.measure.scoring import token_logprobs


def perplexity(model, cfg: dict, device, batch_size: int = 16, progress: bool = False) -> dict:
    """Perplexity on `heldout_tokens.npy`, limited to `eval.perplexity_tokens` tokens.

    The first token of each row has no context and is not scored.
    """
    rows = np.load(Path(cfg["data"]["corpus_dir"]) / "heldout_tokens.npy", mmap_mode="r")
    n_rows = max(1, min(len(rows), cfg["eval"]["perplexity_tokens"] // rows.shape[1]))
    sequences = [rows[i].astype(np.int64) for i in range(n_rows)]
    scored = token_logprobs(
        model, sequences, device, batch_size, desc="perplexity" if progress else None
    )
    total = sum(float(values.sum()) for values in scored)
    count = sum(len(values) for values in scored)
    nll = -total / count
    return {"perplexity": math.exp(nll), "nll": nll, "tokens": count, "rows": n_rows}
