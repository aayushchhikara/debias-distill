"""Batched scoring helpers shared by the evaluators.

Sequences are lists of token ids. Batches are padded on the right; the models are
causal, so padding never changes the values read from real positions.
"""

from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm


def _batches(sequences: Sequence[Sequence[int]], batch_size: int, desc: str | None):
    """Yield (indices, padded ids, lengths), shortest sequences first."""
    order = np.argsort([len(s) for s in sequences], kind="stable")
    starts = range(0, len(order), batch_size)
    for start in tqdm(starts, desc=desc, leave=False, disable=desc is None):
        index = order[start : start + batch_size]
        lengths = torch.tensor([len(sequences[i]) for i in index])
        ids = torch.zeros(len(index), int(lengths.max()), dtype=torch.long)
        for row, i in enumerate(index):
            ids[row, : lengths[row]] = torch.as_tensor(sequences[i], dtype=torch.long)
        yield index, ids, lengths


@torch.no_grad()
def token_logprobs(
    model,
    sequences: Sequence[Sequence[int]],
    device,
    batch_size: int = 128,
    desc: str | None = None,
) -> list[np.ndarray]:
    """log P(token_i | tokens before it) for every i >= 1 of each sequence.

    The first token of a sequence has no context and is not scored, so callers
    start each sequence with the end-of-text token.
    """
    out: list[np.ndarray | None] = [None] * len(sequences)
    for index, ids, lengths in _batches(sequences, batch_size, desc):
        logits = model(ids.to(device)).logits.float()
        targets = ids[:, 1:].to(device)
        logp = logits[:, :-1].log_softmax(-1).gather(-1, targets[..., None])[..., 0].cpu().numpy()
        for row, i in enumerate(index):
            out[i] = logp[row, : lengths[row] - 1].astype(np.float64)
    return out


@torch.no_grad()
def next_token_logprobs(
    model, sequences: Sequence[Sequence[int]], device, batch_size: int = 64, desc: str | None = None
) -> np.ndarray:
    """The next-token log-distribution after each sequence: float32 [n, vocab]."""
    out = None
    for index, ids, lengths in _batches(sequences, batch_size, desc):
        logits = model(ids.to(device)).logits.float()
        last = logits[torch.arange(len(index)), lengths - 1].log_softmax(-1).cpu().numpy()
        if out is None:
            out = np.empty((len(sequences), last.shape[-1]), dtype=np.float32)
        out[index] = last
    return out


@torch.no_grad()
def final_hidden_states(
    model, sequences: Sequence[Sequence[int]], device, batch_size: int = 64, desc: str | None = None
) -> np.ndarray:
    """Hidden state of every layer at each sequence's last token: float32 [n, layers + 1, width]."""
    out = None
    for index, ids, lengths in _batches(sequences, batch_size, desc):
        states = model(ids.to(device), output_hidden_states=True).hidden_states
        rows = torch.arange(len(index))
        last = torch.stack([h[rows, lengths - 1].float() for h in states], dim=1).cpu().numpy()
        if out is None:
            out = np.empty((len(sequences), *last.shape[1:]), dtype=np.float32)
        out[index] = last
    return out
