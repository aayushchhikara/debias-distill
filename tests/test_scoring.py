"""Tests for the batched scoring helpers (tiny random model only)."""

import numpy as np
import pytest
import torch

from debias_distill.measure.scoring import final_hidden_states, next_token_logprobs, token_logprobs

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_model(tiny_lm):
    return tiny_lm()


def test_token_logprobs_match_a_direct_computation(tiny_model):
    rng = np.random.default_rng(0)
    sequences = [rng.integers(0, 50000, size=n).tolist() for n in (3, 9, 5, 9, 2)]
    scored = token_logprobs(tiny_model, sequences, CPU, batch_size=2)
    for sequence, values in zip(sequences, scored):
        ids = torch.tensor([sequence])
        with torch.no_grad():
            logp = tiny_model(ids).logits[0, :-1].log_softmax(-1)
        direct = logp.gather(-1, ids[0, 1:, None])[:, 0].numpy()
        assert values.shape == (len(sequence) - 1,)
        assert np.allclose(values, direct, atol=1e-5)
    again = token_logprobs(tiny_model, sequences, CPU, batch_size=5)  # other padding
    assert all(np.allclose(a, b, atol=1e-5) for a, b in zip(scored, again))


def test_last_position_helpers_ignore_padding(tiny_model):
    sequences = [[5, 6, 7], [5, 6, 7, 8, 9, 10], [11]]
    together = next_token_logprobs(tiny_model, sequences, CPU, batch_size=3)
    alone = next_token_logprobs(tiny_model, sequences, CPU, batch_size=1)
    assert together.shape == (3, 50304)
    assert np.allclose(together, alone, atol=1e-5)
    assert np.allclose(np.exp(together).sum(-1), 1.0, atol=1e-4)
    states = final_hidden_states(tiny_model, sequences, CPU, batch_size=3)
    assert states.shape == (3, 3, 16)  # embeddings + 2 layers
    assert np.allclose(states, final_hidden_states(tiny_model, sequences, CPU, 1), atol=1e-5)
