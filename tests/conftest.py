"""Fixtures shared by the test modules."""

import shutil
from pathlib import Path

import pytest
import torch

from debias_distill.utils.config import load_config
from debias_distill.utils.hub import load_tokenizer

ROOT = Path("data/templates")


@pytest.fixture(scope="session")
def tiny_lm():
    """Factory for a tiny random causal LM with the real vocabulary size."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    def make(seed: int = 0):
        torch.manual_seed(seed)
        config = GPTNeoXConfig(
            vocab_size=50304,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
        return GPTNeoXForCausalLM(config).eval()

    return make


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(load_config("configs/base.yaml"))


@pytest.fixture
def templates_root(tmp_path):
    """A copy of the train and dev splits with a synthetic final split."""
    root = tmp_path / "templates"
    for split in ("train", "dev"):
        shutil.copytree(ROOT / split, root / split)
    final = root / "final"
    final.mkdir()
    (final / "names.tsv").write_text("Zedd\tZoey\n")
    (final / "occupations.txt").write_text("a tester\tneutral\n")
    (final / "templates.tsv").write_text("{subj} tested as\t{subj} tested as\n")
    return root
