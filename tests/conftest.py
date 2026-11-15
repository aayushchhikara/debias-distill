"""Fixtures shared by the test modules."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from debias_distill.data.corpus import build_counterfactuals, sample_corpus
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


class ToyTeacher(torch.nn.Module):
    """A tiny deterministic stand-in for the teacher."""

    def __init__(self, vocab: int):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, 8)
        self.out = torch.nn.Linear(8, vocab)

    def forward(self, ids):
        return SimpleNamespace(logits=4.0 * self.out(self.emb(ids)))


@pytest.fixture
def toy_setup(tmp_path, templates_root, tok):
    """A small corpus with swapped copies, a config pointing at it, and a toy teacher."""
    cfg = load_config(
        "configs/base.yaml",
        overrides={
            "device": "cpu",
            "topk": 8,
            "data": {
                "templates_dir": str(templates_root),
                "corpus_dir": str(tmp_path / "corpus"),
                "cache_dir": str(tmp_path / "cache"),
                "experiments_dir": str(tmp_path / "experiments"),
            },
            "corpus": {"seq_len": 16, "token_budget": 16 * 24},
            "eval": {"perplexity_tokens": 16},
            "teacher_cache": {"batch_pairs": 2, "flush_every": 1},
        },
    )
    neutral = [
        "The weather report predicted rain for the whole week ahead of us.",
        "Prices at the market rose again after the long dry summer.",
        "The train to the coast leaves from platform nine every morning.",
        "Engineers finished the new bridge across the river last month.",
    ]
    docs = [
        "He went to the market and bought apples for his mother.",
        *neutral,
        "She told her brother that the nurse would be late again tonight.",
        *neutral,
    ] * 12
    sample_corpus(cfg, cfg["data"]["corpus_dir"], docs=docs)
    build_counterfactuals(cfg, cfg["data"]["corpus_dir"])
    return cfg, ToyTeacher(len(tok))
