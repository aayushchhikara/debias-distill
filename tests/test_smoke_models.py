"""Load both models on the accelerator and run one forward pass.

Slow (downloads weights on first run). Run with: poetry run pytest -m slow
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from debias_distill.utils.config import load_config
from debias_distill.utils.device import get_device

pytestmark = pytest.mark.slow

CFG = load_config("configs/base.yaml")


@pytest.mark.parametrize("role", ["teacher", "student"])
def test_forward_pass(role):
    name = CFG[role]
    device = get_device(CFG["device"])
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()

    inputs = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    assert logits.shape[:2] == inputs["input_ids"].shape
    assert logits.shape[-1] == model.config.vocab_size
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()


def test_shared_tokenizer():
    t = AutoTokenizer.from_pretrained(CFG["teacher"])
    s = AutoTokenizer.from_pretrained(CFG["student"])
    assert t.get_vocab() == s.get_vocab()
