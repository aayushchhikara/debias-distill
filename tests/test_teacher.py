"""Tests for aligning teacher distributions and caching teacher targets."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from debias_distill.data.corpus import build_counterfactuals, sample_corpus
from debias_distill.data.swaps import load_swap_pairs
from debias_distill.data.templates import load_names, load_occupations
from debias_distill.teacher.align import TokenMap, align, build_token_map
from debias_distill.teacher.cache_targets import (
    SETS,
    cache_teacher_targets,
    compress,
    load_targets,
    pair_targets,
    watch_token_ids,
)
from debias_distill.utils.config import load_config

# Toy vocabulary for the alignment tests.
HE, SHE, HIS, HIM, HER, HERS, NURSE, THE, ACTRESS, ACTOR = range(10)
TOY_MAP = TokenMap.from_edges(
    [(HE, SHE), (SHE, HE), (HIM, HER), (HIS, HER), (HIS, HERS), (HER, HIM), (HER, HIS),
     (HERS, HIS), (ACTRESS, ACTOR)]
)  # fmt: skip


def dist(**mass):
    p = torch.zeros(10)
    for name, value in mass.items():
        p[globals()[name]] = value
    return p


def test_aligned_targets_keep_pronoun_agreement_and_equalise_neutral_tokens():
    # After "She said that" the teacher expects "she"; after "He said that", "he".
    p_orig = dist(SHE=0.6, HE=0.02, NURSE=0.08, THE=0.3)
    p_cf = dist(HE=0.7, SHE=0.01, NURSE=0.02, THE=0.27)
    targets = pair_targets(p_orig[None], p_cf[None], TOY_MAP)

    mean = targets["mean"][0]
    assert mean[SHE] == pytest.approx(0.305) and mean[HE] == pytest.approx(0.36)  # agreement lost

    orig, cf = targets["aligned_orig"][0], targets["aligned_cf"][0]
    assert orig[SHE] == pytest.approx(0.65) and orig[HE] == pytest.approx(0.015)
    assert cf[HE] == pytest.approx(0.65) and cf[SHE] == pytest.approx(0.015)
    for target in (orig, cf, mean):
        assert target[NURSE] == pytest.approx(0.05)  # same for both genders
        assert target.sum() == pytest.approx(1.0)


def test_one_to_many_mass_is_split_by_the_destination():
    p_src = dist(HER=0.4, THE=0.6)
    p_dst = dist(HIS=0.3, HIM=0.1, THE=0.6)
    out = align(p_src, p_dst, TOY_MAP)
    assert out[HIS] == pytest.approx(0.3) and out[HIM] == pytest.approx(0.1)
    assert out[HER] == 0 and out[THE] == pytest.approx(0.6)

    even = align(p_src, dist(THE=1.0), TOY_MAP)  # destination gives no guidance
    assert even[HIS] == pytest.approx(0.2) and even[HIM] == pytest.approx(0.2)


def test_many_to_one_and_one_way_edges():
    out = align(dist(HIM=0.2, HIS=0.3, ACTRESS=0.1, ACTOR=0.4), dist(HER=1.0), TOY_MAP)
    assert out[HER] == pytest.approx(0.5)  # him -> her, his -> her (hers has no destination mass)
    assert out[ACTOR] == pytest.approx(0.5) and out[ACTRESS] == 0  # actor itself is not moved


def test_align_conserves_mass_on_random_batches():
    torch.manual_seed(0)
    p_src = torch.rand(3, 5, 10).softmax(-1)
    p_dst = torch.rand(3, 5, 10).softmax(-1)
    out = align(p_src, p_dst, TOY_MAP)
    assert torch.allclose(out.sum(-1), torch.ones(3, 5), atol=1e-6)
    assert (out >= 0).all()
    assert torch.equal(
        out[..., [NURSE, THE]], p_src[..., [NURSE, THE]]
    )  # unmapped tokens untouched


def test_compress_parts_are_disjoint_and_sum_to_one():
    torch.manual_seed(0)
    p = (torch.randn(2, 4, 200) * 3).softmax(-1)
    watch = torch.tensor([3, 50, 199])
    parts = compress(p, k=8, watch=watch)
    assert parts["ids"].shape == (2, 4, 8) and parts["watch"].shape == (2, 4, 3)
    assert not torch.isin(parts["ids"], watch).any()
    assert (parts["logp"][..., :-1] >= parts["logp"][..., 1:]).all()  # descending
    total = parts["logp"].exp().sum(-1) + parts["watch"].exp().sum(-1) + parts["resid"].exp()
    assert torch.allclose(total, torch.ones(2, 4), atol=1e-5)
    assert torch.allclose(parts["watch"].exp(), p[..., watch], atol=1e-6)


def test_token_map_from_the_real_swap_table(tok):
    table = load_swap_pairs("data/swaps", name_pairs=load_names("train"))
    token_map, unmapped = build_token_map(tok, table)
    one = lambda s: tok(s, add_special_tokens=False)["input_ids"][0]
    edges = set(zip(token_map.src.tolist(), token_map.dst.tolist()))
    for a, b in ((" he", " she"), (" she", " he"), (" He", " She"), (" her", " him"),
                 (" her", " his"), (" his", " hers"), (" actress", " actor")):  # fmt: skip
        assert (one(a), one(b)) in edges, (a, b)
    assert (one(" actor"), one(" actress")) not in edges  # one-way
    male, female = load_names("train")[0]
    assert (one(" " + male), one(" " + female)) in edges
    assert " chairwoman" in unmapped  # several tokens
    assert one(" Miss") not in token_map.sources.tolist()  # reading depends on context
    sizes = dict(zip(token_map.sources.tolist(), token_map.group_size.tolist()))
    assert sizes[one(" her")] == 2 and sizes[one(" he")] == 1


def test_watch_list_is_the_train_occupation_tokens(tok):
    occupations = load_occupations("train")
    ids = watch_token_ids(tok, occupations)
    assert ids == sorted(set(ids)) and 40 <= len(ids) <= 100
    words = {tok.decode([i]).strip() for i in ids}
    assert "nurse" in words and "a" not in words and "an" not in words
    dev_only = load_occupations("dev", new_only=True)
    unseen = {tok.decode([i]).strip() for i in watch_token_ids(tok, dev_only)} - words
    assert unseen  # dev-only occupations add tokens the watch-list does not cover


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
    cfg = load_config(
        "configs/base.yaml",
        overrides={
            "device": "cpu",
            "topk": 8,
            "data": {
                "templates_dir": str(templates_root),
                "corpus_dir": str(tmp_path / "corpus"),
                "cache_dir": str(tmp_path / "cache"),
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


def test_cache_end_to_end_matches_direct_computation(toy_setup, tok):
    cfg, teacher = toy_setup
    out = cache_teacher_targets(cfg, model=teacher)
    meta = json.loads((out / "meta.json").read_text())
    assert (out / "manifest.json").exists() and not (out / "progress.json").exists()

    corpus = cfg["data"]["corpus_dir"]
    train = np.load(f"{corpus}/train_tokens.npy")
    copies = np.load(f"{corpus}/cf_tokens.npy")
    cf_index = np.load(f"{corpus}/cf_index.npy")
    assert 0 < len(copies) < len(train)  # some rows have a copy, some do not
    assert meta["rows"] == {"plain_orig": len(train), **dict.fromkeys(SETS[1:], len(copies))}

    for name in SETS:  # every stored position is a full distribution
        t = load_targets(out, name)
        total = (
            np.exp(t["logp"].astype(np.float64)).sum(-1)
            + np.exp(t["watch"].astype(np.float64)).sum(-1)
            + np.exp(t["resid"].astype(np.float64))
        )
        assert np.allclose(total, 1.0, atol=5e-3), name

    # Recompute one pair directly from full distributions and compare with the cache.
    table = load_swap_pairs(
        cfg["data"]["swaps_dir"], name_pairs=load_names("train", cfg["data"]["templates_dir"])
    )
    token_map, _ = build_token_map(tok, table)
    watch = torch.tensor(meta["watch_token_ids"])
    pair = len(copies) - 1
    with torch.no_grad():
        p = teacher(
            torch.from_numpy(np.stack([train[cf_index[pair]], copies[pair]]).astype(np.int64))
        ).logits.softmax(-1)
    direct = pair_targets(p[:1], p[1:], token_map)
    direct["plain_orig"] = p[:1]
    for name, target in direct.items():
        row = cf_index[pair] if name == "plain_orig" else pair
        stored = load_targets(out, name)
        expected = compress(target, cfg["topk"], watch)
        assert np.array_equal(stored["ids"][row], expected["ids"][0].numpy()), name
        assert np.allclose(stored["logp"][row], expected["logp"][0].numpy(), atol=2e-2), name
        assert np.allclose(
            stored["watch"][row], expected["watch"][0].numpy(), atol=5e-2, rtol=1e-3
        ), name

    assert cache_teacher_targets(cfg, model=teacher) == out  # finished cache is reused


def test_cache_resumes_after_interruption(toy_setup):
    cfg, teacher = toy_setup
    partial = cache_teacher_targets(cfg, model=teacher, max_batches=2)
    progress = json.loads((partial / "progress.json").read_text())
    assert progress["pairs"] == 4 and not (partial / "meta.json").exists()
    resumed = cache_teacher_targets(cfg, model=teacher)
    assert resumed == partial and (resumed / "meta.json").exists()

    fresh_cfg = {**cfg, "data": {**cfg["data"], "cache_dir": cfg["data"]["cache_dir"] + "_fresh"}}
    fresh = cache_teacher_targets(fresh_cfg, model=teacher)
    for name in SETS:
        a, b = load_targets(resumed, name), load_targets(fresh, name)
        for part in a:
            assert np.array_equal(a[part], b[part]), (name, part)


def test_cache_key_changes_with_topk(toy_setup):
    cfg, teacher = toy_setup
    first = cache_teacher_targets(cfg, model=teacher)
    second = cache_teacher_targets({**cfg, "topk": 4}, model=teacher)
    assert first != second
    assert load_targets(second, "mean")["ids"].shape[-1] == 4
