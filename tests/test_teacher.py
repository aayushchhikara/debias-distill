"""Tests for aligning teacher distributions."""

import pytest
import torch

from debias_distill.data.swaps import load_swap_pairs
from debias_distill.data.templates import load_names
from debias_distill.teacher.align import TokenMap, align, build_token_map

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
