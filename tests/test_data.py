"""Tests for substitution lists, templates and corpus preprocessing."""

from collections import Counter
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from debias_distill.data.names import build_name_pairs, count_names, split_pairs, usable

TOKENIZER = "EleutherAI/pythia-70m"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def test_no_place_names_in_candidates():
    places = {"Virginia", "Victoria", "Madison", "Charlotte", "Austin", "Brittany"}
    for g in ("male", "female"):
        names = set(Path(f"data/swaps/name_candidates_{g}.txt").read_text().split())
        assert not names & places


# --- names -------------------------------------------------------------------


def test_name_filter_and_pairing(tok):
    assert usable(tok, "James") and not usable(tok, "Will")  # "will" is a common word
    counts = Counter(
        {
            "James": 100,
            "John": 400,
            "Will": 300,
            "Mary": 380,
            "Patricia": 110,
            "Grace": 300,
            "Ann": 5,
        }
    )
    pairs = build_name_pairs(
        tok, ["James", "John", "Will"], ["Mary", "Patricia", "Grace", "Ann"], counts
    )
    assert pairs == [("John", "Mary"), ("James", "Patricia")]  # closest counts, most frequent first
    # nothing pairs beyond the ratio limit
    assert build_name_pairs(tok, ["John"], ["Patricia"], counts) == []


def test_count_names_whole_words_exact_case():
    text = "Mary met Maryann. mary left. MARY? Mary's book."
    assert count_names(text, ["Mary"])["Mary"] == 2


def test_split_pairs_round_robin():
    pairs = [(f"m{i}", f"f{i}") for i in range(10)]
    splits = split_pairs(pairs)
    assert [len(splits[s]) for s in ("train", "dev", "final")] == [6, 2, 2]
    assert sorted(p for v in splits.values() for p in v) == sorted(pairs)
