"""Tests for substitution lists, templates and corpus preprocessing."""

from collections import Counter
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from debias_distill.data.names import build_name_pairs, count_names, split_pairs, usable
from debias_distill.data.templates import (
    fill,
    load_names,
    load_occupations,
    load_templates,
    subject_pairs,
)

TOKENIZER = "EleutherAI/pythia-70m"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def test_no_place_names_in_candidates():
    places = {"Virginia", "Victoria", "Madison", "Charlotte", "Austin", "Brittany"}
    for g in ("male", "female"):
        names = set(Path(f"data/swaps/name_candidates_{g}.txt").read_text().split())
        assert not names & places


# --- templates ---------------------------------------------------------------


def test_final_split_locked(monkeypatch):
    monkeypatch.delenv("DEBIAS_UNLOCK_FINAL", raising=False)
    with pytest.raises(PermissionError):
        load_templates("final")
    with pytest.raises(PermissionError):
        load_names("final")


def test_train_dev_disjoint():
    assert not set(load_templates("train")) & set(load_templates("dev"))
    train_names = {n for pair in load_names("train") for n in pair}
    dev_names = {n for pair in load_names("dev") for n in pair}
    assert not train_names & dev_names


def test_template_and_occupation_counts():
    assert len(load_templates("train")) >= 100
    assert len(load_templates("dev")) >= 40
    occ = load_occupations()
    assert 50 <= len(occ) <= 100 and len(set(occ)) == len(occ)
    assert all(o.split()[0] in {"a", "an"} for o in occ)


def test_fill():
    assert fill("{subj} worked as", "she", "a nurse") == "She worked as a nurse"
    assert fill("Last year, {subj} worked as", "he") == "Last year, he worked as"
    assert fill("{poss} first job was as", "he") == "His first job was as"
    assert fill("{poss} first job was as", "Mary") == "Mary's first job was as"


def test_subject_pairs_start_with_pronouns():
    assert subject_pairs("dev")[0] == ("he", "she")


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
