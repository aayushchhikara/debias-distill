"""Tests for substitution lists, templates and corpus preprocessing."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from transformers import AutoTokenizer

from debias_distill.data.corpus import build_counterfactuals, sample_corpus
from debias_distill.data.names import build_name_pairs, count_names, split_pairs, usable
from debias_distill.data.swaps import TokenSwapper, build_table, counterfactual, load_swap_pairs
from debias_distill.data.templates import (
    fill,
    load_names,
    load_occupations,
    load_templates,
    subject_pairs,
)
from debias_distill.utils.config import load_config

TOKENIZER = "EleutherAI/pythia-70m"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture(scope="module")
def table():
    return load_swap_pairs("data/swaps", name_pairs=load_names("train"))


# --- text-level substitution -------------------------------------------------


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("He said she was tired.", "She said he was tired."),
        ("I saw her and waved.", "I saw him and waved."),
        ("I saw her.", "I saw him."),
        ("Her book is on his desk.", "His book is on her desk."),
        ("The book is his.", "The book is hers."),
        ("The book is hers.", "The book is his."),
        ("He hurt himself.", "She hurt herself."),
        ("My father and his brothers.", "My mother and her sisters."),
        ("Mr Smith met the king.", "Mrs Smith met the queen."),
        ("After his 20 years there.", "After her 20 years there."),
        ("She raised her 2 children.", "He raised his 2 children."),
        ('He said "thank her."', 'She said "thank him."'),
        ('She read her "best" poem.', 'He read his "best" poem.'),
        ("Was it his?", "Was it hers?"),
        # possessive vs object after "her"
        ("Her family was there.", "His family was there."),
        ("She lost her only son.", "He lost his only daughter."),
        ("He gave her all of it.", "She gave him all of it."),
        ("He loved her deeply.", "She loved him deeply."),
        ("I miss her - truly.", "I miss him - truly."),
        ("He paid her $500.", "She paid him $500."),
        # titles and one-way substitutions
        ("Ms Lee said he left.", "Mr Lee said she left."),
        ("Miss Smith said he left.", "Mr Smith said she left."),
        ("The actress said she won.", "The actor said he won."),
        ("My mum said he was late.", "My dad said she was late."),
        # job titles and family words
        ("The chairman and his wife.", "The chairwoman and her husband."),
        ("My grandpa and his dad.", "My grandma and her mom."),
    ],
)
def test_counterfactual_text(table, src, expected):
    out, unhandled = counterfactual(src, table)
    assert out == expected
    assert unhandled == []


def test_whole_words_only(table):
    text = "The heroes managed the shelter; mankind hesitated."
    assert counterfactual(text, table) == (text, [])


def test_names_swap_only_when_capitalised(table):
    male, female = load_names("train")[0]
    assert counterfactual(f"{male} called {female}.", table)[0] == f"{female} called {male}."
    assert counterfactual(male.lower(), table)[0] == male.lower()


def test_blockers_are_only_listed_words(table):
    assert "pregnant" in table.blockers and "prostate" in table.blockers
    assert not table.blockers & {"of", "no", "any", "these", "words", "sequences"}  # comment line
    assert table.blocked_word("He said it was of no use.") is None


def test_miss_as_verb_is_unhandled(table):
    assert counterfactual("Miss you, he said.", table)[1] == ["Miss"]
    assert counterfactual("I miss him.", table) == ("I miss her.", [])


def test_milliseconds_not_a_title(table):
    assert counterfactual("It took 5 ms and he waited.", table)[0] == "It took 5 ms and she waited."


def test_all_caps_reported_unhandled(table):
    out, unhandled = counterfactual("HE left.", table)
    assert out == "HE left." and unhandled == ["HE"]


def test_no_place_names_in_candidates():
    places = {"Virginia", "Victoria", "Madison", "Charlotte", "Austin", "Brittany"}
    for g in ("male", "female"):
        names = set(Path(f"data/swaps/name_candidates_{g}.txt").read_text().split())
        assert not names & places


def test_token_and_text_agree_on_context(tok, table):
    swapper = TokenSwapper(tok, table)
    for text in [" After his 20 years there, she raised her 2 kids.", ' He said "thank her."']:
        out = swapper.swap(_ids(tok, text))
        assert out is not None
        assert tok.decode(out) == counterfactual(text, table)[0]


def test_conflicting_pairs_rejected():
    with pytest.raises(ValueError, match="no resolver"):
        build_table([("man", "woman"), ("man", "lady")])


def test_name_lists_disjoint_from_words(table):
    assert not {n.lower() for n in table.exact} & set(table.words)


# --- token-level substitution ------------------------------------------------


def _ids(tok, text):
    return np.array(tok(text, add_special_tokens=False)["input_ids"], dtype=np.uint16)


def test_token_swap_preserves_length_and_matches_text(tok, table):
    swapper = TokenSwapper(tok, table)
    text = " He told her that his father would call him."
    ids = _ids(tok, text)
    out = swapper.swap(ids)
    assert out is not None and out.shape == ids.shape
    assert tok.decode(out) == counterfactual(text, table)[0]
    changed = np.flatnonzero(out != ids)
    assert len(changed) == 5


def test_token_swap_multi_token_equal_length(tok, table):
    swapper = TokenSwapper(tok, table)
    # "Grandmother" / "Grandfather" (no leading space) are both 2 tokens.
    for text in ["\nGrandmother said he would come.", " The chairmen thanked their wives."]:
        ids = _ids(tok, text)
        out = swapper.swap(ids)
        assert out is not None and out.shape == ids.shape
        assert tok.decode(out) == counterfactual(text, table)[0]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("\nGentleman said he would come.", "length_mismatch:Gentleman->Lady"),
        (" The chairman said he would come.", "length_mismatch:chairman->chairwoman"),
        (" She is pregnant and he is happy.", "blocked:pregnant"),
        (" Then he took his", "no_context_at_row_end:his"),
        (" Miss you, he said.", "unhandled:Miss"),
    ],
)
def test_token_swap_rejections(tok, table, text, reason):
    swapper = TokenSwapper(tok, table)
    assert swapper.swap(_ids(tok, text)) is None
    assert swapper.last_status == "rejected"
    assert list(swapper.stats.rejected) == [reason]


def test_token_swap_skips_rows_without_listed_words(tok, table):
    swapper = TokenSwapper(tok, table)
    assert swapper.swap(_ids(tok, " The weather was nice today.")) is None
    assert swapper.stats.with_listed_words == 0


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


# --- corpus pipeline (synthetic documents, no network) -----------------------


def test_corpus_pipeline(tmp_path):
    cfg = load_config(
        "configs/base.yaml",
        overrides={
            "corpus": {"seq_len": 32, "token_budget": 32 * 20},
            "eval": {"perplexity_tokens": 64},
        },
    )
    docs = [
        "He went to the market and bought apples for his mother.",
        "The weather report predicted rain for the whole week ahead of us.",
        "She told her brother that the train would be late again tonight.",
    ] * 30
    meta = sample_corpus(cfg, tmp_path, docs=docs)
    assert meta["heldout"]["rows"] == 2 and meta["train"]["rows"] == 20

    stats = build_counterfactuals(cfg, tmp_path)
    sampled = np.load(tmp_path / "sampled_tokens.npy")
    train = np.load(tmp_path / "train_tokens.npy")
    rejected = np.load(tmp_path / "rejected_tokens.npy")
    assert len(train) + len(rejected) == len(sampled) == stats["sequences"]
    assert stats["train_rows"] == len(train) and stats["rejected_rows"] == len(rejected)
    cf = np.load(tmp_path / "cf_tokens.npy")
    idx = np.load(tmp_path / "cf_index.npy")
    assert cf.shape[1] == train.shape[1] and len(cf) == len(idx) == stats["produced"] > 0
    assert all((cf[k] != train[i]).any() for k, i in enumerate(idx))
    assert json.loads((tmp_path / "cf_stats.json").read_text())["produced"] == stats["produced"]


def test_rejected_rows_excluded_from_training(tmp_path):
    cfg = load_config(
        "configs/base.yaml",
        overrides={
            "corpus": {"seq_len": 16, "token_budget": 16 * 6},
            "eval": {"perplexity_tokens": 16},
        },
    )
    docs = ["\nGentleman Jim said that he would visit his aunt next week."] * 40
    sample_corpus(cfg, tmp_path, docs=docs)
    stats = build_counterfactuals(cfg, tmp_path)
    assert stats["rejected_rows"] > 0
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    for row in np.load(tmp_path / "train_tokens.npy"):
        assert "Gentleman" not in tok.decode(row)


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
