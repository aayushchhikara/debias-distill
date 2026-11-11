"""Tests for substitution lists, templates and corpus preprocessing."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from debias_distill.data.corpus import build_counterfactuals, sample_corpus
from debias_distill.data.names import (
    build_corpus_pairs,
    build_name_pairs,
    count_names,
    split_pairs,
    unpaired_candidates,
    usable,
)
from debias_distill.data.swaps import TokenSwapper, build_table, counterfactual, load_swap_pairs
from debias_distill.data.templates import (
    FINAL_FILES,
    FinalUnlock,
    fill,
    load_heldout_names,
    load_names,
    load_occupation_table,
    load_occupations,
    load_stems,
    load_templates,
    read_ledger,
    subject_pairs,
    unlock_final,
    verify_final_hashes,
    write_final_hashes,
)
from debias_distill.utils.config import load_config
from debias_distill.utils.hub import load_tokenizer
from debias_distill.utils.manifest import inputs_match, read_manifest

ROOT = Path("data/templates")
HAS_FINAL = all((ROOT / "final" / name).exists() for name in FINAL_FILES)


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


def test_final_split_locked():
    for loader in (load_templates, load_stems, load_names, load_occupations, subject_pairs):
        with pytest.raises(PermissionError):
            loader("final")
    with pytest.raises(PermissionError):
        load_templates("final", unlock=True)  # only a FinalUnlock from the ledger opens it


def test_unlock_final_writes_ledger_and_checks_hashes(templates_root):
    write_final_hashes(templates_root)
    assert read_ledger(templates_root) == []
    unlock = unlock_final("run-1", templates_root)
    assert isinstance(unlock, FinalUnlock)
    assert load_templates("final", templates_root, unlock) == ["{subj} tested as"]
    assert load_occupations("final", templates_root, unlock, new_only=True) == ["a tester"]
    ledger = read_ledger(templates_root)
    assert len(ledger) == 1 and ledger[0]["run_id"] == "run-1" and "commit" in ledger[0]

    (templates_root / "final" / "names.tsv").write_text("Zedd\tZara\n")
    with pytest.raises(RuntimeError, match="do not match"):
        unlock_final("run-2", templates_root)
    assert len(read_ledger(templates_root)) == 1


@pytest.mark.skipif(not HAS_FINAL, reason="the final split is not distributed")
def test_final_split_matches_recorded_hashes():
    verify_final_hashes()


def test_ledger_is_empty_before_the_final_evaluation():
    assert read_ledger() == []


def test_train_dev_disjoint():
    assert not set(load_templates("train")) & set(load_templates("dev"))
    train_names = {n for pair in load_names("train") for n in pair}
    dev_names = {n for pair in load_names("dev") for n in pair}
    assert not train_names & dev_names


def test_templates_and_stems():
    for split, min_stems in (("train", 18), ("dev", 16)):
        templates, stems = load_templates(split), load_stems(split)
        assert len(templates) == len(stems) == len(set(templates))
        assert len(set(stems)) >= min_stems
        assert all(t.endswith(stem) for t, stem in zip(templates, stems))
    assert not set(load_stems("train")) & set(load_stems("dev"))


def test_occupation_splits():
    train = load_occupations("train")
    dev_new = load_occupations("dev", new_only=True)
    assert len(train) == 40 and len(dev_new) == 12
    assert load_occupations("dev") == train + dev_new and not set(train) & set(dev_new)
    assert all(o.split()[0] in {"a", "an"} for o in train + dev_new)
    table = load_occupation_table("dev")
    assert {group for _, group, _ in table} == {"female_coded", "neutral", "male_coded"}
    for source in ("train", "dev"):  # every group is represented in each split
        assert len({group for _, group, first in table if first == source}) == 3


def test_fill():
    assert fill("{subj} worked as", "she", "a nurse") == "She worked as a nurse"
    assert fill("Last year, {subj} worked as", "he") == "Last year, he worked as"
    assert fill("{poss} first job was as", "he") == "His first job was as"
    assert fill("{poss} first job was as", "Mary") == "Mary's first job was as"


def test_subject_pairs_start_with_pronouns():
    assert subject_pairs("dev")[0] == ("he", "she")


# --- corpus pipeline (synthetic documents, no network) -----------------------


def _pipeline_cfg(templates_root, seq_len, rows, heldout_tokens):
    return load_config(
        "configs/base.yaml",
        overrides={
            "data": {"templates_dir": str(templates_root)},
            "corpus": {"seq_len": seq_len, "token_budget": seq_len * rows},
            "eval": {"perplexity_tokens": heldout_tokens},
        },
    )


def test_corpus_pipeline(tmp_path, templates_root):
    cfg = _pipeline_cfg(templates_root, 32, 20, 64)
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

    manifest = read_manifest(tmp_path)
    assert set(manifest["outputs"]) >= {"train_tokens.npy", "cf_tokens.npy", "cf_index.npy"}
    inputs = {"sampled_tokens.npy": tmp_path / "sampled_tokens.npy"}
    assert manifest["inputs"]["sampled_tokens.npy"]
    assert not inputs_match(tmp_path, inputs)  # the manifest records more inputs than this


def test_rejected_rows_excluded_from_training(tmp_path, templates_root):
    cfg = _pipeline_cfg(templates_root, 16, 6, 16)
    docs = ["\nGentleman Jim said that he would visit his aunt next week."] * 40
    sample_corpus(cfg, tmp_path, docs=docs)
    stats = build_counterfactuals(cfg, tmp_path)
    assert stats["rejected_rows"] > 0
    tok = load_tokenizer(cfg)
    for row in np.load(tmp_path / "train_tokens.npy"):
        assert "Gentleman" not in tok.decode(row)


def test_heldout_names_never_reach_training(tmp_path, templates_root, tok):
    cfg = _pipeline_cfg(templates_root, 16, 12, 16)
    dev_name = load_names("dev")[0][1]
    docs = [
        f"Yesterday {dev_name} went to the market to buy some fresh bread.",  # no gendered word
        "Yesterday Zoey said that she would visit the old museum again.",
        "The weather report predicted rain for the whole week ahead of us.",
    ] * 30
    sample_corpus(cfg, tmp_path, docs=docs)
    stats = build_counterfactuals(cfg, tmp_path)
    assert stats["rejected"]["heldout_name"] > 0
    assert not any(k.startswith("heldout_name:") for k in stats["rejected"])  # names not logged
    kept = tok.batch_decode(np.load(tmp_path / "train_tokens.npy"))
    assert kept and not any(dev_name in text or "Zoey" in text for text in kept)


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


def test_corpus_pairs_cover_every_candidate_outside_the_splits(tok):
    pairs, left = build_corpus_pairs(
        tok, ["John", "David", "Jayden"], ["Kate", "Jane", "Vanessa", "Mary"], taken={"Mary"}
    )
    assert left == [] and len(pairs) == 3
    assert ("Jayden", "Vanessa") in pairs  # both are two tokens with a leading space
    assert "Mary" not in {n for pair in pairs for n in pair}


def test_committed_name_lists_are_consistent(table):
    corpus_names = {
        n
        for row in Path("data/swaps/corpus_names.tsv").read_text().split("\n")
        if row
        for n in row.split("\t")
    }
    train = {n for pair in load_names("train") for n in pair}
    dev = {n for pair in load_names("dev") for n in pair}
    assert not corpus_names & (train | dev)
    assert corpus_names <= set(table.exact)
    if HAS_FINAL:
        assert unpaired_candidates("data/swaps", ROOT) == frozenset()
        assert not load_heldout_names() & set(table.exact)


def test_reject_words_remove_the_row(tok):
    table = load_swap_pairs("data/swaps", rejects={"Zoey": "heldout_name"})
    swapper = TokenSwapper(tok, table)
    assert swapper.swap(_ids(tok, " Zoey bought bread.")) is None
    assert swapper.last_status == "rejected"
    assert dict(swapper.stats.rejected) == {"heldout_name": 1}
    with pytest.raises(ValueError, match="both substituted and rejected"):
        build_table([], name_pairs=[("Adam", "Eve")], rejects={"Eve": "heldout_name"})


def test_split_pairs_round_robin():
    pairs = [(f"m{i}", f"f{i}") for i in range(10)]
    splits = split_pairs(pairs)
    assert [len(splits[s]) for s in ("train", "dev", "final")] == [6, 2, 2]
    assert sorted(p for v in splits.values() for p in v) == sorted(pairs)
