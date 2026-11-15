"""Tests for the evaluators. They use a tiny random model and synthetic splits only."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from debias_distill.data.templates import fill
from debias_distill.measure.benchmarks import crows_pairs_score, fetch_crows_pairs, load_crows_pairs
from debias_distill.measure.bias import (
    TemplateScores,
    score_templates,
    stem_bootstrap,
    summarise,
    with_interval,
)
from debias_distill.measure.capability import perplexity
from debias_distill.measure.compare import compare
from debias_distill.measure.evaluate import evaluate, load_items, save_items
from debias_distill.measure.probe import fold_masks, probe_features
from debias_distill.measure.scoring import token_logprobs
from debias_distill.utils.config import load_config
from debias_distill.utils.hub import load_checkpoint

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_model(tiny_lm):
    return tiny_lm()


@pytest.fixture
def small_root(tmp_path):
    """Two small splits: 4 stems x 2 templates, 2 name pairs, 3 train + 2 dev occupations."""
    root = tmp_path / "templates"
    frames = ["{subj} worked as", "{subj} trained as", "{poss} first job was as", "{subj} is"]
    for split, names, occupations in (
        ("train", "Adam\tBeth\nCarl\tDora\n", "a nurse\tfemale_coded\na pilot\tmale_coded\n"),
        ("dev", "Evan\tFaye\nGlen\tHope\n", "an engineer\tmale_coded\na florist\tfemale_coded\n"),
    ):
        folder = root / split
        folder.mkdir(parents=True)
        lines = [f"{f}\t{f}\n{f}\tLast year, {f}\n" for f in frames]
        (folder / "templates.tsv").write_text("".join(lines))
        (folder / "names.tsv").write_text(names)
        (folder / "occupations.txt").write_text(occupations)
    return root


def scores_from(ratios, stems, split="dev", first=("train", "dev")):
    """TemplateScores with 3 subject pairs whose male-minus-female log-ratio is `ratios[o]`."""
    logp = np.zeros((len(stems), 6, len(ratios)))
    logp[:, 0::2] = np.asarray(ratios, dtype=float)
    occupations = [(f"a job{i}", "neutral", first[i]) for i in range(len(ratios))]
    subjects = ["he", "she", "Al", "Bea", "Cy", "Di"]
    kl = np.zeros((len(stems), 3))
    return TemplateScores(split, list(stems), list(stems), subjects, occupations, logp, logp, kl)


# --- bias score ---------------------------------------------------------------


def test_score_templates_matches_a_direct_computation(tiny_model, tok, small_root):
    scores = score_templates(tiny_model, tok, "dev", CPU, root=small_root)
    assert scores.subjects == ["he", "she", "Evan", "Faye", "Glen", "Hope"]
    assert [o[0] for o in scores.occupations] == ["a nurse", "a pilot", "an engineer", "a florist"]
    assert scores.logp.shape == (8, 6, 4) and scores.template_kl.shape == (8, 3)

    t, s, o = 5, 3, 2  # "Last year, {poss} first job was as", Faye, "an engineer"
    assert scores.templates[t].startswith("Last year, {poss}")
    prompt = tok(fill(scores.templates[t], "Faye"), add_special_tokens=False)["input_ids"]
    occupation = tok(" an engineer", add_special_tokens=False)["input_ids"]
    values = token_logprobs(tiny_model, [[tok.eos_token_id, *prompt, *occupation]], CPU)[0]
    assert scores.logp[t, s, o] == pytest.approx(values[-len(occupation) :].sum(), abs=1e-4)
    assert scores.logp_no_article[t, s, o] == pytest.approx(
        values[-len(occupation) + 1 :].sum(), abs=1e-4
    )
    assert (scores.logp <= scores.logp_no_article).all()
    assert (scores.template_kl >= -1e-6).all() and scores.template_kl.max() > 0
    assert scores.log_ratios().shape == (8, 3, 4)


def test_summary_values_on_known_ratios():
    stems = ["a", "a", "b", "b"]
    scores = scores_from([0.4, -1.0], stems)
    summary = summarise(scores, n_bootstrap=200)
    assert summary["mean_abs_log_ratio"]["value"] == pytest.approx(0.7)
    assert summary["unseen_occupations"]["value"] == pytest.approx(1.0)  # the dev-first one
    assert summary["seen_occupations"]["value"] == pytest.approx(0.4)
    assert summary["signed_by_group"] == {"neutral": pytest.approx(-0.3)}
    assert summary["noise_floor"]["value"] == pytest.approx(0.0)
    # identical templates: the interval collapses onto the mean
    interval = summary["mean_abs_log_ratio"]
    assert interval["ci_low"] == pytest.approx(0.7) and interval["ci_high"] == pytest.approx(0.7)
    assert summary["counts"]["unseen_occupations"] == 1

    assert (
        summarise(scores_from([0.4, -1.0], stems, split="train"), 200)["unseen_occupations"] is None
    )


def test_noise_floor_compares_names_of_the_same_gender():
    scores = scores_from([0.0], ["a", "b"], first=("train",))
    scores.logp[:, 4] += 0.6  # the second male name only
    floor = scores.same_gender_log_ratios()
    assert floor.shape == (2, 4, 1)
    assert np.abs(floor).mean() == pytest.approx(
        0.3
    )  # two male comparisons of 0.6, two female of 0
    assert summarise(scores, 200)["pronoun_pair"]["value"] == pytest.approx(0.0)
    single_pair = TemplateScores(
        "dev", ["t"], ["t"], ["he", "she", "Al", "Bea"], scores.occupations,
        np.zeros((1, 4, 1)), np.zeros((1, 4, 1)), np.zeros((1, 2)),
    )  # fmt: skip
    assert summarise(single_pair, 200)["noise_floor"] is None


def test_bootstrap_resamples_stems_not_templates():
    stems = ["a"] * 6 + ["b"] * 2
    values = np.array([1.0] * 6 + [5.0] * 2)[:, None]
    means = stem_bootstrap(values, stems, n=2000, seed=1)
    assert set(np.round(means, 6)) == {1.0, 2.0, 5.0}  # (a, a), (a, b), (b, b)
    assert np.array_equal(means, stem_bootstrap(values, stems, n=2000, seed=1))
    interval = with_interval(values, stems, n=2000)
    assert interval["value"] == pytest.approx(2.0)
    assert interval["ci_low"] == pytest.approx(1.0) and interval["ci_high"] == pytest.approx(5.0)


# --- capability ---------------------------------------------------------------


def test_perplexity_of_a_uniform_model_is_the_vocabulary_size(tmp_path):
    class Uniform(torch.nn.Module):
        def forward(self, ids):
            return SimpleNamespace(logits=torch.zeros(*ids.shape, 100))

    np.save(tmp_path / "heldout_tokens.npy", np.arange(64, dtype=np.uint16).reshape(4, 16))
    cfg = {"data": {"corpus_dir": str(tmp_path)}, "eval": {"perplexity_tokens": 32}}
    result = perplexity(Uniform(), cfg, CPU)
    assert result["perplexity"] == pytest.approx(100.0, rel=1e-5)
    assert result["rows"] == 2 and result["tokens"] == 30


# --- probe --------------------------------------------------------------------


def test_probe_folds_hold_out_stems_and_pairs_together():
    stem_of = np.repeat(np.arange(8), 12)
    pair_of = np.tile(np.repeat(np.arange(6), 2), 8)
    folds = list(fold_masks(stem_of, pair_of))
    assert len(folds) == 4
    for train, test in folds:
        assert not set(stem_of[train]) & set(stem_of[test])
        assert not set(pair_of[train]) & set(pair_of[test])


def test_probe_finds_a_linear_direction_and_the_control_does_not():
    rng = np.random.default_rng(0)
    stem_of = np.repeat(np.arange(12), 16)
    pair_of = np.tile(np.repeat(np.arange(8), 2), 12)
    labels = np.tile([0, 1], 96)
    noise = rng.normal(size=(192, 8))
    separable = noise.copy()
    separable[:, 3] += 4.0 * (labels * 2 - 1)
    found = probe_features(separable, labels, stem_of, pair_of)
    assert found["accuracy"] > 0.95
    assert 0.2 < found["control_accuracy"] < 0.8
    assert 0.2 < probe_features(noise, labels, stem_of, pair_of)["accuracy"] < 0.8


# --- CrowS-Pairs --------------------------------------------------------------


def test_crows_pairs_loading_and_scoring(tiny_model, tok, tmp_path):
    path = tmp_path / "pairs.csv"
    header = (
        ",sent_more,sent_less,stereo_antistereo,bias_type,annotations,anon_writer,anon_annotators\n"
    )
    path.write_text(
        header
        + '0,"The red door is open.","The blue door is open.",stereo,gender,[],a,[]\n'
        + '1,"A cat sat down.","A dog sat down.",antistereo,gender,[],a,[]\n'
        + '2,"One, two.","Three, four.",stereo,age,[],a,[]\n'
    )
    rows = load_crows_pairs(path)
    assert [row["direction"] for row in rows] == ["stereo", "antistereo"]
    result = crows_pairs_score(tiny_model, tok, rows, CPU, n_bootstrap=100)
    assert result["pairs"] == 2 and result["stereotype_preference"] in (0.0, 0.5, 1.0)
    assert set(result["by_direction"]) == {"antistereo", "stereo"}

    with pytest.raises(RuntimeError, match="recorded hash"):
        fetch_crows_pairs({"path": str(path), "url": "unused", "sha256": "0" * 64})


# --- the evaluator ------------------------------------------------------------


def test_evaluate_writes_a_record_and_items(tiny_model, small_root, tmp_path):
    np.save(
        tmp_path / "heldout_tokens.npy",
        np.random.default_rng(0).integers(0, 50000, size=(3, 32)).astype(np.uint16),
    )
    cfg = load_config(
        "configs/base.yaml",
        {
            "device": "cpu",
            "data": {
                "templates_dir": str(small_root),
                "corpus_dir": str(tmp_path),
                "experiments_dir": str(tmp_path / "experiments"),
            },
            "eval": {"perplexity_tokens": 64},
            "benchmarks": {"crows_pairs": None},
        },
    )
    record = evaluate("student", "dev", cfg, model=tiny_model)
    run_dir = tmp_path / "experiments" / "eval_student_dev"
    assert record["reportable"] and record["run_dir"] == str(run_dir)
    assert record["bias"]["counts"] == {
        "templates": 8, "stems": 4, "subject_pairs": 3, "occupations": 4, "unseen_occupations": 2,
    }  # fmt: skip
    assert record["capability"]["rows"] == 2
    assert len(record["probe"]["layers"]) == 3 and "crows_pairs" not in record
    assert json.loads((run_dir / "record.json").read_text()) == record
    items = load_items(run_dir)
    assert summarise(items) == record["bias"]

    with pytest.raises(FileExistsError):  # a finished run is never overwritten
        evaluate("student", "dev", cfg, model=tiny_model)
    with pytest.raises(PermissionError):  # the final split needs the ledger
        evaluate("student", "final", cfg, model=tiny_model, run_id="final_try")
    train = evaluate("student", "train", cfg, model=tiny_model, parts=("bias",))
    assert not train["reportable"] and train["bias"]["unseen_occupations"] is None


def test_compare_pairs_the_two_sides_over_stems(tmp_path):
    stems = ["a", "a", "b", "c"]

    def run(name, ratios, perplexity_value, split="dev"):
        folder = tmp_path / name
        folder.mkdir()
        save_items(scores_from(ratios, stems, split=split), folder)
        record = {"reportable": split != "train", "capability": {"perplexity": perplexity_value}}
        (folder / "record.json").write_text(json.dumps(record))
        return str(folder)

    reference = [run("ref", [1.0, 1.0], 50.0)]
    seeds = [run("s1", [0.2, 0.4], 51.0), run("s2", [0.4, 0.6], 53.0)]
    result = compare(seeds, reference, n=500)
    assert result["model"]["value"] == pytest.approx(0.4)
    assert result["model"]["per_seed"] == [pytest.approx(0.3), pytest.approx(0.5)]
    assert result["reference"]["value"] == pytest.approx(1.0)
    assert result["difference"]["value"] == pytest.approx(-0.6)
    reduction = result["relative_reduction"]
    assert reduction["value"] == pytest.approx(0.6)
    assert reduction["ci_low"] == pytest.approx(0.6) and reduction["ci_high"] == pytest.approx(0.6)
    assert result["perplexity"]["relative_change"] == pytest.approx(0.04)
    assert result["reportable"]

    with pytest.raises(ValueError, match="different"):
        compare(seeds, [run("train_ref", [1.0, 1.0], 50.0, split="train")])


def test_load_checkpoint_from_a_saved_model(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path / "checkpoint")
    loaded = load_checkpoint({}, str(tmp_path / "checkpoint"), dtype=torch.float32)
    assert sum(p.numel() for p in loaded.parameters()) == sum(
        p.numel() for p in tiny_model.parameters()
    )
    with pytest.raises(FileNotFoundError):
        load_checkpoint({}, str(tmp_path / "missing"))
