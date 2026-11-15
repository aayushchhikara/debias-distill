"""Tests for the distillation loss and the training loop (toy models and a toy cache only)."""

import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from debias_distill.distill.losses import next_token_loss, truncated_kl
from debias_distill.distill.train import (
    HARD,
    batch_order,
    build_examples,
    learning_rate_factor,
    train,
)
from debias_distill.teacher.cache_targets import (
    SETS,
    cache_teacher_targets,
    compress,
    finished_cache,
)
from debias_distill.utils.config import load_config

WATCH = torch.tensor([3, 50, 199])


def teacher_parts(seed: int = 0, k: int = 8):
    torch.manual_seed(seed)
    logits = torch.randn(2, 4, 200) * 3
    return logits, compress(logits.softmax(-1), k=k, watch=WATCH)


def loss(student_logits, parts, **kwargs):
    return truncated_kl(
        student_logits, parts["ids"], parts["logp"], WATCH, parts["watch"], **kwargs
    )


# --- loss ---------------------------------------------------------------------


def test_loss_is_zero_when_the_student_matches_the_teacher():
    logits, parts = teacher_parts()
    assert loss(logits, parts).item() == pytest.approx(0.0, abs=1e-5)
    assert loss(logits, parts, k=3, use_watch=False).item() == pytest.approx(0.0, abs=1e-5)
    assert loss(torch.zeros_like(logits), parts).item() > 0.1


def test_loss_equals_the_kl_over_kept_tokens_and_the_rest_bucket():
    logits, parts = teacher_parts()
    student = torch.randn(2, 4, 200)
    p, q = logits.softmax(-1), student.softmax(-1)

    def bucket_kl(kept: torch.Tensor) -> float:
        p_kept, q_kept = p.gather(-1, kept), q.gather(-1, kept)
        p_rest, q_rest = 1 - p_kept.sum(-1), 1 - q_kept.sum(-1)
        kl = (p_kept * (p_kept / q_kept).log()).sum(-1) + p_rest * (p_rest / q_rest).log()
        return kl.mean().item()

    full = torch.cat([parts["ids"], WATCH.expand(2, 4, 3)], dim=-1)
    assert loss(student, parts).item() == pytest.approx(bucket_kl(full), abs=1e-4)
    assert loss(student, parts, k=3, use_watch=False).item() == pytest.approx(
        bucket_kl(parts["ids"][..., :3]), abs=1e-4
    )
    # a coarser partition can only lose information
    assert loss(student, parts, k=3, use_watch=False).item() <= loss(student, parts).item() + 1e-6


def test_loss_handles_float16_targets_and_can_be_minimised():
    logits, parts = teacher_parts()
    stored = {name: value.to(torch.float16) for name, value in parts.items() if name != "ids"}
    stored["ids"] = parts["ids"]
    student = torch.zeros(2, 4, 200, requires_grad=True)
    optimizer = torch.optim.Adam([student], lr=0.2)
    first = loss(student, stored).item()
    for _ in range(300):
        optimizer.zero_grad()
        value = loss(student, stored)
        value.backward()
        optimizer.step()
    assert value.item() < 0.01 * first
    kept = torch.cat([parts["ids"], WATCH.expand(2, 4, 3)], dim=-1)
    learned = student.detach().softmax(-1).gather(-1, kept)
    assert torch.allclose(learned, logits.softmax(-1).gather(-1, kept), atol=0.02)


def test_loss_stays_finite_when_the_student_puts_all_mass_on_a_kept_token():
    _, parts = teacher_parts()
    student = torch.zeros(2, 4, 200)
    student.scatter_(-1, parts["ids"][..., :1], 80.0)  # probability 1 in float32
    student.requires_grad_()
    value = loss(student, parts)
    value.backward()
    assert torch.isfinite(value) and torch.isfinite(student.grad).all()


def test_next_token_loss_matches_cross_entropy():
    torch.manual_seed(0)
    logits, tokens = torch.randn(2, 5, 30), torch.randint(0, 30, (2, 5))
    expected = -logits[:, :-1].log_softmax(-1).gather(-1, tokens[:, 1:, None]).mean()
    assert next_token_loss(logits, tokens).item() == pytest.approx(expected.item(), abs=1e-6)


# --- examples and schedule ----------------------------------------------------


def test_examples_for_each_arm():
    cf_index = np.array([1, 3])  # rows 1 and 3 have a copy
    name = lambda row: SETS[row[2]]

    a = build_examples("originals", "plain", 5, cf_index)
    assert a.tolist() == [[0, r, SETS.index("plain_orig"), r] for r in range(5)]

    c = build_examples("both", "plain", 5, cf_index)
    assert len(c) == 7 and [name(r) for r in c[5:]] == ["plain_cf", "plain_cf"]
    assert c[5:, [0, 1, 3]].tolist() == [[1, 0, 0], [1, 1, 1]]

    e = build_examples("both", "aligned", 5, cf_index)
    assert [name(r) for r in e] == [
        "plain_orig", "aligned_orig", "plain_orig", "aligned_orig", "plain_orig",
        "aligned_cf", "aligned_cf",
    ]  # fmt: skip
    assert e[[1, 3], 3].tolist() == [0, 1]  # pair index, not row index
    assert e[[0, 2, 4], 3].tolist() == [0, 2, 4]

    d = build_examples("both", "mean", 5, cf_index)
    assert [name(r) for r in d[[1, 3, 5, 6]]] == ["mean"] * 4

    b = build_examples("both", "hard", 5, cf_index)
    assert (b[:, 2] == HARD).all() and len(b) == 7

    with pytest.raises(ValueError):
        build_examples("copies", "plain", 5, cf_index)
    with pytest.raises(ValueError):
        build_examples("both", "average", 5, cf_index)


def test_batch_order_covers_every_example_once_per_pass():
    first = np.concatenate([batch_order(12, 4, i, seed=1) for i in range(3)])
    second = np.concatenate([batch_order(12, 4, i, seed=1) for i in range(3, 6)])
    assert sorted(first) == sorted(second) == list(range(12))
    assert first.tolist() != second.tolist()
    assert batch_order(12, 4, 1, seed=1).tolist() == first[4:8].tolist()
    assert batch_order(12, 4, 1, seed=2).tolist() != first[4:8].tolist()


def test_learning_rate_schedule():
    factors = [learning_rate_factor(s, warmup=4, total=20) for s in range(20)]
    assert factors[:4] == [0.25, 0.5, 0.75, 1.0]
    assert factors[4] == 1.0 and all(a >= b for a, b in itertools.pairwise(factors[4:]))
    assert 0 <= factors[-1] < 0.05


# --- training loop ------------------------------------------------------------


@pytest.fixture
def toy_run(toy_setup):
    cfg, teacher = toy_setup
    cache_teacher_targets(cfg, model=teacher)
    distill = {
        "steps": 6, "grad_accum": 2, "batch_size": 2, "warmup_steps": 1, "lr": 1e-3,
        "checkpoint_every": 2, "log_every": 1,
    }  # fmt: skip
    return load_config("configs/base.yaml", {**cfg, "distill": distill})


def with_distill(cfg: dict, **settings) -> dict:
    out = copy.deepcopy(cfg)
    out["distill"].update(settings)
    return out


def weights(path) -> dict:
    from safetensors.torch import load_file

    return load_file(f"{path}/model.safetensors")


def metrics(cfg, run_id) -> list[dict]:
    path = Path(cfg["data"]["experiments_dir"]) / run_id / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_training_runs_and_reduces_the_loss(toy_run, tiny_lm):
    cfg = with_distill(toy_run, steps=30, lr=3e-3)
    path = train(cfg, model=tiny_lm())
    logged = metrics(cfg, "armA_seed1234")
    assert [m["step"] for m in logged] == list(range(1, 31))
    assert np.mean([m["loss"] for m in logged[-5:]]) < 0.9 * np.mean(
        [m["loss"] for m in logged[:5]]
    )
    summary = json.loads((Path(path).parent / "train_summary.json").read_text())
    assert summary["steps"] == 30 and summary["rows_per_step"] == 4
    assert train(cfg, model=tiny_lm()) == path  # finished runs are not repeated


def test_interrupted_training_resumes_to_the_same_weights(toy_run, tiny_lm):
    full = train(toy_run, run_id="full", model=tiny_lm())

    class Interrupted(RuntimeError):
        pass

    model = tiny_lm()
    calls = {"n": 0}

    def stop_during_step_five(module, args):
        calls["n"] += 1
        if calls["n"] > 8:  # two forward passes per step
            raise Interrupted

    hook = model.register_forward_pre_hook(stop_during_step_five)
    with pytest.raises(Interrupted):
        train(toy_run, run_id="resumed", model=model)
    hook.remove()

    resumed = train(toy_run, run_id="resumed", model=tiny_lm())
    assert [m["step"] for m in metrics(toy_run, "resumed")] == [1, 2, 3, 4, 5, 6]
    a, b = weights(full), weights(resumed)
    assert all(torch.allclose(a[name], b[name], atol=1e-6) for name in a)
    meta_path = Path(toy_run["data"]["experiments_dir"]) / "resumed" / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert len(meta["resumed"]) == 1


@pytest.mark.parametrize(
    "data,target", [("both", "plain"), ("both", "mean"), ("both", "aligned"), ("both", "hard")]
)
def test_every_arm_trains(toy_run, tiny_lm, data, target):
    cfg = with_distill(toy_run, arm="X", data=data, target=target, steps=2)
    path = train(cfg, model=tiny_lm())
    assert path.endswith("armX_seed1234/checkpoint")
    summary = json.loads((Path(path).parent / "train_summary.json").read_text())
    assert summary["examples"] > len(np.load(f"{cfg['data']['corpus_dir']}/train_tokens.npy"))
    assert np.isfinite(summary["final_loss"])


def test_training_needs_a_finished_cache(toy_setup):
    cfg, _ = toy_setup
    with pytest.raises(FileNotFoundError, match="no finished teacher cache"):
        finished_cache(cfg)
