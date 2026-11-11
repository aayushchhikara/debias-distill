"""Unit tests for config, seeding, device selection and run logging."""

import json
import random

import numpy as np
import pytest
import torch
import yaml

from debias_distill.utils.config import load_config
from debias_distill.utils.device import get_device
from debias_distill.utils.runlog import RunLogger
from debias_distill.utils.seed import set_seed


def test_load_base_config():
    cfg = load_config("configs/base.yaml")
    assert cfg["teacher"] == "EleutherAI/pythia-410m"
    assert cfg["student"] == "EleutherAI/pythia-70m"


def test_overrides_deep_merge():
    cfg = load_config("configs/base.yaml", overrides={"distill": {"lr": 3e-4}})
    assert cfg["distill"]["lr"] == 3e-4
    assert cfg["distill"]["batch_size"] == 8  # sibling key preserved


def test_base_inheritance(tmp_path):
    (tmp_path / "parent.yaml").write_text(yaml.safe_dump({"a": 1, "nested": {"x": 1, "y": 2}}))
    (tmp_path / "child.yaml").write_text(
        yaml.safe_dump({"base": "parent.yaml", "nested": {"y": 3}})
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"a": 1, "nested": {"x": 1, "y": 3}}


def test_success_criteria_file():
    crit = load_config("configs/success_criteria.yaml")
    assert crit == {
        "min_bias_reduction": 0.50,
        "max_capability_drop": 0.05,
        "must_hold_on_split": "final",
    }


def test_manifest_detects_changed_inputs(tmp_path):
    from debias_distill.utils.manifest import config_hash, inputs_match, write_manifest

    source = tmp_path / "in.txt"
    source.write_text("a")
    (tmp_path / "out.txt").write_text("result")
    manifest = write_manifest(tmp_path, {"in": source}, {"k": 1}, outputs=["out.txt"])
    assert manifest["config_sha256"] == config_hash({"k": 1}) and "out.txt" in manifest["outputs"]
    assert inputs_match(tmp_path, {"in": source})
    source.write_text("b")
    assert not inputs_match(tmp_path, {"in": source})


def test_set_seed_reproducible():
    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item()

    set_seed(123)
    first = draw()
    set_seed(123)
    assert draw() == first


def test_get_device():
    assert get_device("cpu").type == "cpu"
    expected = "mps" if torch.backends.mps.is_available() else "cpu"
    assert get_device("mps").type == expected


def test_runlogger_writes_files(tmp_path):
    cfg = {"seed": 7, "device": "cpu"}
    with RunLogger("run1", cfg, root=tmp_path) as log:
        log.log(0, loss=1.5)
        log.log(1, loss=1.2)

    run = tmp_path / "run1"
    assert yaml.safe_load((run / "config.yaml").read_text()) == cfg
    meta = json.loads((run / "meta.json").read_text())
    assert meta["seed"] == 7 and meta["finished"] is not None
    assert "commit" in meta["git"]
    lines = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["loss"] for r in lines] == [1.5, 1.2]


def test_runlogger_refuses_existing_dir(tmp_path):
    RunLogger("dup", {}, root=tmp_path).close()
    with pytest.raises(FileExistsError):
        RunLogger("dup", {}, root=tmp_path)
