"""Fixtures shared by the test modules."""

import shutil
from pathlib import Path

import pytest

from debias_distill.utils.config import load_config
from debias_distill.utils.hub import load_tokenizer

ROOT = Path("data/templates")


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(load_config("configs/base.yaml"))


@pytest.fixture
def templates_root(tmp_path):
    """A copy of the train and dev splits with a synthetic final split."""
    root = tmp_path / "templates"
    for split in ("train", "dev"):
        shutil.copytree(ROOT / split, root / split)
    final = root / "final"
    final.mkdir()
    (final / "names.tsv").write_text("Zedd\tZoey\n")
    (final / "occupations.txt").write_text("a tester\tneutral\n")
    (final / "templates.tsv").write_text("{subj} tested as\t{subj} tested as\n")
    return root
