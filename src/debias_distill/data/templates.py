"""Evaluation templates, split into train / dev / final.

Layout:
  data/templates/occupations.txt        one occupation per line, with its article
  data/templates/<split>/templates.txt  one template per line; contains {subj} or {poss}
                                        and ends where the occupation continuation starts
  data/templates/<split>/names.tsv      male<TAB>female name pairs for that split

The final split is locked: loading it raises unless `unlock_final=True` or the
environment variable DEBIAS_UNLOCK_FINAL=1 is set.
"""

import os
from pathlib import Path

SPLITS = ("train", "dev", "final")
UNLOCK_ENV = "DEBIAS_UNLOCK_FINAL"
PRONOUN_PAIR = ("he", "she")
_POSSESSIVE = {"he": "his", "she": "her"}


def _check_split(split: str, unlock_final: bool) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "final" and not (unlock_final or os.environ.get(UNLOCK_ENV) == "1"):
        raise PermissionError(
            "the final split is locked; pass unlock_final=True or set "
            f"{UNLOCK_ENV}=1 only for the one-time final evaluation"
        )


def _lines(path: Path) -> list[str]:
    return [
        ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")
    ]


def load_templates(
    split: str, root: str | Path = "data/templates", unlock_final: bool = False
) -> list[str]:
    """Load templates for a split."""
    _check_split(split, unlock_final)
    templates = _lines(Path(root) / split / "templates.txt")
    for t in templates:
        if ("{subj}" in t) == ("{poss}" in t):
            raise ValueError(f"template needs exactly one of {{subj}} / {{poss}}: {t!r}")
    return templates


def load_names(
    split: str, root: str | Path = "data/templates", unlock_final: bool = False
) -> list[tuple[str, str]]:
    """Load (male, female) name pairs for a split."""
    _check_split(split, unlock_final)
    return [tuple(ln.split("\t")) for ln in _lines(Path(root) / split / "names.tsv")]


def load_occupations(root: str | Path = "data/templates") -> list[str]:
    return _lines(Path(root) / "occupations.txt")


def subject_pairs(
    split: str, root: str | Path = "data/templates", unlock_final: bool = False
) -> list[tuple[str, str]]:
    """(male, female) subjects for a split: the pronoun pair plus that split's names."""
    return [PRONOUN_PAIR, *load_names(split, root, unlock_final)]


def fill(template: str, subject: str, occupation: str | None = None) -> str:
    """Fill {subj}/{poss} with `subject` and optionally append the occupation."""
    poss = _POSSESSIVE.get(subject.lower(), f"{subject}'s")
    text = template.replace("{subj}", subject).replace("{poss}", poss)
    text = text[0].upper() + text[1:]
    return f"{text} {occupation}" if occupation else text
