"""Evaluation templates, names and occupations, split into train / dev / final.

Layout under the templates root (`data/templates/`):
  <split>/templates.tsv     stem<TAB>template; the template contains {subj} or {poss}
                            and ends where the occupation continuation starts
  <split>/names.tsv         male<TAB>female name pairs used only in that split
  <split>/occupations.txt   occupation<TAB>group, for occupations first used in that split
  final.sha256              SHA-256 of the final split's files (the files stay untracked)
  final_ledger.jsonl        one line per unlock of the final split

Split roles:
  train  finds hotspots for the closed loop; never used to report a method's score
  dev    selects hyperparameters, decides stopping, and is reported before the final run
  final  evaluated once, at the end, through `unlock_final`

A split is evaluated on its own occupations plus those of the earlier splits, so
dev and final each contain occupations that no training signal has targeted.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from debias_distill.utils.manifest import file_sha256, git_state

SPLITS = ("train", "dev", "final")
FINAL_FILES = ("names.tsv", "occupations.txt", "templates.tsv")
HASH_FILE = "final.sha256"
LEDGER_FILE = "final_ledger.jsonl"
PRONOUN_PAIR = ("he", "she")
_POSSESSIVE = {"he": "his", "she": "her"}


@dataclass(frozen=True)
class FinalUnlock:
    """Proof that the final split was unlocked through the ledger. Create with `unlock_final`."""

    run_id: str


# --- final split integrity ---------------------------------------------------


def final_split_hashes(root: str | Path = "data/templates") -> dict[str, str]:
    return {name: file_sha256(Path(root) / "final" / name) for name in FINAL_FILES}


def write_final_hashes(root: str | Path = "data/templates") -> Path:
    """Record the final split's hashes. Run once, when the split is frozen."""
    path = Path(root) / HASH_FILE
    lines = [f"{digest}  {name}\n" for name, digest in final_split_hashes(root).items()]
    path.write_text("".join(lines))
    return path


def verify_final_hashes(root: str | Path = "data/templates") -> None:
    """Raise if the final split's files differ from the recorded hashes."""
    recorded = {}
    for line in (Path(root) / HASH_FILE).read_text().splitlines():
        digest, name = line.split()
        recorded[name] = digest
    if recorded != final_split_hashes(root):
        raise RuntimeError("the final split's files do not match the recorded hashes")


def unlock_final(run_id: str, root: str | Path = "data/templates") -> FinalUnlock:
    """Unlock the final split for one evaluation run and record it in the ledger."""
    verify_final_hashes(root)
    entry = {
        "run_id": run_id,
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        **git_state(),
    }
    with (Path(root) / LEDGER_FILE).open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return FinalUnlock(run_id)


def read_ledger(root: str | Path = "data/templates") -> list[dict]:
    path = Path(root) / LEDGER_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- loaders -----------------------------------------------------------------


def _check_split(split: str, unlock: FinalUnlock | None) -> None:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "final" and not isinstance(unlock, FinalUnlock):
        raise PermissionError(
            "the final split is locked; call unlock_final(run_id) for the one-time final evaluation"
        )


def _rows(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            rows.append(line.rstrip("\n").split("\t"))
    return rows


def _template_rows(split: str, root: str | Path, unlock: FinalUnlock | None) -> list[list[str]]:
    _check_split(split, unlock)
    rows = _rows(Path(root) / split / "templates.tsv")
    for row in rows:
        if len(row) != 2:
            raise ValueError(f"expected stem<TAB>template, got {row!r}")
        if ("{subj}" in row[1]) == ("{poss}" in row[1]):
            raise ValueError(f"template needs exactly one of {{subj}} / {{poss}}: {row[1]!r}")
    return rows


def load_templates(
    split: str, root: str | Path = "data/templates", unlock: FinalUnlock | None = None
) -> list[str]:
    """Templates for a split."""
    return [template for _, template in _template_rows(split, root, unlock)]


def load_stems(
    split: str, root: str | Path = "data/templates", unlock: FinalUnlock | None = None
) -> list[str]:
    """The stem of each template, in the same order as `load_templates`.

    Templates sharing a stem differ only in their context prefix, so they are not
    independent samples: confidence intervals are bootstrapped over stems.
    """
    return [stem for stem, _ in _template_rows(split, root, unlock)]


def load_names(
    split: str, root: str | Path = "data/templates", unlock: FinalUnlock | None = None
) -> list[tuple[str, str]]:
    """(male, female) name pairs for a split."""
    _check_split(split, unlock)
    return [(a, b) for a, b in _rows(Path(root) / split / "names.tsv")]


def load_heldout_names(root: str | Path = "data/templates") -> frozenset[str]:
    """All dev and final names, for excluding them from training data.

    This is the only read of the final split outside the final evaluation. Callers
    must use the result only as an exclusion set and must not log its members.
    """
    names: set[str] = set()
    for split in ("dev", "final"):
        for pair in _rows(Path(root) / split / "names.tsv"):
            names.update(pair)
    return frozenset(names)


def load_occupation_table(
    split: str,
    root: str | Path = "data/templates",
    unlock: FinalUnlock | None = None,
    new_only: bool = False,
) -> list[tuple[str, str, str]]:
    """(occupation, group, first_split) rows evaluated on `split`.

    Includes the occupations of earlier splits unless `new_only` is set.
    """
    _check_split(split, unlock)
    sources = [split] if new_only else SPLITS[: SPLITS.index(split) + 1]
    table = []
    for source in sources:
        for occupation, group in _rows(Path(root) / source / "occupations.txt"):
            table.append((occupation, group, source))
    return table


def load_occupations(
    split: str,
    root: str | Path = "data/templates",
    unlock: FinalUnlock | None = None,
    new_only: bool = False,
) -> list[str]:
    """Occupations (with their article) evaluated on `split`."""
    return [row[0] for row in load_occupation_table(split, root, unlock, new_only)]


def subject_pairs(
    split: str, root: str | Path = "data/templates", unlock: FinalUnlock | None = None
) -> list[tuple[str, str]]:
    """(male, female) subjects for a split: the pronoun pair plus that split's names."""
    return [PRONOUN_PAIR, *load_names(split, root, unlock)]


def fill(template: str, subject: str, occupation: str | None = None) -> str:
    """Fill {subj}/{poss} with `subject` and optionally append the occupation."""
    poss = _POSSESSIVE.get(subject.lower(), f"{subject}'s")
    text = template.replace("{subj}", subject).replace("{poss}", poss)
    text = text[0].upper() + text[1:]
    return f"{text} {occupation}" if occupation else text


def main() -> None:
    ap = argparse.ArgumentParser(description="Check or record the final split's file hashes.")
    ap.add_argument("--root", default="data/templates")
    ap.add_argument("--write-final-hash", action="store_true", help="record the current hashes")
    args = ap.parse_args()
    if args.write_final_hash:
        print(f"wrote {write_final_hashes(args.root)}")
    else:
        verify_final_hashes(args.root)
        print("final split matches the recorded hashes")


if __name__ == "__main__":
    main()
