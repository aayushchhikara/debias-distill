"""First-name pairs: frequency-matched pairs for the evaluation splits, and extra
pairs used only when substituting the training corpus.

Split pairs (`data/templates/<split>/names.tsv`) are frozen. They were built from
candidates that are a single token with a leading space and are not also common
words, paired so that each pair occurs about equally often in the sampled corpus
(count ratio <= MAX_RATIO), then assigned to splits round-robin by frequency.
Rewriting them needs `--regenerate-splits`, which also changes the final split.

Corpus-only pairs (`data/swaps/corpus_names.tsv`) cover candidate names that are in
no split, so those names are swapped in training text too. They need no frequency
match. A pair must have the same token length with a leading space; pairs that also
match without one (sentence-initial position) are preferred.

Usage:
  python -m debias_distill.data.names                        # report only
  python -m debias_distill.data.names --write-corpus-pairs
  python -m debias_distill.data.names --regenerate-splits    # guarded; avoid after freezing
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from debias_distill.data.swaps import CORPUS_NAMES_FILE
from debias_distill.data.templates import SPLITS, _rows
from debias_distill.utils.config import load_config
from debias_distill.utils.hub import load_tokenizer

SPLIT_PATTERN = ("train", "train", "train", "dev", "final")  # 60 / 20 / 20
MAX_RATIO = 2.0
MIN_COUNT = 20


def _ids(tokenizer, s: str) -> list[int]:
    return tokenizer(s, add_special_tokens=False)["input_ids"]


def usable(tokenizer, name: str) -> bool:
    return len(_ids(tokenizer, " " + name)) == 1 and len(_ids(tokenizer, " " + name.lower())) > 1


def count_names(text: str, names: list[str]) -> Counter:
    """Exact-case whole-word occurrence counts of each name in `text`."""
    found = Counter(re.findall(r"\b[A-Z][a-z]+\b", text))
    return Counter({n: found[n] for n in names})


def build_name_pairs(
    tokenizer,
    male: list[str],
    female: list[str],
    counts: Counter,
    max_ratio: float = MAX_RATIO,
    min_count: int = MIN_COUNT,
) -> list[tuple[str, str]]:
    """Pair names with the closest corpus counts (greedy, smallest log-ratio first)."""

    def keep(names: list[str]) -> list[str]:
        return [n for n in dict.fromkeys(names) if counts[n] >= min_count and usable(tokenizer, n)]

    m, f = keep(male), keep(female)
    limit = math.log(max_ratio)
    candidates = sorted(
        (abs(math.log(counts[a] / counts[b])), a, b)
        for a in m
        for b in f
        if abs(math.log(counts[a] / counts[b])) <= limit
    )
    used: set[str] = set()
    pairs = []
    for _, a, b in candidates:
        if a not in used and b not in used:
            pairs.append((a, b))
            used.update((a, b))
    return sorted(pairs, key=lambda p: (-(counts[p[0]] + counts[p[1]]), p))


def split_pairs(pairs: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {"train": [], "dev": [], "final": []}
    for i, pair in enumerate(pairs):
        out[SPLIT_PATTERN[i % len(SPLIT_PATTERN)]].append(pair)
    return out


def build_corpus_pairs(
    tokenizer, male: list[str], female: list[str], taken: set[str], counts: Counter | None = None
) -> tuple[list[tuple[str, str]], list[str]]:
    """Pair the candidate names not in `taken`; return (pairs, names left unpaired).

    Two passes: first names with the same token length both with and without a
    leading space, then the rest by the leading-space length alone. Within a group
    the most frequent names are paired with each other.
    """
    counts = counts or Counter()

    def length(name: str, space: bool) -> int:
        return len(_ids(tokenizer, (" " if space else "") + name))

    def rest(names: list[str]) -> list[str]:
        free = [n for n in dict.fromkeys(names) if n not in taken]
        return sorted(free, key=lambda n: (-counts[n], n))

    pairs: list[tuple[str, str]] = []
    m, f = rest(male), rest(female)
    for key in (lambda n: (length(n, True), length(n, False)), lambda n: length(n, True)):
        groups_m, groups_f = defaultdict(list), defaultdict(list)
        for name in m:
            groups_m[key(name)].append(name)
        for name in f:
            groups_f[key(name)].append(name)
        m, f = [], []
        for k in sorted(set(groups_m) | set(groups_f), key=str):
            gm, gf = groups_m.get(k, []), groups_f.get(k, [])
            pairs.extend(zip(gm, gf))
            m.extend(gm[len(gf) :])
            f.extend(gf[len(gm) :])
    return pairs, sorted(m + f)


def unpaired_candidates(swap_dir: str | Path, templates_dir: str | Path) -> frozenset[str]:
    """Candidate names that are in no split and have no corpus-only partner."""
    swap_dir = Path(swap_dir)
    paired = {n for pair in _rows(swap_dir / CORPUS_NAMES_FILE) for n in pair}
    for split in SPLITS:
        paired.update(n for pair in _rows(Path(templates_dir) / split / "names.tsv") for n in pair)
    candidates = set()
    for gender in ("male", "female"):
        candidates.update((swap_dir / f"name_candidates_{gender}.txt").read_text().split())
    return frozenset(candidates - paired)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--write-corpus-pairs", action="store_true")
    ap.add_argument(
        "--regenerate-splits",
        action="store_true",
        help="rewrite the train/dev/final name pairs (changes the frozen final split)",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    swaps = Path(cfg["data"]["swaps_dir"])
    templates = Path(cfg["data"]["templates_dir"])
    corpus = Path(cfg["data"]["corpus_dir"])

    tok = load_tokenizer(cfg)
    male = (swaps / "name_candidates_male.txt").read_text().split()
    female = (swaps / "name_candidates_female.txt").read_text().split()
    text = "\n".join(tok.batch_decode(np.load(corpus / "sampled_tokens.npy")))
    counts = count_names(text, male + female)

    if args.regenerate_splits:
        pairs = build_name_pairs(tok, male, female, counts)
        (corpus / "name_counts.json").write_text(json.dumps(dict(counts.most_common()), indent=2))
        for split, members in split_pairs(pairs).items():
            path = templates / split / "names.tsv"
            path.write_text("".join(f"{a}\t{b}\n" for a, b in members))
            print(f"{split:5}: {len(members)} pairs -> {path}")

    # Names of all three splits are read here only to keep them out of the corpus-only pairs.
    taken = {n for s in SPLITS for pair in _rows(templates / s / "names.tsv") for n in pair}
    corpus_pairs, left = build_corpus_pairs(tok, male, female, taken, counts)
    print(f"corpus-only pairs: {len(corpus_pairs)}; candidates left unpaired: {len(left)}")
    if args.write_corpus_pairs:
        path = swaps / CORPUS_NAMES_FILE
        path.write_text("".join(f"{a}\t{b}\n" for a, b in corpus_pairs))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
