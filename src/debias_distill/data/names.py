"""Build paired first-name lists and split them across train / dev / final.

Candidates come from data/swaps/name_candidates_{male,female}.txt. A name is kept
only if it is a single token with a leading space (its usual in-sentence form)
and its lowercase form is NOT a single token (filters names that are also common
words, e.g. "Will", "Mark", "Grace").

Male and female names are paired so that each pair occurs about equally often in
the sampled corpus (count ratio <= `max_ratio`); names with no partner in range
are dropped. Pairs are then ordered by frequency and assigned to splits
round-robin, so each split spans the same frequency range.

Requires the sampled corpus (`sampled_tokens.npy`) for the counts.
Usage: python -m debias_distill.data.names
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from debias_distill.utils.config import load_config

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    swaps = Path(cfg["data"]["swaps_dir"])
    templates = Path(cfg["data"]["templates_dir"])
    corpus = Path(cfg["data"]["corpus_dir"])

    tok = AutoTokenizer.from_pretrained(cfg["student"])
    male = (swaps / "name_candidates_male.txt").read_text().split()
    female = (swaps / "name_candidates_female.txt").read_text().split()
    text = "\n".join(tok.batch_decode(np.load(corpus / "sampled_tokens.npy")))
    counts = count_names(text, male + female)
    pairs = build_name_pairs(tok, male, female, counts)

    (corpus / "name_counts.json").write_text(json.dumps(dict(counts.most_common()), indent=2))
    ratios = [counts[a] / counts[b] for a, b in pairs]
    print(
        f"{len(pairs)} pairs; male/female count ratio median {np.median(ratios):.2f}, "
        f"range {min(ratios):.2f}-{max(ratios):.2f}"
    )
    for split, split_pairs_ in split_pairs(pairs).items():
        path = templates / split / "names.tsv"
        path.write_text("".join(f"{a}\t{b}\n" for a, b in split_pairs_))
        print(f"{split:5}: {len(split_pairs_)} pairs -> {path}")


if __name__ == "__main__":
    main()
