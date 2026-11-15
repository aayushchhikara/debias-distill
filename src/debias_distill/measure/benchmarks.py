"""External bias benchmarks. Evaluation only: never a source of training data.

CrowS-Pairs (Nangia et al., 2020; CC BY-SA 4.0) pairs a more stereotypical sentence
with a less stereotypical one. The score is the share of pairs in which the model
gives the more stereotypical sentence the higher likelihood; 0.5 means no preference.
The file is downloaded to a gitignored folder at a pinned commit and checked against
a recorded hash. Its sentences are never printed or written to run records.
"""

import csv
import urllib.request
from pathlib import Path

import numpy as np

from debias_distill.measure.scoring import token_logprobs
from debias_distill.utils.manifest import file_sha256


def fetch_crows_pairs(settings: dict) -> Path:
    """Download the CrowS-Pairs file named in the config (if absent) and verify its hash."""
    path = Path(settings["path"])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(settings["url"], path)
    if file_sha256(path) != settings["sha256"]:
        raise RuntimeError(f"{path} does not match the recorded hash")
    return path


def load_crows_pairs(path: str | Path, bias_type: str = "gender") -> list[dict]:
    """Rows of one bias type: sent_more, sent_less and the pair's direction."""
    with Path(path).open(newline="") as f:
        return [
            {
                "sent_more": row["sent_more"],
                "sent_less": row["sent_less"],
                "direction": row["stereo_antistereo"],
            }
            for row in csv.DictReader(f)
            if row["bias_type"] == bias_type
        ]


def crows_pairs_score(
    model, tokenizer, rows: list[dict], device, n_bootstrap: int = 10_000, seed: int = 0
) -> dict:
    """Share of pairs where the more stereotypical sentence is likelier, with a 95% interval."""
    bos = tokenizer.eos_token_id

    def loglik(key: str) -> np.ndarray:
        sequences = [
            [bos, *tokenizer(row[key], add_special_tokens=False)["input_ids"]] for row in rows
        ]
        return np.array([v.sum() for v in token_logprobs(model, sequences, device, batch_size=64)])

    prefers = loglik("sent_more") > loglik("sent_less")
    direction = np.array([row["direction"] for row in rows])
    draws = np.random.default_rng(seed).integers(0, len(rows), size=(n_bootstrap, len(rows)))
    low, high = np.percentile(prefers[draws].mean(1), [2.5, 97.5])
    return {
        "stereotype_preference": float(prefers.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "by_direction": {
            str(d): float(prefers[direction == d].mean()) for d in sorted(set(direction))
        },
        "pairs": len(rows),
    }
