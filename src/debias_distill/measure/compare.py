"""Compare two models (or arms) from their evaluation runs.

Each side is one or more evaluation run folders: several folders are the seeds of
one arm, and an arm's score is the mean over its seeds. Intervals come from the
same stem resamples on both sides, so the difference and the relative reduction
are paired over templates. Seed-to-seed spread is reported next to them.

Usage:
  python -m debias_distill.measure.compare \\
      --model experiments/eval_armA_seed1_dev experiments/eval_armA_seed2_dev \\
      --reference experiments/eval_student_dev
"""

import argparse
import json
from pathlib import Path

import numpy as np

from debias_distill.measure.bias import N_BOOTSTRAP, stem_bootstrap
from debias_distill.measure.evaluate import load_items


def _side(run_dirs: list[str]) -> dict:
    runs = [load_items(d) for d in run_dirs]
    records = [json.loads((Path(d) / "record.json").read_text()) for d in run_dirs]
    first = runs[0]
    for run in runs[1:]:
        same = (run.split, run.templates, run.subjects, run.occupations) == (
            first.split,
            first.templates,
            first.subjects,
            first.occupations,
        )
        if not same:
            raise ValueError("runs were evaluated on different items")
    per_seed = np.stack([np.abs(run.log_ratios()) for run in runs])
    perplexities = [r["capability"]["perplexity"] for r in records if "capability" in r]
    return {
        "split": first.split,
        "stems": first.stems,
        "items": per_seed.mean(0),  # seed-averaged |log-ratio| per (template, pair, occupation)
        "per_seed": [float(x.mean()) for x in per_seed],
        "perplexity": float(np.mean(perplexities)) if perplexities else None,
        "reportable": all(r["reportable"] for r in records),
    }


def _interval(draws: np.ndarray, value: float) -> dict:
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"value": float(value), "ci_low": float(low), "ci_high": float(high)}


def compare(
    model_runs: list[str], reference_runs: list[str], n: int = N_BOOTSTRAP, seed: int = 0
) -> dict:
    """Bias score of both sides, their paired difference and the relative reduction."""
    model, reference = _side(model_runs), _side(reference_runs)
    if (model["split"], model["stems"]) != (reference["split"], reference["stems"]):
        raise ValueError("the two sides were evaluated on different splits or templates")
    m = stem_bootstrap(model["items"], model["stems"], n, seed)
    r = stem_bootstrap(reference["items"], reference["stems"], n, seed)
    m_value, r_value = model["items"].mean(), reference["items"].mean()
    out = {
        "split": model["split"],
        "reportable": model["reportable"] and reference["reportable"],
        "model": {**_interval(m, m_value), "per_seed": model["per_seed"]},
        "reference": {**_interval(r, r_value), "per_seed": reference["per_seed"]},
        "difference": _interval(m - r, m_value - r_value),
        "relative_reduction": _interval(1 - m / r, 1 - m_value / r_value),
    }
    if model["perplexity"] and reference["perplexity"]:
        out["perplexity"] = {
            "model": model["perplexity"],
            "reference": reference["perplexity"],
            "relative_change": model["perplexity"] / reference["perplexity"] - 1,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", required=True, help="evaluation run folders (seeds)")
    ap.add_argument("--reference", nargs="+", required=True)
    args = ap.parse_args()
    print(json.dumps(compare(args.model, args.reference), indent=2))


if __name__ == "__main__":
    main()
