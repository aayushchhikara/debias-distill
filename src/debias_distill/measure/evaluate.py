"""Evaluate one model on one split with every metric.

`evaluate(checkpoint, split, cfg)` returns a metrics record and writes it, with the
per-item scores, to a new folder under `experiments/`. `checkpoint` is "teacher" or
"student" (the pinned models of the config) or the path of a saved student.

Split roles: scores on the train split exist only to find hotspots for the closed
loop and are marked as not reportable. The final split needs a `FinalUnlock` and is
not reachable from the command line.

Usage:
  python -m debias_distill.measure.evaluate --checkpoint teacher --split dev
  python -m debias_distill.measure.evaluate --checkpoint experiments/<run>/checkpoint --split dev
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from debias_distill.data.templates import FinalUnlock
from debias_distill.measure.benchmarks import crows_pairs_score, fetch_crows_pairs, load_crows_pairs
from debias_distill.measure.bias import TemplateScores, score_templates, summarise
from debias_distill.measure.capability import perplexity
from debias_distill.measure.probe import probe_accuracy
from debias_distill.utils.config import load_config
from debias_distill.utils.device import get_device
from debias_distill.utils.hub import load_checkpoint, load_tokenizer
from debias_distill.utils.manifest import config_hash
from debias_distill.utils.runlog import RunLogger

PARTS = ("bias", "perplexity", "probe", "crows_pairs")


def save_items(scores: TemplateScores, run_dir: Path) -> None:
    """Write the per-item scores behind the bias metrics."""
    np.savez_compressed(
        run_dir / "bias_items.npz",
        logp=scores.logp,
        logp_no_article=scores.logp_no_article,
        template_kl=scores.template_kl,
    )
    index = {
        "split": scores.split,
        "templates": scores.templates,
        "stems": scores.stems,
        "subjects": scores.subjects,
        "occupations": scores.occupations,
    }
    (run_dir / "bias_index.json").write_text(json.dumps(index, indent=2))


def load_items(run_dir: str | Path) -> TemplateScores:
    """Read back the per-item scores of an evaluation run."""
    run_dir = Path(run_dir)
    index = json.loads((run_dir / "bias_index.json").read_text())
    arrays = np.load(run_dir / "bias_items.npz")
    return TemplateScores(
        index["split"],
        index["templates"],
        index["stems"],
        index["subjects"],
        [tuple(row) for row in index["occupations"]],
        arrays["logp"],
        arrays["logp_no_article"],
        arrays["template_kl"],
    )


def evaluate(
    checkpoint: str,
    split: str,
    cfg: dict,
    run_id: str | None = None,
    unlock: FinalUnlock | None = None,
    parts: tuple[str, ...] = PARTS,
    model=None,
    progress: bool = False,
) -> dict:
    """Evaluate `checkpoint` on `split` and return the metrics record.

    `model` overrides loading the checkpoint (used by tests and by callers that
    already hold the model in memory).
    """
    label = Path(checkpoint).parent.name if Path(checkpoint).is_dir() else checkpoint
    run_id = run_id or f"eval_{label}_{split}"
    root = cfg["data"]["templates_dir"]
    device = get_device(cfg["device"])
    tokenizer = load_tokenizer(cfg)
    if model is None:
        model = load_checkpoint(cfg, checkpoint, dtype=torch.float32)
    model = model.to(device).eval()

    record: dict = {
        "checkpoint": checkpoint,
        "split": split,
        "reportable": split != "train",
        "parameters": sum(p.numel() for p in model.parameters()),
        "config_sha256": config_hash(cfg),
    }
    with RunLogger(run_id, cfg, root=cfg["data"].get("experiments_dir", "experiments")) as run:
        if "bias" in parts:
            scores = score_templates(
                model, tokenizer, split, device, root, unlock, progress=progress
            )
            save_items(scores, run.dir)
            record["bias"] = summarise(scores)
        if "perplexity" in parts:
            record["capability"] = perplexity(model, cfg, device, progress=progress)
        if "probe" in parts:
            record["probe"] = probe_accuracy(model, tokenizer, split, device, root, unlock)
        if "crows_pairs" in parts and cfg.get("benchmarks", {}).get("crows_pairs"):
            rows = load_crows_pairs(fetch_crows_pairs(cfg["benchmarks"]["crows_pairs"]))
            record["crows_pairs"] = crows_pairs_score(model, tokenizer, rows, device)
        record["run_dir"] = str(run.dir)
        (run.dir / "record.json").write_text(json.dumps(record, indent=2))
    return record


def headline(record: dict) -> dict:
    """The few numbers worth printing after a run."""
    out = {"checkpoint": record["checkpoint"], "split": record["split"]}
    if "bias" in record:
        for key in ("mean_abs_log_ratio", "noise_floor", "unseen_occupations", "template_kl"):
            out[key] = record["bias"][key]
    if "capability" in record:
        out["perplexity"] = record["capability"]["perplexity"]
    if "probe" in record:
        out["probe_max_accuracy"] = record["probe"]["max_accuracy"]
        out["probe_control"] = record["probe"]["mean_control_accuracy"]
    if "crows_pairs" in record:
        out["crows_pairs"] = record["crows_pairs"]["stereotype_preference"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help='"teacher", "student" or a saved model')
    ap.add_argument("--split", default=None, choices=("train", "dev"))
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--parts", nargs="+", default=PARTS, choices=PARTS)
    args = ap.parse_args()
    cfg = load_config(args.config)
    record = evaluate(
        args.checkpoint,
        args.split or cfg["eval"]["split"],
        cfg,
        run_id=args.run_id,
        parts=tuple(args.parts),
        progress=True,
    )
    print(json.dumps(headline(record), indent=2))
    print(f"record: {record['run_dir']}/record.json")


if __name__ == "__main__":
    main()
