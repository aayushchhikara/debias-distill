"""Train the student on cached teacher targets.

An arm is a choice of data and targets (`distill.data`, `distill.target`):
  data    originals       training rows only
          both            training rows and their gender-swapped copies
  target  plain           the teacher's own distribution for each row
          mean            the average of the distributions for a row and its copy
          aligned         the same average after aligning gendered tokens
          hard            next-token labels, no teacher
Rows without a copy always use the plain target. Every arm takes the same number
of optimizer steps. A run is checkpointed and resumes when started again with the
same run id.

Usage:
  python -m debias_distill.distill.train --seed 1
  python -m debias_distill.distill.train --config configs/arms/e.yaml --seed 1
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from debias_distill.distill.losses import next_token_loss, truncated_kl
from debias_distill.teacher.cache_targets import SETS, finished_cache, load_targets
from debias_distill.utils.config import load_config
from debias_distill.utils.device import get_device
from debias_distill.utils.hub import load_model
from debias_distill.utils.runlog import RunLogger
from debias_distill.utils.seed import set_seed

TARGET_SETS = {  # target -> (set for originals with a copy, set for copies)
    "plain": ("plain_orig", "plain_cf"),
    "mean": ("mean", "mean"),
    "aligned": ("aligned_orig", "aligned_cf"),
}
HARD = -1


def build_examples(data: str, target: str, n_rows: int, cf_index: np.ndarray) -> np.ndarray:
    """One line per training example: (is_copy, token row, target set, target row).

    The target set indexes `SETS`, or is -1 for hard labels.
    """
    if data not in ("originals", "both"):
        raise ValueError(f"unknown distill.data {data!r}")
    if target != "hard" and target not in TARGET_SETS:
        raise ValueError(f"unknown distill.target {target!r}")
    rows = np.arange(n_rows)
    pairs = np.arange(len(cf_index))
    plain = SETS.index("plain_orig")
    originals = np.stack([np.zeros_like(rows), rows, np.full(n_rows, plain), rows], axis=1)
    copies = np.stack([np.ones_like(pairs), pairs, np.zeros_like(pairs), pairs], axis=1)
    if target == "hard":
        originals[:, 2] = copies[:, 2] = HARD
    else:
        for_originals, for_copies = (SETS.index(name) for name in TARGET_SETS[target])
        if for_originals != plain:  # the pair-level sets are indexed by pair, not by row
            originals[cf_index, 2] = for_originals
            originals[cf_index, 3] = pairs
        copies[:, 2] = for_copies
    return originals if data == "originals" else np.concatenate([originals, copies])


def batch_order(n_examples: int, batch_size: int, index: int, seed: int) -> np.ndarray:
    """Example indices of micro-batch `index`: a fresh seeded shuffle for each pass over the data."""
    per_epoch = n_examples // batch_size
    epoch, position = divmod(index, per_epoch)
    order = np.random.default_rng([seed, epoch]).permutation(n_examples)
    return order[position * batch_size : (position + 1) * batch_size]


class Batches:
    """Reads token rows and their cached targets."""

    def __init__(self, cfg: dict, cache_dir: Path, device):
        corpus = Path(cfg["data"]["corpus_dir"])
        self.tokens = (
            np.load(corpus / "train_tokens.npy", mmap_mode="r"),
            np.load(corpus / "cf_tokens.npy", mmap_mode="r"),
        )
        self.cf_index = np.load(corpus / "cf_index.npy")
        self.targets = {name: load_targets(cache_dir, name) for name in SETS}
        meta = json.loads((cache_dir / "meta.json").read_text())
        self.watch_ids = torch.tensor(meta["watch_token_ids"], dtype=torch.long, device=device)
        self.device = device

    def load(self, examples: np.ndarray) -> dict[str, torch.Tensor]:
        tokens = np.stack([self.tokens[is_copy][row] for is_copy, row, _, _ in examples])
        batch = {"tokens": torch.from_numpy(tokens.astype(np.int64)).to(self.device)}
        if examples[0, 2] != HARD:
            for part in ("ids", "logp", "watch"):
                values = np.stack(
                    [self.targets[SETS[s]][part][row] for _, _, s, row in examples]
                ).astype(np.int64 if part == "ids" else np.float32)
                batch[part] = torch.from_numpy(values).to(self.device)
        return batch


def loss_for(model, batch: dict[str, torch.Tensor], watch_ids: torch.Tensor, settings: dict):
    logits = model(batch["tokens"]).logits
    if "ids" not in batch:
        return next_token_loss(logits, batch["tokens"])
    return truncated_kl(
        logits,
        batch["ids"],
        batch["logp"],
        watch_ids,
        batch["watch"],
        k=settings.get("k"),
        use_watch=settings.get("use_watch", True),
    )


def learning_rate_factor(step: int, warmup: int, total: int) -> float:
    """Linear warm-up, then cosine decay to zero."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def train(cfg: dict, run_id: str | None = None, model=None, progress: bool = False) -> str:
    """Train the student for `cfg` and return the path of the saved model.

    `model` overrides the pretrained student (used by tests).
    """
    settings = cfg["distill"]
    seed = cfg["seed"]
    run_id = run_id or f"arm{settings['arm']}_seed{seed}"
    run_dir = Path(cfg["data"].get("experiments_dir", "experiments")) / run_id
    final, state_path = run_dir / "checkpoint", run_dir / "state.pt"
    if final.exists():
        return str(final)  # finished earlier

    set_seed(seed)
    device = get_device(cfg["device"])
    cache_dir = finished_cache(cfg)
    batches = Batches(cfg, cache_dir, device)
    examples = build_examples(
        settings["data"], settings["target"], len(batches.tokens[0]), batches.cf_index
    )
    if model is None:
        model = load_model(cfg, "student", dtype=torch.float32)
    model = model.to(device).train()

    steps, accum, batch_size = settings["steps"], settings["grad_accum"], settings["batch_size"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: learning_rate_factor(s, settings["warmup_steps"], steps)
    )
    start = 0
    if state_path.exists():
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start = state["step"]

    def save_state(step: int) -> None:
        state = {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(state, state_path.with_suffix(".tmp"))
        state_path.with_suffix(".tmp").replace(state_path)

    root = cfg["data"].get("experiments_dir", "experiments")
    with RunLogger(run_id, cfg, root=root, resume=run_dir.exists()) as run:
        began = time.monotonic()
        last_loss = float("nan")
        for step in tqdm(range(start, steps), initial=start, total=steps, disable=not progress):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for part in range(accum):
                index = batch_order(len(examples), batch_size, step * accum + part, seed)
                batch = batches.load(examples[np.sort(index)])
                loss = loss_for(model, batch, batches.watch_ids, settings) / accum
                loss.backward()
                total += loss.item()
            if not math.isfinite(total):
                raise FloatingPointError(f"loss is {total} at step {step}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            last_loss = total
            if (step + 1) % settings["log_every"] == 0 or step + 1 == steps:
                run.log(step + 1, loss=total, lr=scheduler.get_last_lr()[0])
            if (step + 1) % settings["checkpoint_every"] == 0 and step + 1 < steps:
                save_state(step + 1)

        model.eval()
        model.save_pretrained(final)
        summary = {
            "run_id": run_id,
            "arm": settings["arm"],
            "seed": seed,
            "steps": steps,
            "rows_per_step": accum * batch_size,
            "examples": len(examples),
            "final_loss": last_loss,
            "cache": str(cache_dir),
            "seconds_this_session": round(time.monotonic() - began, 1),
        }
        (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
        state_path.unlink(missing_ok=True)
    return str(final)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    overrides: dict = {"distill": {}}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.lr is not None:
        overrides["distill"]["lr"] = args.lr
    if args.steps is not None:
        overrides["distill"]["steps"] = args.steps
    cfg = load_config(args.config, overrides)
    print(f"saved model: {train(cfg, run_id=args.run_id, progress=True)}")


if __name__ == "__main__":
    main()
