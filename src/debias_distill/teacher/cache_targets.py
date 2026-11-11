"""Run the teacher once and cache its next-token targets.

For every training row and its gender-swapped copy the cache holds five target sets:

  plain_orig    the teacher's distribution on each original row          [N rows]
  plain_cf      the teacher's distribution on each swapped copy          [K rows]
  mean          the position-wise average of a row and its copy          [K rows]
  aligned_orig  average after mapping the copy into the original's frame [K rows]
  aligned_cf    average after mapping the original into the copy's frame [K rows]

Rows 0..K-1 of the last four sets correspond to `cf_tokens[k]`, whose original is
`train_tokens[cf_index[k]]`. Originals without a copy only have `plain_orig`.

A full distribution has about 50k entries per position, so each target is stored as
  <set>.watch.npy  float16 [rows, T, W]  log-prob of every watch-list token
  <set>.ids.npy    uint16  [rows, T, k]  the k most likely tokens not on the watch-list
  <set>.logp.npy   float16 [rows, T, k]  their log-probs, in descending order
  <set>.resid.npy  float16 [rows, T]     log of the mass on all remaining tokens
The watch-list is the tokens of the train occupations: occupations rarely reach the
top k, and they are where the bias is measured. The three parts are disjoint and
their probabilities sum to 1.

The pass runs in chunks and resumes from `progress.json` if interrupted.

Usage: python -m debias_distill.teacher.cache_targets [--config configs/base.yaml]
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from debias_distill.data.swaps import load_swap_pairs
from debias_distill.data.templates import load_names, load_occupations
from debias_distill.teacher.align import align, build_token_map
from debias_distill.utils.config import load_config
from debias_distill.utils.device import get_device
from debias_distill.utils.hub import load_model, load_tokenizer
from debias_distill.utils.manifest import file_sha256, write_manifest

PAIR_SETS = ("plain_cf", "mean", "aligned_orig", "aligned_cf")
SETS = ("plain_orig", *PAIR_SETS)
PARTS = ("watch", "ids", "logp", "resid")
MIN_PROB = 1e-30
MIN_RESIDUAL = 1e-10


def watch_token_ids(tokenizer, occupations: list[str]) -> list[int]:
    """Token ids of the occupation words (article dropped), sorted and unique."""
    ids: set[int] = set()
    for occupation in occupations:
        words = occupation.split(" ", 1)[1]
        ids.update(tokenizer(" " + words, add_special_tokens=False)["input_ids"])
    return sorted(ids)


def compress(p: torch.Tensor, k: int, watch: torch.Tensor) -> dict[str, torch.Tensor]:
    """Reduce probabilities [..., vocab] to watch-list values, top-k of the rest, and the residual."""
    watch_p = p[..., watch]
    rest = p.clone()
    rest[..., watch] = -1.0
    top_p, top_ids = rest.topk(k, dim=-1)
    residual = 1.0 - top_p.sum(-1) - watch_p.sum(-1)
    return {
        "watch": watch_p.clamp_min(MIN_PROB).log(),
        "ids": top_ids,
        "logp": top_p.clamp_min(MIN_PROB).log(),
        "resid": residual.clamp_min(MIN_RESIDUAL).log(),
    }


def pair_targets(p_orig: torch.Tensor, p_cf: torch.Tensor, token_map) -> dict[str, torch.Tensor]:
    """The four pair-level target distributions for a batch of rows and their copies."""
    return {
        "plain_cf": p_cf,
        "mean": 0.5 * (p_orig + p_cf),
        "aligned_orig": 0.5 * (p_orig + align(p_cf, p_orig, token_map)),
        "aligned_cf": 0.5 * (p_cf + align(p_orig, p_cf, token_map)),
    }


class TargetStore:
    """Memory-mapped arrays for the five target sets."""

    def __init__(self, path: Path, rows: dict[str, int], seq_len: int, k: int, n_watch: int):
        self.path = path
        shapes = {
            "watch": (seq_len, n_watch),
            "ids": (seq_len, k),
            "logp": (seq_len, k),
            "resid": (seq_len,),
        }
        self.arrays: dict[tuple[str, str], np.memmap] = {}
        for name in SETS:
            for part in PARTS:
                file = path / f"{name}.{part}.npy"
                dtype = np.uint16 if part == "ids" else np.float16
                shape = (rows[name], *shapes[part])
                if file.exists():
                    array = np.load(file, mmap_mode="r+")
                    if array.shape != shape:
                        raise ValueError(f"{file} has shape {array.shape}, expected {shape}")
                else:
                    array = np.lib.format.open_memmap(file, mode="w+", dtype=dtype, shape=shape)
                self.arrays[name, part] = array

    def write(self, name: str, rows: np.ndarray, parts: dict[str, torch.Tensor]) -> None:
        for part, value in parts.items():
            array = self.arrays[name, part]
            array[rows] = value.cpu().numpy().astype(array.dtype)

    def flush(self) -> None:
        for array in self.arrays.values():
            array.flush()


def load_targets(cache_dir: str | Path, name: str, mode: str = "r") -> dict[str, np.ndarray]:
    """Open one target set of a finished cache as memory-mapped arrays keyed by part."""
    cache_dir = Path(cache_dir)
    return {part: np.load(cache_dir / f"{name}.{part}.npy", mmap_mode=mode) for part in PARTS}


def _cache_key(cfg: dict, inputs: dict[str, Path], watch: list[int]) -> str:
    payload = {
        "teacher": cfg["teacher"],
        "revision": cfg.get("teacher_revision"),
        "topk": cfg["topk"],
        "watch": watch,
        "inputs": {name: file_sha256(path) for name, path in sorted(inputs.items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def cache_teacher_targets(cfg: dict, model=None, max_batches: int | None = None) -> Path:
    """Build (or resume) the target cache for `cfg` and return its directory.

    `model` overrides the teacher (used by tests). `max_batches` stops early, leaving a
    resumable cache.
    """
    corpus = Path(cfg["data"]["corpus_dir"])
    swaps_dir, templates_dir = Path(cfg["data"]["swaps_dir"]), Path(cfg["data"]["templates_dir"])
    tokenizer = load_tokenizer(cfg, "teacher")
    table = load_swap_pairs(swaps_dir, name_pairs=load_names("train", templates_dir))
    token_map, unmapped = build_token_map(tokenizer, table)
    watch = watch_token_ids(tokenizer, load_occupations("train", templates_dir))

    inputs = {name: corpus / name for name in ("train_tokens.npy", "cf_tokens.npy", "cf_index.npy")}
    inputs.update({f"swaps/{p.name}": p for p in sorted(swaps_dir.iterdir()) if p.is_file()})
    inputs["names/train"] = templates_dir / "train" / "names.tsv"
    inputs["occupations/train"] = templates_dir / "train" / "occupations.txt"
    out = Path(cfg["data"]["cache_dir"]) / _cache_key(cfg, inputs, watch)
    meta_path, progress_path = out / "meta.json", out / "progress.json"
    if meta_path.exists():
        return out  # finished earlier with identical inputs
    out.mkdir(parents=True, exist_ok=True)

    train = np.load(inputs["train_tokens.npy"], mmap_mode="r")
    copies = np.load(inputs["cf_tokens.npy"], mmap_mode="r")
    cf_index = np.load(inputs["cf_index.npy"])
    n_rows, seq_len = train.shape
    n_pairs = len(copies)
    singles = np.setdiff1d(np.arange(n_rows), cf_index)
    k = cfg["topk"]

    rows = {"plain_orig": n_rows, **dict.fromkeys(PAIR_SETS, n_pairs)}
    store = TargetStore(out, rows, seq_len, k, len(watch))
    progress = {"pairs": 0, "singles": 0, "positions": 0, "mapped_mass": 0.0, "stored_mass": 0.0}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())

    device = get_device(cfg["device"])
    settings = cfg.get("teacher_cache", {})
    if model is None:
        dtype = torch.bfloat16 if device.type != "cpu" else torch.float32
        model = load_model(cfg, "teacher", dtype=dtype)
    model = model.to(device).eval()
    token_map = token_map.to(device)
    watch_t = torch.tensor(watch, dtype=torch.long, device=device)
    batch_pairs = settings.get("batch_pairs", 8)
    flush_every = settings.get("flush_every", 20)

    def probabilities(token_rows: np.ndarray) -> torch.Tensor:
        ids = torch.from_numpy(np.ascontiguousarray(token_rows).astype(np.int64)).to(device)
        return model(ids).logits.float().softmax(-1)

    def checkpoint() -> None:
        store.flush()
        progress_path.write_text(json.dumps(progress))

    batches = 0
    total = n_pairs + len(singles)
    with (
        torch.no_grad(),
        tqdm(total=total, initial=progress["pairs"] + progress["singles"], unit="row") as bar,
    ):
        while progress["pairs"] < n_pairs and (max_batches is None or batches < max_batches):
            pair_rows = np.arange(progress["pairs"], min(progress["pairs"] + batch_pairs, n_pairs))
            orig_rows = cf_index[pair_rows]
            p = probabilities(np.concatenate([train[orig_rows], copies[pair_rows]]))
            p_orig, p_cf = p[: len(pair_rows)], p[len(pair_rows) :]
            stored = compress(p_orig, k, watch_t)
            store.write("plain_orig", orig_rows, stored)
            for name, target in pair_targets(p_orig, p_cf, token_map).items():
                store.write(name, pair_rows, compress(target, k, watch_t))
            positions = p_orig.shape[0] * p_orig.shape[1]
            progress["positions"] += positions
            progress["mapped_mass"] += p_orig[..., token_map.sources].sum().item()
            progress["stored_mass"] += positions - stored["resid"].exp().sum().item()
            progress["pairs"] += len(pair_rows)
            batches += 1
            bar.update(len(pair_rows))
            if batches % flush_every == 0:
                checkpoint()
        while (
            progress["pairs"] == n_pairs
            and progress["singles"] < len(singles)
            and (max_batches is None or batches < max_batches)
        ):
            rows_ = singles[progress["singles"] : progress["singles"] + 2 * batch_pairs]
            store.write("plain_orig", rows_, compress(probabilities(train[rows_]), k, watch_t))
            progress["singles"] += len(rows_)
            batches += 1
            bar.update(len(rows_))
            if batches % flush_every == 0:
                checkpoint()
    checkpoint()
    if progress["pairs"] < n_pairs or progress["singles"] < len(singles):
        return out  # stopped early; resumable

    positions = max(progress["positions"], 1)
    meta = {
        "teacher": cfg["teacher"],
        "teacher_revision": cfg.get("teacher_revision"),
        "topk": k,
        "seq_len": seq_len,
        "rows": rows,
        "watch_token_ids": watch,
        "token_edges": len(token_map.src),
        "unmapped_forms": unmapped,
        # Measured on the originals that have a copy:
        "mean_mass_on_mapped_tokens": progress["mapped_mass"] / positions,
        "mean_mass_stored": progress["stored_mass"] / positions,
    }
    write_manifest(out, inputs, cfg)
    meta_path.write_text(json.dumps(meta, indent=2))
    progress_path.unlink()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the teacher once and cache its targets.")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--max-batches", type=int, default=None, help="stop early (resumable)")
    args = ap.parse_args()
    out = cache_teacher_targets(load_config(args.config), max_batches=args.max_batches)
    meta = out / "meta.json"
    if meta.exists():
        summary = json.loads(meta.read_text())
        keys = ("rows", "token_edges", "mean_mass_on_mapped_tokens", "mean_mass_stored")
        print(json.dumps({key: summary[key] for key in keys}, indent=2))
        print(f"cache complete: {out}")
    else:
        print(f"stopped early; run again to resume: {out}")


if __name__ == "__main__":
    main()
