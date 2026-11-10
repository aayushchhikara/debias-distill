"""Text corpus sampling and preprocessing.

Outputs (in `data.corpus_dir`, gitignored):
  heldout_tokens.npy   uint16 [M, seq_len]  documents disjoint from the training slice
  sampled_tokens.npy   uint16 [S, seq_len]  raw training slice as sampled
  train_tokens.npy     uint16 [N, seq_len]  training rows: sampled rows minus rejected ones
  cf_tokens.npy        uint16 [K, seq_len]  substituted variants (same length as originals)
  cf_index.npy         int64  [K]           row in train_tokens each variant came from
  rejected_tokens.npy  uint16 [R, seq_len]  rows with listed words that could not be fully
                                            substituted; excluded from training, kept for analysis
  sample_meta.json     source, sizes, document counts
  cf_stats.json        substitution counts and rejection reasons

Usage:
  python -m debias_distill.data.corpus                    # sample + variants
  python -m debias_distill.data.corpus --token-budget 200000 --out /tmp/x   # quick trial
  python -m debias_distill.data.corpus --skip-sample      # rebuild variants only

Variants use the train name pairs, which are derived from the sampled slice. After a
fresh sample, run `python -m debias_distill.data.names`, then rebuild with --skip-sample.
"""

import argparse
import itertools
import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from debias_distill.data.swaps import TokenSwapper, load_swap_pairs
from debias_distill.data.templates import load_names
from debias_distill.utils.config import load_config


def iter_documents(cfg: dict) -> Iterator[str]:
    """Yield document texts from the configured dataset's parquet files, in file and row-group order.

    Reads row groups over HTTP range requests, so only the needed part of each file is fetched.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    c = cfg["corpus"]
    fs = HfFileSystem()
    files = sorted(fs.glob(f"datasets/{c['name']}@{c['revision']}/{c['data_dir']}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files for {c['name']}@{c['revision']}/{c['data_dir']}")
    for path in files:
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                yield from pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()


def tokenize_documents(
    docs: Iterable[str], tokenizer, batch_docs: int = 256
) -> Iterator[list[int]]:
    """Lazily tokenize documents in batches, yielding one EOS-terminated id list per document."""
    it = iter(docs)
    eos = tokenizer.eos_token_id
    try:
        while batch := list(itertools.islice(it, batch_docs)):
            for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                yield [*ids, eos]
    finally:
        if hasattr(it, "close"):
            it.close()  # close the document source (open file handles)


def pack(token_docs: Iterator[list[int]], seq_len: int, n_tokens: int) -> tuple[np.ndarray, int]:
    """Concatenate tokenized documents into full rows of `seq_len` until `n_tokens` are collected.

    Pulls documents from the shared iterator only as needed, so a later call continues
    with unseen documents. Returns (rows, documents consumed); the tail of the last
    consumed document is discarded.
    """
    n_rows = max(1, n_tokens // seq_len)
    rows = np.empty((n_rows, seq_len), dtype=np.uint16)
    buf: list[int] = []
    filled, consumed = 0, 0
    with tqdm(total=n_rows, unit="row", leave=False) as bar:
        while filled < n_rows:
            doc = next(token_docs, None)
            if doc is None:
                raise RuntimeError(f"ran out of documents after {filled}/{n_rows} rows")
            consumed += 1
            buf.extend(doc)
            while len(buf) >= seq_len and filled < n_rows:
                rows[filled] = buf[:seq_len]
                del buf[:seq_len]
                filled += 1
                bar.update(1)
    return rows, consumed


def sample_corpus(cfg: dict, out_dir: str | Path, docs: Iterable[str] | None = None) -> dict:
    """Collect the held-out slice first, then the training slice, from one document stream."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg["student"])
    assert len(tok) < np.iinfo(np.uint16).max, "vocabulary does not fit in uint16"

    seq_len = cfg["corpus"]["seq_len"]
    token_docs = tokenize_documents(docs if docs is not None else iter_documents(cfg), tok)
    heldout, n_heldout_docs = pack(token_docs, seq_len, cfg["eval"]["perplexity_tokens"])
    train, n_train_docs = pack(token_docs, seq_len, cfg["corpus"]["token_budget"])
    token_docs.close()

    np.save(out / "heldout_tokens.npy", heldout)
    np.save(out / "sampled_tokens.npy", train)
    meta = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": cfg["corpus"]["name"],
        "revision": cfg["corpus"]["revision"],
        "data_dir": cfg["corpus"]["data_dir"],
        "tokenizer": cfg["student"],
        "seq_len": seq_len,
        "heldout": {"rows": len(heldout), "tokens": heldout.size, "documents": n_heldout_docs},
        "train": {
            "file": "sampled_tokens.npy",
            "rows": len(train),
            "tokens": train.size,
            "documents": n_train_docs,
        },
    }
    (out / "sample_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def build_counterfactuals(cfg: dict, out_dir: str | Path) -> dict:
    """Produce length-preserving substituted variants of training rows that contain listed words."""
    out = Path(out_dir)
    tok = AutoTokenizer.from_pretrained(cfg["student"])
    table = load_swap_pairs(
        cfg["data"]["swaps_dir"], name_pairs=load_names("train", cfg["data"]["templates_dir"])
    )
    swapper = TokenSwapper(tok, table)

    sampled = np.load(out / "sampled_tokens.npy")
    seq_len = sampled.shape[1]
    keep, variants, index, rejected = [], [], [], []
    for row in tqdm(sampled, unit="row", leave=False):
        cf = swapper.swap(row)
        if swapper.last_status == "rejected":
            rejected.append(row)
            continue
        if cf is not None:
            variants.append(cf)
            index.append(len(keep))
        keep.append(row)

    def as_rows(rows: list[np.ndarray]) -> np.ndarray:
        return np.array(rows, dtype=np.uint16).reshape(-1, seq_len)

    np.save(out / "train_tokens.npy", as_rows(keep))
    np.save(out / "cf_tokens.npy", as_rows(variants))
    np.save(out / "cf_index.npy", np.array(index, dtype=np.int64))
    np.save(out / "rejected_tokens.npy", as_rows(rejected))
    stats = swapper.stats.to_dict()
    stats["train_rows"] = len(keep)
    stats["train_tokens"] = len(keep) * seq_len
    stats["rejected_rows"] = len(rejected)
    (out / "cf_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample the corpus and build substituted variants.")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default=None, help="output dir (default: data.corpus_dir)")
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--skip-sample", action="store_true", help="reuse existing sampled_tokens.npy")
    args = ap.parse_args()

    overrides = {"corpus": {"token_budget": args.token_budget}} if args.token_budget else None
    cfg = load_config(args.config, overrides)
    out = args.out or cfg["data"]["corpus_dir"]

    if not args.skip_sample:
        meta = sample_corpus(cfg, out)
        print(json.dumps({k: meta[k] for k in ("heldout", "train")}, indent=2))
    stats = build_counterfactuals(cfg, out)
    keys = (
        "sequences",
        "with_listed_words",
        "produced",
        "rejected_rows",
        "train_rows",
        "train_tokens",
    )
    summary = {k: stats[k] for k in keys}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
