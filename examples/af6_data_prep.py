"""EXP-AF-006 data prep: build openwebtext + wikitext-test token caches.

Builds three caches (numpy int64 token streams, same layout as the
wikitext-103 train cache used by every experiment since EXP-AF-001):

  /tmp/openwebtext_train_ids.npy   train portion of 1 OWT parquet shard
  /tmp/openwebtext_test_ids.npy    held-out 1% tail of that shard
  /tmp/wikitext103_test_ids.npy    wikitext-103 TEST split (for the
                                   corpus-ppl sanity cross-check
                                   against lm-eval's wikitext ppl)

Downloads parquet directly via huggingface_hub (the datasets library
is unreliable in this Python 3.14 venv — see distill_run.py). Records
shard filenames + sha256 + token counts in the output manifest.

Usage:

    python examples/af6_data_prep.py --out-manifest /tmp/af6_data_prep.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

MODEL = "allenai/OLMo-1B-0724-hf"
OWT_SHARD_INDEX = 0          # first shard is plenty (~50M tokens)
OWT_TEST_FRACTION = 0.01     # held-out tail


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _tokenize_texts(texts, tokenizer) -> np.ndarray:
    eot = tokenizer.eos_token_id or 0
    all_ids: list[int] = []
    for text in texts:
        if not text or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.append(eot)
        all_ids.extend(ids)
    return np.asarray(all_ids, dtype=np.int64)


def _write_cache(arr: np.ndarray, path: Path) -> dict:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing cache: {path}")
    np.save(path, arr, allow_pickle=False)
    return {
        "path": str(path),
        "tokens": int(arr.size),
        "sha256": _sha256_file(path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-manifest", type=Path, required=True)
    p.add_argument("--owt-train", type=Path,
                   default=Path("/tmp/openwebtext_train_ids.npy"))
    p.add_argument("--owt-test", type=Path,
                   default=Path("/tmp/openwebtext_test_ids.npy"))
    p.add_argument("--wt-test", type=Path,
                   default=Path("/tmp/wikitext103_test_ids.npy"))
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    record = {
        "prep_pid": os.getpid(),
        "prep_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer_id": MODEL,
        "eot_policy": "appended per non-empty text (mirrors distill_run)",
        "caches": {},
        "sources": {},
    }

    # ---- openwebtext (Skylion007 mirror, plaintext parquet shards) ----
    owt_files = sorted(
        f for f in list_repo_files("Skylion007/openwebtext",
                                   repo_type="dataset")
        if f.startswith("plaintext/") and f.endswith(".parquet")
        and "train" in f
    )
    if not owt_files:
        raise SystemExit("no openwebtext plaintext train parquets found")
    shard = owt_files[OWT_SHARD_INDEX]
    print(f"[af6-prep] downloading {shard} ...", flush=True)
    shard_path = hf_hub_download(
        repo_id="Skylion007/openwebtext", filename=shard,
        repo_type="dataset",
    )
    record["sources"]["openwebtext_shard"] = {
        "file": shard, "sha256": _sha256_file(Path(shard_path)),
    }
    texts = pq.read_table(shard_path, columns=["text"]).column(
        "text").to_pylist()
    print(f"[af6-prep] tokenizing {len(texts):,} OWT rows ...", flush=True)
    ids = _tokenize_texts(texts, tokenizer)
    n_test = max(1, int(ids.size * OWT_TEST_FRACTION))
    train_ids, test_ids = ids[:-n_test], ids[-n_test:]
    record["caches"]["openwebtext_train"] = _write_cache(
        train_ids, args.owt_train)
    record["caches"]["openwebtext_test"] = _write_cache(
        test_ids, args.owt_test)

    # ---- wikitext-103 test split ----
    wt_test_path = hf_hub_download(
        repo_id="wikitext",
        filename="wikitext-103-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset",
    )
    record["sources"]["wikitext_test_shard"] = {
        "file": "wikitext-103-raw-v1/test-00000-of-00001.parquet",
        "sha256": _sha256_file(Path(wt_test_path)),
    }
    texts = pq.read_table(wt_test_path, columns=["text"]).column(
        "text").to_pylist()
    print(f"[af6-prep] tokenizing {len(texts):,} wikitext test rows ...",
          flush=True)
    wt_ids = _tokenize_texts(texts, tokenizer)
    record["caches"]["wikitext103_test"] = _write_cache(wt_ids,
                                                        args.wt_test)

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(record, indent=2))
    print(f"[af6-prep] manifest written to {args.out_manifest}",
          flush=True)
    print("[af6-prep] OK.", flush=True)


if __name__ == "__main__":
    main()
