"""AF1-R token-cache auditor - AF8 governance helper.

EXP-AF-001-R (clean reproduction of EXP-AF-001) requires an
**independently generated** wikitext-103 token cache: the new cache
file MUST NOT match AF1's recorded cache SHA256 (provenance notary).

Usage on legion:

    python examples/audit_af1_reproduction.py \\
        --af1-cache-sha256 <sha256 from AF1 ARTIFACTS.json> \\
        --out-path   /home/andrew-jochl/TORUS/runs/a/EXP-AF-001-R/<ts>/wikitext103_train_ids.npy \\
        --manifest   runs/a/EXP-AF-001-R/<ts>/cache_provenance.json

Exits non-zero if:
  - the wikitext-103 parquet shards cannot be located or hashed;
  - the resulting cache file's sha256 equals the AF1 cache sha256
    (provenance mismatch - auditor fails loud rather than allow
    cross-experiment cache reuse);
  - the output directory is missing or unwritable;
  - the target cache file already exists (refuses to overwrite).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _provenance_check(new_sha: str, af1_sha: str) -> None:
    """AF8 governance notary: the new cache MUST NOT match AF1's.

    Raises SystemExit on provenance mismatch. Pure function: takes two
    SHA256 hex strings and exits non-zero if they collide.
    """
    if new_sha == af1_sha:
        sys.exit(
            "PROVENANCE VIOLATION: the regenerated cache matches AF1's "
            "cache byte-for-byte. The AF8 governance requires an "
            "independently generated cache. Aborting."
        )


def _build_wikitext_cache(out_path: Path) -> dict:
    """Re-tokenize the wikitext-103 train split, mirroring distill_run.

    Returns a provenance dict with: token count, cache file SHA256,
    cache path, and SHA256 of every input parquet shard. Side effect:
    writes `out_path` (must not preexist).
    """
    import os
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    repo_id = "wikitext"
    shard_paths = [
        hf_hub_download(
            repo_id=repo_id,
            filename=f"wikitext-103-raw-v1/train-{i:05d}-of-00002.parquet",
            repo_type="dataset",
        )
        for i in range(2)
    ]
    shard_shas = {p: _sha256_file(Path(p)) for p in shard_paths}
    tables = [pq.read_table(p, columns=["text"]) for p in shard_paths]
    texts = sum((t.column("text").to_pylist() for t in tables), [])

    # Mirror distill_run's eot-appended tokenization.
    tok = AutoTokenizer.from_pretrained("allenai/OLMo-1B-0724-hf")
    eot = tok.eos_token_id or 0
    all_ids: list[int] = []
    for text in texts:
        if not text.strip():
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ids.append(eot)
        all_ids.extend(ids)

    arr = np.asarray(all_ids, dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr, allow_pickle=False)

    cache_sha = _sha256_file(out_path)
    return {
        "token_count": int(arr.size),
        "wikitext_cache_sha256": cache_sha,
        "wikitext_cache_path": str(out_path),
        "parquet_shards_sha256": shard_shas,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--af1-cache-sha256", required=True,
        help="SHA256 of the AF1 token cache file. Must NOT match the "
             "new cache's SHA256 (AF8 governance provenance check).",
    )
    p.add_argument(
        "--out-path", type=Path, required=True,
        help="Where to write the regenerated .npy cache.",
    )
    p.add_argument(
        "--manifest", type=Path, required=True,
        help="JSON file to write the provenance record to.",
    )
    args = p.parse_args()

    if args.out_path.exists():
        sys.exit(f"refusing to overwrite existing file: {args.out_path}")

    print(
        f"[af1-r-audit] regenerating wikitext-103 train cache at "
        f"{args.out_path}",
        flush=True,
    )
    record = _build_wikitext_cache(args.out_path)
    print(
        f"[af1-r-audit] new cache sha256: "
        f"{record['wikitext_cache_sha256']}",
        flush=True,
    )
    print(
        f"[af1-r-audit] af1  cache sha256: {args.af1_cache_sha256}",
        flush=True,
    )
    # AF8 governance notary: refuse to proceed if the SHA collides.
    _provenance_check(record["wikitext_cache_sha256"], args.af1_cache_sha256)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(record, indent=2))
    print(
        f"[af1-r-audit] provenance recorded to {args.manifest}",
        flush=True,
    )
    print(
        "[af1-r-audit] OK: cache sha256 != af1 cache sha256",
        flush=True,
    )


if __name__ == "__main__":
    main()
