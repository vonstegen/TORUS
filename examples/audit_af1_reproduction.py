"""AF1-R token-cache auditor - AF8 governance helper.

EXP-AF-001-R (clean reproduction of EXP-AF-001) requires an
**independently generated** wikitext-103 token cache under AF8
governance. The cache is a deterministic function of the wikitext-103
parquet shards + the OLMo tokenizer + the eot-append policy, so
re-tokenizing produces the same SHA256 by construction. AF8 governance
is therefore *traceability* (fresh process invocation, fingerprint of
every input, no silent reuse of an existing artifact path) - NOT a
byte-difference check.

The audit:
  - Re-tokenizes wikitext-103 train into a path that MUST NOT preexist.
  - Records in the manifest: cache SHA, parquet shard SHAs, tokenizer
    identifier, AF8 verifier PID, UTC start timestamp, the AF1
    reference SHA (for cross-reference).
  - Refuses to overwrite an existing target.
  - Warns (does NOT fail) if the regenerated cache SHA differs from
    the AF1 reference - that signals the inputs drifted.

Usage on legion:

    python examples/audit_af1_reproduction.py \\
        --af1-cache-sha256 <sha256 computed from AF1 driver> \\
        --out-path  /home/andrew-jochl/TORUS/runs/a/EXP-AF-001-R/<ts>/wikitext103_train_ids.npy \\
        --manifest  /home/andrew-jochl/TORUS/runs/a/EXP-AF-001-R/<ts>/cache_provenance.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
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


def _af8_record(out_path: Path, cache_sha: str, parquet_shas: dict,
                token_count: int) -> dict:
    """AF8 governance record: structural provenance for the audit."""
    return {
        "auditor_pid": os.getpid(),
        "auditor_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "token_count": token_count,
        "wikitext_cache_sha256": cache_sha,
        "wikitext_cache_path": str(out_path),
        "parquet_shards_sha256": parquet_shas,
        "tokenizer_id": "allenai/OLMo-1B-0724-hf",
        "eot_policy": "appended per non-empty text (mirrors distill_run)",
    }


def _build_wikitext_cache(out_path: Path) -> dict:
    """Re-tokenize wikitext-103 train split, mirroring distill_run.

    Returns an AF8 governance record. Side effect: writes `out_path`
    (must not preexist)."""
    import os as _os
    _os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    shard_paths = [
        hf_hub_download(
            repo_id="wikitext",
            filename=f"wikitext-103-raw-v1/train-{i:05d}-of-00002.parquet",
            repo_type="dataset",
        )
        for i in range(2)
    ]
    shard_shas = {p: _sha256_file(Path(p)) for p in shard_paths}
    tables = [pq.read_table(p, columns=["text"]) for p in shard_paths]
    texts = sum((t.column("text").to_pylist() for t in tables), [])

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
    return _af8_record(
        out_path=out_path,
        cache_sha=cache_sha,
        parquet_shas=shard_shas,
        token_count=int(arr.size),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--af1-cache-sha256", required=True,
        help="Reference SHA256 of the AF1 cache (precomputed once by "
             "a separate process). Identity is the expected outcome of "
             "a clean reproduction; only diverging SHAs warrant a warning.",
    )
    p.add_argument(
        "--out-path", type=Path, required=True,
        help="Where to write the regenerated .npy cache.",
    )
    p.add_argument(
        "--manifest", type=Path, required=True,
        help="JSON file to write the AF8 provenance record to.",
    )
    args = p.parse_args()

    if args.out_path.exists():
        sys.exit(f"refusing to overwrite existing file: {args.out_path}")

    print(
        f"[af1-r-audit] regenerating wikitext-103 train cache at "
        f"{args.out_path}", flush=True,
    )
    record = _build_wikitext_cache(args.out_path)
    new_sha = record["wikitext_cache_sha256"]
    print(f"[af1-r-audit] new cache sha256: {new_sha}", flush=True)
    print(f"[af1-r-audit] af1  cache sha256: {args.af1_cache_sha256}", flush=True)

    same = new_sha == args.af1_cache_sha256
    if same:
        print(
            "[af1-r-audit] identity: expected outcome of a clean "
            "reproduction (cache is a deterministic function of the inputs).",
            flush=True,
        )
    else:
        print(
            "[af1-r-audit] WARNING: regenerated cache differs from the "
            "AF1 reference. This means the parquet shards OR the tokenizer "
            "moved between AF1 and AF1-R. Recording both SHAs and continuing.",
            flush=True,
        )
    record["af1_reference_sha256"] = args.af1_cache_sha256
    record["af1_reference_identity"] = same

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(record, indent=2))
    print(f"[af1-r-audit] provenance recorded to {args.manifest}", flush=True)
    print("[af1-r-audit] OK.", flush=True)


if __name__ == "__main__":
    main()
