"""Tests for examples/audit_af1_reproduction.py.

The AF8 governance for AF1-R is *traceability*, not byte-difference:
re-tokenizing a deterministic corpus yields the same SHA by
construction. These tests pin what the audit MUST guarantee: a
non-overwriting write, a complete provenance record with SHA
fingerprints, and a fingerprint identical to hashlib. The hard
governance stop is the refuse-to-overwrite check.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, EXAMPLES / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit_af1_reproduction")


def test_af8_record_has_required_keys() -> None:
    """The provenance dict must carry every field an audit reader needs
    to verify an AF8-clean run without the audit script itself."""
    rec = audit._af8_record(
        out_path=Path("/tmp/x.npy"),
        cache_sha="abc",
        parquet_shas={"/tmp/a.parquet": "pqr"},
        token_count=42,
    )
    for k in (
        "auditor_pid",
        "auditor_utc",
        "token_count",
        "wikitext_cache_sha256",
        "wikitext_cache_path",
        "parquet_shards_sha256",
        "tokenizer_id",
        "eot_policy",
    ):
        assert k in rec, f"missing AF8 field: {k}"


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    payload = b"hello world\n"
    p = tmp_path / "blob.bin"
    p.write_bytes(payload)
    import hashlib
    expected = hashlib.sha256(payload).hexdigest()
    assert audit._sha256_file(p) == expected


def test_main_refuses_to_overwrite_existing(tmp_path: Path) -> None:
    """Refuse-to-overwrite is the hard AF8 governance stop: the auditor
    must NOT silently clobber an artifact another run is depending on."""
    target = tmp_path / "wikitext103_train_ids.npy"
    target.write_bytes(b"already here")
    out = subprocess.run(
        [
            sys.executable, str(EXAMPLES / "audit_af1_reproduction.py"),
            "--af1-cache-sha256", "deadbeef" * 8,
            "--out-path", str(target),
            "--manifest", str(tmp_path / "manifest.json"),
        ],
        capture_output=True, text=True,
    )
    assert out.returncode != 0
    combined = out.stderr + out.stdout
    assert "refusing to overwrite" in combined


def test_main_help_runs() -> None:
    subprocess.run(
        [
            sys.executable, str(EXAMPLES / "audit_af1_reproduction.py"),
            "--help",
        ],
        check=True, capture_output=True, text=True,
    )
