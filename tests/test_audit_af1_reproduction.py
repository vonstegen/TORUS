"""Tests for examples/audit_af1_reproduction.py.

These pin the AF8 governance notary: the regenerated cache file MUST
NOT match AF1's recorded cache SHA256. The check is a pure function
(SystemExit on collision), so the test does not need to contact HF.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_provenance_check_distinct_shas_passes() -> None:
    """Two different SHAs -> notary returns normally (no exception)."""
    audit._provenance_check("aaa", "bbb")


def test_provenance_check_identical_shas_exits() -> None:
    """Matching SHAs trigger SystemExit. This is the AF8 governance
    hard-stop: any future change that disables the check would let a
    cross-experiment cache reuse through, which the framework forbids.
    """
    with pytest.raises(SystemExit) as excinfo:
        audit._provenance_check("same", "same")
    msg = str(excinfo.value)
    assert "PROVENANCE VIOLATION" in msg
    assert "AF8 governance" in msg


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    """The cache-file hashing helper must compute exactly what hashlib
    would on the same content, otherwise the comparison with the
    ARTIFACTS.json sha256 is unsound."""
    payload = b"hello world\n"
    p = tmp_path / "blob.bin"
    p.write_bytes(payload)
    import hashlib
    expected = hashlib.sha256(payload).hexdigest()
    assert audit._sha256_file(p) == expected


def test_main_help_runs(tmp_path: Path) -> None:
    """Just covers --help: real runs need HF + parquet and are not
    unit-tested here."""
    subprocess.run(
        [sys.executable, str(EXAMPLES / "audit_af1_reproduction.py"),
         "--help"],
        check=True, capture_output=True, text=True,
    )
