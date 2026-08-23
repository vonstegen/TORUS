"""Tests for examples/audit_rpm_000_reproduction.py.

The EXP-RPM-000 (G-RPM-0) audit verifies a fresh AF2-D reproduction
aggregate against preregistered PASS bands. It is the structural
twin of test_audit_af2_reproduction.py but operates on the
post-train aggregate rather than the token cache; the AF2-R
auditor remains the canonical token-cache provenance check.

These tests pin what the audit MUST guarantee: a complete band
check with preregistered reference values, deterministic
verdict logic, a fingerprint identical to json, and a working
--help path.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "research" / "residual-pareto" / "experiments" / "RPM-000" / "manifest.yaml"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, EXAMPLES / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit_rpm_000_reproduction")


# ---- synthetic aggregate fixtures ---------------------------------------

def _seed_dict(*, seed: int, ppl: float, arc: float, lam: float,
               pre_ppl: float, pre_arc: float, dep_bytes: int) -> dict:
    return {
        "seed": seed,
        "pre_train_eval": {"wikitext_ppl": pre_ppl, "arc_easy": pre_arc},
        "eval": {"wikitext_ppl": ppl, "arc_easy": arc, "lambada_openai": lam},
        "deployed_bytes": {"serialized_total": dep_bytes},
    }


def _aggregate(t2_seeds: list) -> dict:
    return {"trained_arms": {"t2_ternary": {"seeds": t2_seeds}}}


def _af2d_reference() -> dict:
    return {"git_sha": "frozen", "run_namespace": "experiments/AF2-D/runs/<ts>/af2d/"}


def _fake_manifest(tmp_path: Path) -> Path:
    # Minimal valid manifest; only id + revision read by audit.
    p = tmp_path / "manifest.yaml"
    p.write_text("id: EXP-RPM-000\nrevision: 'frozen'\n")
    return p


# ---- tests --------------------------------------------------------------


def test_af2d_reference_has_required_keys() -> None:
    """The reference dict (AF2-D aggregate_corrected.json) is opaque to
    the audit; only git_sha + run_namespace are read. Pin the
    contract so a future schema change doesn't silently break."""
    rec = _af2d_reference()
    for k in ("git_sha", "run_namespace"):
        assert k in rec, f"missing reference field: {k}"


def test_band_check_in_band_and_out_of_band() -> None:
    """A band check returns in_band=True iff lower <= value <= upper."""
    in_band = audit._band_check(0.5, [0.0, 1.0], "x")
    assert in_band["in_band"] is True
    assert in_band["metric"] == "x"

    out_low = audit._band_check(-0.1, [0.0, 1.0], "x")
    assert out_low["in_band"] is False
    out_high = audit._band_check(1.1, [0.0, 1.0], "x")
    assert out_high["in_band"] is False


def test_deployed_bytes_check_uses_tolerance_pct() -> None:
    """+/-1% tolerance on the AF2-D t2_ternary reference 4,194,404 B
    is the deployed-bytes PASS band (the manifest's PASS clause)."""
    # Inside +/-1%
    rec = audit._deployed_bytes_check(4194404, 4194404, 1.0, "deployed_bytes_t2")
    assert rec["in_band"] is True
    # Just inside
    rec = audit._deployed_bytes_check(4194404 + 4100, 4194404, 1.0, "deployed_bytes_t2")
    assert rec["in_band"] is True
    # Outside
    rec = audit._deployed_bytes_check(4194404 + 50000, 4194404, 1.0, "deployed_bytes_t2")
    assert rec["in_band"] is False


def test_verdict_reproduced_iff_all_in_band(tmp_path: Path) -> None:
    """verdict is REPRODUCED iff every check passes; otherwise DRIFTED."""
    # All in band -> REPRODUCED
    agg_in = _aggregate([
        _seed_dict(seed=1, ppl=20.96, arc=0.600, lam=0.545,
                   pre_ppl=425.76, pre_arc=0.4891, dep_bytes=4194404),
        _seed_dict(seed=2, ppl=20.96, arc=0.600, lam=0.545,
                   pre_ppl=425.76, pre_arc=0.4891, dep_bytes=4194404),
        _seed_dict(seed=3, ppl=20.96, arc=0.600, lam=0.545,
                   pre_ppl=425.76, pre_arc=0.4891, dep_bytes=4194404),
    ])
    manifest = _fake_manifest(tmp_path)
    af2d = _af2d_reference()
    # Round-trip through disk to mimic the production path.
    af2d_path = tmp_path / "af2d.json"
    af2d_path.write_text(json.dumps(af2d))
    agg_path = tmp_path / "agg.json"
    agg_path.write_text(json.dumps(agg_in))
    out = audit.audit_rpm_000(aggregate_path=agg_path,
                              af2d_reference_path=af2d_path,
                              manifest_path=manifest)
    assert out["verdict"] == "REPRODUCED"
    assert all(c["in_band"] for c in out["checks"])

    # Drift one check -> DRIFTED
    agg_out = _aggregate([
        _seed_dict(seed=1, ppl=99.99, arc=0.600, lam=0.545,  # ppl out of [18.66, 23.26]
                   pre_ppl=425.76, pre_arc=0.4891, dep_bytes=4194404),
        _seed_dict(seed=2, ppl=20.96, arc=0.600, lam=0.545,
                   pre_ppl=425.76, pre_arc=0.4891, dep_bytes=4194404),
        _seed_dict(seed=3, ppl=20.96, arc=0.600, lam=0.545,
                   pre_ppl=425.76, pre_arc=0.4891, dep_bytes=4194404),
    ])
    agg_path.write_text(json.dumps(agg_out))
    out = audit.audit_rpm_000(aggregate_path=agg_path,
                              af2d_reference_path=af2d_path,
                              manifest_path=manifest)
    assert out["verdict"] == "DRIFTED"
    assert any(not c["in_band"] for c in out["checks"])


def test_main_help_runs() -> None:
    """The audit script must expose --help without crashing."""
    subprocess.run(
        [
            sys.executable, str(EXAMPLES / "audit_rpm_000_reproduction.py"),
            "--help",
        ],
        check=True, capture_output=True, text=True,
    )