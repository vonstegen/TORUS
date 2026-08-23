"""Tests for examples/audit_rpm_000_reproduction.py.

The EXP-RPM-000 (G-RPM-0) audit verifies a fresh AF2-D reproduction
aggregate against preregistered PASS bands. It is the structural
twin of test_audit_af2_reproduction.py but operates on the
post-train aggregate + per-seed pre_train_eval.json rather than
the token cache; the AF2-R auditor remains the canonical
token-cache provenance check.

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

def _tasks(ppl: float, arc: float, lam: float, vals_ppl=(20.96, 20.96, 20.96),
           vals_arc=(0.600, 0.600, 0.600), vals_lam=(0.545, 0.545, 0.545)) -> dict:
    return {
        "wikitext": {"mean": ppl, "stderr": 0.1, "values": list(vals_ppl)},
        "arc_easy": {"mean": arc, "stderr": 0.001, "values": list(vals_arc)},
        "lambada_openai": {"mean": lam, "stderr": 0.001, "values": list(vals_lam)},
    }


def _aggregate(t2_ppl=20.96, t2_arc=0.600, t2_lam=0.545,
               matched=(4199318, 4199318, 4199318)) -> dict:
    return {
        "trained_arms": {
            "t2_ternary": {
                "n": 3,
                "matched_bytes": list(matched),
                "matched_bytes_target": 4194404,
                "tasks": _tasks(t2_ppl, t2_arc, t2_lam),
            }
        }
    }


def _pre_train_eval(seed: int, ppl: float, arc: float) -> dict:
    return {
        "seed": seed,
        "tasks": {
            "wikitext": {"metric": "word_perplexity,none", "value": ppl},
            "arc_easy": {"metric": "acc_norm,none", "value": arc},
            "lambada_openai": {"metric": "acc,none", "value": 0.25},
        },
    }


def _seed_dir(tmp_path: Path, seed: int, pre_ppl: float, pre_arc: float) -> Path:
    sd = tmp_path / f"seed-{seed:03d}"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "pre_train_eval.json").write_text(json.dumps(_pre_train_eval(seed, pre_ppl, pre_arc)))
    return sd


def _af2d_reference() -> dict:
    return {"git_sha": "frozen", "run_dir": "experiments/AF2-D/runs/<ts>/af2d/"}


def _fake_manifest(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text("id: EXP-RPM-000\nrevision: 'frozen'\n")
    return p


# ---- tests --------------------------------------------------------------


def test_af2d_reference_has_required_keys() -> None:
    """The reference dict (AF2-D aggregate_corrected.json) is opaque to
    the audit; only git_sha + run_dir are read. Pin the contract so
    a future schema change doesn't silently break."""
    rec = _af2d_reference()
    for k in ("git_sha", "run_dir"):
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
    """+/-1% tolerance on the AF2-D t2_ternary reference 4,199,318 B
    is the deployed-bytes PASS band (the manifest's PASS clause)."""
    # Inside +/-1%
    rec = audit._deployed_bytes_check(4199318, 4199318, 1.0, "deployed_bytes_t2")
    assert rec["in_band"] is True
    # Just inside
    rec = audit._deployed_bytes_check(4199318 + 4100, 4199318, 1.0, "deployed_bytes_t2")
    assert rec["in_band"] is True
    # Outside
    rec = audit._deployed_bytes_check(4199318 + 50000, 4199318, 1.0, "deployed_bytes_t2")
    assert rec["in_band"] is False


def test_verdict_reproduced_iff_all_in_band(tmp_path: Path) -> None:
    """verdict is REPRODUCED iff every check passes; otherwise DRIFTED."""
    # Build per-seed pre_train_eval with values inside the band.
    for s, (p, a) in enumerate([(425.76, 0.4891)] * 3, start=1):
        _seed_dir(tmp_path, s, p, a)

    # All in band -> REPRODUCED
    agg_in = _aggregate()
    manifest = _fake_manifest(tmp_path)
    af2d = _af2d_reference()
    af2d_path = tmp_path / "af2d.json"
    af2d_path.write_text(json.dumps(af2d))
    agg_path = tmp_path / "agg.json"
    agg_path.write_text(json.dumps(agg_in))
    out = audit.audit_rpm_000(aggregate_path=agg_path,
                              af2d_reference_path=af2d_path,
                              manifest_path=manifest,
                              runs_dir=tmp_path)
    assert out["verdict"] == "REPRODUCED", json.dumps(out, indent=2)
    assert all(c["in_band"] for c in out["checks"])

    # Drift one check (post-train ppl out of [18.66, 23.26]) -> DRIFTED
    agg_out = _aggregate(t2_ppl=99.99)
    agg_path.write_text(json.dumps(agg_out))
    out = audit.audit_rpm_000(aggregate_path=agg_path,
                              af2d_reference_path=af2d_path,
                              manifest_path=manifest,
                              runs_dir=tmp_path)
    assert out["verdict"] == "DRIFTED"
    assert any(not c["in_band"] for c in out["checks"])

    # Drift pre-train ppl (out of [400, 460]) -> DRIFTED
    for s in range(1, 4):
        (tmp_path / f"seed-{s:03d}" / "pre_train_eval.json").write_text(
            json.dumps(_pre_train_eval(s, 100.0, 0.5))  # ppl out of band
        )
    agg_path.write_text(json.dumps(agg_in))
    out = audit.audit_rpm_000(aggregate_path=agg_path,
                              af2d_reference_path=af2d_path,
                              manifest_path=manifest,
                              runs_dir=tmp_path)
    assert out["verdict"] == "DRIFTED"
    assert not out["checks"][0]["in_band"]


def test_avg_pre_train_averages_across_seeds(tmp_path: Path) -> None:
    """Per-seed pre_train_eval is at seed-XXX/pre_train_eval.json;
    _avg_pre_train returns the mean across seeds."""
    _seed_dir(tmp_path, 1, 425.0, 0.49)
    _seed_dir(tmp_path, 2, 426.0, 0.48)
    _seed_dir(tmp_path, 3, 425.0, 0.50)
    ppl, arc = audit._avg_pre_train(tmp_path)
    assert abs(ppl - 425.3333333333) < 1e-6
    assert abs(arc - 0.49) < 1e-6


def test_main_help_runs() -> None:
    """The audit script must expose --help without crashing."""
    subprocess.run(
        [
            sys.executable, str(EXAMPLES / "audit_rpm_000_reproduction.py"),
            "--help",
        ],
        check=True, capture_output=True, text=True,
    )