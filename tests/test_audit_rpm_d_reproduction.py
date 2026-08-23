"""Tests for examples/audit_rpm_d_reproduction.py.

The per-regime Stage 1 audit verifies a fresh RPM-D<n> aggregate
against preregistered bands + the RPM-006 axis (trained vs
random T2 separation). It is the structural twin of
test_audit_rpm_000_reproduction.py but extended to all8 arms
and the cross-arm z-test.
"""
from __future__ import annotations

import importlib.util
import json
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


audit = _load("audit_rpm_d_reproduction")


# ---- synthetic aggregate + manifest fixtures --------------------------

def _tasks(ppl: float, arc: float, lam: float, ppl_vals, arc_vals, lam_vals) -> dict:
    return {
        "wikitext": {"mean": ppl, "stderr": 0.1, "values": list(ppl_vals)},
        "arc_easy": {"mean": arc, "stderr": 0.001, "values": list(arc_vals)},
        "lambada_openai": {"mean": lam, "stderr": 0.001, "values": list(lam_vals)},
    }


def _arm(name: str, matched: int, ppl: float, arc: float, lam: float,
         ppl_vals, arc_vals, lam_vals) -> dict:
    return {
        "n": 3,
        "matched_bytes": [matched, matched, matched],
        "matched_bytes_target": 4194404,
        "tasks": _tasks(ppl, arc, lam, ppl_vals, arc_vals, lam_vals),
    }


def _aggregate(
    t2_ppl=20.0, t2_arc=0.600, t2_lam=0.545,
    int4_ppl=120.0, int8_ppl=80.0, lora_ppl=22.0, dense_ppl=42.0,
    random_t2_ppl=400.0, random_lora_ppl=400.0, no_corr_ppl=425.0,
    matched=4199318,
) -> dict:
    """Synthetic aggregate that should pass all checks."""
    return {
        "trained_arms": {
            "t2_ternary": _arm("t2", matched, t2_ppl, t2_arc, t2_lam,
                               [t2_ppl] * 3, [t2_arc] * 3, [t2_lam] * 3),
            "int4_residual": _arm("i4", matched, int4_ppl, 0.50, 0.30,
                                    [int4_ppl] * 3, [0.50] * 3, [0.30] * 3),
            "int8_residual": _arm("i8", matched, int8_ppl, 0.55, 0.35,
                                    [int8_ppl] * 3, [0.55] * 3, [0.35] * 3),
            "lora": _arm("lo", matched, lora_ppl, 0.61, 0.56,
                         [lora_ppl] * 3, [0.61] * 3, [0.56] * 3),
            "dense_adapter": _arm("de", matched, dense_ppl, 0.62, 0.55,
                                   [dense_ppl] * 3, [0.62] * 3, [0.55] * 3),
        },
        "untrained_controls": {
            "random_t2_ternary": _arm("rt", matched, random_t2_ppl, 0.495, 0.255,
                                        [random_t2_ppl] * 3, [0.495] * 3,
                                        [0.255] * 3),
            "random_lora": _arm("rl", matched, random_lora_ppl, 0.50, 0.30,
                                  [random_lora_ppl] * 3, [0.50] * 3, [0.30] * 3),
        },
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


def _seed_dir(tmp_path: Path, seed: int, ppl: float, arc: float) -> Path:
    sd = tmp_path / f"seed-{seed:03d}"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "pre_train_eval.json").write_text(
        json.dumps(_pre_train_eval(seed, ppl, arc))
    )
    return sd


def _fake_manifest(tmp_path: Path, threshold: float | None) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(
        f"id: EXP-RPM-Dtest\n"
        f"damage_ptq:\n"
        f"  threshold: {'null' if threshold is None else threshold}\n"
        f"  pre_train_eval_check: 'synthetic'\n"
        f"target_deployed_bytes: 4194404\n"
        f"matched_bytes_tolerance_pct: 1.0\n"
        f"training:\n  n_steps: 500\n"
    )
    return p


# ---- tests -------------------------------------------------------------


def test_recovery_check_pass() -> None:
    """Post-train ppl <= 1.5x pre-train ppl passes."""
    pre_ppl = 425.0
    trained = {
        "tasks": {
            "wikitext": {
                "mean": 20.0, "stderr": 1.5, "values": [19.0, 21.0, 20.0]
            }
        }
    }
    rec = audit._recovery_check(pre_ppl, trained)
    assert rec["in_band"] is True
    assert abs(rec["ratio"] - 20.0 / 425.0) < 1e-6


def test_recovery_check_fail() -> None:
    pre_ppl = 425.0
    trained = {
        "tasks": {
            "wikitext": {
                "mean": 1000.0, "stderr": 5.0, "values": [990, 1010, 1000]
            }
        }
    }
    rec = audit._recovery_check(pre_ppl, trained)
    assert rec["in_band"] is False
    assert rec["ratio"] > 1.5


def test_representation_signal_pass() -> None:
    """Trained ppl = 20, random ppl = 400; both with stderr=1.0.
    z = (400 - 20) / sqrt(1^2 + 1^2) = ~268 sigma. Should pass easily.
    """
    trained = {
        "tasks": {
            "wikitext": {"values": [19, 20, 21]},
            "arc_easy": {"values": [0.59, 0.60, 0.61]},
            "lambada_openai": {"values": [0.54, 0.545, 0.55]},
        }
    }
    random = {
        "tasks": {
            "wikitext": {"values": [399, 400, 401]},
            "arc_easy": {"values": [0.49, 0.495, 0.50]},
            "lambada_openai": {"values": [0.25, 0.255, 0.26]},
        }
    }
    rec = audit._representation_signal_check(trained, random)
    assert rec["in_band"] is True
    assert rec["metrics"]["wikitext"]["passes_2sigma"] is True
    assert rec["metrics"]["arc_easy"]["passes_2sigma"] is True
    assert rec["metrics"]["lambada_openai"]["passes_2sigma"] is True


def test_representation_signal_fail() -> None:
    """Trained ppl ~ random ppl -> no separation."""
    trained = {
        "tasks": {
            "wikitext": {"values": [400, 401, 402]},
            "arc_easy": {"values": [0.495, 0.495, 0.495]},
            "lambada_openai": {"values": [0.25, 0.25, 0.26]},
        }
    }
    random = {
        "tasks": {
            "wikitext": {"values": [400, 401, 402]},
            "arc_easy": {"values": [0.495, 0.495, 0.495]},
            "lambada_openai": {"values": [0.25, 0.25, 0.26]},
        }
    }
    rec = audit._representation_signal_check(trained, random)
    assert rec["in_band"] is False


def test_matched_bytes_check_pass() -> None:
    """All per-seed matched_bytes within +/-1% of 4194404."""
    arm = {"matched_bytes": [4199318, 4199318, 4199318]}
    rec = audit._matched_bytes_check(arm, 4194404, 1.0)
    assert rec["in_band"] is True


def test_matched_bytes_check_fail() -> None:
    arm = {"matched_bytes": [5000000, 4199318, 4199318]}
    rec = audit._matched_bytes_check(arm, 4194404, 1.0)
    assert rec["in_band"] is False
    assert 5000000 in rec["violations"]


def test_per_arm_summary_extracts_correct_fields() -> None:
    agg = _aggregate()
    summary = audit._per_arm_summary(agg)
    assert "t2_ternary" in summary["trained"]
    assert summary["trained"]["t2_ternary"]["matched_bytes"] == [4199318] * 3
    assert "wikitext" in summary["trained"]["t2_ternary"]["tasks"]
    assert "random_t2_ternary" in summary["untrained"]
    assert summary["untrained"]["random_t2_ternary"]["matched_bytes"] == [4199318] * 3


def test_full_audit_pass_plus_path(tmp_path: Path) -> None:
    """Full audit integration test: synthetic data, all checks pass.
    Pre-train ppl 425 (D5-style damage); trained ppl 20, arc 0.600,
    lambada 0.545. Random ppl 400. -> PASS_PLUS (recovery + RPM-006).
    """
    for s, (p, a) in enumerate([(425.0, 0.4891)] * 3, start=1):
        _seed_dir(tmp_path, s, p, a)

    agg = _aggregate(t2_ppl=20.0, t2_arc=0.600, t2_lam=0.545,
                     random_t2_ppl=400.0)
    manifest = _fake_manifest(tmp_path, threshold=0.7)

    agg_path = tmp_path / "agg.json"
    agg_path.write_text(json.dumps(agg))

    out = audit.audit_rpm_d(aggregate_path=agg_path, manifest_path=manifest,
                            runs_dir=tmp_path)
    assert out["verdict"] in ("PASS", "PASS_PLUS")
    if out["verdict"] == "PASS_PLUS":
        assert out["rep_signal"]["in_band"] is True


def test_full_audit_fail_when_no_recovery(tmp_path: Path) -> None:
    """If trained ppl > 1.5x starting ppl, verdict = FAIL."""
    for s in range(1, 4):
        _seed_dir(tmp_path, s, 425.0, 0.4891)
    # t2_ternary does NOT recover: 900 ppl > 1.5x 425 = 637.5
    agg = _aggregate(t2_ppl=900.0, random_t2_ppl=400.0)
    manifest = _fake_manifest(tmp_path, threshold=0.7)
    agg_path = tmp_path / "agg.json"
    agg_path.write_text(json.dumps(agg))
    out = audit.audit_rpm_d(aggregate_path=agg_path, manifest_path=manifest,
                            runs_dir=tmp_path)
    assert out["verdict"] == "FAIL"


def test_main_help_runs() -> None:
    """The audit script must expose --help without crashing."""
    subprocess.run(
        [
            sys.executable, str(EXAMPLES / "audit_rpm_d_reproduction.py"),
            "--help",
        ],
        check=True, capture_output=True, text=True,
    )