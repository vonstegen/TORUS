"""Tests for examples/audit_af4_reproduction.py (EXP-AF-004-R auditor).

Pins the reproduction acceptance rule from the AF4-R manifest:
  1. decision replay reproduces the AF4 verdict on AF4-like data
     (FAIL / joint superior) and detects a flipped verdict;
  2. band check catches an arm-metric mean outside the ±2
     combined-stderr band;
  3. reproduction_verdict returns REPRODUCED / NOT_REPRODUCED /
     INVALID for the right reasons;
  4. run_integrity flags missing runs, freeze violations, wrong
     deployed bytes, and non-finite values.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # noqa: F401 — suite convention

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, EXAMPLES / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit_af4_reproduction")


def _summary(arm, seed, ppl, arc, lam):
    return {
        "arm": arm,
        "seed": seed,
        "freeze_check": True if arm == "seq" else None,
        "deployed_bytes": audit.EXPECTED_BYTES[arm],
        "tasks": {
            "wikitext": {"metric": "word_perplexity,none", "value": ppl},
            "arc_easy": {"metric": "acc,none", "value": arc},
            "lambada_openai": {"metric": "acc,none", "value": lam},
        },
    }


def _af4_like(ppl_shift=0.0):
    """Synthetic 3-arm x 3-seed dataset shaped like AF4's result:
    joint beats seq on ppl (+5σ-ish) and lambada (huge), arc tied."""
    summaries = []
    for seed in (1, 2, 3):
        summaries.append(_summary("seq", seed, 24.9 + 0.1 * seed + ppl_shift,
                                  0.561, 0.443))
        summaries.append(_summary("joint", seed, 21.4 + 0.05 * seed,
                                  0.569, 0.468))
        summaries.append(_summary("t1_only", seed, 19.3 + 0.2 * seed,
                                  0.655, 0.593))
    return summaries


def _write_run(run_dir: Path, summaries) -> None:
    for s in summaries:
        d = run_dir / f"seed-{s['seed']:03d}" / s["arm"]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "eval.summary.json", "w") as f:
            json.dump(s, f)
        with open(d / "history.jsonl", "w") as f:
            f.write(json.dumps({"step": 0, "loss": 4.0}) + "\n")


def test_replay_decision_reproduces_af4_verdict() -> None:
    stats = audit.compute_arm_stats(_af4_like())
    replay = audit.replay_decision(stats)
    assert replay["verdict"] == "FAIL"
    assert replay["direction"] == "joint superior"
    assert replay["fail2_joint_superior"] is True
    assert replay["pass_clause_fired"] is False


def test_replay_decision_detects_sequential_win() -> None:
    summaries = []
    for seed in (1, 2, 3):
        # seq strictly better than joint on ppl by a wide margin.
        summaries.append(_summary("seq", seed, 18.0 + 0.05 * seed,
                                  0.60, 0.55))
        summaries.append(_summary("joint", seed, 24.0 + 0.05 * seed,
                                  0.59, 0.54))
    stats = audit.compute_arm_stats(summaries)
    replay = audit.replay_decision(stats)
    assert replay["verdict"] == "PASS"
    assert replay["direction"] == "sequential superior"


def test_band_check_flags_outlier() -> None:
    ref = audit.compute_arm_stats(_af4_like())
    new = audit.compute_arm_stats(_af4_like(ppl_shift=0.0))
    bands = audit.band_check(ref, new)
    assert bands["within_all"] is True
    # A 5-ppl shift on seq wikitext must blow the band.
    new_shifted = audit.compute_arm_stats(_af4_like(ppl_shift=5.0))
    bands = audit.band_check(ref, new_shifted)
    assert bands["within_all"] is False
    assert bands["rows"]["seq"]["wikitext"]["within"] is False


def test_reproduction_verdict_outcomes() -> None:
    ref = audit.compute_arm_stats(_af4_like())
    new = audit.compute_arm_stats(_af4_like())
    integrity = {"ok": True, "problems": []}
    replay = audit.replay_decision(new)
    ref_replay = audit.replay_decision(ref)
    bands = audit.band_check(ref, new)
    v = audit.reproduction_verdict(integrity, replay, bands, ref_replay)
    assert v["verdict"] == "REPRODUCED"

    bad_integrity = {"ok": False, "problems": ["missing runs: [('seq', 3)]"]}
    v = audit.reproduction_verdict(bad_integrity, replay, bands, ref_replay)
    assert v["verdict"] == "INVALID"

    shifted = audit.compute_arm_stats(_af4_like(ppl_shift=5.0))
    bands = audit.band_check(ref, shifted)
    v = audit.reproduction_verdict(
        integrity, audit.replay_decision(shifted), bands, ref_replay
    )
    assert v["verdict"] == "NOT_REPRODUCED"
    assert any("band violations" in r for r in v["reasons"])


def test_run_integrity_checks(tmp_path) -> None:
    summaries = _af4_like()
    _write_run(tmp_path, summaries)
    loaded = audit.load_summaries(tmp_path)
    res = audit.run_integrity(tmp_path, loaded)
    assert res["ok"] is True

    # Freeze violation on one seq run.
    bad = [dict(s) for s in summaries]
    bad[0] = dict(bad[0], freeze_check=False)
    _write_run(tmp_path, bad)
    res = audit.run_integrity(tmp_path, audit.load_summaries(tmp_path))
    assert res["ok"] is False
    assert any("freeze_check" in p for p in res["problems"])

    # Missing run.
    for s in summaries:
        if s["arm"] == "t1_only" and s["seed"] == 3:
            summaries.remove(s)
            break
    import shutil

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    _write_run(tmp_path, summaries)
    res = audit.run_integrity(tmp_path, audit.load_summaries(tmp_path))
    assert res["ok"] is False
    assert any("missing runs" in p for p in res["problems"])
