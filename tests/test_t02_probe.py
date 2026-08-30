"""Tests for the EXP-RPM-T02-PROBE frozen rules.

Covers the pure rule functions in examples/t02_regime_probe.py:

  - per-task drop bar: >= max(3 x stderr_max, 0.02)
  - regime qualify rule: >= 1 of 4 tasks below the bar
  - D1p near-FP16 verification gate (>= 3 of 4 tasks within
    2 x stderr)
  - frozen regime selection: largest summed drop; ties -> more
    severe threshold
  - probe summary decision mapping: REGIMES_FOUND / NO_REGIME /
    INVALID
"""

import importlib.util
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


t02 = _load("t02_regime_probe")

TASKS = ["hellaswag", "winogrande", "boolq", "openbookqa"]

# FP16 reference: stable scores with small stderr.
FP16 = {
    "hellaswag": (0.6600, 0.005),
    "winogrande": (0.6150, 0.004),
    "boolq": (0.6600, 0.006),
    "openbookqa": (0.3600, 0.010),
}

# Near-FP16 regime (reproduces T01's D1p diagnosis).
D1P_NEAR = {
    "hellaswag": (0.6590, 0.005),
    "winogrande": (0.6155, 0.004),
    "boolq": (0.6580, 0.006),
    "openbookqa": (0.3570, 0.010),
}

# Clearly damaging regime: two tasks drop well past the bar.
D4P_DAMAGED = {
    "hellaswag": (0.4000, 0.005),   # drop 0.26
    "winogrande": (0.3000, 0.004),  # drop 0.315
    "boolq": (0.6300, 0.006),       # drop 0.03 (below 3x stderr+abs? 0.03 >= max(0.018,0.02) -> qualifies too)
    "openbookqa": (0.3500, 0.010),  # drop 0.01 -> no
}


def test_per_task_drop_below_bar_is_zero():
    drop = t02.per_task_drop(0.6600, 0.005, 0.6580, 0.005)
    assert drop == 0.0


def test_per_task_drop_above_bar():
    drop = t02.per_task_drop(0.6600, 0.005, 0.4000, 0.005)
    assert drop == pytest.approx(0.26)


def test_per_task_drop_abs_floor():
    # stderr near zero: the 0.02 absolute floor still applies.
    drop = t02.per_task_drop(0.6600, 0.0001, 0.6300, 0.0001)
    assert drop == pytest.approx(0.03)


def test_regime_qualifies_damaged():
    v = t02.regime_qualifies(FP16, D4P_DAMAGED, TASKS)
    assert v["qualifying"] is True
    assert v["per_task_drops"]["hellaswag"] == pytest.approx(0.26)
    assert v["per_task_drops"]["openbookqa"] == 0.0


def test_regime_not_qualifying_near_fp16():
    v = t02.regime_qualifies(FP16, D1P_NEAR, TASKS)
    assert v["qualifying"] is False
    assert v["summed_drop"] == 0.0


def test_d1p_gate_passes_near_fp16():
    g = t02.d1p_gate_ok(FP16, D1P_NEAR, TASKS)
    assert g["gate_ok"] is True
    assert g["n_tasks_near_fp16"] == 4


def test_d1p_gate_fails_when_damaged():
    g = t02.d1p_gate_ok(FP16, D4P_DAMAGED, TASKS)
    assert g["gate_ok"] is False


def _results(regime_scores_by_id):
    out = {
        "fp16": {"regime_id": "fp16", "scores": FP16,
                 "verdict": {"qualifying": False, "per_task_drops": {},
                             "summed_drop": 0.0}},
    }
    for rid, scores in regime_scores_by_id.items():
        out[rid] = {
            "regime_id": rid,
            "scores": scores,
            "verdict": t02.regime_qualifies(FP16, scores, TASKS),
        }
    return out


def test_select_regime_largest_drop():
    results = _results({"D1p": D1P_NEAR, "D4p": D4P_DAMAGED})
    sel = t02.select_regime(results)
    assert sel["selected"] == "D4p"
    assert sel["qualifying_regimes"] == ["D4p"]


def test_select_regime_tie_prefers_more_severe():
    alt = dict(D4P_DAMAGED)
    results = _results({"D4p": alt, "D5p": D4P_DAMAGED})
    sel = t02.select_regime(results)
    assert sel["selected"] == "D5p"
    assert set(sel["qualifying_regimes"]) == {"D4p", "D5p"}


def test_select_regime_none():
    results = _results({"D1p": D1P_NEAR})
    sel = t02.select_regime(results)
    assert sel["selected"] is None
    assert sel["qualifying_regimes"] == []


def test_probe_summary_regimes_found():
    results = _results({"D1p": D1P_NEAR, "D4p": D4P_DAMAGED})
    s = t02.build_probe_summary(results, TASKS)
    assert s["probe_valid"] is True
    assert s["decision"] == "REGIMES_FOUND"
    assert s["selection"]["selected"] == "D4p"


def test_probe_summary_no_regime():
    results = _results({"D1p": D1P_NEAR})
    s = t02.build_probe_summary(results, TASKS)
    assert s["decision"] == "NO_REGIME"


def test_probe_summary_invalid_when_d1p_gate_fails():
    # If even D1p is damaged, the environment/instrument drifted.
    results = _results({"D1p": D4P_DAMAGED, "D4p": D4P_DAMAGED})
    s = t02.build_probe_summary(results, TASKS)
    assert s["probe_valid"] is False
    assert s["decision"] == "INVALID"
