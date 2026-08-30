"""Tests for the EXP-RPM-T02-PROBE frozen rules.

Covers the pure rule functions in examples/t02_regime_probe.py:

  - per-task drop bar: >= max(3 x stderr_max, 0.02)
  - regime qualify rule: >= 1 of 4 tasks below the bar
  - T01-REPRO verification gate (AMENDED): gauss02 near-FP16 on
    >= 3 of 4 tasks (T01's actual eval regime) -> else INVALID
  - frozen regime selection: largest summed drop; ties -> more
    severe threshold; gauss02 never a candidate
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
    "hellaswag": (0.6614, 0.0047),
    "winogrande": (0.6172, 0.0137),
    "boolq": (0.6621, 0.0083),
    "openbookqa": (0.3560, 0.0214),
}

# T01-REPRO (gauss02): T01's actual eval regime — near-FP16.
GAUSS02_NEAR = {
    "hellaswag": (0.6590, 0.0047),
    "winogrande": (0.6160, 0.0137),
    "boolq": (0.6600, 0.0083),
    "openbookqa": (0.3540, 0.0214),
}

# Near-FP16 regime (hypothetical).
D1P_NEAR = {
    "hellaswag": (0.6590, 0.005),
    "winogrande": (0.6155, 0.004),
    "boolq": (0.6580, 0.006),
    "openbookqa": (0.3570, 0.010),
}

# Clearly damaging regime: three tasks drop well past the bar.
D5P_DAMAGED = {
    "hellaswag": (0.4256, 0.0049),  # drop ~0.236
    "winogrande": (0.5501, 0.0140),  # drop ~0.067
    "boolq": (0.5691, 0.0087),       # drop ~0.093
    "openbookqa": (0.2980, 0.0205),  # drop ~0.058
}

D4P_DAMAGED = {
    "hellaswag": (0.4415, 0.0050),
    "winogrande": (0.5485, 0.0140),
    "boolq": (0.5633, 0.0087),
    "openbookqa": (0.3080, 0.0207),
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
    v = t02.regime_qualifies(FP16, D5P_DAMAGED, TASKS)
    assert v["qualifying"] is True
    assert v["per_task_drops"]["hellaswag"] > 0.2
    assert v["per_task_drops"]["winogrande"] > 0.0


def test_regime_not_qualifying_near_fp16():
    v = t02.regime_qualifies(FP16, D1P_NEAR, TASKS)
    assert v["qualifying"] is False
    assert v["summed_drop"] == 0.0


def test_t01_repro_gate_passes_near_fp16():
    g = t02.t01_repro_gate_ok(FP16, GAUSS02_NEAR, TASKS)
    assert g["gate_ok"] is True
    assert g["n_tasks_near_fp16"] == 4


def test_t01_repro_gate_fails_when_damaged():
    g = t02.t01_repro_gate_ok(FP16, D5P_DAMAGED, TASKS)
    assert g["gate_ok"] is False


def _results(regime_scores_by_id, gauss02=GAUSS02_NEAR):
    out = {
        "fp16": {"regime_id": "fp16", "scores": FP16,
                 "verdict": {"qualifying": False, "per_task_drops": {},
                             "summed_drop": 0.0}},
        "gauss02": {"regime_id": "gauss02", "scores": gauss02,
                    "verdict": {"qualifying": False,
                                "per_task_drops": {},
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
    results = _results({"D4p": D4P_DAMAGED, "D5p": D5P_DAMAGED})
    sel = t02.select_regime(results)
    assert sel["selected"] == "D5p"
    assert set(sel["qualifying_regimes"]) == {"D4p", "D5p"}


def test_select_regime_tie_prefers_more_severe():
    results = _results({"D4p": D4P_DAMAGED, "D5p": D4P_DAMAGED})
    sel = t02.select_regime(results)
    assert sel["selected"] == "D5p"


def test_select_regime_none():
    results = _results({"D1p": D1P_NEAR})
    sel = t02.select_regime(results)
    assert sel["selected"] is None
    assert sel["qualifying_regimes"] == []


def test_gauss02_never_a_candidate():
    # Even if gauss02's scores happened to show a drop, the frozen
    # exclusion keeps it out of the candidate set.
    results = _results({}, gauss02=D5P_DAMAGED)
    sel = t02.select_regime(results)
    assert sel["selected"] is None


def test_probe_summary_regimes_found():
    results = _results({"D4p": D4P_DAMAGED, "D5p": D5P_DAMAGED})
    s = t02.build_probe_summary(results, TASKS)
    assert s["probe_valid"] is True
    assert s["decision"] == "REGIMES_FOUND"
    assert s["selection"]["selected"] == "D5p"


def test_probe_summary_no_regime():
    results = _results({"D1p": D1P_NEAR})
    s = t02.build_probe_summary(results, TASKS)
    assert s["decision"] == "NO_REGIME"


def test_probe_summary_invalid_when_gate_fails():
    # If gauss02 itself is damaged, the environment/instrument
    # drifted -> INVALID.
    results = _results({"D5p": D5P_DAMAGED}, gauss02=D5P_DAMAGED)
    s = t02.build_probe_summary(results, TASKS)
    assert s["probe_valid"] is False
    assert s["decision"] == "INVALID"
