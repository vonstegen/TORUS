"""Tests for the EXP-RPM-T02 frozen threshold rules.

Covers the pure rule functions in examples/t02_af5_tournament.py:

  - sd-of-difference formulation
  - base verification gate vs the frozen probe D5p scores
  - frozen T01 pass thresholds (4 rules, all must hold)
  - frozen fail triggers
  - decision mapping PASS / FAIL

Synthetic data; no torch.
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


t02 = _load("t02_af5_tournament")

TASKS = ["hellaswag", "winogrande", "boolq", "openbookqa"]

SE_SMALL = 0.005


def _arm_map(t2=(0.60, SE_SMALL), r2=(0.55, SE_SMALL),
             rlora=(0.54, SE_SMALL), trained=None):
    trained = trained or {
        "int4_residual": (0.40, SE_SMALL),
        "int8_residual": (0.58, SE_SMALL),
        "lora": (0.57, SE_SMALL),
        "dense_adapter": (0.56, SE_SMALL),
    }
    return {
        "t2_ternary": t2,
        "random_t2_ternary": r2,
        "random_lora": rlora,
        **trained,
    }


def _by_task(pass_like=True):
    if pass_like:
        return {t: _arm_map() for t in TASKS}
    # t2 ~ random everywhere -> all rules miss
    return {t: _arm_map(t2=(0.55, SE_SMALL)) for t in TASKS}


def test_sd_of_difference():
    assert t02.sd_of_difference(0.003, 0.004) == pytest.approx(0.005)
    assert t02.sd_of_difference(None, 0.004) == pytest.approx(0.004)
    assert t02.sd_of_difference(None, None) == 0.0

def test_base_gate_passes_in_band():
    base = {
        "hellaswag": (0.4256, 0.0049),
        "winogrande": (0.5501, 0.0140),
        "boolq": (0.5691, 0.0087),
        "openbookqa": (0.2980, 0.0205),
    }
    g = t02.base_gate_ok(base, TASKS)
    assert g["gate_ok"] is True
    assert g["n_tasks_in_band"] == 4


def test_base_gate_fails_off_band():
    base = {
        "hellaswag": (0.6614, 0.0047),   # FP16-like: drifted
        "winogrande": (0.6172, 0.0137),  # FP16-like: drifted
        "boolq": (0.5691, 0.0087),
        "openbookqa": (0.2980, 0.0205),
    }
    g = t02.base_gate_ok(base, TASKS)
    assert g["gate_ok"] is False
    assert "hellaswag" not in g["in_band_tasks"]

def test_thresholds_pass_when_all_rules_hold():
    v = t02.apply_thresholds(_by_task(pass_like=True))
    assert v["decision"] == "PASS"
    assert all(ok for ok, _ in v["rules"].values())
    assert not any(fired for fired, _ in v["fail_triggers"].values())


def test_thresholds_fail_when_t2_matches_random():
    v = t02.apply_thresholds(_by_task(pass_like=False))
    assert v["decision"] == "FAIL"
    # r1/r2/r4 miss (t2 ~ random), r3 may hold (best trained is low)
    assert v["rules"]["r1_t2_vs_random_t2_ge_1sd"][0] is False
    assert v["fail_triggers"]["f1_t2_below_chance"][1] == 0


def test_thresholds_fail_trigger_t2_below_chance():
    by_task = {t: _arm_map(t2=(0.50, SE_SMALL)) for t in TASKS}
    v = t02.apply_thresholds(by_task)
    assert v["decision"] == "FAIL"
    fired, n = v["fail_triggers"]["f1_t2_below_chance"]
    assert fired is True and n == 4


def test_thresholds_rule3_ties_ok():
    # t2 ties the best trained comparator -> rule 3 counts it.
    by_task = {t: _arm_map(trained={
        "int4_residual": (0.40, SE_SMALL),
        "int8_residual": (0.60, SE_SMALL),
        "lora": (0.57, SE_SMALL),
        "dense_adapter": (0.56, SE_SMALL),
    }) for t in TASKS}
    v = t02.apply_thresholds(by_task)
    assert v["rules"]["r3_t2_wins_or_ties_best_trained"][0] is True


def test_thresholds_partial_pass_is_fail():
    # 2 of 4 tasks separate -> rules miss -> FAIL even though no
    # fail trigger fires.
    by_task = {}
    for i, t in enumerate(TASKS):
        if i < 2:
            by_task[t] = _arm_map()
        else:
            by_task[t] = _arm_map(t2=(0.55, SE_SMALL))
    v = t02.apply_thresholds(by_task)
    assert v["decision"] == "FAIL"
    assert v["rules"]["r1_t2_vs_random_t2_ge_1sd"][0] is False
    assert v["fail_triggers"]["f1_t2_below_chance"][0] is False
