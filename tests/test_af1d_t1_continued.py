"""Tests for the EXP-AF-001-D frozen bar rules.

Covers the pure rule functions in examples/af1d_t1_continued.py:

  - sd-of-difference
  - acceptance-bar item 1 application: PASS (T2 beats arm A by
    >2sd on >= 2 metrics, no regression), FAIL (arm A beats T2 by
    >2sd on >= 1 metric), NULL otherwise
  - arm B z-scores recorded descriptively (no gate)

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


af1d = _load("af1d_t1_continued")


def test_sd_of_difference():
    assert af1d.sd_of_difference(0.003, 0.004) == pytest.approx(0.005)
    assert af1d.sd_of_difference(None, 0.004) == pytest.approx(0.004)


def test_pass_when_t2_beats_armA_on_two_metrics():
    # arm A far worse than T2 on wikitext and lambada, within 2sd on
    # arc_easy (no regression).
    arm_a = {
        "wikitext": (120.0, 5.0),
        "arc_easy": (0.590, 0.004),
        "lambada_openai": (0.30, 0.01),
    }
    arm_b = {
        "wikitext": (150.0, 5.0),
        "arc_easy": (0.570, 0.004),
        "lambada_openai": (0.25, 0.01),
    }
    v = af1d.apply_bars(arm_a, arm_b)
    assert v["decision"] == "PASS"
    assert v["n_pass_metrics"] == 2
    assert v["regressions"] == []


def test_fail_when_armA_beats_t2():
    arm_a = {
        "wikitext": (15.0, 1.0),   # arm A better than T2's 20.96
        "arc_easy": (0.62, 0.004),
        "lambada_openai": (0.58, 0.003),
    }
    arm_b = {
        "wikitext": (25.0, 1.0),
        "arc_easy": (0.60, 0.004),
        "lambada_openai": (0.55, 0.003),
    }
    v = af1d.apply_bars(arm_a, arm_b)
    assert v["decision"] == "FAIL"


def test_null_when_no_separation():
    # arm A ~ T2 on every metric (within 2sd): no separation.
    arm_a = {
        "wikitext": (21.5, 1.0),
        "arc_easy": (0.598, 0.004),
        "lambada_openai": (0.543, 0.003),
    }
    arm_b = {
        "wikitext": (22.0, 1.0),
        "arc_easy": (0.597, 0.004),
        "lambada_openai": (0.542, 0.003),
    }
    v = af1d.apply_bars(arm_a, arm_b)
    assert v["decision"] == "NULL"


def test_regression_blocks_pass():
    # T2 crushes arm A on 2 metrics but arm A beats T2 by >2sd on
    # arc_easy -> no PASS (regression rule).
    arm_a = {
        "wikitext": (150.0, 5.0),
        "arc_easy": (0.65, 0.004),
        "lambada_openai": (0.30, 0.01),
    }
    arm_b = {
        "wikitext": (160.0, 5.0),
        "arc_easy": (0.64, 0.004),
        "lambada_openai": (0.28, 0.01),
    }
    v = af1d.apply_bars(arm_a, arm_b)
    assert "arc_easy" in v["regressions"]
    assert v["decision"] != "PASS"


def test_armB_recorded_but_ungated():
    # Arm B much worse than T2 -> still does not drive the decision;
    # only arm A gates.
    arm_a = {
        "wikitext": (120.0, 5.0),
        "arc_easy": (0.590, 0.004),
        "lambada_openai": (0.30, 0.01),
    }
    arm_b = {
        "wikitext": (400.0, 5.0),
        "arc_easy": (0.50, 0.004),
        "lambada_openai": (0.25, 0.01),
    }
    v = af1d.apply_bars(arm_a, arm_b)
    assert v["decision"] == "PASS"
    assert v["z_t2_minus_armB"]["wikitext"] > 10
