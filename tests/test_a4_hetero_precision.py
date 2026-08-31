"""Tests for the EXP-A4-001 frozen bar rules.

Covers the pure rule functions in examples/a4_hetero_precision.py:

  - per-metric direction z (wikitext ppl lower-is-better)
  - H-bar: hetero beats uniform ternary by >2sd on >=2/3, no
    >=2sd regression
  - C1-bar: hetero not worse than INT8 on any metric AND bytes
    <= INT8 bytes
  - C2-check: hetero's continuation deficit not worse than INT8's
    by >2sd on any metric
  - PASS iff all three
"""

import importlib.util
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


a4 = _load("a4_hetero_precision")


def _means(hetero=(25.0, 1.0), uniform=(60.0, 2.0), int8=(24.0, 1.0),
           cont=(13.5, 0.5)):
    return {
        "hetero_ternary": {
            "wikitext": (hetero[0], hetero[1]),
            "arc_easy": (0.62, 0.004),
            "lambada_openai": (0.58, 0.003),
        },
        "uniform_ternary": {
            "wikitext": (uniform[0], uniform[1]),
            "arc_easy": (0.55, 0.004),
            "lambada_openai": (0.50, 0.003),
        },
        "int8_uniform": {
            "wikitext": (int8[0], int8[1]),
            "arc_easy": (0.615, 0.004),
            "lambada_openai": (0.575, 0.003),
        },
        "int4_uniform": {
            "wikitext": (30.0, 1.0),
            "arc_easy": (0.60, 0.004),
            "lambada_openai": (0.56, 0.003),
        },
        "fp16_continue": {
            "wikitext": (cont[0], cont[1]),
            "arc_easy": (0.64, 0.004),
            "lambada_openai": (0.60, 0.003),
        },
        "fp16_reference": {
            "wikitext": (13.09, 0.01),
            "arc_easy": (0.607, 0.004),
            "lambada_openai": (0.610, 0.003),
        },
    }


def test_z_direction_wikitext_lower_better():
    # a ppl 20 vs b ppl 40 -> a better -> z positive.
    z = a4.z_better((20.0, 1.0), (40.0, 1.0), "wikitext")
    assert z > 0
    z2 = a4.z_better((0.6, 0.01), (0.5, 0.01), "arc_easy")
    assert z2 > 0


def test_pass_when_all_bars_hold():
    bars = a4.apply_bars(_means(),
                         {"hetero_ternary": 500_000_000,
                          "int8_uniform": 900_000_000,
                          "int4_uniform": 480_000_000,
                          "uniform_ternary": 480_000_000,
                          "fp16_continue": 2_400_000_000,
                          "fp16_reference": 2_400_000_000})
    assert bars["decision"] == "PASS"
    assert bars["h_bar"] and bars["c1_bar"] and bars["c2_check"]

def test_fail_when_h_bar_misses():
    # hetero ~ uniform on every metric (no advantage) -> H-bar fails.
    means = _means(hetero=(58.0, 1.0), uniform=(60.0, 2.0))
    for m in ("arc_easy", "lambada_openai"):
        u = means["uniform_ternary"][m]
        means["hetero_ternary"][m] = (u[0] - 0.001, u[1])
    bars = a4.apply_bars(means,
                         {"hetero_ternary": 500_000_000,
                          "int8_uniform": 900_000_000})
    assert bars["decision"] == "FAIL"
    assert bars["h_bar"] is False
    # hetero ~ uniform also loses to int8 on capability here:
    assert bars["c1_bar"] is False

def test_fail_when_hetero_loses_to_int8_on_capability():
    means = _means(hetero=(40.0, 0.5), int8=(24.0, 1.0))
    bars = a4.apply_bars(means,
                         {"hetero_ternary": 500_000_000,
                          "int8_uniform": 900_000_000})
    assert bars["decision"] == "FAIL"
    assert bars["c1_capability_ok"] is False


def test_fail_when_hetero_bigger_than_int8():
    bars = a4.apply_bars(_means(),
                         {"hetero_ternary": 950_000_000,
                          "int8_uniform": 900_000_000})
    assert bars["decision"] == "FAIL"
    assert bars["c1_bytes_ok"] is False
    assert bars["c1_capability_ok"] is True


def test_fail_when_c2_check_misses():
    # hetero much further from continuation than int8 is.
    means = _means(hetero=(60.0, 1.0), int8=(24.0, 1.0),
                   cont=(13.5, 0.5))
    bars = a4.apply_bars(means,
                         {"hetero_ternary": 500_000_000,
                          "int8_uniform": 900_000_000})
    assert bars["c2_check"] is False
    assert bars["decision"] == "FAIL"


def test_pass_requires_all_three():
    bars = a4.apply_bars(_means(),
                         {"hetero_ternary": 500_000_000,
                          "int8_uniform": 900_000_000})
    assert bars["decision"] == ("PASS" if (bars["h_bar"]
                                           and bars["c1_bar"]
                                           and bars["c2_check"])
                                else "FAIL")
