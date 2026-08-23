"""Tests for examples/audit_rpm_d_cross_regime.py.

The cross-regime audit verifies the RPM-002 damage-dependence
hypothesis: T2 value (trained-vs-random z-score) is non-decreasing
across 3 or more consecutive damage regimes.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
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


audit = _load("audit_rpm_d_cross_regime")


def _arm_with_variance(name, ppl, ppl_vals, arc, arc_vals, lam, lam_vals):
    """Per-arm with non-trivial per-seed variance (avoids the
    diff_se=1e-9 fallback that makes z-scores blow up)."""
    return {
        "matched_bytes": [4199318] * 3,
        "tasks": {
            "wikitext": {
                "mean": ppl, "stderr": 1.0, "values": list(ppl_vals)
            },
            "arc_easy": {
                "mean": arc, "stderr": 0.01, "values": list(arc_vals)
            },
            "lambada_openai": {
                "mean": lam, "stderr": 0.01, "values": list(lam_vals)
            },
        },
    }


def _vals(mean, sep):
    return [mean - sep * 0.1, mean, mean + sep * 0.1]


def _regime_aggregate_growing(n):
    """6 regimes where the trained-vs-random separation grows with n
    (gap increases across all 3 metrics)."""
    sep_ppl = 100.0 + n * 50.0
    sep_arc = 0.05 + n * 0.02
    sep_lam = 0.10 + n * 0.05
    return {
        "trained_arms": {
            "t2_ternary": _arm_with_variance(
                "t2", 20.0, _vals(20.0, 1.0),
                0.600, _vals(0.600, 0.005),
                0.545, _vals(0.545, 0.005),
            ),
        },
        "untrained_controls": {
            "random_t2_ternary": _arm_with_variance(
                "rt", 20.0 + sep_ppl, _vals(20.0 + sep_ppl, 1.0),
                0.495 - sep_arc, _vals(0.495 - sep_arc, 0.005),
                0.255 - sep_lam, _vals(0.255 - sep_lam, 0.005),
            ),
            "no_correction": _arm_with_variance(
                "nc", 50.0 + n * 100, _vals(50.0 + n * 100, 1.0),
                0.50, [0.49, 0.50, 0.51],
                0.30, [0.29, 0.30, 0.31],
            ),
        },
    }


def _regime_aggregate_shrinking(n):
    """6 regimes where the trained-vs-random separation shrinks with n
    (gap decreases across all 3 metrics)."""
    sep_ppl = max(300.0 - n * 50.0, 10.0)
    sep_arc = max(0.10 - n * 0.015, 0.005)
    sep_lam = max(0.20 - n * 0.030, 0.005)
    return {
        "trained_arms": {
            "t2_ternary": _arm_with_variance(
                "t2", 20.0, _vals(20.0, 1.0),
                0.600, _vals(0.600, 0.005),
                0.545, _vals(0.545, 0.005),
            ),
        },
        "untrained_controls": {
            "random_t2_ternary": _arm_with_variance(
                "rt", 20.0 + sep_ppl, _vals(20.0 + sep_ppl, 1.0),
                0.495 + (0.10 - sep_arc), _vals(0.495 + (0.10 - sep_arc), 0.005),
                0.255 + (0.20 - sep_lam), _vals(0.255 + (0.20 - sep_lam), 0.005),
            ),
            "no_correction": _arm_with_variance(
                "nc", 50.0 + n * 100, _vals(50.0 + n * 100, 1.0),
                0.50, [0.49, 0.50, 0.51],
                0.30, [0.29, 0.30, 0.31],
            ),
        },
    }


def _aggregate_basic(t2_ppl, random_ppl, no_corr_ppl):
    """Plain aggregate for non-cross-regime tests."""
    return {
        "trained_arms": {
            "t2_ternary": _arm_with_variance(
                "t2", t2_ppl, _vals(t2_ppl, 1.0),
                0.600, _vals(0.600, 0.005),
                0.545, _vals(0.545, 0.005),
            ),
        },
        "untrained_controls": {
            "random_t2_ternary": _arm_with_variance(
                "rt", random_ppl, _vals(random_ppl, 1.0),
                0.495, _vals(0.495, 0.005),
                0.255, _vals(0.255, 0.005),
            ),
            "no_correction": _arm_with_variance(
                "nc", no_corr_ppl, _vals(no_corr_ppl, 1.0),
                0.50, [0.49, 0.50, 0.51],
                0.30, [0.29, 0.30, 0.31],
            ),
        },
    }


def test_per_regime_metrics_extracts_z_scores(tmp_path):
    """z = (random_ppl - trained_ppl) / pooled_stderr.

    With trained mean=20 (values 19,20,21; stderr ~1.0) and random
    mean=400 (values 399,400,401; stderr ~1.0):
      diff_se = sqrt(1.0^2 + 1.0^2) ~ 1.41
      z ~ (400 - 20) / 1.41 ~ 269 sigma.
    """
    agg_path = tmp_path / "aggregate.json"
    agg_path.write_text(json.dumps(_aggregate_basic(20.0, 400.0, 425.0)))
    m = audit._per_regime_metrics(agg_path)
    assert m["pre_train_ppl"] == 425.0
    assert m["trained_t2_ppl"] == 20.0
    assert m["random_t2_ppl"] == 400.0
    assert m["recovery_ratio"] == 20.0 / 425.0
    assert m["metrics"]["wikitext"] > 100


def test_regime_damage_order_sorts_by_digit():
    paths = [
        Path("/x/EXP-RPM-D3/20260823T000000Z/af2d"),
        Path("/x/EXP-RPM-D0/20260823T000000Z/af2d"),
        Path("/x/EXP-RPM-D5/20260823T000000Z/af2d"),
        Path("/x/EXP-RPM-D1/20260823T000000Z/af2d"),
    ]
    ordered = audit._regime_damage_order(paths)
    assert [p.parent.parent.name for p in ordered] == [
        "EXP-RPM-D0", "EXP-RPM-D1", "EXP-RPM-D3", "EXP-RPM-D5"
    ]


def test_cross_regime_audit_pass_when_monotonic(tmp_path):
    """6 regimes where the trained-vs-random separation grows with
    damage severity across all 3 capability metrics. PASS expected."""
    regimes = []
    for n in range(6):
        agg = _regime_aggregate_growing(n)
        reg_dir = tmp_path / f"EXP-RPM-D{n}" / "ts" / "af2d"
        reg_dir.mkdir(parents=True)
        (reg_dir / "aggregate.json").write_text(json.dumps(agg))
        regimes.append(reg_dir)
    result = audit.audit_cross_regime(regimes)
    assert result["n_regimes"] == 6
    assert result["verdict"] == "PASS"
    runs = [c["non_decreasing_run"] for c in result["metric_checks"].values()]
    assert max(runs) >= 3


def test_cross_regime_audit_fail_when_decreasing_z(tmp_path):
    """6 regimes where the trained-vs-random separation DECREASES with
    damage severity across all 3 metrics. FAIL expected."""
    regimes = []
    for n in range(6):
        agg = _regime_aggregate_shrinking(n)
        reg_dir = tmp_path / f"EXP-RPM-D{n}" / "ts" / "af2d"
        reg_dir.mkdir(parents=True)
        (reg_dir / "aggregate.json").write_text(json.dumps(agg))
        regimes.append(reg_dir)
    result = audit.audit_cross_regime(regimes)
    assert result["verdict"] == "FAIL"


def test_cross_regime_audit_fail_with_only_two_regimes(tmp_path):
    """Fewer than three regimes - FAIL (RPM-002 needs 3+ regimes)."""
    regimes = []
    for n in range(2):
        agg = _aggregate_basic(20.0, 400.0, 425.0)
        reg_dir = tmp_path / f"EXP-RPM-D{n}" / "ts" / "af2d"
        reg_dir.mkdir(parents=True)
        (reg_dir / "aggregate.json").write_text(json.dumps(agg))
        regimes.append(reg_dir)
    result = audit.audit_cross_regime(regimes)
    assert result["verdict"] == "FAIL"
    for c in result["metric_checks"].values():
        assert c["in_band"] is False
        assert "fewer than 3" in c.get("note", "")


def test_per_regime_metrics_accepts_directory_path(tmp_path):
    """The audit should accept either the agg.json file OR its
    containing directory."""
    agg_path = tmp_path / "aggregate.json"
    agg_path.write_text(json.dumps(_aggregate_basic(20.0, 400.0, 425.0)))
    m = audit._per_regime_metrics(tmp_path)
    assert m["pre_train_ppl"] == 425.0


def test_main_help_runs():
    subprocess.run(
        [
            sys.executable, str(EXAMPLES / "audit_rpm_d_cross_regime.py"),
            "--help",
        ],
        check=True, capture_output=True, text=True,
    )