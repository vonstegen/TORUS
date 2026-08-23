"""Cross-regime RPM-002 auditor.

Verifies the Stage 1 (EXP-RPM-D0..D5) cross-regime aggregate
against the RPM-002 damage-dependence hypothesis:

  H-RPM-002: T2 value increases with base damage severity. The
  relative gain of trained T2 over random T2 (and over no-
  correction) is non-decreasing across at least three consecutive
  damage regimes.

The cross-regime comparison is per-arm:
  - For each regime D<n> in D0..D5:
    - Load aggregate.json.
    - Compute trained-vs-random T2 z-score on each capability
      metric (wikitext, arc, lambada).
    - Compute trained t2_ternary recovery ratio
      (post_train_ppl / pre_train_ppl).
  - Sort regimes by their preregistered damage knob (D0 = mildest
    damage; D5 = most damage).
  - PASS: the trained-vs-random z-score on at least one metric is
    non-decreasing across >=3 consecutive regimes.
  - FAIL: the trained-vs-random z-score is non-increasing across
    ALL consecutive regime pairs (no positive trend).

Usage::

    python examples/audit_rpm_d_cross_regime.py \\
        --rpm-d-dirs runs/r/EXP-RPM-D0/<ts>/af2d,runs/r/EXP-RPM-D1/<ts>/af2d,... \\
        --out rpm_d_cross_regime_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean_stderr(values):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if n == 1:
        return mean, 0.0
    return mean, float(arr.std(ddof=1) / np.sqrt(n))


def _per_regime_metrics(agg_path: Path) -> dict:
    """Extract the key per-regime metrics for cross-regime comparison.

    Accepts either the aggregate.json file or the directory containing
    it (looks for aggregate.json inside the directory).
    """
    if agg_path.is_dir():
        agg_path = agg_path / "aggregate.json"
    agg = _load_json(agg_path)
    out = {"source": str(agg_path), "metrics": {}}

    # Per-seed pre-train ppl from seed-XXX/pre_train_eval.json (the
    # verified damaged-base state BEFORE adapter training). Use this
    # as the pre-train baseline since the driver does NOT run a
    # no_correction arm.
    runs_dir = agg_path.parent
    pre_ppls = []
    for seed_dir in sorted(runs_dir.glob("seed-*")):
        p = seed_dir / "pre_train_eval.json"
        if p.exists():
            pre_ppls.append(_load_json(p)["tasks"]["wikitext"]["value"])
    if pre_ppls:
        out["pre_train_ppl"] = sum(pre_ppls) / len(pre_ppls)
    else:
        out["pre_train_ppl"] = float("nan")
    trained_t2 = agg.get("trained_arms", {}).get("t2_ternary")
    random_t2 = agg.get("untrained_controls", {}).get("random_t2_ternary")
    if not trained_t2 or not random_t2:
        return out

    out["trained_t2_ppl"] = trained_t2["tasks"]["wikitext"]["mean"]
    out["random_t2_ppl"] = random_t2["tasks"]["wikitext"]["mean"]

    for task in ("wikitext", "arc_easy", "lambada_openai"):
        t_vals = trained_t2["tasks"][task]["values"]
        r_vals = random_t2["tasks"][task]["values"]
        if not t_vals or not r_vals:
            continue
        t_mean, t_se = _mean_stderr(t_vals)
        r_mean, r_se = _mean_stderr(r_vals)
        diff_se = (t_se ** 2 + r_se ** 2) ** 0.5
        if diff_se == 0:
            diff_se = 1e-9
        lower_is_better = (task == "wikitext")
        if lower_is_better:
            z = (r_mean - t_mean) / diff_se
        else:
            z = (t_mean - r_mean) / diff_se
        out["metrics"][task] = float(z)

    if out["pre_train_ppl"] > 0:
        out["recovery_ratio"] = (
            out["trained_t2_ppl"] / out["pre_train_ppl"]
        )
    return out


def _regime_damage_order(paths) -> list:
    """Sort regimes D0..D5 by their preregistered damage threshold.

    D0 (no damage) < D1 (threshold=0.0) < ... < D5 (threshold=0.7).
    The paths are expected to be in EXP-RPM-D<n> order; if not, we
    extract the digit from the path and sort.
    """
    def get_d_key(p: Path) -> int:
        for part in str(p).replace("\\", "/").split("/"):
            if part.startswith("EXP-RPM-D"):
                tail = part[len("EXP-RPM-D"):]
                if tail.isdigit():
                    return int(tail)
        return 999
    return sorted(paths, key=get_d_key)


def audit_cross_regime(rpm_d_dirs: list[Path]) -> dict:
    """Compute the cross-regime RPM-002 audit."""
    regime_metrics = []
    for p in _regime_damage_order(rpm_d_dirs):
        m = _per_regime_metrics(p)
        m["regime_path"] = str(p)
        regime_metrics.append(m)

    metric_checks = {}
    for task in ("wikitext", "arc_easy", "lambada_openai"):
        z_scores = [
            r["metrics"].get(task, float("nan"))
            for r in regime_metrics
        ]
        valid = [(i, z) for i, z in enumerate(z_scores) if not np.isnan(z)]
        if len(valid) < 3:
            metric_checks[task] = {
                "z_scores": z_scores,
                "non_decreasing_run": 0,
                "in_band": False,
                "note": "fewer than 3 regimes with data",
            }
            continue
        max_run = 1
        cur_run = 1
        for j in range(1, len(valid)):
            if valid[j][1] >= valid[j - 1][1]:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1
        metric_checks[task] = {
            "z_scores": z_scores,
            "non_decreasing_run": max_run,
            "in_band": max_run >= 3,
        }

    any_pass = any(c["in_band"] for c in metric_checks.values())
    n_regimes = len(regime_metrics)

    audit_record = {
        "auditor": "examples/audit_rpm_d_cross_regime.py",
        "n_regimes": n_regimes,
        "regime_metrics": regime_metrics,
        "metric_checks": metric_checks,
        "verdict": "PASS" if any_pass else "FAIL",
        "note": (
            "RPM-002: PASS if trained-vs-random z-score is non-decreasing "
            "across >=3 consecutive regimes on any capability metric. "
            "FAIL if no metric shows this trend."
        ),
    }
    return audit_record


def main() -> int:
    p = argparse.ArgumentParser(description="RPM-002 cross-regime auditor")
    p.add_argument(
        "--rpm-d-dirs", required=True, type=str,
        help="Comma-separated list of EXP-RPM-D<n>/<ts>/af2d paths",
    )
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    dirs = [Path(d) for d in args.rpm_d_dirs.split(",")]
    audit = audit_cross_regime(dirs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True))

    print(f"RPM-002 cross-regime audit ({audit['n_regimes']} regimes): "
          f"verdict={audit['verdict']}")
    for task, c in audit["metric_checks"].items():
        print(f"  {task}: non-decreasing run = {c['non_decreasing_run']} "
              f"({'PASS' if c['in_band'] else 'no'} bar)")
    return 0 if audit["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())