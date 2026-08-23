"""Per-regime EXP-RPM-D<n> auditor.

Verifies the Stage 1 (EXP-RPM-D0..D5) per-regime aggregate against
the preregistered bands:

  1. Pre-train ppl lands in the regime's nominal_band
     (recorded as an observed covariate; deviation noted but does
     not invalidate per RPM proposal section 5).
  2. All trained arms within +/-1% matched-bytes tolerance.
  3. Trained t2_ternary recovers the damaged base substantially
     (post-train ppl <= 1.5x starting ppl).
  4. PASS+: trained t2_ternary beats random_t2_ternary by >2
     stderr-of-difference on >=1 capability metric (RPM-006 axis).
     Note: per-seed stderr for D1-D5 may be larger than AF2-D's
     since each regime is novel; >2 stderr on the cross-arm
     (B-A) difference is the standard rule.

The cross-regime audit (RPM-002 axis) is a separate script
(audit_rpm_d_cross_regime.py) that compares the per-regime
aggregate.json files against each other.

Usage on legion::

    python examples/audit_rpm_d_reproduction.py \\
        --aggregate runs/r/EXP-RPM-D<n>/<ts>/af2d/aggregate.json \\
        --manifest research/residual-pareto/experiments/EXP-RPM-D<n>/manifest.yaml \\
        --out runs/r/EXP-RPM-D<n>/<ts>/af2d/rpm_d_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean_stderr(values: list[float]) -> tuple:
    """Return (mean, stderr-of-the-mean) for a list of per-seed values."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    # stderr of the mean = sample std / sqrt(n).
    if n == 1:
        return mean, 0.0
    stderr = float(arr.std(ddof=1) / np.sqrt(n))
    return mean, stderr


def _per_arm_summary(aggregate: dict) -> dict:
    """Restructure AF2-D-shaped aggregate.json into a per-arm map.

    aggregate structure (verified against EXP-AF-002-D):
      trained_arms.<arm>.{
        n: int, matched_bytes: [int, ...], matched_bytes_target: int,
        tasks.<task>.{mean, stderr, values: [float, ...]}
      }
      untrained_controls.<arm>.{...}
    """
    out = {"trained": {}, "untrained": {}}
    for arm_name, arm in aggregate.get("trained_arms", {}).items():
        out["trained"][arm_name] = {
            "matched_bytes": arm.get("matched_bytes", []),
            "matched_bytes_target": arm.get("matched_bytes_target"),
            "tasks": {
                t: arm["tasks"][t]
                for t in ("wikitext", "arc_easy", "lambada_openai")
                if t in arm.get("tasks", {})
            },
        }
    for arm_name, arm in aggregate.get("untrained_controls", {}).items():
        out["untrained"][arm_name] = {
            "matched_bytes": arm.get("matched_bytes", []),
            "matched_bytes_target": arm.get("matched_bytes_target"),
            "tasks": {
                t: arm["tasks"][t]
                for t in ("wikitext", "arc_easy", "lambada_openai")
                if t in arm.get("tasks", {})
            },
        }
    return out


def _avg_pre_train(runs_dir: Path) -> tuple:
    """Average per-seed pre_train_eval.json across the regime's seeds.

    Returns (mean_pre_ppl, mean_pre_arc).
    """
    pre_ppls, pre_arcs = [], []
    for seed_dir in sorted(runs_dir.glob("seed-*")):
        p = seed_dir / "pre_train_eval.json"
        if not p.exists():
            continue
        d = _load_json(p)
        pre_ppls.append(d["tasks"]["wikitext"]["value"])
        pre_arcs.append(d["tasks"]["arc_easy"]["value"])
    if not pre_ppls:
        raise ValueError(f"no seed-XXX/pre_train_eval.json under {runs_dir}")
    return sum(pre_ppls) / len(pre_ppls), sum(pre_arcs) / len(pre_arcs)


def _matched_bytes_check(arm_data: dict, ref_target: int, tol_pct: float) -> dict:
    """Check that all per-seed matched_bytes are within +/-tol_pct of ref_target."""
    tol = ref_target * (tol_pct / 100.0)
    bytes_list = arm_data.get("matched_bytes", [])
    if not bytes_list:
        return {
            "metric": "matched_bytes",
            "arm": "(unknown)",
            "in_band": False,
            "note": "no matched_bytes recorded",
        }
    violations = [
        b for b in bytes_list if abs(b - ref_target) > tol
    ]
    return {
        "metric": "matched_bytes",
        "arm_match": bytes_list,
        "reference": ref_target,
        "tolerance_pct": tol_pct,
        "in_band": len(violations) == 0,
        "violations": violations,
    }


def _recovery_check(
    pre_train_ppl: float,
    trained_t2: dict,
) -> dict:
    """Check trained t2_ternary recovers the damaged base substantially.

    PASS bar: post-train ppl <= 1.5x the starting ppl.
    """
    post_ppl_mean, post_ppl_stderr = _mean_stderr(
        trained_t2["tasks"]["wikitext"]["values"]
    )
    ratio = (
        post_ppl_mean / pre_train_ppl
        if pre_train_ppl > 0
        else float("nan")
    )
    return {
        "metric": "recovery",
        "pre_train_ppl": pre_train_ppl,
        "post_train_ppl_mean": post_ppl_mean,
        "post_train_ppl_stderr": post_ppl_stderr,
        "ratio": ratio,
        "in_band": ratio <= 1.5,
    }


def _representation_signal_check(
    trained_t2: dict,
    random_t2: dict,
) -> dict:
    """RPM-006 axis: trained t2_ternary vs random_t2_ternary separation.

    PASS: trained beats random by >2 stderr-of-difference on >=1 of
    (wikitext ppl, arc_easy, lambada_openai).
    """
    out = {"metric": "representation_signal", "metrics": {}}
    any_pass = False
    for task in ("wikitext", "arc_easy", "lambada_openai"):
        t_vals = trained_t2["tasks"][task]["values"]
        r_vals = random_t2["tasks"][task]["values"]
        if not t_vals or not r_vals:
            continue
        t_mean, t_se = _mean_stderr(t_vals)
        r_mean, r_se = _mean_stderr(r_vals)
        # pooled stderr-of-difference (assumes equal n; if n differs,
        # use conservative: sqrt(t_se^2 + r_se^2)).
        diff_se = (t_se ** 2 + r_se ** 2) ** 0.5
        if diff_se == 0:
            diff_se = 1e-9
        # For ppl: trained should beat random (lower is better).
        # For acc: trained should beat random (higher is better).
        lower_is_better = (task == "wikitext")
        if lower_is_better:
            z = (r_mean - t_mean) / diff_se
        else:
            z = (t_mean - r_mean) / diff_se
        passed = z > 2.0
        if passed:
            any_pass = True
        out["metrics"][task] = {
            "trained_mean": t_mean,
            "trained_stderr": t_se,
            "random_mean": r_mean,
            "random_stderr": r_se,
            "z_score": float(z),
            "passes_2sigma": passed,
        }
    out["in_band"] = any_pass
    return out


def audit_rpm_d(
    *, aggregate_path: Path, manifest_path: Path, runs_dir: Path
) -> dict:
    """Run all per-regime checks against the EXP-RPM-D<n> aggregate."""
    aggregate = _load_json(aggregate_path)
    manifest = _load_yaml(manifest_path)

    arms = _per_arm_summary(aggregate)
    nominal_band = manifest["damage_ptq"].get(
        "pre_train_eval_check", ""
    )  # not strictly structured; we re-derive from threshold below

    # Pre-train ppl from per-seed pre_train_eval.json.
    pre_ppl_mean, pre_arc_mean = _avg_pre_train(runs_dir)

    # Nominal band: re-read from manifest's pre_train_eval_check
    # (a free-text field); for the audit we record the band as
    # whatever the threshold-targeted band was.
    threshold = manifest["damage_ptq"]["threshold"]
    if threshold is None:
        nominal_band = [13.0, 15.0]   # D0 FP16 baseline
    else:
        # Use the proposal's nominal bands as a guide; not enforced.
        # The actual pre_train_ppl_mean is recorded; the band is the
        # nominal TARGET. The verdict notes deviation but does not
        # invalidate.
        if threshold == 0.0:
            nominal_band = [13.0, 30.0]
        elif threshold == 0.3:
            nominal_band = [30.0, 80.0]
        elif threshold == 0.5:
            nominal_band = [80.0, 200.0]
        elif threshold == 0.6:
            nominal_band = [200.0, 350.0]
        elif threshold == 0.7:
            nominal_band = [300.0, 500.0]
        else:
            nominal_band = [13.0, 1000.0]

    pre_train_check = {
        "metric": "pre_train_ppl",
        "value": pre_ppl_mean,
        "nominal_band": nominal_band,
        "in_band": nominal_band[0] <= pre_ppl_mean <= nominal_band[1],
        "note": (
            "recorded as observed covariate; deviation noted but does "
            "not invalidate per RPM proposal section 5"
        ),
    }

    # Matched-bytes tolerance for each trained arm.
    target = manifest["target_deployed_bytes"]
    tol = manifest["matched_bytes_tolerance_pct"]
    mb_checks = {
        arm: _matched_bytes_check(arms["trained"][arm], target, tol)
        for arm in arms["trained"]
    }
    all_mb_pass = all(c["in_band"] for c in mb_checks.values())

    # Trained t2_ternary recovery (PASS bar).
    trained_t2 = arms["trained"].get("t2_ternary")
    random_t2 = arms["untrained"].get("random_t2_ternary")
    checks = [pre_train_check]
    if trained_t2:
        checks.append(_recovery_check(pre_ppl_mean, trained_t2))
    for arm_name, c in mb_checks.items():
        c["arm"] = arm_name
        checks.append(c)

    # RPM-006 representation-signal axis (PASS+ bar).
    rep_signal = None
    if trained_t2 and random_t2:
        rep_signal = _representation_signal_check(trained_t2, random_t2)
        checks.append(rep_signal)

    # Verdict logic: PASS if recovery + matched-bytes hold.
    # PASS_PLUS if RPM-006 axis also fires.
    recovery_ok = (
        trained_t2 is not None
        and checks[1]["in_band"]   # recovery check is checks[1]
    )
    rep_signal_ok = rep_signal is not None and rep_signal["in_band"]
    if recovery_ok and all_mb_pass:
        verdict = "PASS_PLUS" if rep_signal_ok else "PASS"
    else:
        verdict = "FAIL"

    audit_record = {
        "auditor": "examples/audit_rpm_d_reproduction.py",
        "manifest_id": manifest["id"],
        "manifest_threshold": threshold,
        "n_seeds": manifest.get("training", {}).get("n_steps"),  # placeholder
        "checks": checks,
        "rep_signal": rep_signal,
        "verdict": verdict,
    }
    return audit_record


def main() -> int:
    p = argparse.ArgumentParser(description="EXP-RPM-D<n> per-regime auditor")
    p.add_argument("--aggregate", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--runs-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    audit = audit_rpm_d(
        aggregate_path=args.aggregate,
        manifest_path=args.manifest,
        runs_dir=args.runs_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True))

    n_pass = sum(1 for c in audit["checks"] if c.get("in_band"))
    n_total = len(audit["checks"])
    print(
        f"{audit['manifest_id']} audit: {n_pass}/{n_total} checks in band; "
        f"verdict={audit['verdict']}"
    )
    return 0 if audit["verdict"] in ("PASS", "PASS_PLUS") else 1


if __name__ == "__main__":
    sys.exit(main())