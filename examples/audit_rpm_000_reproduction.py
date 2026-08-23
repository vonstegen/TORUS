"""EXP-RPM-000 reference-lock + reproduction auditor.

The RPM program's G-RPM-0 gate (EXP-RPM-000) requires a fresh
reproduction of AF2-D's two headline measurements under AF8 governance:

  (1) damaged starting state: pre-train wikitext ppl in [400, 460]
      AND arc_easy in [0.45, 0.58];
  (2) trained t2_ternary correction: post-train wikitext ppl in
      [18.66, 23.26] (the AF2-D reference 20.96 +/- 1.5 sigma at
      n=3) AND arc_easy in [0.594, 0.606] AND lambada_openai in
      [0.541, 0.549].

Cost-vector B must be within +/-1% of 4,194,404 B for the
t2_ternary arm.

This script is the structural twin of ``audit_af2_reproduction.py``
but it audits the AGGREGATE.json against the EXP-RPM-000 PASS
bands rather than auditing the token cache. The two audits are
independent (cache traceability is verified by the AF2-D audit;
this verifies the AF2-D reproduction itself).

Usage on legion::

    python examples/audit_rpm_000_reproduction.py \\
        --aggregate runs/r/RPM-000/<ts>/aggregate.json \\
        --af2d-reference <path to AF2-D aggregate_corrected.json> \\
        --manifest research/residual-pareto/experiments/RPM-000/manifest.yaml \\
        --out runs/r/RPM-000/<ts>/rpm000_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

# Reference values from EXP-AF-002-D (legion, git 330e8b3, run
# experiments/AF2-D/runs/20260823T092339Z/af2d/, n=3 seeds).
# These are the bands preregistered in EXP-RPM-000/manifest.yaml.
AF2D_REFERENCE = {
    "pre_train_ppl": {"mean": 425.76, "band": [400.0, 460.0]},
    "pre_train_arc_easy": {"mean": 0.4891, "band": [0.45, 0.58]},
    "trained_t2_ppl": {"mean": 20.96, "band": [18.66, 23.26]},
    "trained_t2_arc_easy": {"mean": 0.600, "band": [0.594, 0.606]},
    "trained_t2_lambada": {"mean": 0.545, "band": [0.541, 0.549]},
    "deployed_bytes_t2": {"mean": 4194404, "tolerance_pct": 1.0},
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _band_check(value: float, band: list, name: str) -> dict:
    in_band = band[0] <= value <= band[1]
    return {
        "metric": name,
        "value": value,
        "band": band,
        "in_band": in_band,
    }


def _deployed_bytes_check(value: int, ref: int, tolerance_pct: float, name: str) -> dict:
    tol = ref * (tolerance_pct / 100.0)
    in_band = abs(value - ref) <= tol
    return {
        "metric": name,
        "value": value,
        "reference": ref,
        "tolerance_pct": tolerance_pct,
        "in_band": in_band,
    }


def _compute_rpm000_verdict(audit_checks: list) -> str:
    """Verdict: REPRODUCED iff all in_band checks pass."""
    if all(c["in_band"] for c in audit_checks):
        return "REPRODUCED"
    return "DRIFTED"


def audit_rpm_000(*, aggregate_path: Path, af2d_reference_path: Path,
                  manifest_path: Path) -> dict:
    """Verify the EXP-RPM-000 reproduction aggregate against bands.

    Reads the per-seed aggregate produced by af2_storage_tournament.py
    (t2_ternary arm only, --damage-ptq --pre-train-eval) and the
    AF2-D reference aggregate; checks each metric against the
    preregistered band.
    """
    aggregate = _load_json(aggregate_path)
    manifest = _load_yaml(manifest_path)
    af2d = _load_json(af2d_reference_path)

    # Compute per-seed means across n=3 seeds for the checks.
    # aggregate.json layout mirrors AF2-D's: trained_arms[arm] = {seeds: [...]}
    seeds = aggregate.get("trained_arms", {}).get("t2_ternary", {}).get("seeds", [])
    if not seeds:
        raise ValueError("aggregate.json missing trained_arms.t2_ternary.seeds")

    pre_ppl_vals = [s["pre_train_eval"]["wikitext_ppl"] for s in seeds
                    if "pre_train_eval" in s]
    pre_arc_vals = [s["pre_train_eval"]["arc_easy"] for s in seeds
                    if "pre_train_eval" in s]
    post_ppl_vals = [s["eval"]["wikitext_ppl"] for s in seeds if "eval" in s]
    post_arc_vals = [s["eval"]["arc_easy"] for s in seeds if "eval" in s]
    post_lam_vals = [s["eval"]["lambada_openai"] for s in seeds if "eval" in s]
    dep_bytes_vals = [s["deployed_bytes"]["serialized_total"] for s in seeds
                      if "deployed_bytes" in s]

    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    pre_ppl_mean = mean(pre_ppl_vals)
    pre_arc_mean = mean(pre_arc_vals)
    post_ppl_mean = mean(post_ppl_vals)
    post_arc_mean = mean(post_arc_vals)
    post_lam_mean = mean(post_lam_vals)
    dep_bytes_mean = int(mean(dep_bytes_vals))

    checks = [
        _band_check(pre_ppl_mean, AF2D_REFERENCE["pre_train_ppl"]["band"],
                    "pre_train_ppl"),
        _band_check(pre_arc_mean, AF2D_REFERENCE["pre_train_arc_easy"]["band"],
                    "pre_train_arc_easy"),
        _band_check(post_ppl_mean, AF2D_REFERENCE["trained_t2_ppl"]["band"],
                    "trained_t2_ppl"),
        _band_check(post_arc_mean, AF2D_REFERENCE["trained_t2_arc_easy"]["band"],
                    "trained_t2_arc_easy"),
        _band_check(post_lam_mean, AF2D_REFERENCE["trained_t2_lambada"]["band"],
                    "trained_t2_lambada"),
        _deployed_bytes_check(dep_bytes_mean,
                              AF2D_REFERENCE["deployed_bytes_t2"]["mean"],
                              AF2D_REFERENCE["deployed_bytes_t2"]["tolerance_pct"],
                              "deployed_bytes_t2"),
    ]
    verdict = _compute_rpm000_verdict(checks)

    audit_record = {
        "auditor": "examples/audit_rpm_000_reproduction.py",
        "manifest_id": manifest["id"],
        "manifest_revision": manifest.get("revision", ""),
        "frozen_af2d_sha": af2d.get("git_sha", ""),
        "frozen_af2d_run": af2d.get("run_namespace", ""),
        "n_seeds": len(seeds),
        "checks": checks,
        "verdict": verdict,
    }
    return audit_record


def main() -> int:
    p = argparse.ArgumentParser(description="EXP-RPM-000 reproduction auditor")
    p.add_argument("--aggregate", required=True, type=Path,
                   help="Path to runs/r/RPM-000/<ts>/aggregate.json")
    p.add_argument("--af2d-reference", required=True, type=Path,
                   help="Path to AF2-D aggregate_corrected.json (the reference)")
    p.add_argument("--manifest", required=True, type=Path,
                   help="Path to EXP-RPM-000 manifest.yaml")
    p.add_argument("--out", required=True, type=Path,
                   help="Path to write rpm000_audit.json")
    args = p.parse_args()

    audit = audit_rpm_000(aggregate_path=args.aggregate,
                          af2d_reference_path=args.af2d_reference,
                          manifest_path=args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True))

    # Print a one-line summary + the verdict.
    n_pass = sum(1 for c in audit["checks"] if c["in_band"])
    n_total = len(audit["checks"])
    print(f"EXP-RPM-000 audit: {n_pass}/{n_total} checks in band; verdict={audit['verdict']}")
    return 0 if audit["verdict"] == "REPRODUCED" else 1


if __name__ == "__main__":
    sys.exit(main())
