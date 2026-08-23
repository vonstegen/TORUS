"""EXP-RPM-000 reference-lock + reproduction auditor.

The RPM program's G-RPM-0 gate (EXP-RPM-000) requires a fresh
reproduction of AF2-D's two headline measurements under AF8 governance:

  (1) damaged starting state: pre-train wikitext ppl in [400, 460]
      AND arc_easy in [0.45, 0.58];
  (2) trained t2_ternary correction: post-train wikitext ppl in
      [17.91, 24.01] (the AF2-D reference 20.96 +/- 2 sigma at
      n=3) AND arc_easy in [0.592, 0.608] AND lambada_openai in
      [0.539, 0.551].

Cost-vector B must be within +/-1% of 4,199,318 B (the AF2-D
target_deployed_bytes is 4,194,404; the actual value is
4,199,318 due to scale-metadata overhead — within the +/-1%
tolerance.

This script is the structural twin of ``audit_af2_reproduction.py``
but it audits the AGGREGATE.json against the EXP-RPM-000 PASS
bands rather than auditing the token cache. The two audits are
independent (cache traceability is verified by the AF2-D audit;
this verifies the AF2-D reproduction itself).

Source layout expected::

    runs/r/RPM-000/<ts>/
        aggregate.json
        seed-001/pre_train_eval.json
        seed-002/pre_train_eval.json
        seed-003/pre_train_eval.json

The aggregate.json layout mirrors EXP-AF-002-D's
``aggregate_corrected.json``: ``trained_arms.<arm>.tasks.<task>.{mean,
stderr, values}`` and ``trained_arms.<arm>.matched_bytes``.

Usage on legion::

    python examples/audit_rpm_000_reproduction.py \\
        --aggregate runs/r/RPM-000/<ts>/aggregate.json \\
        --af2d-reference <path to AF2-D aggregate_corrected.json> \\
        --manifest research/residual-pareto/experiments/RPM-000/manifest.yaml \\
        --runs-dir runs/r/RPM-000/<ts>/ \\
        --out runs/r/RPM-000/<ts>/rpm000_audit.json
"""
from __future__ import annotations

import argparse
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
    # trained_t2 bands are AF2-D reference +/- 2 standard deviations
    # (the standard rule per OPERATING-PLAN §11 v2.3 and AF2-D
    # manifest: "untrained control by >2 stderr"; we apply the same
    # 2σ rule to the reproduction bands so a within-2σ drift is
    # considered reproduced, not DRIFTED).
    "trained_t2_ppl": {"mean": 20.96, "band": [17.91, 24.01]},
    "trained_t2_arc_easy": {"mean": 0.600, "band": [0.592, 0.608]},
    "trained_t2_lambada": {"mean": 0.545, "band": [0.539, 0.551]},
    "deployed_bytes_t2": {"mean": 4199318, "tolerance_pct": 1.0},
}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _band_check(value: float, band: list, name: str) -> dict:
    in_band = band[0] <= value <= band[1]
    return {"metric": name, "value": value, "band": band, "in_band": in_band}


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
    return "REPRODUCED" if all(c["in_band"] for c in audit_checks) else "DRIFTED"


def _avg_pre_train(runs_dir: Path) -> tuple:
    """Average the per-seed pre_train_eval.json files.

    Returns (mean_pre_ppl, mean_pre_arc). Per-seed pre_train_eval is
    stored at ``seed-XXX/pre_train_eval.json`` (one file per seed,
    not per arm — the damaged base is the same for all arms within
    a seed).
    """
    pre_ppls = []
    pre_arcs = []
    for seed_dir in sorted(runs_dir.glob("seed-*")):
        p = seed_dir / "pre_train_eval.json"
        if not p.exists():
            raise FileNotFoundError(f"missing pre_train_eval.json: {p}")
        d = _load_json(p)
        pre_ppls.append(d["tasks"]["wikitext"]["value"])
        pre_arcs.append(d["tasks"]["arc_easy"]["value"])
    if not pre_ppls:
        raise ValueError(f"no seed-XXX/pre_train_eval.json under {runs_dir}")
    return (
        sum(pre_ppls) / len(pre_ppls),
        sum(pre_arcs) / len(pre_arcs),
    )


def audit_rpm_000(*, aggregate_path: Path, af2d_reference_path: Path,
                  manifest_path: Path, runs_dir: Path) -> dict:
    """Verify the EXP-RPM-000 reproduction aggregate against bands."""
    aggregate = _load_json(aggregate_path)
    manifest = _load_yaml(manifest_path)
    af2d = _load_json(af2d_reference_path)

    # Post-train: from aggregate.json trained_arms.t2_ternary.tasks.<task>.mean
    # and matched_bytes (list of int).
    t2 = aggregate.get("trained_arms", {}).get("t2_ternary")
    if t2 is None:
        raise ValueError("aggregate.json missing trained_arms.t2_ternary")
    tasks = t2.get("tasks")
    if not tasks:
        raise ValueError("aggregate.json trained_arms.t2_ternary missing tasks")
    matched_bytes = t2.get("matched_bytes")
    if not matched_bytes:
        raise ValueError("aggregate.json trained_arms.t2_ternary missing matched_bytes")
    if "wikitext" not in tasks or "arc_easy" not in tasks or "lambada_openai" not in tasks:
        raise ValueError("aggregate.json missing required tasks (wikitext/arc_easy/lambada_openai)")

    post_ppl_mean = tasks["wikitext"]["mean"]
    post_arc_mean = tasks["arc_easy"]["mean"]
    post_lam_mean = tasks["lambada_openai"]["mean"]
    dep_bytes_mean = int(sum(matched_bytes) / len(matched_bytes))

    # Pre-train: average the per-seed pre_train_eval.json files.
    pre_ppl_mean, pre_arc_mean = _avg_pre_train(runs_dir)

    checks = [
        _band_check(pre_ppl_mean, AF2D_REFERENCE["pre_train_ppl"]["band"], "pre_train_ppl"),
        _band_check(pre_arc_mean, AF2D_REFERENCE["pre_train_arc_easy"]["band"], "pre_train_arc_easy"),
        _band_check(post_ppl_mean, AF2D_REFERENCE["trained_t2_ppl"]["band"], "trained_t2_ppl"),
        _band_check(post_arc_mean, AF2D_REFERENCE["trained_t2_arc_easy"]["band"], "trained_t2_arc_easy"),
        _band_check(post_lam_mean, AF2D_REFERENCE["trained_t2_lambada"]["band"], "trained_t2_lambada"),
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
        "frozen_af2d_run": af2d.get("run_dir", ""),
        "n_seeds": len(matched_bytes),
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
    p.add_argument("--runs-dir", required=True, type=Path,
                   help="Path to runs/r/RPM-000/<ts>/ (parent of seed-XXX/)")
    p.add_argument("--out", required=True, type=Path,
                   help="Path to write rpm000_audit.json")
    args = p.parse_args()

    audit = audit_rpm_000(aggregate_path=args.aggregate,
                          af2d_reference_path=args.af2d_reference,
                          manifest_path=args.manifest,
                          runs_dir=args.runs_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2, sort_keys=True))

    n_pass = sum(1 for c in audit["checks"] if c["in_band"])
    n_total = len(audit["checks"])
    print(f"EXP-RPM-000 audit: {n_pass}/{n_total} checks in band; verdict={audit['verdict']}")
    return 0 if audit["verdict"] == "REPRODUCED" else 1


if __name__ == "__main__":
    sys.exit(main())