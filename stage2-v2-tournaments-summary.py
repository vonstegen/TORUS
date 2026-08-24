"""Stage 2 v2 tournament post-hoc summary.

The driver's `aggregate()` filters by `matched_bytes_passed=True`,
which removes all arms at non-down_proj sites (v_proj, q_proj) because
the byte target is registered for the down_proj (4.19 MB) and
attention projections don't match. This post-hoc aggregator includes
ALL arms regardless of matched_bytes, since the Pareto analysis
across arm types is the point of the v2 protocol (even when the
arm's deployed byte count is different).

Reads every EXP-RPM-{SITE}-GAUSS/{ts}/seed-N/{arm}/eval.summary.json,
computes per-arm mean +/- stderr across the 3 seeds, and writes a
combined site tournament summary at:
  research/residual-pareto/experiments/STAGE2-V2-TOURNAMENTS-SUMMARY.json
"""

import json
import statistics
from pathlib import Path

import yaml


BASE = Path("/home/andrew-jochl/TORUS")
RUNS_DIR = BASE / "runs" / "r"

SITES = {
    "EXP-RPM-L0-V-GAUSS": "model.layers.0.self_attn.v_proj",
    "EXP-RPM-L15-GAUSS":  "model.layers.15.mlp.down_proj",
}
TASKS = ["wikitext", "arc_easy", "lambada_openai"]


def per_arm_summary(exp_id: str, target_module: str) -> dict:
    site_dir = RUNS_DIR / exp_id
    if not site_dir.exists():
        return {"exp_id": exp_id, "error": "no runs directory"}
    # Pick the timestamp dir with the most populated seed-XXX dirs.
    ts_dirs = [p for p in site_dir.iterdir() if p.is_dir()]
    best = None
    best_count = -1
    for ts in ts_dirs:
        n = sum(1 for p in ts.rglob("eval.summary.json"))
        if n > best_count:
            best_count = n
            best = ts
    if best is None:
        return {"exp_id": exp_id, "error": "no timestamp directory"}
    ts = best
    seed_dirs = sorted([p for p in ts.iterdir()
                        if p.is_dir() and p.name.startswith("seed-")])
    per_arm = {}
    for sd in seed_dirs:
        for es_path in sd.rglob("eval.summary.json"):
            try:
                d = json.loads(es_path.read_text())
            except Exception:
                continue
            arm = es_path.parent.name
            per_arm.setdefault(arm, []).append(d)
    out = {"exp_id": exp_id, "target_module": target_module,
            "timestamp": ts.name, "n_seeds": len(seed_dirs),
            "per_arm": {}}
    for arm, summaries in per_arm.items():
        entry = {"n_cells": len(summaries)}
        for t in TASKS:
            vals = [s.get("tasks", {}).get(t, {}).get("value")
                     for s in summaries]
            vals = [v for v in vals if v is not None]
            if vals:
                entry.setdefault("tasks", {})[t] = {
                    "n": len(vals),
                    "mean": statistics.fmean(vals),
                    "stderr": (statistics.stdev(vals) / len(vals) ** 0.5
                                if len(vals) > 1 else 0.0),
                    "values": vals,
                }
        out["per_arm"][arm] = entry
    return out


def main():
    combined = []
    for exp_id, target_module in SITES.items():
        s = per_arm_summary(exp_id, target_module)
        combined.append(s)
    out = BASE / "research" / "residual-pareto" / "experiments" / \
        "STAGE2-V2-TOURNAMENTS-SUMMARY.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(combined, indent=2))
    for s in combined:
        print(f"\n=== {s['exp_id']} ({s['target_module']}) ===")
        print(f"  timestamp={s.get('timestamp', '?')} n_seeds={s.get('n_seeds', '?')}")
        for arm, entry in s.get("per_arm", {}).items():
            t = entry.get("tasks", {})
            ppl = t.get("wikitext", {}).get("mean", "—")
            arc = t.get("arc_easy", {}).get("mean", "—")
            lam = t.get("lambada_openai", {}).get("mean", "—")
            print(f"  {arm:20s} ppl={ppl:.4f}  arc={arc:.4f}  "
                  f"lambada={lam:.4f}  (n={entry.get('n_cells', '?')})"
                  if isinstance(ppl, float) else
                  f"  {arm:20s} — (n={entry.get('n_cells', '?')})")
    print(f"\ncombined: {out}")


if __name__ == "__main__":
    main()