"""Stage 2 v3 tournament post-hoc summary (L15 down_proj @ sigma=0.5).

Reads every EXP-RPM-L15-GAUSS-V3/{ts}/seed-N/{arm}/eval.summary.json,
computes per-arm mean +/- stderr across the 3 seeds, and writes a
combined tournament summary at:
  research/residual-pareto/experiments/STAGE2-V3-TOURNAMENTS-SUMMARY.json

Includes all arms (trained + random).
"""

import json
import statistics
from pathlib import Path


BASE = Path("/home/andrew-jochl/TORUS")
RUNS_DIR = BASE / "runs" / "r"
SITES = {
    "EXP-RPM-L15-GAUSS-V3": "model.layers.15.mlp.down_proj",
    "EXP-RPM-L15-GAUSS-V4": "model.layers.15.mlp.down_proj",
    "EXP-RPM-L15-TWN-V5": "model.layers.15.mlp.down_proj",
}
TASKS = ["wikitext", "arc_easy", "lambada_openai"]


def per_arm_summary(exp_id: str, target_module: str) -> dict:
    site_dir = RUNS_DIR / exp_id
    if not site_dir.exists():
        return {"exp_id": exp_id, "error": "no runs directory"}
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
        "STAGE2-V3-TOURNAMENTS-SUMMARY.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(combined, indent=2))
    for s in combined:
        print(f"\n=== {s['exp_id']} ({s['target_module']}) ===")
        print(f"  timestamp={s.get('timestamp', '?')} n_seeds={s.get('n_seeds', '?')}")
        for arm, entry in s.get("per_arm", {}).items():
            t = entry.get("tasks", {})
            ppl = t.get("wikitext", {}).get("mean", float("nan"))
            arc = t.get("arc_easy", {}).get("mean", float("nan"))
            lam = t.get("lambada_openai", {}).get("mean", float("nan"))
            print(f"  {arm:20s} ppl={ppl:.4f}  arc={arc:.4f}  "
                  f"lambada={lam:.4f}  (n={entry.get('n_cells', '?')})")
    site_keys = sorted(s.get("exp_id", "") for s in combined)
    out_name = "_".join(site_keys) + "_SUMMARY.json" if len(site_keys) > 1 else f"{site_keys[0]}_SUMMARY.json"
    out = BASE / "research" / "residual-pareto" / "experiments" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
