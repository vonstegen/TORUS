"""AF3 robustness auditor — EXP-AF-003 (initialization robustness matrix).

Scans the EXP-AF-003 run tree (sigma-*/seed-*/t2_ternary cells),
computes per-level statistics, and applies the FROZEN classification
rules from experiments/AF3/manifest.yaml:

  Cell success:    post-train wikitext ppl <= 100 (AF2-D's frozen
                   recovery bar, 4.3x from the ~430 damage point).
  Level success:   all 3 seeds succeed.
  ROBUST:          all 5 non-zero sigma levels succeed AND
                   spread ratio (worst level mean ppl / best) <= 2.
  MODERATELY SENSITIVE: >= 3 of 5 non-zero levels succeed, OR (5/5
                   succeed with spread ratio <= 5).
  FRAGILE:         otherwise (< 3 of 5 levels succeed, OR spread > 5).
  sigma=0:         structural control, reported separately and
                   excluded from the level counts. Expected to fail
                   (near-dead start); success would falsify the
                   dead-zone model and is recorded as a finding.

Also emits the capability cross-check (arc_easy / lambada per level):
if a level "succeeds" on ppl while its acc metrics sit at the
damaged-base level, the audit flags PROXY/CAPABILITY divergence.

Usage:

    python examples/audit_af3_robustness.py \
        --run-dir runs/a/EXP-AF-003/<ts> --out <run-dir>/audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SIGMA_LEVELS = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
NONZERO_LEVELS = [s for s in SIGMA_LEVELS if s > 0]
SEEDS = [11, 22, 33]
SUCCESS_PPL_BAR = 100.0          # AF2-D's frozen recovery bar
SPREAD_ROBUST_MAX = 2.0
SPREAD_MODERATE_MAX = 5.0
MIN_LEVELS_MODERATE = 3
TASKS = ("wikitext", "arc_easy", "lambada_openai")
# AF2-D damaged-base reference (EXP-AF-002-D pre-train, seed-mean):
DAMAGED_BASE = {"wikitext": 425.76, "arc_easy": 0.4891,
                "lambada_openai": 0.2418}


def sigma_dir_name(sigma: float) -> str:
    return f"sigma-{sigma:g}"


def load_cells(run_dir: Path) -> list[dict]:
    """Load every sigma-*/seed-*/t2_ternary/eval.summary.json cell."""
    cells = []
    for path in sorted(
        run_dir.glob("sigma-*/seed-*/t2_ternary/eval.summary.json")
    ):
        with open(path) as f:
            s = json.load(f)
        sigma = float(path.parts[-4].split("-", 1)[1])
        tasks = {
            t: s.get("tasks", {}).get(t, {}).get("value") for t in TASKS
        }
        cells.append({
            "sigma": sigma,
            "seed": int(s["seed"]),
            "t2_init_sigma": s.get("t2_init_sigma"),
            "ppl": tasks["wikitext"],
            "arc_easy": tasks["arc_easy"],
            "lambada_openai": tasks["lambada_openai"],
            "deployed_bytes": s.get("matched_bytes_actual"),
            "is_untrained_control": s.get("is_untrained_control"),
            "path": str(path),
        })
    return cells


def level_stats(cells: list[dict]) -> dict:
    """sigma -> per-level aggregate over seeds."""
    levels: dict[float, list[dict]] = {}
    for c in cells:
        levels.setdefault(c["sigma"], []).append(c)
    out = {}
    for sigma, rows in sorted(levels.items()):
        ppls = [r["ppl"] for r in rows if r["ppl"] is not None]
        arcs = [r["arc_easy"] for r in rows if r["arc_easy"] is not None]
        lams = [r["lambada_openai"] for r in rows
                if r["lambada_openai"] is not None]
        successes = [r["ppl"] is not None and r["ppl"] <= SUCCESS_PPL_BAR
                     for r in rows]
        out[sigma] = {
            "n_seeds": len(rows),
            "seeds": sorted(r["seed"] for r in rows),
            "ppl_values": ppls,
            "ppl_mean": float(np.mean(ppls)) if ppls else None,
            "ppl_std": float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0,
            "ppl_min": min(ppls) if ppls else None,
            "ppl_max": max(ppls) if ppls else None,
            "arc_easy_mean": float(np.mean(arcs)) if arcs else None,
            "lambada_openai_mean": float(np.mean(lams)) if lams else None,
            "n_success": int(sum(successes)),
            "failure_rate": 1.0 - (sum(successes) / len(successes)
                                   if successes else 0.0),
            "level_success": bool(successes) and all(successes),
        }
    return out


def classify(levels: dict) -> dict:
    """Apply the frozen classification rules."""
    nonzero = {s: lvl for s, lvl in levels.items() if s > 0}
    missing = [s for s in NONZERO_LEVELS if s not in nonzero]
    n_success = sum(1 for lvl in nonzero.values() if lvl["level_success"])
    means = [lvl["ppl_mean"] for lvl in nonzero.values()
             if lvl["level_success"] and lvl["ppl_mean"] is not None]
    spread = (max(means) / min(means)) if len(means) >= 2 else None

    if not missing and n_success == len(NONZERO_LEVELS) and spread is not None:
        if spread <= SPREAD_ROBUST_MAX:
            classification = "ROBUST"
        elif spread <= SPREAD_MODERATE_MAX:
            classification = "MODERATELY SENSITIVE"
        else:
            classification = "FRAGILE"
    elif n_success >= MIN_LEVELS_MODERATE and not missing:
        classification = "MODERATELY SENSITIVE"
    else:
        classification = "FRAGILE"

    zero = levels.get(0.0)
    zero_control = None
    if zero is not None:
        zero_succeeds = zero["level_success"]
        zero_control = {
            "level_success": zero_succeeds,
            "ppl_mean": zero["ppl_mean"],
            "finding": (
                "sigma=0 SUCCEEDED — the near-dead-start model is "
                "falsified at this site/budget"
                if zero_succeeds else
                "sigma=0 failed as expected (near-dead start; the "
                "correction cannot grow from a zero init in 500 steps)"
            ),
        }

    return {
        "classification": classification,
        "levels_succeeded": n_success,
        "levels_total": len(NONZERO_LEVELS),
        "levels_missing": [f"{s:g}" for s in missing],
        "spread_ratio": spread,
        "zero_control": zero_control,
    }


def capability_crosscheck(levels: dict) -> dict:
    """Flag levels whose ppl 'success' is not backed by acc metrics
    above the damaged base (PROXY/CAPABILITY divergence rule)."""
    flagged = []
    for sigma, lvl in sorted(levels.items()):
        if not lvl["level_success"]:
            continue
        arc = lvl["arc_easy_mean"]
        lam = lvl["lambada_openai_mean"]
        arc_ok = arc is not None and arc > DAMAGED_BASE["arc_easy"]
        lam_ok = lam is not None and lam > DAMAGED_BASE["lambada_openai"]
        if not (arc_ok and lam_ok):
            flagged.append({
                "sigma": sigma,
                "ppl_mean": lvl["ppl_mean"],
                "arc_easy_mean": arc,
                "lambada_openai_mean": lam,
                "note": "ppl success without acc-metric recovery above "
                        "the damaged base — PROXY IMPROVEMENT / "
                        "CAPABILITY NOT VALIDATED for this level",
            })
    return {"divergent_levels": flagged}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cells = load_cells(args.run_dir)
    expected = {(s, seed) for s in SIGMA_LEVELS for seed in SEEDS}
    seen = {(c["sigma"], c["seed"]) for c in cells}
    missing = sorted(expected - seen)
    levels = level_stats(cells)
    result = classify(levels)
    crosscheck = capability_crosscheck(levels)
    integrity = {
        "n_cells": len(cells),
        "expected_cells": len(expected),
        "missing_cells": [f"{s:g}/{seed}" for s, seed in missing],
        "sigma_flag_consistent": all(
            c["t2_init_sigma"] is None
            or abs(c["t2_init_sigma"] - c["sigma"]) < 1e-12
            for c in cells
        ),
        "deployed_bytes_uniform": len({
            c["deployed_bytes"] for c in cells
            if c["deployed_bytes"] is not None
        }) <= 1,
    }
    out = {
        "experiment_id": "EXP-AF-003",
        "run_dir": str(args.run_dir),
        "integrity": integrity,
        "levels": {f"{s:g}": lvl for s, lvl in sorted(levels.items())},
        "classification": result,
        "capability_crosscheck": crosscheck,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "classification": result["classification"],
        "levels_succeeded": result["levels_succeeded"],
        "spread_ratio": result["spread_ratio"],
        "n_divergent_levels": len(crosscheck["divergent_levels"]),
        "missing_cells": integrity["missing_cells"],
    }, indent=2))
    print(f"[af3-audit] audit written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
