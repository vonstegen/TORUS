"""AF6b context-robustness auditor — EXP-AF-006b.

Applies the frozen thresholds from experiments/AF6b/manifest.yaml to
the 12 cells (regimes seq16/seq128/seq256/owt x seeds {1,2,3}):

  integrity:    12/12 cells, uniform deployed bytes, finite histories.
  reference:    seq128 (wikitext) mean trained ppl in [17.91, 24.01]
                (RPM-000 band) — else INVALID (recipe drift).
  Q1 window:    a regime is window-robust if all 3 seeds reach
                wikitext ppl <= 100; failing at >= 2 seeds ->
                window-sensitive. Interpretation rule: if seq16
                succeeds AND materially beats seq128 (>2 combined
                stderrs), label 'recovers, step-confounded'.
  Q2 corpus:    (a) cross transfer: owt-trained cells reach wikitext
                ppl <= 100 at all 3 seeds;
                (b) own-corpus recovery ratio
                (damaged_owt - trained_owt) / (damaged_owt - fp16_owt)
                >= 0.5 (mean across seeds), with fp16_owt/damaged_owt
                from the run's verification.json;
                (c) seq128 cells' owt-test ppl recorded as covariate.

Usage:

    python examples/audit_af6_context.py \
        --run-dir runs/a/EXP-AF-006b/<ts> --out <run-dir>/audit.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REGIMES = ("seq16", "seq128", "seq256", "owt")
SEEDS = (1, 2, 3)
PPL_BAR = 100.0
REFERENCE_BAND = (17.91, 24.01)
RECOVERY_RATIO_BAR = 0.5


def load_cells(run_dir: Path) -> dict:
    """regime -> list of cell dicts (seed, ppl, arc, lam, cross_ppl)."""
    cells: dict[str, list[dict]] = {}
    for regime in REGIMES:
        rows = []
        for path in sorted(
            run_dir.glob(f"{regime}/seed-*/t2_ternary/eval.summary.json")
        ):
            with open(path) as f:
                s = json.load(f)
            tasks = s.get("tasks", {})
            cross = s.get("cross_corpus_ppl") or {}
            rows.append({
                "seed": int(s["seed"]),
                "ppl": tasks.get("wikitext", {}).get("value"),
                "arc_easy": tasks.get("arc_easy", {}).get("value"),
                "lambada_openai": tasks.get("lambada_openai", {}).get("value"),
                "cross_owt_ppl": cross.get("value"),
                "deployed_bytes": s.get("matched_bytes_actual"),
                "n_steps": s.get("n_steps"),
                "seq_len": s.get("seq_len"),
                "path": str(path),
            })
        cells[regime] = rows
    return cells


def _stats(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "stderr": float(arr.std(ddof=1) / math.sqrt(arr.size))
        if arr.size > 1 else 0.0,
        "values": [float(v) for v in vals],
    }


def check_integrity(cells: dict, run_dir: Path) -> dict:
    problems = []
    for regime in REGIMES:
        rows = cells.get(regime, [])
        if len(rows) != 3:
            problems.append(f"{regime}: {len(rows)}/3 cells")
        for r in rows:
            if r["ppl"] is None or not math.isfinite(r["ppl"]):
                problems.append(f"{regime} seed {r['seed']}: ppl {r['ppl']}")
    byte_sets = {r["deployed_bytes"] for rows in cells.values()
                 for r in rows if r["deployed_bytes"] is not None}
    if len(byte_sets) > 1:
        problems.append(f"deployed bytes not uniform: {sorted(byte_sets)}")
    for hist in run_dir.glob("*/seed-*/t2_ternary/history*.jsonl"):
        for line in hist.read_text().splitlines():
            if not math.isfinite(json.loads(line)["loss"]):
                problems.append(f"non-finite loss in {hist}")
                break
    return {"ok": not problems, "problems": problems}


def evaluate_reference(cells: dict) -> dict:
    ppls = [r["ppl"] for r in cells["seq128"]]
    st = _stats(ppls)
    lo, hi = REFERENCE_BAND
    return {
        "stats": st,
        "band": list(REFERENCE_BAND),
        "reproduced": bool(lo <= st["mean"] <= hi),
    }


def evaluate_q1(cells: dict) -> dict:
    regimes = {}
    for regime in ("seq16", "seq128", "seq256"):
        ppls = [r["ppl"] for r in cells[regime]]
        n_success = sum(1 for p in ppls if p <= PPL_BAR)
        regimes[regime] = {
            "ppl_stats": _stats(ppls),
            "n_success": n_success,
            "window_robust": n_success == 3,
            "window_sensitive": n_success <= 1,
        }
    # Frozen interpretation rule: seq16 success AND materially better
    # than seq128 (>2 combined stderrs) -> 'recovers, step-confounded'.
    a = regimes["seq16"]["ppl_stats"]
    b = regimes["seq128"]["ppl_stats"]
    se = math.sqrt(a["stderr"] ** 2 + b["stderr"] ** 2)
    z = (b["mean"] - a["mean"]) / se if se > 0 else None
    if not regimes["seq16"]["window_robust"]:
        label = "window-sensitive" if regimes["seq16"]["window_sensitive"] \
            else "partially window-sensitive"
    elif z is not None and z > 2:
        label = "recovers, step-confounded"
    else:
        label = "window-robust"
    return {"regimes": regimes, "seq16_vs_seq128_z": z,
            "seq16_label": label}


def evaluate_q2(cells: dict, verification: dict) -> dict:
    owt_ppls = [r["ppl"] for r in cells["owt"]]
    n_cross = sum(1 for p in owt_ppls if p <= PPL_BAR)
    trained_owt = [r["cross_owt_ppl"] for r in cells["owt"]
                   if r["cross_owt_ppl"] is not None]
    fp16_owt = verification["fp16_owt_test_ppl"]
    damaged_owt = verification["damaged_owt_test_ppl"]
    ratios = [
        (damaged_owt - t) / (damaged_owt - fp16_owt) for t in trained_owt
    ]
    ratio_stats = _stats(ratios) if ratios else None
    wt_trained_cross = [r["cross_owt_ppl"] for r in cells["seq128"]
                        if r["cross_owt_ppl"] is not None]
    return {
        "cross_capability": {
            "ppl_stats": _stats(owt_ppls),
            "n_success": n_cross,
            "transfer_holds": n_cross == 3,
        },
        "own_corpus_recovery": {
            "ratios": ratios,
            "ratio_stats": ratio_stats,
            "bar": RECOVERY_RATIO_BAR,
            "holds": bool(ratio_stats
                          and ratio_stats["mean"] >= RECOVERY_RATIO_BAR),
            "fp16_owt_test_ppl": fp16_owt,
            "damaged_owt_test_ppl": damaged_owt,
        },
        "wt_trained_cross_direction_covariate": {
            "owt_test_ppl_values": wt_trained_cross,
            "note": "first measurement; no bar (frozen at PROPOSE)",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    cells = load_cells(args.run_dir)
    with open(args.run_dir / "verification.json") as f:
        verification = json.load(f)

    integrity = check_integrity(cells, args.run_dir)
    reference = evaluate_reference(cells)
    q1 = evaluate_q1(cells)
    q2 = evaluate_q2(cells, verification)

    if not integrity["ok"]:
        verdict = "INVALID"
    elif not reference["reproduced"]:
        verdict = "INVALID"
    else:
        verdict = "DECIDED"

    out = {
        "experiment_id": "EXP-AF-006b",
        "run_dir": str(args.run_dir),
        "integrity": integrity,
        "reference_reproduction": reference,
        "q1_window_robustness": q1,
        "q2_corpus_transfer": q2,
        "verdict": verdict,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({
        "verdict": verdict,
        "integrity_ok": integrity["ok"],
        "reference_mean_ppl": reference["stats"]["mean"],
        "reference_reproduced": reference["reproduced"],
        "seq16_label": q1["seq16_label"],
        "q1_regimes": {r: v["n_success"] for r, v in q1["regimes"].items()},
        "q2_transfer": q2["cross_capability"]["transfer_holds"],
        "q2_recovery_ratio": (q2["own_corpus_recovery"]["ratio_stats"]
                              or {}).get("mean"),
    }, indent=2))
    print(f"[af6-audit] audit written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
