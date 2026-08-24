"""Stage 2 v2 CAL pilot summarizer.

Reads every EXP-RPM-*-GAUSS-CAL/{timestamp}/sigma-*/seed-*/pre_train_eval.json,
computes the per-site sigma -> ppl mapping (mean +/- stderr across 3 seeds),
writes a site_cal_summary.json per site, and produces a single
combined `stage2_v2_cal_summary.json` covering all 4 sites.

Per the preregistered kill criteria:
  QUALIFYING iff >= 3 sigma values produce ppl in distinct
  reproducibility bands (round(ppl,0)) separated by >= 1 ppl unit
  AND spanning >= 2 ppl units total.

The summary output records the qualification verdict per site.
"""

import json
import statistics
import sys
from pathlib import Path

import yaml


BASE = Path("/home/andrew-jochl/TORUS")
RUNS_DIR = BASE / "runs" / "r"

SITES = {
    "EXP-RPM-AF2D-GAUSS-CAL": "model.layers.0.mlp.down_proj",
    "EXP-RPM-L15-GAUSS-CAL":  "model.layers.15.mlp.down_proj",
    "EXP-RPM-L0-Q-GAUSS-CAL": "model.layers.0.self_attn.q_proj",
    "EXP-RPM-L0-V-GAUSS-CAL": "model.layers.0.self_attn.v_proj",
}


def parse_sigma(d: Path) -> float:
    """sigma-00_05 -> 0.05."""
    name = d.name
    if not name.startswith("sigma-"):
        return float("nan")
    body = name[len("sigma-"):]
    return float(body.replace("_", "."))


def load_ppl(pre_train_eval_path: Path) -> float | None:
    try:
        d = json.loads(pre_train_eval_path.read_text())
    except Exception:
        return None
    wt = d.get("tasks", {}).get("wikitext")
    if not wt:
        return None
    return float(wt["value"])


def summarize_site(exp_id: str, target_module: str) -> dict:
    site_dir = RUNS_DIR / exp_id
    if not site_dir.exists():
        return {"exp_id": exp_id, "target_module": target_module,
                "error": "no runs directory"}

    # The most-recent timestamp dir wins (any earlier runs are stale).
    ts_dirs = sorted([p for p in site_dir.iterdir() if p.is_dir()],
                     reverse=True)
    if not ts_dirs:
        return {"exp_id": exp_id, "target_module": target_module,
                "error": "no timestamp directory"}
    ts = ts_dirs[0]

    sigma_dirs = sorted([p for p in ts.iterdir()
                         if p.is_dir() and p.name.startswith("sigma-")])
    rows = []
    for sd in sigma_dirs:
        sigma = parse_sigma(sd)
        seed_dirs = sorted([p for p in sd.iterdir()
                            if p.is_dir() and p.name.startswith("seed-")])
        ppls = []
        for sdir in seed_dirs:
            pte = sdir / "pre_train_eval.json"
            if pte.exists():
                ppl = load_ppl(pte)
                if ppl is not None:
                    ppls.append(ppl)
        if not ppls:
            continue
        rows.append({
            "sigma": sigma,
            "n_seeds": len(ppls),
            "ppl_values": ppls,
            "ppl_mean": statistics.fmean(ppls),
            "ppl_stderr": (statistics.stdev(ppls) / len(ppls) ** 0.5
                           if len(ppls) > 1 else 0.0),
            "ppl_min": min(ppls),
            "ppl_max": max(ppls),
            "ppl_range": max(ppls) - min(ppls),
        })

    # Apply kill criteria.
    qualifying_bands = set()
    if rows:
        max_ppl = max(r["ppl_mean"] for r in rows)
        min_ppl = min(r["ppl_mean"] for r in rows)
        span = max_ppl - min_ppl
        for r in rows:
            band = round(r["ppl_mean"])
            qualifying_bands.add(band)
    n_distinct = len(qualifying_bands)
    is_qualifying = (n_distinct >= 3 and span >= 2.0)

    summary = {
        "exp_id": exp_id,
        "target_module": target_module,
        "timestamp": ts.name,
        "n_sigmas": len(rows),
        "sigma_to_ppl": rows,
        "ppl_min_overall": (min(r["ppl_mean"] for r in rows)
                              if rows else None),
        "ppl_max_overall": (max(r["ppl_mean"] for r in rows)
                              if rows else None),
        "ppl_span": (max(r["ppl_mean"] for r in rows)
                     - min(r["ppl_mean"] for r in rows) if rows else None),
        "n_distinct_ppl_bands": n_distinct,
        "qualifying": is_qualifying,
        "kill_criteria": (
            "QUALIFYING iff >= 3 sigma values produce ppl in distinct "
            "reproducibility bands (round(ppl,0)) AND span >= 2 ppl units"
        ),
    }
    out = ts / "site_cal_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    combined = []
    for exp_id, target_module in SITES.items():
        s = summarize_site(exp_id, target_module)
        combined.append(s)
        print(f"{exp_id:30s}  qual={s.get('qualifying', '?')}  "
              f"sigmas={s.get('n_sigmas', '?')}  "
              f"span={s.get('ppl_span', '?')}")
    out = BASE / "research" / "residual-pareto" / "experiments" / \
        "stage2_v2_cal_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(combined, indent=2))
    print(f"combined: {out}")


if __name__ == "__main__":
    main()