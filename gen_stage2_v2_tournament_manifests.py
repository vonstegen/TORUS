"""Generate Stage 2 v2 tournament manifests.

These are written BEFORE knowing which sites qualify (per the
preregistered protocol). Each QUALIFYING site gets a tournament at
the σ value chosen from its CAL site_cal_summary.json (preregistered
selection: σ that produces a ppl within the middle band of the
sigma->ppl curve; ties broken by lowest σ that produces >= 1 ppl-unit
increase over the σ=0.0 reference).

Each tournament runs the same 7 trained arms + 2 random controls as
Stage 1 / 1.5: t2_ternary, int4_residual, int8_residual, lora,
dense_adapter, random_t2_ternary, random_lora. Identical n_steps=500,
batch_size=4, seq_len=128, lr=1e-3 to Stage 1.5. Metric keys:
wikitext, arc_easy, lambada_openai.

The manifest generator is run AFTER the CAL pilot completes; it reads
research/residual-pareto/experiments/stage2_v2_cal_summary.json and
emits one manifest per QUALIFYING site (one per site_id).
"""

import json
import os
import sys
import time
from pathlib import Path

import yaml


BASE = Path(os.environ.get("TORUS_BASE", "/home/andrew-jochl/TORUS"))
CAL_SUMMARY = BASE / "research" / "residual-pareto" / "experiments" / \
    "stage2_v2_cal_summary.json"


def select_sigma(rows: list) -> tuple[float, str]:
    """Pick the σ for the tournament from the preregistered rule.

    Rule (per the user's direction):
      - Pick the σ whose ppl_mean is in the middle band of the
        sigma->ppl curve (median), excluding σ=0.0 (no damage).
      - Tie-break by the lowest σ that produces >= 1 ppl-unit
        increase over σ=0.0 reference.

    Returns (sigma, reason).
    """
    if not rows:
        raise ValueError("empty rows")
    by_sigma = {round(r["sigma"], 4): r["ppl_mean"] for r in rows}
    sigma_0 = by_sigma.get(0.0)
    if sigma_0 is None:
        raise ValueError("missing sigma=0.0 reference")
    sorted_ppls = sorted(by_sigma.items(), key=lambda kv: kv[1])
    median = sorted_ppls[len(sorted_ppls) // 2][0]
    if median == 0.0:
        # No mid-band; pick lowest σ producing >=1 ppl-unit increase.
        candidates = [s for s, p in by_sigma.items()
                      if s != 0.0 and (p - sigma_0) >= 1.0]
        if not candidates:
            return 0.0, "no informative band; tournament aborted"
        median = min(candidates)
    return median, "middle band of sigma->ppl curve"


def make_manifest(site_id: str, target_module: str, sigma: float,
                  reason: str) -> dict:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "id": f"EXP-RPM-{site_id.upper()}",
        "title": f"Stage 2 v2 tournament — {target_module} @ sigma={sigma}",
        "track": "A",
        "subtrack": "Stage 2 v2 (Gaussian damage tournament)",
        "tier": "discovery",
        "status": "PROPOSED",
        "date_proposed": ts,
        "owner": "research harness",
        "claims_addressed": [],
        "rationale": (
            f"Stage 2 v2 tournament at QUALIFYING site {site_id}. "
            f"σ={sigma} selected from preregistered rule: {reason}. "
            f"Identical protocol to Stage 1 / 1.5 (7 trained arms + "
            f"2 random controls, n_steps=500, batch_size=4, "
            f"seq_len=128, lr=1e-3)."
        ),
        "hypothesis": (
            "trained T2 TERNARY adapter ≫ random T2 TERNARY adapter "
            "on the damaged base at this site, replicating the Stage 1 "
            "architecture-vs-training finding under Gaussian damage."
        ),
        "kill_criteria": (
            "Tournament runs to completion (no early kill). Verdict is "
            "decided by the post-hoc aggregate.json comparison of trained "
            "vs random arms across wikitext/arc_easy/lambada_openai, "
            "matching the Stage 1 / 1.5 protocol."
        ),
        "expected_artifacts": [
            "runs/r/EXP-RPM-{SITE_ID}/{timestamp}/seed-{n}/{arm}/"
            "eval.summary.json (per-cell)",
            "runs/r/EXP-RPM-{SITE_ID}/{timestamp}/aggregate.json "
            "(per-site trained-vs-random comparison)",
        ],
        "artifact_paths": [],
        "experiment": {
            "model": "allenai/OLMo-1B-hf",
            "target_module": target_module,
            "arms": ["t2_ternary", "int4_residual", "int8_residual",
                      "lora", "dense_adapter",
                      "random_t2_ternary", "random_lora"],
            "seeds": [1, 2, 3],
            "n_steps": 500,
            "batch_size": 4,
            "seq_len": 128,
            "lr": 1e-3,
            "momentum": 0.9,
            "grad_clip": 1.0,
            "tasks": "wikitext,arc_easy,lambada_openai",
            "damage_gaussian": True,
            "damage_sigma": sigma,
            "damage_seed": 0,
            "pre_train_eval": True,
        },
        "supersedes": None,
        "next_permitted_experiment": (
            "EXP-RPM-{SITE_ID} post-hoc random-arm eval "
            "(eval_untrained_arms.py) for the sigma ppl band, followed "
            "by RPM-006 decision on the per-site z-score."
        ),
    }


def main():
    if not CAL_SUMMARY.exists():
        print(f"missing {CAL_SUMMARY}; cannot run until CAL pilot completes")
        sys.exit(1)
    cal_data = json.loads(CAL_SUMMARY.read_text())
    if not isinstance(cal_data, list):
        print("cal summary is not a list")
        sys.exit(1)

    out_dir = BASE / "research" / "residual-pareto" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    qualifying_count = 0
    for site in cal_data:
        exp_id = site["exp_id"]
        site_id = exp_id.replace("EXP-RPM-", "").replace("-CAL", "").lower()
        target_module = site["target_module"]
        if not site.get("qualifying", False):
            print(f"SKIP: {site_id} not qualifying "
                  f"(span={site.get('ppl_span')}, bands={site.get('n_distinct_ppl_bands')})")
            continue
        try:
            sigma, reason = select_sigma(site.get("sigma_to_ppl", []))
        except ValueError as e:
            print(f"SKIP: {site_id} cannot select σ ({e})")
            continue
        if sigma == 0.0:
            print(f"SKIP: {site_id} no informative band")
            continue
        m = make_manifest(site_id, target_module, sigma, reason)
        d = out_dir / m["id"]
        d.mkdir(parents=True, exist_ok=True)
        p = d / "manifest.yaml"
        p.write_text(yaml.safe_dump(m, sort_keys=False, default_flow_style=False))
        print(f"wrote {p}  sigma={sigma}  reason={reason}")
        qualifying_count += 1

    if qualifying_count == 0:
        print("WARNING: no qualifying sites; no tournaments to preregister")


if __name__ == "__main__":
    main()