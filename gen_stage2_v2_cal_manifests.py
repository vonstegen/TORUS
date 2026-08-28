"""Generate Stage 2 v2 CAL-only pilot manifests.

Sites (preregistered):
  1. model.layers.0.mlp.down_proj       (AF2-D reference)
  2. model.layers.8.mlp.down_proj       (was TWN-degenerate; retest under Gaussian)
  3. model.layers.15.mlp.down_proj      (was TWN-degenerate; retest under Gaussian)
  4. model.layers.0.self_attn.q_proj    (NEW: attention site)
  5. model.layers.0.self_attn.v_proj    (NEW: attention site)

Damage recipe (preregistered): deterministic Gaussian weight noise
  W' = W + sigma * std(W) * eps, eps ~ N(0,1) from torch.Generator(seed).

Sigma values (preregistered):
  0.0, 0.05, 0.10, 0.20, 0.50, 1.00 (6 points, log-spaced light end).

Seeds: 1, 2, 3 (same as Stage 2 v1).

Total cells per site: 6 sigmas x 3 seeds = 18.
Total cells: 5 sites x 18 = 90.

Each cell: 3 min (eval-only, no training). Total: ~4.5 hours on Legion.

Kill criteria (preregistered, before any data):
  - Site is QUALIFYING iff at least 3 sigma values produce ppl in
    distinct reproducibility bands (band = round(ppl, 0)) separated by
    >= 1 ppl unit AND spanning >= 2 ppl units total. Tighter bands
    would conflate with seed variance; the >= 2 ppl span is the
    minimum to claim an informative axis.
  - Tournament proceeds ONLY at QUALIFYING sites; non-qualifying sites
    are ABORTED with reason "axis too narrow".
"""

import sys
import time
from pathlib import Path

import yaml
import os
BASE = Path(os.environ.get("TORUS_BASE", "/home/andrew-jochl/TORUS"))

SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]
SEEDS = [1, 2, 3]

SITES = [
    ("af2d-gauss",     "model.layers.0.mlp.down_proj",
     "AF2-D reference under Gaussian noise (was TWN ppl 88-1524 across D0-D5')"),
    ("L15-gauss",      "model.layers.15.mlp.down_proj",
     "Layer 15 down_proj under Gaussian noise (TWN was degenerate; ppl 14.10-15.49)"),
    ("L0-q-gauss",     "model.layers.0.self_attn.q_proj",
     "NEW attention site: layer 0 self_attn.q_proj under Gaussian noise"),
    ("L0-v-gauss",     "model.layers.0.self_attn.v_proj",
     "NEW attention site: layer 0 self_attn.v_proj under Gaussian noise"),
]


def make_manifest(site_id: str, target_module: str, rationale: str):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "id": f"EXP-RPM-{site_id.upper()}-CAL",
        "title": f"Stage 2 v2 per-site Gaussian CAL — {target_module}",
        "track": "A",
        "subtrack": "Stage 2 v2 (Gaussian damage pilot)",
        "tier": "discovery",
        "status": "PROPOSED",
        "date_proposed": ts,
        "owner": "research harness",
        "claims_addressed": [],
        "rationale": (
            f"Per-site Gaussian CAL pilot for Stage 2 v2. {rationale}. "
            f"Damage recipe: W' = W + sigma * std(W) * eps with eps ~ N(0,1) "
            f"from torch.Generator(seed). Sigma values {SIGMAS}; seeds {SEEDS}; "
            f"6 x 3 = 18 cells per site. Pre-train eval only (no training) "
            f"to characterize the sigma -> ppl mapping BEFORE any tournament. "
            f"wikitext-only CAL to bound compute; tournament stage adds "
            f"arc_easy / lambada_openai on QUALIFYING sites only."
        ),
        "hypothesis": (
            "Gaussian weight noise produces an informative ppl axis at sites "
            "where the TWN recipe is degenerate. At least 3 distinct sigma "
            "values per site should yield reproducible ppl bands spanning "
            ">= 2 ppl units total."
        ),
        "kill_criteria": (
            "Site is QUALIFYING iff >= 3 sigma values produce ppl in distinct "
            "reproducibility bands (round(ppl,0)) separated by >= 1 ppl unit "
            "AND spanning >= 2 ppl units total. Tournament proceeds ONLY at "
            "QUALIFYING sites; non-qualifying sites are ABORTED with reason "
            "'axis too narrow' (reproducible but uninformative)."
        ),
        "expected_artifacts": [
            "runs/r/EXP-RPM-{SITE_ID}-CAL/<timestamp>/sigma-<v>/seed-<n>/"
            "pre_train_eval.json (per-seed ppl on damaged base)",
            "runs/r/EXP-RPM-{SITE_ID}-CAL/<timestamp>/sigma-<v>/aggregate.json "
            "(per-sigma ppl mean +/- stderr across 3 seeds)",
            "runs/r/EXP-RPM-{SITE_ID}-CAL/<timestamp>/site_cal_summary.json "
            "(site-level: sigma -> ppl mapping, qualification verdict)",
        ],
        "artifact_paths": [],
        "contamination_risks": [],
        "experiment": {
            "model": "allenai/OLMo-1B-0724-hf",
            "target_module": target_module,
            "arms": ["t2_ternary"],  # CAL only; pre-train eval
            "seeds": SEEDS,
            "n_steps": 0,  # eval-only
            "batch_size": 4,
            "seq_len": 128,
            "lr": 0.0,
            "tasks": "wikitext,arc_easy,lambada_openai",
            "damage_gaussian": True,
            "damage_sigmas": SIGMAS,
            "damage_seed": 0,
            "pre_train_eval": True,
        },
        "supersedes": None,
        "next_permitted_experiment": (
            "EXP-RPM-{SITE_ID} tournament at QUALIFYING sites, sigma chosen "
            "from the calibration site's reproducible mid-band. Tournament "
            "rejects the site if no reproducible mid-band exists."
        ),
    }


def main():
    out_dir = BASE / "research" / "residual-pareto" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    for site_id, target_module, rationale in SITES:
        m = make_manifest(site_id, target_module, rationale)
        exp_id = m["id"]
        d = out_dir / exp_id
        d.mkdir(parents=True, exist_ok=True)
        p = d / "manifest.yaml"
        p.write_text(yaml.safe_dump(m, sort_keys=False, default_flow_style=False))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()