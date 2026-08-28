"""Per-site CAL manifests for Stage 2.

EXP-RPM-{L15,L8}-CAL: per-site threshold->ppl mapping.

Run BEFORE the EXP-RPM-Lxx tournament to find the threshold that
matches the AF2-D ppl target (~425) on each new layer site.
"""
from __future__ import annotations
from pathlib import Path

import yaml

SITES = [
    {
        "id": "EXP-RPM-L15-CAL",
        "target_module": "model.layers.15.mlp.down_proj",
        "label": "MLP down_proj, late (layer 15)",
    },
    {
        "id": "EXP-RPM-L8-CAL",
        "target_module": "model.layers.8.mlp.down_proj",
        "label": "MLP down_proj, middle (layer 8)",
    },
]


def make_manifest(site):
    sid = site["id"]
    target_module = site["target_module"]
    return {
        "id": sid,
        "track": "A",
        "subtrack": f"RPM Stage 2 per-site CAL (site {sid})",
        "tier": "calibration",
        "claims_addressed": ["RPM-001", "RPM-002", "RPM-006"],
        "title": f"Stage 2 per-site CAL — {site['label']}",
        "owner": "research-harness",
        "date_proposed": "2026-08-24",
        "status": "PROPOSED",
        "decision": None,
        "grade": None,
        "hypothesis": (
            f"On {target_module}, the EXP-RPM-CAL threshold->ppl "
            "mapping is qualitatively similar to the AF2-D layer "
            "(model.layers.0.mlp.down_proj) mapping from the "
            "earlier CAL: thresholds 0.0-0.5 produce ppl in the "
            "DEGENERATE region (ppl ~1500), thresholds 0.6-1.0 "
            "produce a smooth ppl gradient from ~700 down to ~88."
            " The threshold that produces ppl closest to the AF2-D "
            "reference (ppl ~425) will be used for the Stage 2 "
            "tournament at this site."
        ),
        "rationale": (
            "EXP-RPM-CAL (already DECIDED on the AF2-D layer) "
            "showed that the damage-axis basis must be the OBSERVED "
            "ppl, not the threshold knob. Per-site CAL extends this "
            "to other layer sites so that Stage 2 tournaments "
            "compare apples-to-apples across layers (same ppl "
            "damage target)."
        ),
        "revision": "",
        "run_namespace": f"runs/r/{sid}/<timestamp>/af2d/",
        "environment": (
            "legion; x86_64; 2x TITAN RTX; env-lock.txt recorded at "
            "run start. AF8 governance: new namespace per site; "
            f"fresh process per site."
        ),
        "model": "allenai/OLMo-1B-0724-hf",
        "target_module": target_module,
        "damage_ptq": {
            "threshold_values": [
                0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            ],
            "group_size": 128,
            "calibrate_norm": False,
            "applies": True,
        },
        "arms": {
            "trained": ["t2_ternary"],
            "untrained_controls": [],
            "rationale": (
                "t2_ternary used only as a vehicle for "
                "--pre-train-eval (the driver captures pre-train eval "
                "via the t2_ternary arm; we ignore post-train results "
                "and extract pre_train_eval.json)."
            ),
        },
        "training": {
            "corpus": "wikitext-103 train split",
            "token_cache": (
                f"runs/r/{sid}/<ts>/wikitext103_train_ids.npy "
                "(sha256 captured; AF8 independent re-tokenization)"
            ),
            "n_steps": 500,
            "batch_size": 4,
            "seq_len": 128,
            "base_state": (
                f"damaged-PTQ applied to {target_module} at each "
                "preregistered threshold; per-seed pre-train eval "
                "captured BEFORE any adapter training."
            ),
            "eval_dtype": "float16",
            "eval_tasks": ["wikitext"],
            "eval_limit": None,
        },
        "matched_bytes_tolerance_pct": 1.0,
        "target_deployed_bytes": 4194404,
        "quantitative_thresholds": {
            "id": f"{sid}-v1",
            "pass": [
                "All 33 cells complete (11 thresholds × 3 seeds).",
                "Per-threshold ppl reproducible to displayed "
                "precision (stderr = 0 across seeds; deterministic "
                "eval).",
                "At least 4 distinct ppl bands observed across the "
                "11 thresholds (so the per-site axis is informative).",
            ],
            "fail": [
                "Fewer than 33 cells complete (driver crash).",
                "All 11 thresholds produce the same ppl (axis "
                "uninformative on this site, similar to AF2-D D1-D3 "
                "collapse).",
            ],
        },
        "decision_logic_summary": (
            "For each of 11 preregistered thresholds × 3 seeds, apply "
            f"--damage-ptq to {target_module} and capture "
            "pre_train_eval.json. Aggregate to per-threshold ppl "
            f"table. Output: runs/r/{sid}/<ts>/af2d/aggregate.json "
            "with the threshold->ppl mapping for this site."
        ),
        "stop_conditions": [
            "All 33 cells complete.",
            "Driver crash on this target module -> record as site-"
            "specific limitation; do NOT proceed to Stage 2 "
            "tournament without a usable CAL.",
        ],
        "expected_artifacts": [
            f"runs/r/{sid}/<ts>/af2d/aggregate.json",
            f"runs/r/{sid}/<ts>/af2d/driver.log",
            f"runs/r/{sid}/<ts>/af2d/seed-{{001,002,003}}/t2_ternary/pre_train_eval.json",
        ],
        "artifact_paths": [],
        "artifacts_manifest": f"runs/r/{sid}/<ts>/ARTIFACTS.json",
        "contamination_risks": [
            f"Token cache under runs/r/{sid}/<ts>/; sha256 captured; "
            "one writer per namespace.",
            "Damage mode reproducible from frozen driver SHA + "
            "preregistered knobs.",
        ],
        "freeze_exception": {
            "change": (
                "Per-site CAL reuses the Stage 1 driver verbatim. NO "
                "code modifications. The driver already supports "
                "--target-module for any Linear module path."
            ),
            "justification": (
                "Calibration is eval-only (no adapter training). The "
                "driver's pre_train_eval path captures the ppl "
                "required for threshold->ppl mapping."
            ),
            "approved_by": "harness rule: feature work permitted when "
            "required to execute a registered experiment; no "
            "feature work needed.",
        },
        "result_summary": "",
        "confidence_and_reproduction": "",
        "next_permitted_experiment": "",
        "experiments_blocked": [],
        "conclusion": "",
        "supersedes": None,
    }


def main():
    out_dir = Path("/home/andrew-jochl/TORUS/research/residual-pareto"
                    "/experiments")
    for site in SITES:
        sid = site["id"]
        m = make_manifest(site)
        site_dir = out_dir / sid
        site_dir.mkdir(parents=True, exist_ok=True)
        out_path = site_dir / "manifest.yaml"
        out_path.write_text(
            yaml.safe_dump(m, sort_keys=False, default_flow_style=False))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()