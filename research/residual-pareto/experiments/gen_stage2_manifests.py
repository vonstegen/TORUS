"""Stage 2 (EXP-RPM-Lxx) manifests: layer-category sweep.

Stage 2 is the immediate next gate per rev 2.15 corrective.
Per the reviewer:
- Tests whether the T2 effect exists outside AF2-D / down_proj.
- Required evidence for RPM-006 PASS+ (≥2 layer categories).
- Supports Track B B1 unlock.

Stage 2 design (this script generates the manifests):

  Site A (already done): model.layers.0.mlp.down_proj
    - Verified in Stage 1 D5' + Stage 1.5 D5' (AF2-D reference).
    - Architecture category: MLP down_proj, layer 0 (early).

  Site B (NEW): model.layers.15.mlp.down_proj
    - Architecture category: MLP down_proj, layer 15 (late).
    - Different weight statistics from layer 0 (different input
      distribution; different depth-trained features).
    - Same geometry (8192 -> 2048) so the driver works without
      modification.

  Site C (NEW): model.layers.8.mlp.down_proj
    - Architecture category: MLP down_proj, layer 8 (middle).
    - Intermediate position between layer 0 and layer 15.

3 sites × 1 regime (the AF2-D reference threshold=0.7 OR the
CAL-matched threshold from site-specific CAL) × 7 arms × 3 seeds = 21
runs per site, 63 runs total. Plus per-site CAL pre-experiments:
3 sites × 11 thresholds × 3 seeds × 1 arm × ~30s = ~16 min per site,
~50 min total.

Per-site CAL design: same as EXP-RPM-CAL (eval-only, --pre-train-eval,
no adapter). Per-site output: ppl-by-threshold table; pick threshold
that matches the AF2-D ppl ~425.

Total time estimate:
  CAL ×3 sites:     ~50 min
  Tournament ×3 sites: ~105 min (35 min each)
  Post-hoc eval ×3 sites: ~85 min (28 min × 3)
  Total: ~4 hours on Legion (2x TITAN RTX)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import yaml

SITES = [
    {
        "id": "EXP-RPM-L15",
        "target_module": "model.layers.15.mlp.down_proj",
        "label": "MLP down_proj, late (layer 15)",
        "rationale": (
            "Layer 15 = final MLP block before lm_head; deepest-trained "
            "features; different weight statistics from layer 0 "
            "(AF2-D). Tests whether T2 separates from random on late-"
            "layer features."
        ),
    },
    {
        "id": "EXP-RPM-L8",
        "target_module": "model.layers.8.mlp.down_proj",
        "label": "MLP down_proj, middle (layer 8)",
        "rationale": (
            "Layer 8 = middle MLP block; intermediate depth. Bridges "
            "between early-layer (AF2-D, layer 0) and late-layer (L15) "
            "results. Tests T2 effect at intermediate depth."
        ),
    },
]

# Frozen driver SHA (Stage 1, NOT modified)
DRIVER_SHA = "692e8ee"

# Default knobs (matching Stage 1 D5' / AF2-D reference recipe)
COMMON_KWARGS = dict(
    model="allenai/OLMo-1B-0724-hf",
    n_steps=500,
    batch_size=4,
    seq_len=128,
    lr=1e-3,
    momentum=0.9,
    grad_clip=1.0,
    matched_bytes_target=4194404,
    matched_bytes_tolerance_pct=1.0,
    damage_group_size=128,
    eval_dtype="float16",
    dtype="float16",
    tasks="wikitext,arc_easy,lambada_openai",
    matched_bytes_tolerance=1.0,
)


def make_manifest(site):
    target_module = site["target_module"]
    sid = site["id"]
    return {
        "id": sid,
        "track": "A",
        "subtrack": f"RPM Stage 2 layer sweep (site {sid})",
        "tier": "confirmation",
        "claims_addressed": ["RPM-001", "RPM-002", "RPM-006"],
        "title": f"RPM Stage 2 layer sweep — {site['label']}",
        "owner": "research-harness",
        "date_proposed": "2026-08-24",
        "status": "PROPOSED",
        "decision": None,
        "grade": None,
        "hypothesis": (
            f"At the {site['label']} site ({target_module}), with damage "
            "applied at the threshold selected by the per-site CAL to "
            "match the AF2-D ppl target (~425), the matched-storage "
            "tournament reproduces the trained ≫ random separation "
            "(RPM-006 PASS at this site) AND T2 IS Pareto-optimal on "
            "the joint (3 cap × 5 cost B/F/O/M/L) vector (RPM-001 "
            "tentative PASS at this site). RPM-002's registered "
            "monotonicity test is NOT required at this stage — Stage "
            "2 only tests the architecture-vs-training gap and "
            "Pareto status at the AF2-D-equivalent damage point."
        ),
        "rationale": site["rationale"],
        "revision": "",
        "run_namespace": f"runs/r/{sid}/<timestamp>/af2d/",
        "environment": (
            "legion; x86_64; 2x TITAN RTX; env-lock.txt recorded at run "
            "start. AF8 governance: new namespace per site; fresh "
            f"process per site; independent token cache per site."
        ),
        "model": COMMON_KWARGS["model"],
        "target_module": target_module,
        "damage_ptq": {
            "threshold": None,  # filled by per-site CAL
            "group_size": COMMON_KWARGS["damage_group_size"],
            "calibrate_norm": False,
            "applies": True,
            "calibration_source": (
                f"runs/r/EXP-RPM-{sid[8:]}-CAL/<ts>/"  # per-site CAL
            ),
            "pre_train_eval_check": (
                "Per-site CAL must complete first to characterize "
                "threshold->ppl mapping on this site. The tournament "
                "uses the threshold that produces ppl closest to "
                "AF2-D (~425)."
            ),
        },
        "arms": {
            "trained": [
                "t2_ternary", "int4_residual", "int8_residual",
                "lora", "dense_adapter",
            ],
            "untrained_controls": ["random_t2_ternary", "random_lora"],
        },
        "training": {
            "corpus": "wikitext-103 train split",
            "token_cache": (
                f"runs/r/{sid}/<ts>/wikitext103_train_ids.npy "
                "(sha256 captured; AF8 independent re-tokenization)"
            ),
            "optimizer": "SGD lr=1e-3 momentum=0.9 grad-clip=1.0",
            "objective": "next-token cross-entropy on the damaged base",
            "n_steps": COMMON_KWARGS["n_steps"],
            "batch_size": COMMON_KWARGS["batch_size"],
            "seq_len": COMMON_KWARGS["seq_len"],
            "base_frozen": True,
            "base_state": (
                f"damaged-PTQ applied to {target_module} at the "
                "threshold selected by per-site CAL"
            ),
            "eval_dtype": COMMON_KWARGS["eval_dtype"],
            "eval_tasks": COMMON_KWARGS["tasks"].split(","),
            "eval_limit": None,
        },
        "matched_bytes_tolerance_pct": COMMON_KWARGS[
            "matched_bytes_tolerance_pct"
        ],
        "target_deployed_bytes": COMMON_KWARGS["matched_bytes_target"],
        "quantitative_thresholds": {
            "id": f"{sid}-v1",
            "pass": [
                f"Damaged starting state ppl within ~25% of the "
                "AF2-D reference (ppl 425.76); observation only, "
                "not a hard criterion.",
                "All trained arms within ±1% matched-bytes tolerance.",
                "Trained t2_ternary recovers the damaged base "
                "(post-train ppl ≤ 3× the starting ppl).",
                "Trained t2_ternary SEPARATES from random_t2_ternary "
                "by ≥2σ on at least one of arc_easy / lambada_openai "
                "(RPM-006 site-level PASS).",
            ],
            "pass_plus": [
                "Trained t2_ternary SEPARATES from random_t2_ternary "
                "by ≥2σ on BOTH arc_easy AND lambada_openai.",
                "T2 IS Pareto-optimal on the joint (3 cap × 5 cost) "
                "vector at this site (RPM-001 site-level PASS).",
            ],
            "fail": [
                "Matched-bytes tolerance violation for any trained arm.",
                "Trained t2_ternary does not separate from "
                "random_t2_ternary by ≥2σ on any capability metric at "
                "this site (RPM-006 site-level FAIL).",
            ],
        },
        "decision_logic_summary": (
            f"Run all 7 arms × 3 seeds at this site ({target_module}) "
            "with damage applied at the per-site CAL-matched "
            "threshold. PASS criteria per quantitative_thresholds."
        ),
        "stop_conditions": [
            "All 21 runs complete (7 arms × 3 seeds).",
            "Any matched-bytes tolerance violation -> INVALIDATED, "
            "rerun.",
            "T2 does not separate from random by ≥2σ on any metric -> "
            "site-level FAIL recorded.",
        ],
        "expected_artifacts": [
            f"runs/r/{sid}/<ts>/af2d/aggregate.json",
            f"runs/r/{sid}/<ts>/af2d/driver.log",
            f"runs/r/{sid}/<ts>/af2d/seed-001/t2_ternary/eval.summary.json",
            f"runs/r/{sid}/<ts>/af2d/seed-001/random_t2_ternary/eval.summary.json",
        ],
        "artifact_paths": [],
        "artifacts_manifest": (
            f"runs/r/{sid}/<ts>/ARTIFACTS.json"
        ),
        "contamination_risks": [
            "Token cache under runs/r/<site>/<ts>/; sha256 captured; "
            "one writer per namespace.",
            "Damage mode is a function of (threshold, group_size, "
            "calibrate_norm) applied to the target module; "
            "reproducible from frozen driver SHA + preregistered knobs.",
            "Independent re-tokenization per site; AF8 governance.",
        ],
        "freeze_exception": {
            "change": (
                "Stage 2 reuses the Stage 1 driver (commit 692e8ee) "
                "verbatim. NO code modifications. The driver already "
                "supports arbitrary down_proj-equivalent target "
                "modules via the --target-module CLI flag (the driver "
                "builds SiteAdapters based on the patched module's "
                "weight shape, but assumes input=intermediate_size, "
                "output=hidden_size). We restrict Stage 2 sites to "
                "down_proj layers which satisfy this assumption."
            ),
            "justification": (
                "Stage 2 tests the architecture-vs-training signal "
                "across layer categories without modifying the "
                "driver. Layer-category diversity is achieved by "
                "varying the LAYER INDEX (0, 8, 15) while keeping the "
                "ARCHITECTURE CATEGORY fixed (MLP down_proj). If "
                "future Stage 2+ experiments require different "
                "architecture categories (e.g. attention projections "
                "with input=hidden_size, output=hidden_size), the "
                "driver will need to be extended to support those "
                "shapes — that will be a separate freeze-exception "
                "preregistration."
            ),
            "approved_by": (
                "harness rule: feature work permitted when required "
                "to execute a registered experiment; in this case "
                "no feature work is needed."
            ),
        },
        "result_summary": "",
        "confidence_and_reproduction": "",
        "next_permitted_experiment": "",
        "experiments_blocked": [],
        "conclusion": "",
        "supersedes": None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path,
                     default=Path("/home/andrew-jochl/TORUS/research/"
                                  "residual-pareto/experiments"))
    args = ap.parse_args()

    for site in SITES:
        sid = site["id"]
        m = make_manifest(site)
        out_dir = args.out_dir / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "manifest.yaml"
        out_path.write_text(
            yaml.safe_dump(m, sort_keys=False, default_flow_style=False))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()