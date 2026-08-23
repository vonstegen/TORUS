"""Generate EXP-RPM-D0..D5 manifests for Stage 1 damage sweep.

Each manifest is a record with:
  - frozen damage knobs (threshold; group_size; calibrate_norm)
  - frozen arms (8 = 5 trained + 2 untrained controls + no_correction)
  - 3 seeds
  - preregistered ppl band (nominal target; the actual ppl is recorded
    as an observed covariate)

The Stage 1 design uses threshold as the monotonic damage axis.
Group_size=128 and calibrate_norm=False are fixed (matching AF2-D's
recipe). D0 is the FP16 reference (no damage).

This script writes 6 YAML manifests under
research/residual-pareto/experiments/RPM-D{0..5}/manifest.yaml.
"""

import yaml
from pathlib import Path

REPO = Path("/tmp/TORUS")
OUT_BASE = REPO / "research" / "residual-pareto" / "experiments"

# Stage 1 design (frozen at preregistration 2026-08-23).
# Each regime: threshold value at group_size=128, calibrate_norm=False
# (matching AF2-D's recipe). D0 has no damage.
REGIMES = [
    {
        "id": "EXP-RPM-D0",
        "subtrack": "RPM Stage 1 damage sweep - D0 calibrated FP16 reference",
        "threshold": None,
        "nominal_band": [13.0, 15.0],
        "rationale_band": "FP16 baseline; preregistered ppl from EXP-A-001 (13.09).",
    },
    {
        "id": "EXP-RPM-D1",
        "subtrack": "RPM Stage 1 damage sweep - D1 mild damage (threshold=0.0)",
        "threshold": 0.0,
        "nominal_band": [13.0, 30.0],
        "rationale_band": "Sign-rounding only (no zeroing). Mildest possible damage at group_size=128. Actual ppl is observed.",
    },
    {
        "id": "EXP-RPM-D2",
        "subtrack": "RPM Stage 1 damage sweep - D2 light damage (threshold=0.3)",
        "threshold": 0.3,
        "nominal_band": [30.0, 80.0],
        "rationale_band": "Light TWN zeroing (~30% sparsity in TWN-normalized space). Actual ppl is observed.",
    },
    {
        "id": "EXP-RPM-D3",
        "subtrack": "RPM Stage 1 damage sweep - D3 moderate damage (threshold=0.5)",
        "threshold": 0.5,
        "nominal_band": [80.0, 200.0],
        "rationale_band": "Moderate TWN zeroing. Actual ppl is observed.",
    },
    {
        "id": "EXP-RPM-D4",
        "subtrack": "RPM Stage 1 damage sweep - D4 heavy damage (threshold=0.6)",
        "threshold": 0.6,
        "nominal_band": [200.0, 350.0],
        "rationale_band": "Heavy TWN zeroing. Actual ppl is observed.",
    },
    {
        "id": "EXP-RPM-D5",
        "subtrack": "RPM Stage 1 damage sweep - D5 catastrophic / AF2-D reference (threshold=0.7)",
        "threshold": 0.7,
        "nominal_band": [300.0, 500.0],
        "rationale_band": "AF2-D's reference recipe. EXP-RPM-000 measured ppl ~425.",
    },
]

COMMON = dict(
    track="A",
    tier="discovery",
    claims_addressed=["RPM-001", "RPM-002"],
    title="RPM Stage 1 damage sweep (one regime)",
    owner="research-harness",
    date_proposed="2026-08-23",
    status="PROPOSED",
    decision=None,
    grade=None,
    model="allenai/OLMo-1B-0724-hf",
    target_module="model.layers.0.mlp.down_proj",
    n_steps=500,
    batch_size=4,
    seq_len=128,
    lr=1e-3,
    momentum=0.9,
    grad_clip=1.0,
    arms_trained=[
        "t2_ternary",
        "int4_residual",
        "int8_residual",
        "lora",
        "dense_adapter",
    ],
    arms_untrained=["random_t2_ternary", "random_lora"],
    arms_no_correction=["no_correction"],
    seeds=[1, 2, 3],
    matched_bytes_tolerance_pct=1.0,
    target_deployed_bytes=4194404,
    group_size=128,
    calibrate_norm=False,
)


def make_manifest(regime):
    """Build the manifest dict for one regime."""
    has_damage = regime["threshold"] is not None
    threshold = regime["threshold"]

    if has_damage:
        threshold_clause = (
            f"with threshold={threshold} (group_size={COMMON['group_size']}, "
            f"calibrate_norm={COMMON['calibrate_norm']})"
        )
    else:
        threshold_clause = "with NO damage (FP16 reference; D0)"

    hypothesis = (
        f"At the AF2-D layer (model.layers.0.mlp.down_proj) "
        f"{threshold_clause}, the matched-storage tournament "
        f"reproduces the AF2-D arm ordering (t2_ternary ties "
        f"dense_adapter on capability metrics at matched bytes) AND "
        f"the trained t2_ternary arm recovers the damaged base "
        f"substantially. The damaged-base starting ppl lands in the "
        f"preregistered band {regime['nominal_band']} (observed "
        f"covariate; deviation noted but does not invalidate per "
        f"RPM proposal section 5). This regime preregisters damage "
        f"severity only; cross-regime result interpretation is via "
        f"RPM-002 (damage-dependence)."
    )

    rationale = (
        f"Stage 1 of the RPM program (proposal section 5). "
        f"{regime['rationale_band']} "
        f"The damage mechanism is parameterized by preregistered knobs "
        f"(threshold={threshold}, group_size={COMMON['group_size']}, "
        f"calibrate_norm={COMMON['calibrate_norm']}); the resulting ppl "
        f"is recorded as an observed covariate, not tuned post-hoc. "
        f"Within-regime comparison: trained t2_ternary vs "
        f"random_t2_ternary (RPM-006 axis) and trained t2_ternary vs "
        f"dense_adapter (RPM-001 cost-vector axis)."
    )

    decision_logic = (
        f"Run all arms x 3 seeds under AF8 governance. Verify:\n"
        f"  (a) damaged starting state ppl in {regime['nominal_band']} "
        f"(observation; deviation noted but does not invalidate);\n"
        f"  (b) matched-bytes tolerance holds for all trained arms;\n"
        f"  (c) trained t2_ternary recovers the damaged base substantially.\n"
        f"Aggregate over n=3 seeds per arm. The Stage 1 cross-regime "
        f"comparison (RPM-002) is computed AFTER all 6 regimes complete."
    )

    manifest = {
        "id": regime["id"],
        "track": COMMON["track"],
        "subtrack": regime["subtrack"],
        "tier": COMMON["tier"],
        "claims_addressed": COMMON["claims_addressed"],
        "title": COMMON["title"],
        "owner": COMMON["owner"],
        "date_proposed": COMMON["date_proposed"],
        "status": COMMON["status"],
        "decision": COMMON["decision"],
        "grade": COMMON["grade"],
        "hypothesis": hypothesis,
        "rationale": rationale,
        "revision": (
            "EXP-RPM-000-reproduced driver (687f3f5). Frozen at RUN-start."
        ),
        "run_namespace": (
            f"runs/r/{regime['id']}/<timestamp>/seed-00X/<arm>/"
        ),
        "environment": (
            "legion; x86_64; 2x TITAN RTX; env-lock.txt recorded at run start"
        ),
        "model": COMMON["model"],
        "target_module": COMMON["target_module"],
        "damage_ptq": {
            "threshold": threshold,
            "group_size": COMMON["group_size"],
            "calibrate_norm": COMMON["calibrate_norm"],
            "applies": has_damage,
            "pre_train_eval_check": (
                f"For {regime['id']}: pre-train wikitext ppl lands in "
                f"{regime['nominal_band']} (nominal target). Deviation "
                f"is recorded as an observed covariate; does NOT "
                f"invalidate the run (per RPM proposal section 5)."
            ),
        },
        "arms": {
            "trained": COMMON["arms_trained"],
            "untrained_controls": COMMON["arms_untrained"],
            "no_correction": COMMON["arms_no_correction"],
        },
        "training": {
            "corpus": "wikitext-103 train split",
            "token_cache": (
                f"runs/r/{regime['id']}/<ts>/wikitext103_train_ids.npy "
                "(sha256 captured; AF8 independent re-tokenization)"
            ),
            "optimizer": "SGD lr=1e-3 momentum=0.9 grad-clip=1.0",
            "objective": "next-token cross-entropy on the damaged base",
            "n_steps": COMMON["n_steps"],
            "batch_size": COMMON["batch_size"],
            "seq_len": COMMON["seq_len"],
            "base_frozen": True,
            "base_state": (
                "damaged-PTQ (per --damage-ptq with the regime's "
                "preregistered knobs)"
                if has_damage else "FP16 reference (no damage; D0)"
            ),
            "eval_dtype": "float16",
            "eval_tasks": ["wikitext", "arc_easy", "lambada_openai"],
            "eval_limit": None,
        },
        "matched_bytes_tolerance_pct": COMMON["matched_bytes_tolerance_pct"],
        "target_deployed_bytes": COMMON["target_deployed_bytes"],
        "quantitative_thresholds": {
            "id": f"{regime['id']}-v1",
            "pass": [
                f"Damaged starting state ppl in {regime['nominal_band']} "
                f"(observed covariate).",
                "All trained arms within +/-1% matched-bytes tolerance.",
                "Trained t2_ternary recovers the damaged base (post-train "
                "ppl <= 1.5x the starting ppl).",
            ],
            "pass_plus": [
                "Trained t2_ternary beats random_t2_ternary by >2 "
                "stderr-of-difference on >=1 capability metric (the "
                "RPM-006 representation-signal axis).",
                "Trained t2_ternary is Pareto-optimal against the full "
                "registered comparator set (RPM-001 full-comparator "
                "rule, evaluated at this regime's cost-vector point).",
            ],
            "fail": [
                "Matched-bytes tolerance violation for any trained arm.",
                "Trained t2_ternary does not recover (post-train ppl > "
                "3x the starting ppl).",
            ],
        },
        "decision_logic_summary": decision_logic,
        "stop_conditions": [
            "All runs complete: 5 trained x 3 seeds + 2 untrained x 3 "
            "seeds + 1 no_correction x 3 seeds = 24 runs total.",
            "Any matched-bytes tolerance violation -> that arm "
            "INVALIDATED, re-execute.",
            "Any pre-train ppl outside nominal_band -> recorded as "
            "observation, run continues (regime band is a target, not "
            "a hard validation criterion).",
        ],
        "expected_artifacts": [
            "per-(seed, arm): history.jsonl, eval.summary.json, "
            "eval.full.json, adapter.npz + meta, deployed_bytes.json, "
            "cost_vector.json",
            "per-seed: pre_train_eval.json (capturing the regime's "
            "verified damaged-base state BEFORE adapter training)",
            "aggregate.json: trained-arms summary, untrained-arms "
            "panel, cost-vector table, Pareto frontier points, "
            "damage-mode verification, regime verdict",
            "provenance.json + env-lock.txt + driver.log + ARTIFACTS.json",
        ],
        "artifact_paths": [],
        "artifacts_manifest": (
            f"runs/r/{regime['id']}/<timestamp>/ARTIFACTS.json"
        ),
        "contamination_risks": [
            f"Token cache under runs/r/{regime['id']}/<ts>/; sha256 "
            "captured; one writer per namespace.",
            "Damage mode is a function of (threshold, group_size, "
            "calibrate_norm) applied to the target module; reproducible "
            "from the frozen driver SHA + preregistered knobs.",
            f"Concurrent writers: the {regime['id']} writer is the only "
            f"process touching runs/r/{regime['id']}/.",
        ],
        "freeze_exception": {
            "change": (
                "New manifest per regime; no driver modifications. The "
                "Stage 1 sweep uses the existing --damage-ptq / "
                "--damage-threshold knobs from AF2-D's recipe "
                "(group_size=128, calibrate_norm=False) with threshold "
                "varied across regimes."
            ),
            "justification": (
                "Required to execute the preregistered RPM Stage 1 "
                "damage sweep (EXP-RPM-D0..D5). The driver is invoked "
                "as-is; the only per-regime variation is the "
                "--damage-threshold flag."
            ),
            "approved_by": (
                "harness rule: feature work permitted when required to "
                "execute a registered experiment (OPERATING-PLAN section 3)"
            ),
        },
        "result_summary": "",
        "confidence_and_reproduction": "",
        "next_permitted_experiment": "",
        "experiments_blocked": [],
        "conclusion": "",
        "supersedes": None,
    }
    return manifest


def main():
    for regime in REGIMES:
        manifest = make_manifest(regime)
        out_dir = OUT_BASE / regime["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "manifest.yaml"
        out_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False)
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()