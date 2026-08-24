"""Generate EXP-RPM-D{0..5}'  (Stage 1.5) manifests.

Each manifest is a record with:
  - frozen damage knob (threshold; group_size; calibrate_norm)
  - frozen observed-ppl band (from EXP-RPM-CAL)
  - frozen arms (5 trained + 2 untrained controls)
  - 3 seeds
  - 21 runs per regime (no_correction arm not supported by driver)

The Stage 1.5 design uses the OBSERVED ppl (from EXP-RPM-CAL) as
the damage-axis basis. Each regime's preregistered knob is the
threshold that produces that observed ppl on the AF2-D layer.
This replaces Stage 1's threshold axis (where 3 of 6 knobs collapsed
to the same ppl 1524.80).
"""
import yaml
from pathlib import Path

REPO = Path("/tmp/TORUS")
OUT_BASE = REPO / "research" / "residual-pareto" / "experiments"

# Stage 1.5 design (frozen at preregistration 2026-08-24 after
# EXP-RPM-CAL calibration). 6 regimes spanning the observed-ppl
# range from CAL. D0' is the FP16 reference (no damage).
REGIMES = [
    {
        "id": "EXP-RPM-D0p",
        "threshold": None,
        "calibrated_ppl": None,
        "nominal_band": [13.0, 15.0],
        "rationale_band": "FP16 reference baseline; calibrated ppl ~13.09 from EXP-A-001.",
    },
    {
        "id": "EXP-RPM-D1p",
        "threshold": 1.0,
        "calibrated_ppl": 88.31,
        "nominal_band": [50.0, 150.0],
        "rationale_band": "Light damage (threshold=1.0; ppl 88.31 from CAL). Threshold=1.0 is essentially no quantization but enough to register the kernel.",
    },
    {
        "id": "EXP-RPM-D2p",
        "threshold": 0.9,
        "calibrated_ppl": 203.60,
        "nominal_band": [150.0, 260.0],
        "rationale_band": "Moderate-light damage (threshold=0.9; ppl 203.60 from CAL).",
    },
    {
        "id": "EXP-RPM-D3p",
        "threshold": 0.8,
        "calibrated_ppl": 303.06,
        "nominal_band": [260.0, 360.0],
        "rationale_band": "Moderate damage (threshold=0.8; ppl 303.06 from CAL).",
    },
    {
        "id": "EXP-RPM-D4p",
        "threshold": 0.7,
        "calibrated_ppl": 429.55,
        "nominal_band": [360.0, 500.0],
        "rationale_band": "Heavy damage (threshold=0.7; ppl 429.55 from CAL = AF2-D reference).",
    },
    {
        "id": "EXP-RPM-D5p",
        "threshold": 0.6,
        "calibrated_ppl": 697.29,
        "nominal_band": [550.0, 850.0],
        "rationale_band": "Severe damage (threshold=0.6; ppl 697.29 from CAL).",
    },
]

COMMON = dict(
    track="A",
    tier="confirmation",
    claims_addressed=["RPM-001", "RPM-002", "RPM-006"],
    title="RPM Stage 1.5 damage sweep (observed-ppl axis, one regime)",
    owner="research-harness",
    date_proposed="2026-08-24",
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
        "t2_ternary", "int4_residual", "int8_residual",
        "lora", "dense_adapter",
    ],
    arms_untrained=["random_t2_ternary", "random_lora"],
    seeds=[1, 2, 3],
    matched_bytes_tolerance_pct=1.0,
    target_deployed_bytes=4194404,
    group_size=128,
    calibrate_norm=False,
)


def make_manifest(regime):
    has_damage = regime["threshold"] is not None
    threshold = regime["threshold"]

    if has_damage:
        threshold_clause = (
            f"with threshold={threshold} (group_size={COMMON['group_size']}, "
            f"calibrate_norm={COMMON['calibrate_norm']})"
        )
    else:
        threshold_clause = "with NO damage (FP16 reference; D0')"

    hypothesis = (
        f"At the AF2-D layer ({COMMON['target_module']}) {threshold_clause}, "
        f"the matched-storage tournament reproduces the AF2-D arm ordering "
        f"(t2 ties dense on capability at matched bytes) AND T2 IS Pareto-optimal "
        f"vs the complete frozen comparator set on the joint (3 cap × 5 cost "
        f"B/F/O/M/L) vector. The damaged-base starting ppl lands in the band "
        f"{regime['nominal_band']} (centered on the EXP-RPM-CAL-calibrated ppl "
        f"{regime['calibrated_ppl']}); deviation noted but does not invalidate "
        f"(per RPM proposal §5). This regime is one of six in the Stage 1.5 "
        f"observed-ppl axis (calibrated via EXP-RPM-CAL on 2026-08-24)."
    )

    rationale = (
        f"Stage 1.5 of the RPM program (proposal §5). "
        f"{regime['rationale_band']} "
        f"The damage mechanism is parameterized by preregistered knobs "
        f"(threshold={threshold}, group_size={COMMON['group_size']}, "
        f"calibrate_norm={COMMON['calibrate_norm']}); the resulting ppl "
        f"is recorded as an observed covariate. The damage-axis basis is "
        f"OBSERVED ppl (from EXP-RPM-CAL on the AF2-D layer), NOT the "
        f"threshold knob — this addresses Stage 1 finding F1 (threshold axis "
        f"was non-monotonic and partially uninformative)."
    )

    decision_logic = (
        f"Run all arms × 3 seeds under AF8 governance. Verify:\n"
        f"  (a) damaged starting state ppl ∈ {regime['nominal_band']} "
        f"(centered on CAL ppl {regime['calibrated_ppl']}; observation, "
        f"deviation noted but does not invalidate);\n"
        f"  (b) matched-bytes tolerance holds for all trained arms;\n"
        f"  (c) trained t2_ternary recovers the damaged base substantially.\n"
        f"Aggregate over n=3 seeds per arm. Cross-regime RPM-002 / RPM-006 "
        f"verdicts require random_t2_ternary evals; **those are still skipped "
        f"by the driver** (Stage 1 data gap unfixed). If Stage 1.5 is "
        f"launched as-is, RPM-002 and RPM-006 will again be UNRESOLVED. "
        f"The post-hoc eval of random_t2_ternary adapters from Stage 1 "
        f"and Stage 1.5 remains a separate task."
    )

    manifest = {
        "id": regime["id"],
        "track": COMMON["track"],
        "subtrack": "RPM Stage 1.5 (observed-ppl axis, EXP-RPM-CAL-calibrated)",
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
        "revision": "",
        "run_namespace": f"runs/r/{regime['id']}/<timestamp>/af2d/",
        "environment": "legion; x86_64; 2x TITAN RTX; env-lock.txt recorded at run start",
        "model": COMMON["model"],
        "target_module": COMMON["target_module"],
        "damage_ptq": {
            "threshold": threshold,
            "group_size": COMMON["group_size"],
            "calibrate_norm": COMMON["calibrate_norm"],
            "applies": has_damage,
            "calibrated_ppl": regime["calibrated_ppl"],
            "calibration_source": "EXP-RPM-CAL runs/r/EXP-RPM-CAL/20260824T000924Z/ (2026-08-24)",
            "pre_train_eval_check": (
                f"For {regime['id']}: pre-train wikitext ppl lands in "
                f"{regime['nominal_band']} (centered on CAL ppl "
                f"{regime['calibrated_ppl']}; observation, deviation noted "
                f"but does NOT invalidate)."
            ),
        },
        "arms": {
            "trained": COMMON["arms_trained"],
            "untrained_controls": COMMON["arms_untrained"],
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
                "preregistered knobs; threshold chosen from EXP-RPM-CAL "
                "to target a specific observed-ppl band)"
                if has_damage else "FP16 reference (no damage; D0')"
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
                f"Damaged starting state ppl ∈ {regime['nominal_band']} "
                f"(observed covariate; centered on CAL ppl "
                f"{regime['calibrated_ppl']}).",
                "All trained arms within ±1% matched-bytes tolerance.",
                "Trained t2_ternary recovers the damaged base "
                "(post-train ppl ≤ 1.5× the starting ppl).",
            ],
            "pass_plus": [
                "Trained t2_ternary beats random_t2_ternary by >2 "
                "stderr-of-difference on ≥1 capability metric (RPM-006 "
                "axis). **NOTE:** random_t2_ternary evals are SKIPPED by "
                "the driver, so this check requires post-hoc eval of the "
                "random_t2_ternary adapter.npz.",
                "Trained t2_ternary is Pareto-optimal vs the full "
                "registered comparator set on the joint (3 cap × 5 cost) "
                "vector (RPM-001 axis).",
            ],
            "fail": [
                "Matched-bytes tolerance violation for any trained arm.",
                "Trained t2_ternary does not recover (post-train ppl > "
                "3× the starting ppl).",
            ],
        },
        "decision_logic_summary": decision_logic,
        "stop_conditions": [
            "All 21 runs complete (5 trained × 3 seeds + 2 untrained × 3 "
            "seeds = 21 runs total; no_correction arm skipped — driver "
            "doesn't implement it).",
            "Any matched-bytes tolerance violation -> that arm "
            "INVALIDATED, re-execute.",
            f"Any pre-train ppl outside {regime['nominal_band']} -> "
            "recorded as observation, run continues (regime band is a "
            "target, not a hard validation criterion).",
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
        "artifacts_manifest": f"runs/r/{regime['id']}/<timestamp>/ARTIFACTS.json",
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
            "change": "New manifest; no driver modifications. Uses the "
            "existing --damage-ptq / --damage-threshold / --pre-train-eval "
            "flags from the Stage 1 driver. Threshold value chosen per "
            "regime from EXP-RPM-CAL's threshold→ppl function so the "
            "damage axis is calibrated to observed ppl, not the uninformative "
            "threshold knob.",
            "justification": "Required to execute the preregistered RPM "
            "Stage 1.5 damage sweep (EXP-RPM-D0'..D5') using the observed-ppl "
            "axis from EXP-RPM-CAL.",
            "approved_by": "harness rule: feature work permitted when "
            "required to execute a registered experiment (OPERATING-PLAN "
            "section 3)",
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