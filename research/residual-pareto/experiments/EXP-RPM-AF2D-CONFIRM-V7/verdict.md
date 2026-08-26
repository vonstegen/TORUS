# EXP-RPM-AF2D-CONFIRM-V7 Verdict — Stage 2 v7 Boundary Confirmation

**Date:** 2026-08-26
**Run namespace:** `runs/r/EXP-RPM-AF2D-CONFIRM-V7/`
**Manifest:** `research/residual-pareto/experiments/EXP-RPM-AF2D-CONFIRM-V7/manifest.yaml`
**Git commit:** (this verdict's commit)
**Driver:** unchanged from Stage 2 v6 (`stage2-v7-launch.sh` + `eval_random_v7.py`)

---

## Decision: **CONFIRMED**

The Stage 2 v6 finding — that the LRN + TSP operating band at AF2-D TWN damage-severity is the full preregistered range `{0.6, 0.7, 0.8, 0.9, 1.0}` — **reproduces under fresh seeds {4, 5, 6} at three preregistered confirmation thresholds: lower boundary 0.6, interior 0.8, upper boundary 1.0**.

Scientific status upgraded:

> **"Reproduced operating band across the full preregistered AF2-D/TWN severity range."**

The band is now supported by two independent seed sets (v6 seeds {1, 2, 3} + v7 seeds {4, 5, 6}).

---

## Hypothesis

Per `manifest.yaml`:

> At AF2-D layer (`model.layers.0.mlp.down_proj`) under TWN damage at `thr ∈ {0.6, 0.8, 1.0}`, the boundary criterion (trained T2 ≥+2σ vs random T2 on ≥2 of 3 capability metrics, no ≥2σ regression elsewhere) is reproduced with **FRESH seeds (4, 5, 6)**.

Same criterion is reproduced for TSP (trained T2 vs random LoRA).

---

## Cells

| threshold | seeds | tournament arms | base-eval | post-hoc random arm eval | total |
| --- | --- | --- | --- | --- | --- |
| 0.6 | {4,5,6} | 12 | 3 | 6 | 21 |
| 0.8 | {4,5,6} | 12 | 3 | 6 | 21 |
| 1.0 | {4,5,6} | 12 | 3 | 6 | 21 |
| **TOTAL** | | **36** | **9** | **18** | **63** |

All cells produced an `eval.summary.json` (60-80 KB each) with full lm-eval-harness output. Total run wall time on Legion: ~3 hours (sequential, single GPU). Total raw artifact volume: ~5 MB across `eval.summary.json`.

---

## Results

### V7 trained T2 vs random T2 (LRN axis) — fresh seeds {4, 5, 6}

| thr | wikitext_z | arc_easy_z | lambada_z | LRN_active |
| ---: | ---: | ---: | ---: | :---: |
| 0.6 | +67.41 | +15.81 | +30.31 | **YES** |
| 0.8 | +135.99 | +20.90 | +52.28 | **YES** |
| 1.0 | +82.02 | +17.22 | +294.88 | **YES** |

(Positive z = trained T2 is better. Wikitext z-scores are sign-corrected because wikitext word_perplexity is "lower is better".)

### V7 trained T2 vs random LoRA (TSP axis) — fresh seeds {4, 5, 6}

| thr | wikitext_z | arc_easy_z | lambada_z | TSP_active |
| ---: | ---: | ---: | ---: | :---: |
| 0.6 | +52.33 | +16.37 | +30.07 | **YES** |
| 0.8 | +119.88 | +25.93 | +49.92 | **YES** |
| 1.0 | +66.15 | +22.22 | +279.35 | **YES** |

### V6 baseline — original seeds {1, 2, 3} (commit 75c20d3)

| thr | LRN_active | TSP_active |
| ---: | :---: | :---: |
| 0.6 | YES | YES |
| 0.7 | YES | YES |
| 0.8 | YES | YES |
| 0.9 | YES | YES |
| 1.0 | YES | YES |

### V7 raw means (n=3 fresh seeds)

| thr | arm | wikitext ppl | arc_easy acc | lambada acc |
| ---: | --- | ---: | ---: | ---: |
| 0.6 | **t2_ternary** | **26.62** | **0.590** | **0.552** |
| 0.6 | random_t2_ternary | 680.72 | 0.483 | 0.240 |
| 0.6 | random_lora | 534.33 | 0.479 | 0.243 |
| 0.8 | **t2_ternary** | **17.33** | **0.609** | **0.550** |
| 0.8 | random_t2_ternary | 298.19 | 0.504 | 0.284 |
| 0.8 | random_lora | 264.92 | 0.479 | 0.296 |
| 1.0 | **t2_ternary** | **17.54** | **0.570** | **0.533** |
| 1.0 | random_t2_ternary | 83.41 | 0.505 | 0.275 |
| 1.0 | random_lora | 70.66 | 0.486 | 0.289 |

FP16 baseline: ppl 13.09, arc_easy 0.6073, lambada 0.6095.

### Statistical note on z-score magnitudes

Z-scores are large (15-300σ) because the random arm is deterministic across seeds (zero variance by construction: random_t2_ternary and random_lora are computed from fixed random seeds and produce identical adapter weights at every seed index). The relevant noise in the z-score is the **trained-arm-only stderr**, which gets pooled into a small denominator. The qualitative conclusion (trained T2 recovers ppl 17-26, random controls stay 70-680 across all three preregistered thresholds) is robust to stderr convention. The trained arm has finite seed-variance because SGD noise + adapter init randomization differ across seeds.

---

## What this confirms

1. **The AF2-D/TWN operating band is reproducible under a second independent seed set.** The band does not depend on the specific seed set {1, 2, 3} used in v6; seeds {4, 5, 6} reproduce the same qualitative pattern at three preregistered thresholds.
2. **The lower boundary (thr=0.6), interior (thr=0.8), and upper boundary (thr=1.0) of the preregistered band all reproduce.** The band is not just an artifact of the v6 seed draw.
3. **Both LRN and TSP axes reproduce.** The architecture-vs-training signal and the ternary-vs-LoRA signal both survive independent confirmation.
4. **The z-score magnitude at v7 (15-300σ) is of the same order as v6 (10-1500σ).** The trained arm effect is dominant across both seed sets; the random arm variance is structurally zero in both.

## What this does NOT confirm

1. **The band is bounded at thr=1.0.** The preregistration terminates at thr=1.0 because that was the upper end of the sweep; this is not an LRN failure boundary. Stage 2 v7 cannot and does not test that.
2. **LRN behavior at other damage types.** TWN is one specific damage mechanism. Gaussian sigma=0.20/0.50/1.00 (Stage 2 v2/v3/v4) and held-out capability tasks (Stage 4 EXP-RPM-T01) showed LRN absent or inverted. The v7 confirmation is for the TWN damage type only.
3. **Other layers.** The v7 confirmation is at `model.layers.0.mlp.down_proj` (AF2-D) only. Layer-sensitivity (the original Phase 1 EXP-A-011 mandate) remains to be addressed.
4. **Random arm variance.** The deterministic-across-seeds nature of the random arm means the z-scores are inflated relative to a hypothetical setting where the random arm had independent variance. The qualitative conclusion survives this inflation, but the exact z-score magnitudes are not directly comparable to conventional ±2σ tests.

---

## Update to claims

### A-RP-LRN
- **Pre-v7 state:** CONFIRMED_AT_AF2D_TWN_BAND (v6 alone)
- **Post-v7 state:** **CONFIRMED_AT_AF2D_TWN_BAND** — confirmed by TWO independent seed sets (v6 seeds {1, 2, 3} + v7 seeds {4, 5, 6})
- Scientific status upgraded from "full preregistered range observed" to "reproduced operating band".

### A-RP-TSP
- **Pre-v7 state:** CONFIRMED_AT_AF2D_TWN_BAND (v6 alone)
- **Post-v7 state:** **CONFIRMED_AT_AF2D_TWN_BAND** — same upgrade

### A-RP-002
- Unchanged. Composite claim preserved exactly as the v6 audit left it. Decomposition note remains.

---

## Next permitted experiment

With the v7 confirmation, the next scientific question is the **damage-mechanism axis**:

\[ P(\text{learned residual signal matters} \mid \text{damage mechanism}, \text{layer}, \text{task}, \text{budget}) \]

This is the Track B gating question. It cannot be answered by extending the AF2-D/TWN sweep further; it requires a separate experiment that varies the damage mechanism (TWN vs Gaussian vs held-out-task-induced vs structural) at a fixed operating point within the now-confirmed AF2-D/TWN band.

**Recommended next experiment:**

- **EXP-RPM-DAMAGE-TYPE-001** (proposed, not yet preregistered): at AF2-D layer, TWN thr=0.7 (operating-band interior), compare LRN z-score under matched seed sets for {TWN, Gaussian sigma=0.5, Gaussian sigma=1.0} damage types. Preregister whether LRN-positive is preserved across damage types or specific to TWN.

Track B's gating question is now well-formed: the operating band is bounded in {site × damage type × severity} space, and the next experiment should map the band across damage type.

---

## Standing rules respected

- ✅ Preregistered thresholds and criteria frozen before v7 run.
- ✅ No parameter changes from v6 (same driver, same eval protocol, same matched-storage budget).
- ✅ Fresh seeds only (not in {1, 2, 3}).
- ✅ No range expansion (confirmation only; thr=0.7 and 0.9 deliberately excluded).
- ✅ All cells audited (cost-vector, matched-bytes, pre-train-eval).
- ✅ Decision output includes hypothesis, result, grade, decision, confidence/reproduction status, next permitted experiment, experiments explicitly blocked.
- ✅ Commit + push per lifecycle transition.

---

## Audit trail

- `runs/r/EXP-RPM-AF2D-CONFIRM-V7/threshold-{0.6,0.8,1.0}/<ts>/seed-{004,005,006}/{t2_ternary,lora,random_t2_ternary,random_lora}/eval.summary.json` — 36 tournament cells + 18 post-hoc random arm cells + 9 base-eval cells (base-eval stored in `<ts>-base/seed-NNN/t2_ternary/eval.summary.json`)
- `runs/r/_logs/stage2-v7-confirmation/EXP-RPM-AF2D-CONFIRM-V7_threshold-{0.6,0.8,1.0}_<ts>.log` — per-threshold log
- `runs/r/_logs/stage2-v7-master.log` — launcher master log
- `runs/r/_logs/stage2-v7-random-eval.log` — post-hoc random arm eval log
- `analyze_v7.py` — analysis script (committed; reproduces numbers in this verdict)
- `stage2-v7-launch.sh` — launcher (committed)
- `eval_random_v7.py` — post-hoc random arm eval script (committed)

---

## Experiments blocked

Until the v7 confirmation is registered in INDEX.md/ROADMAP.md/CHANGELOG.md:

- ❌ Any Track B adaptive-precision experiment using A-RP-LRN as a gate dependency beyond AF2-D/TWN
- ❌ Any expansion of the AF2-D/TWN sweep to additional sites or layers (that's EXP-A-011 / Stage 3 territory)
- ❌ Any damage-type variation (that's the proposed EXP-RPM-DAMAGE-TYPE-001)

The freeze remains ACTIVE for code changes not required to register this verdict.