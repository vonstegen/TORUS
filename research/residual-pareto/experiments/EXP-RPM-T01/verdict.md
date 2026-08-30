# EXP-RPM-T01 Verdict — Stage 4 AF5 Held-Out Task Robustness

**Date:** 2026-08-25
**Run namespace:** `runs/r/EXP-RPM-T01/20260825T192400Z`
**Manifest:** `research/residual-pareto/experiments/EXP-RPM-T01/manifest.yaml`
**Driver SHA:** `eb90ccb` (current; manifest-only commit). Stage 1 / 1.5 driver
SHA `692e8ee` untouched. Stage 5 driver SHA `caa5572` reused for the patched
forward logic.

## Hypothesis

At AF2-D layer (`model.layers.0.mlp.down_proj`) under D1p Gaussian damage
(threshold=1.0, CAL ppl=88.31) with seed-001:

1. Trained T2 separates from random T2 by ≥+1σ on ≥3 of 4 held-out tasks.
2. Trained T2 separates from random_lora by ≥+1σ on ≥3 of 4 held-out tasks.
3. Trained T2 wins or ties with best trained comparator on ≥3 of 4 tasks.

If all three hold, **AF5 (downstream-transfer gate) is satisfied** and
Track B B1 unlocks per OPERATING-PLAN §5 v2.3.

## Setup

- **Site:** `model.layers.0.mlp.down_proj` (AF2-D reference)
- **Damage:** Gaussian, sigma=0.20, seed=0 (matches Stage 1.5 D1p)
- **Seed:** 1 (only seed; capability eval deterministic given identical bytes)
- **Comparator set (7 arms):** t2_ternary, int4_residual, int8_residual, lora,
  dense_adapter, random_t2_ternary, random_lora
- **Held-out tasks (4):**
  - **hellaswag** (acc_norm): commonsense sentence completion, 4-choice MC
  - **winogrande** (acc): pronoun resolution, 2-choice
  - **boolq** (acc): yes/no boolean QA, reading comprehension
  - **openbookqa** (acc_norm): open-book science QA, 4-choice MC
- **Eval protocol:** lm-eval-harness, batch_size=16, full task set, fp16
- **Inputs reused:** Stage 1.5 D1p seed-001 adapters (sha256-pinned in
  `ARTIFACTS.json`)

## Results

### Per-arm per-task accuracy (D1p seed-001)

| arm | hellaswag acc_norm | winogrande acc | boolq acc | openbookqa acc_norm |
|---|---:|---:|---:|---:|
| **t2_ternary** | **0.6604** | 0.6148 | **0.6609** | 0.3620 |
| int4_residual | 0.4006 | 0.5438 | 0.5370 | 0.2880 |
| int8_residual | 0.6453 | 0.5991 | 0.6483 | 0.3560 |
| lora | 0.6593 | 0.6077 | **0.6609** | 0.3660 |
| dense_adapter | 0.6599 | **0.6156** | 0.6584 | **0.3700** |
| random_t2_ternary | 0.6617 | 0.6117 | 0.6581 | 0.3560 |
| random_lora | 0.6616 | 0.6085 | 0.6550 | 0.3520 |

### T2 vs random controls (architecture-vs-training-signal axis)

| task | t2 | random_t2 | Δ | trained stderr proxy | z (proxy) |
|---|---:|---:|---:|---:|---:|
| hellaswag | 0.6604 | 0.6617 | **−0.0013** | 0.0653 | −0.02 |
| winogrande | 0.6148 | 0.6117 | +0.0032 | 0.0180 | +0.18 |
| boolq | 0.6609 | 0.6581 | +0.0028 | 0.0310 | +0.09 |
| openbookqa | 0.3620 | 0.3560 | +0.0060 | 0.0205 | +0.29 |

**T2 vs random_lora:** similar — T2 wins on 3 of 4 tasks by tiny amounts
(0.006-0.010 acc) and loses on hellaswag by 0.001. All z-scores <0.5σ.

**Trained_max per task:**
- hellaswag: **random_t2** 0.6617 > T2 0.6604 (random wins by 0.001)
- winogrande: **dense_adapter** 0.6156 ≈ T2 0.6148 (T2 ties within 0.001)
- boolq: **T2 = lora** 0.6609 (tie at top)
- openbookqa: **dense_adapter** 0.3700 > T2 0.3620 (dense wins by 0.008)

### T2 vs trained_mean (architecture-vs-training on the joint trained set)

| task | T2 | trained_mean | Δ (T2 − mean) |
|---|---:|---:|---:|
| hellaswag | 0.6604 | 0.6146 | **+0.046** |
| winogrande | 0.6148 | 0.5988 | +0.016 |
| boolq | 0.6609 | 0.6373 | +0.024 |
| openbookqa | 0.3620 | 0.3497 | +0.012 |

T2 exceeds the **mean** of trained arms on all 4 tasks because int4_residual
and int8_residual underperform (int4 catastrophically on hellaswag at
0.4006). This is a misleading framing — T2 ties random_t2 on every
held-out task; the "above mean" finding is just int4 being bad.

## Pass/fail threshold check

Preregistered thresholds (manifest.yaml):

### Pass thresholds

1. **T2 vs random_t2 ≥+1σ on ≥3 of 4** — **FAIL.** T2 vs random_t2 ≥+1σ
   on **0 of 4 tasks** (max z = +0.29σ on openbookqa).
2. **T2 vs random_lora ≥+1σ on ≥3 of 4** — **FAIL.** T2 vs random_lora
   ≥+1σ on **0 of 4 tasks** (max z = +0.49σ on openbookqa).
3. **T2 wins or ties best trained on ≥3 of 4** — **FAIL.** T2 ties or
   wins best trained on **2 of 4 tasks** (winogrande ties, boolq ties).
4. **T2 above chance (> random_t2_mean) on ≥3 of 4** — **PASS.**
   Trained T2 > random_t2 on 3 of 4 tasks (loses by 0.0013 on hellaswag
   only).

### Fail thresholds

1. **T2 below chance on ≥2 of 4** — **NOT TRIGGERED.** T2 > random_t2 on
   3 of 4.
2. **T2 ≪ random_lora (>2σ below) on ≥2 of 4** — **NOT TRIGGERED.** T2 >
   random_lora on 3 of 4 (loses by 0.0012 on hellaswag only).

## Effect on RPM-001 / A-RP-002 / Track B

**This is a substantive finding.** The Stage 4 result **does not change
RPM-001's CONFIRMED_PASS status** (RPM-001 is about Pareto dominance on
the joint cost vector, which Stage 5 confirmed on the storage/training/
latency/energy dimensions). But it **does close the AF5 path for Track B
B1 unlock** at AF2-D / D1p / seed-001.

### Why T2 doesn't separate from random on held-out tasks

The Stage 1.5 architecture-vs-training-signal finding (T2 ≫ random on
damaged bases) used **wikitext/arc_easy/lambada_openai** — three tasks
where the Stage 1 D1p baseline was already partially-recovered
(Stage 1.5 ppl 17.7 = 2.4x over FP16 ppl 13.1; arc_easy 0.564; lambada
0.537). On these partially-recovered bases, T2's training signal could
amplify the residual.

At **D1p / held-out tasks**, the situation is different:
- **hellaswag** is robust to mild damage: random_t2 (no signal) and
  trained T2 (signal) both reach ~0.66, essentially the FP16 baseline.
  The task doesn't require the residual signal because the damaged model
  already answers commonsense questions correctly.
- **boolq, winogrande, openbookqa** show similar robustness to mild
  damage — random and trained are within 0.003-0.010 acc.

**Implication:** the trained-vs-random signal is **damage-conditional**.
It manifests on tasks where the damaged base has lost capability
(wikitext ppl jumps 13.1 → 17.7; arc_easy recovers from 0.25 to 0.56;
lambada recovers from 0.001 to 0.537). On tasks where the damaged base
is already near the FP16 baseline (hellaswag, winogrande, boolq,
openbookqa), the residual correction is unnecessary and the
architecture-vs-training signal disappears.

### Track B B1 unlock status

**Track B B1 stays locked.** AF5 (downstream-transfer gate) was the
remaining unlock condition per OPERATING-PLAN §5 v2.3. The pass threshold
required T2 to show task-relevant value **above chance on held-out
tasks**. T2 doesn't fail below chance (no FAIL fires), but it doesn't
**separate from random** at ≥+1σ, which was the preregistered bar.

### Stage 1 / 1.5 finding: not contradicted

The Stage 1 + 1.5 verdict (RPM-006 z-scores +19σ to +262σ on arc_easy
and lambada across 10 damaged regimes) **is not contradicted** by this
Stage 4 result. Those z-scores were on arc_easy/lambada at damage
regimes where T2 trained and recovered meaningful capability. Stage 4
shows that on tasks that **don't require recovery** (because the
damaged base is still near baseline), the architecture-vs-training
signal vanishes.

This is consistent with the broader Stage 2 v2 finding that trained
T2 ≈ random T2 at L15 and L0-v under Gaussian damage at σ=0.20 —
when the damage is mild, the trained-arm signal doesn't manifest.

### Stage 2 v3 priority

The Stage 5 verdict already identified that Stage 2 v3 (higher σ on
L15 down_proj) is the next concrete step. Stage 4 reinforces this:
**at higher damage, the held-out task set will likely differentiate
trained vs random** (because the damaged base will lose capability
on those tasks too). Stage 2 v3 at σ=0.50 (CAL ppl=13.75 — still mild)
or σ=0.70 (CAL ppl=429.55 — severe) is the natural next step.

## Constraints / what remains open

- **Single-seed:** only seed-001 was measured. Per-seed variance is
  small for arc_easy/lambada at D1p (~0.001-0.004), but the
  architecture-vs-training-signal question deserves a multi-seed
  confirmation before claiming the Stage 4 result generalizes.
- **Single site:** AF2-D only. Layer generalization blocked at Stage 2
  v2 (L15, L0-v showed trained ≈ random under σ=0.20 Gaussian).
- **Single damage regime:** D1p only. Higher σ may open up the
  architecture-vs-training signal on held-out tasks.
- **Four held-out tasks:** hellaswag, winogrande, boolq, openbookqa.
  Other task families (math, code, multilingual) were not tested.

## Effect on Track B gating (summary)

Per ROADMAP Phase 4 and OPERATING-PLAN §5 v2.3:

| Condition | Status |
|---|---|
| A-RP-001 CONFIRMED_PASS | **FAIL** (Stage 1.5 verdict). Track B B1 prerequisite rewrite under §5 v2.3 substitutes A-RP-002 PROVISIONAL_PASS. |
| A-RP-002 PROVISIONAL_PASS or above | **PASS** (CONFIRMED_PASS via AF2/AF2-R; PASS+ via AF2-D) |
| AF5 task-relevant T2 above threshold | **FAIL** at AF2-D / D1p / seed-001 / 4 held-out tasks. Pass threshold was T2 ≥+1σ vs random on ≥3 of 4; observed 0 of 4. |
| AF8-clean CONFIRMED on A-RP-002 or A-RP-003 | A-RP-002 has AF8-clean CONFIRMED (AF2-R). |
| ≥2 layer categories Pareto | **FAIL** (Stage 2 v2: L15 and L0-v both NOT QUALIFYING) |
| Systems measurements don't eliminate advantage | **PASS** (Stage 5 EXP-RPM-SYS) |
| ≥1 stable budget region | **PASS** (Stage 1 + 1.5) |

**Track B B1 stays locked.** Three conditions remain unsatisfied:
AF5, ≥2 layer categories Pareto, and the A-RP-001 (or A-RP-002 PROV)
+ AF5 + AF8-clean triple under §5 v2.3.

## Driver changes (committed during this work)

- `examples/eval_held_out_tasks.py` (new): Stage 4 task-robustness
  harness. Reloads model + patches per arm + per task to avoid state
  carry-over. Computes per-task aggregates in `held_out_summary.json`.

## Tests

239/244 pass. No new tests added in this commit; the driver changes
are scoped to the Stage 4 harness.

## Reproducibility

```bash
PYTHONPATH=/home/andrew-jochl/TORUS nohup .venv/bin/python \
    examples/eval_held_out_tasks.py \
    --arms t2_ternary,int4_residual,int8_residual,lora,dense_adapter,random_t2_ternary,random_lora \
    > runs/r/_logs/exp-rpm-t01.log 2>&1 &
```

Total runtime: ~28 min on Legion cuda:0 (cuda:1 must be idle for
fair power sampling if a Stage 5 follow-on is run concurrently).
Output at:
`runs/r/EXP-RPM-T01/<ts>/held_out_summary.json` + `per_arm/<arm>/<task>/`.

## Artifacts (sha256-pinned in ARTIFACTS.json)

- `runs/r/EXP-RPM-T01/20260825T192400Z/held_out_summary.json`
- `runs/r/EXP-RPM-T01/20260825T192400Z/per_arm/<arm>/summary.json` (7)
- `runs/r/EXP-RPM-T01/20260825T192400Z/per_arm/<arm>/<task>/eval.summary.json` (28)
- `runs/r/EXP-RPM-T01/20260825T192400Z/per_arm/<arm>/<task>/eval.full.json` (28)
- `runs/r/EXP-RPM-T01/20260825T192400Z/ARTIFACTS.json` (sha256 manifest)
- `research/residual-pareto/experiments/EXP-RPM-T01/manifest.yaml` (this verdict's manifest)
- `research/residual-pareto/experiments/EXP-RPM-T01/verdict.md` (this file)
## CORRECTION 2026-08-30 — regime-mismatch finding (EXP-RPM-T02-PROBE)

T01's manifest lists `damage_regime: D1p, threshold: 1.0` (TWN), but
T01's eval driver (`examples/eval_held_out_tasks.py` →
`eval_untrained_arms_v2.py`, `damage_target_module_gaussian`)
applied **Gaussian σ=0.20** to the base at eval time. Per the
Stage 2 v2 CAL pilot that is ppl 13.13 ≈ FP16 — an essentially
undamaged base. T01's adapters were D1p-trained (sha256 match
against the Stage 1.5 artifact), evaluated on a near-FP16 base.

Consequence: T01's diagnosis "the damaged base was already near FP16
on those tasks — nothing to recover" is **true for T01's actual
(Gaussian σ=0.2) eval regime, not for TWN damage**. The TWN
damaged bases degrade held-out capability at every severity:
EXP-RPM-T02-PROBE (20260830T204622Z) measured hellaswag
0.6614→0.4256, boolq 0.6621→0.5691, winogrande 0.6172→0.5501 at
D5p (thr 0.6). The AF5 rerun is preregistered as EXP-RPM-T02 at
D5p with the frozen T01 thresholds.
