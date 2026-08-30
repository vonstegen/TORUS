# Verdict — EXP-AF-001-D — damaged-start T1-only continuation vs T2 plane

**Date:** 2026-08-30
**Run:** `runs/a/EXP-AF-001-D/20260830T222136Z/` on legion
**Manifest:** `research/track-a-residual-ternary/residual-falsification/experiments/AF1-D/manifest.yaml`
**Driver SHA:** `8a0bb4c` + eval-extraction fix
**Question:** acceptance-bar item 1 (suite doc §15): does the T2
plane beat an equal-budget T1-only continuation — here, the
damaged-start analog of AF1 (AF1's arm A was PRISTINE-FP16
continuation; this arm starts from the TWN-damaged base)?

## Integrity

- Pre-train band gate 3/3 seeds: damaged base ppl 429.55 ∈
  [400, 460] (bit-reproducible across seeds — deterministic
  damage).
- 6/6 arm evals; matched budget by construction (identical
  sampler, optimizer, steps); no kills; arm-A instability kill NOT
  fired (27.3 ≪ 425.76); ~21 min wall (≤ 6 GPU-h cap).

## Results

| seed | arm A ppl | arm A arc | arm A lamb | arm B ppl | arm B arc | arm B lamb |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 44.05 | 0.6528 | 0.6010 | 23.05 | 0.6145 | 0.5319 |
| 2 | 17.92 | 0.6646 | 0.5898 | 18.38 | 0.6279 | 0.5451 |
| 3 | 20.05 | 0.6566 | 0.5806 | 4052.04 | 0.5535 | 0.2459 |

Means: arm A ppl 27.34 ± 8.38, arc 0.658 ± 0.004, lambada
0.590 ± 0.006; arm B ppl 1364 ± 1344 (seed 3 diverged), arc
0.599 ± 0.023, lambada 0.441 ± 0.098. Frozen T2: 20.96 ± 1.53,
0.600 ± 0.004, 0.545 ± 0.003.

Frozen bars (T2 minus arm A, sd-of-difference, per-metric
direction): wikitext +0.75σ (T2 better, within noise), arc_easy
**−10.94σ**, lambada **−6.88σ** (arm A better). Fail trigger:
arm A beats T2 by >2σ on ≥1 metric → **FAIL**.

## Interpretation (recorded without rescue)

1. **The T1-only continuation beats the T2 plane at matched
   budget.** Whole-model FP16 training from the damaged state
   recovers the site to ppl ~27 and pushes capability to
   0.658/0.590 — above the frozen T2 means on both capability
   metrics by large margins, and statistically tied on ppl.
   Acceptance-bar item 1 FAILS at the evidenced site.
2. **T2's real advantage is its cost structure, not its
   capability.** The plane achieves 20.96 ppl / 0.600 / 0.545 with
   a 4.2 MB frozen-base correction; the continuation achieves its
   result by repairing the base in FP16 (no storage win). The
   suite doc's item 1 is the training-budget axis — on that axis,
   the plane loses.
3. **Arm B (ternary continuation, matched deployment) is
   seed-fragile**: seeds 1-2 match T2's recovery (23.05/18.38 ppl),
   seed 3 diverges to ppl 4052. Descriptive only, no gate.

## Grade

**B** — clean execution, frozen bars applied, decisive
falsification of the acceptance bar at the tested site.

## Decision

**DECIDED FAIL.** Per suite doc §15's downgrade instruction
("downgrade the conclusion appropriately rather than attempting to
rescue the architecture with additional complexity"): the T2
correction plane does NOT outperform the equal-budget T1-only
continuation at its evidenced site. Combined with EXP-RPM-T02
(T2 loses to int8 at matched storage on held-out tasks), the
correction mechanism's evidenced niche is now precisely bounded:
real value over the random structural prior (LRN) and over no
correction at all, but NOT competitive with retraining the base
nor with the equal-storage int8 correction.

## Next permitted experiment

The 2026-08-30 steering chain is complete (CAL discovery → AF5
rerun → T1-only). Track B remains locked on conditions 3 and 4,
both now with definitive evidence.
