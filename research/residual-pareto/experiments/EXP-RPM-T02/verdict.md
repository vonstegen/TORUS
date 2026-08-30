# Verdict — EXP-RPM-T02 — AF5 tournament at D5p

**Date:** 2026-08-30
**Run:** `runs/r/EXP-RPM-T02/20260830T211744Z/` on legion
**Manifest:** `research/residual-pareto/experiments/EXP-RPM-T02/manifest.yaml`
**Driver SHA:** `230d0e7` (new TWN-capable driver; T01 instrument untouched)
**Regime:** D5p — TWN thr 0.6, group 128, AF2-D down_proj, base ppl
697.29 (frozen by EXP-RPM-T02-PROBE's selection rule).
**Adapters:** sha256-pinned Stage 1.5 D5p seed-001 artifacts
(`runs/r/EXP-RPM-D5p/20260824T144239Z`; hashes in ARTIFACTS.json).

## Question

Does trained T2 show task-relevant value on held-out tasks above the
frozen AF5 threshold, at a regime where the damaged base actually
loses held-out capability? (Track B condition 3.)

## Integrity

- Base gate PASS 4/4: damaged base bit-matches the probe's frozen
  D5p scores (hellaswag 0.4256, winogrande 0.5501, boolq 0.5691,
  openbookqa 0.2980).
- 32/32 cells; 9/9 unit tests on the frozen threshold application;
  no kills; ~40 min wall (≤ 4 GPU-h cap).

## Results

| task | base (D5p) | t2 | random_t2 | random_lora | int8 | t2 z vs r_t2 |
|---|---:|---:|---:|---:|---:|---:|
| hellaswag | 0.4256 | 0.5853 | 0.4336 | 0.4343 | 0.5979 | **+21.76σ** |
| winogrande | 0.5501 | 0.5620 | 0.5454 | 0.5414 | 0.5912 | +0.84σ |
| boolq | 0.5691 | 0.5835 | 0.5654 | 0.5569 | 0.5972 | +1.48σ |
| openbookqa | 0.2980 | 0.3300 | 0.2940 | 0.3060 | 0.3120 | +1.23σ |

Frozen threshold verdict:

- r1 T2 vs random_t2 ≥ +1σ on ≥3/4: **TRUE** (3/4; winogrande
  +0.84σ misses)
- r2 T2 vs random_lora ≥ +1σ on ≥3/4: **TRUE** (3/4; openbookqa
  +0.81σ misses)
- r3 T2 wins/ties best trained comparator on ≥3/4: **FALSE**
  (0/4 — int8_residual wins hellaswag/winogrande/boolq; lora wins
  openbookqa)
- r4 T2 above chance on ≥3/4: **TRUE** (4/4)
- Fail triggers: none fired.

## Interpretation (recorded without rescue)

1. **The architecture-vs-training signal manifests on held-out
   tasks at a regime with real damage.** Trained T2 recovers
   hellaswag 0.426→0.585 (+21.76σ over random T2) — the first
   held-out-task LRN evidence in the program, and the exact
   opposite of T01's null. T01's null is now fully explained:
   its Gaussian-σ=0.2 eval base was near-FP16 (nothing to
   recover); at D5p there is catastrophic damage and trained T2
   recovers a large share of it where the random structural prior
   does not (+0.008 on hellaswag).
2. **But the frozen AF5 gate is not met.** T2 loses to the best
   trained comparator on all 4 tasks (int8_residual on 3,
   lora on 1). T2's held-out value is real but not competitive at
   matched storage.
3. No fail trigger fired — this is a threshold-miss, not a
   decisive fail: T2 is above chance everywhere and never far
   below random_lora.

## Grade

**B** — clean confirmation-tier execution, decisive threshold
application, the T01 regime-mismatch correction now fully
evidenced.

## Decision

**DECIDED FAIL.** Frozen rule: PASS iff ALL four hold; r3 missed
0/4. **Track B condition 3 remains BLOCKED** — now with correct
evidence: held-out-task value exists (T2 ≫ random at D5p) but is
below the preregistered AF5 threshold (T2 loses to int8 on every
held-out task).

## Next permitted experiment

Per the steering order: the dedicated T1-only test (its own
preregistration). Condition 4 (second site) remains blocked by the
discovery null; condition 3 now has its definitive evidence.
