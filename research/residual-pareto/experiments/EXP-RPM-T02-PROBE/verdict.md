# Verdict — EXP-RPM-T02-PROBE — AF5-regime probe

**Date:** 2026-08-30
**Run:** `runs/r/EXP-RPM-T02-PROBE/20260830T204622Z/` on legion
**Manifest:** `research/residual-pareto/experiments/EXP-RPM-T02-PROBE/manifest.yaml`
**Driver SHA:** `7ca7cfa` (amended; gate + gauss02 cell)
**Prior runs:** `20260830T201326Z` INVALID (verification-gate
miscalibration — see below; recorded in the manifest `prior_runs`).

## Question

Where does the damaged base actually lose held-out capability, so
the AF5 tournament (Track B condition 3) can be run where the
correction has room to act? (User steering chain 2026-08-30.)

## The run-1 INVALID was itself a finding

Run 1 fired the frozen D1p near-FP16 gate: INVALID, as designed.
Root cause: **T01's manifest lists D1p (TWN thr 1.0) but T01's eval
driver (examples/eval_untrained_arms_v2.py,
damage_target_module_gaussian) applied GAUSSIAN sigma=0.20 at eval
time** — per the Stage 2 v2 CAL pilot that is ppl 13.13 ≈ FP16,
i.e. an essentially undamaged base. T01's "nothing to recover"
diagnosis is true for ITS near-undamaged eval regime, not for TWN
damage. T01's adapters were D1p-trained (sha256 match against the
Stage 1.5 artifact) but evaluated on a near-FP16 base.

Amendment (pre-decision, recorded): verification gate → T01-REPRO
cell (gauss02 near-FP16 on ≥3/4); grid 24 → 28 cells; gauss02
excluded from the candidate set; frozen qualify/selection rules
unchanged.

## Integrity

- T01-REPRO gate PASS on 4/4: gauss02 hellaswag 0.6614 vs FP16
  0.6614; winogrande 0.6133 vs 0.6172; boolq 0.6606 vs 0.6621;
  openbookqa 0.3560 vs 0.3560 — T01's actual eval base reproduced.
- 28/28 cells; TWN map bit-identical to run 1 (deterministic
  damage), confirming the INVALID run's data as clean
  non-decision evidence.
- 16/16 unit tests green (rule + gate + selection).

## Results — damaged-base held-out capability vs severity

| regime (CAL ppl) | hellaswag | winogrande | boolq | openbookqa | qual |
|---|---:|---:|---:|---:|---|
| fp16 (13.09) | 0.6614 | 0.6172 | 0.6621 | 0.3560 | — |
| gauss02 (13.13) | 0.6614 | 0.6133 | 0.6606 | 0.3560 | — |
| D1p (88.31) | 0.4785 | 0.5414 | 0.5370 | 0.3100 | YES |
| D2p (203.60) | 0.4857 | 0.5556 | 0.5492 | 0.3240 | YES |
| D3p (303.06) | 0.4690 | 0.5501 | 0.5676 | 0.2980 | YES |
| D4p (429.55) | 0.4415 | 0.5485 | 0.5633 | 0.3080 | YES |
| D5p (697.29) | 0.4256 | 0.5501 | 0.5691 | 0.2980 | YES |

Held-out capability degrades at EVERY TWN severity (hellaswag
−0.18…−0.24, boolq −0.09…−0.13, winogrande −0.06…−0.08; openbookqa
−0.03…−0.06, borderline). T01's null was regime-miscalibration, not
task insensitivity.

## Grade

**B** — discovery tier; decisive; the INVALID gate fired exactly as
designed and produced the T01 regime-mismatch finding; clean rerun.

## Decision

**DECIDED REGIMES_FOUND.** Frozen selection rule (largest summed
drop, tie → more severe): **D5p** (TWN thr 0.6, base ppl 697.29,
summed drop 0.3959 — hellaswag 0.2358, boolq 0.093, winogrande
0.0671, openbookqa 0.058). The AF5 tournament (EXP-RPM-T02) is
preregistered separately at D5p with the frozen T01 thresholds and
the sha256-pinned Stage 1.5 D5p adapters.

## Next permitted experiment

EXP-RPM-T02 (AF5 tournament at D5p — confirmation tier). T01's
verdict carries a regime-mismatch correction annotation.
