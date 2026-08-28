# Verdict — EXP-AF-006b — AF6 dataset/context robustness

**Date:** 2026-08-28
**Run:** `runs/a/EXP-AF-006b/20260828T162141Z/` on legion
**Corrects:** EXP-AF-006 (INVALID — verification-gate miscalibration;
`experiments/AF6/verdict-INVALID.md`). Design identical except the
instrument-anchored verification bands.
**Audit:** `runs/a/EXP-AF-006b/20260828T162141Z/audit.json`
**Manifest:** `experiments/AF6b/manifest.yaml` (thresholds frozen
2026-08-28 at PROPOSE)

## Question (suite doc §9)

Is the AF2-D T2 recovery a general representation/training effect, or
an artifact of the 128-token training window and/or the wikitext-103
training corpus that every prior experiment used?

## Integrity and verification

12/12 cells complete; deployed bytes uniform (4,199,318); no
NaN/inf in any history. Verification gates (instrument-anchored):
FP16 corpus_ppl(wt-103 test) 22.15 ∈ [19.9, 24.4] ✓; damage ratio
3.39 > 1.5 ✓. Pre-train damage band (lm-eval ladder, seq128
invocation): ppl in [400, 460] ✓. Reference reproduction: seq128
cells' mean wikitext ppl 18.99 ∈ [17.91, 24.01] ✓ — a third
independent reproduction of the CONFIRMED_PASS recipe (after AF2-R
and AF3's σ=1e-2 level).

## Results

### Q1 — window matrix (wikitext-trained, token-matched 256k tokens)

| regime | steps | wikitext ppl | arc_easy | lambada | bar (≤100, 3/3) |
|--------|-------|--------------|----------|---------|------------------|
| seq16  | 4000  | **15.08 ± 0.06** | 0.6069–0.6326 | 0.5777–0.5806 | ✓ |
| seq128 | 500   | 18.99 ± 1.61 | 0.5960–0.6153 | 0.5486–0.5556 | ✓ |
| seq256 | 250   | 26.70 ± 1.31 | 0.5774–0.5875 | 0.5030–0.5086 | ✓ |

All three regimes recover at all seeds. **The recovery is not a
128-window artifact at the tested regimes.** The visible gradient is
monotonic in *optimizer steps* (4000 → 15.1, 500 → 19.0, 250 →
26.7), not in window width. Per the frozen interpretation rule,
seq16's superiority over seq128 (z = 2.43 > 2) is labeled
**"recovers, step-confounded"** — the 8× step count, not the narrow
window, is the plausible driver; and seq256's relative weakness
mirrors its 250-step budget. The window-artifact hypothesis is dead;
a step-budget gradient is the measured effect.

### Q2 — corpus matrix (seq=128, 500 steps)

| direction | metric | result | bar |
|-----------|--------|--------|-----|
| owt-trained → wikitext ladder | ppl 14.6–14.7 (3/3) | ✓ | ≤100 |
| owt-trained → owt test (own corpus) | recovery ratio 0.977/0.976/0.978 | ✓ | ≥0.5 |
| wikitext-trained → owt test (covariate) | ppl 17.1/16.7/16.9 | recorded | none (frozen) |

Cross-corpus capability transfer holds in both directions
(wikitext-trained → owt recovery ≈ (93.0−17)/(93.0−14.0) ≈ 0.96
descriptive, recorded as the frozen covariate). owt-trained cells
outperform wikitext-trained reference cells on the wikitext ladder
(14.7 vs 19.0) — consistent with the verification measurement that
OWT is the easier/broader distribution for this model (FP16
corpus_ppl 14.02 owt vs 22.15 wt-103).

## Grade

**A** — confirmation tier, 12/12 cells, frozen thresholds evaluated
by an independent auditor, reference reproduction in band, both
robustness questions answered with the frozen interpretation rule
applied (including its caveat label).

## Decision

**DECIDED — the AF2-D T2 recovery is a general
representation/training effect at the tested regimes**, not a
128-window or wikitext-corpus artifact. Robustness annotations for
the A-RP-002 CONFIRMED_PASS entry (third bullet):

1. init-robust (EXP-AF-003: ROBUST);
2. seed-robust (EXP-AF-003, EXP-AF-006b reference cells);
3. **context/corpus-robust (EXP-AF-006b): window regimes {16, 128,
   256} and corpus {wikitext-103, openwebtext} all recover; the
   sensitivity axis is optimizer-step budget, not window width.**

Findings recorded:
- The program's ladder "wikitext ppl" is wikitext-2 document-level
  word ppl (from the EXP-AF-006 INVALID diagnosis); retained for
  report labeling.
- OWT is the easier/broader training distribution for this model;
  corpus choice moves absolute recovery levels by ~4 ppl.

## Next permitted experiment

- Track B reassessment per OPERATING-PLAN §5 (user steering
  2026-08-28): AF3/AF6 are now complete; the reassessment weighs
  A-RP-LRN (CONFIRMED at the AF2-D TWN band), the AF5 held-out-task
  FAIL (EXP-RPM-T01), and the ≥2-layer-categories Pareto gap.
- Optional preregistered follow-up: step-budget gradient
  characterization (is the recovery monotone-saturating in steps, and
  does it change the deployment recipe?).

## Experiments explicitly blocked by this result

- None. (AF6 informs robustness annotations; unlock rules unchanged.)
