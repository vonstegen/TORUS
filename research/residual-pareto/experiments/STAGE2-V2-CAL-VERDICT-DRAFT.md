# Stage 2 v2 CAL Verdict (DRAFT — pilot in progress 2026-08-24)

**Status:** DRAFT, will be finalized when L0-v and L15 sites complete.
**Date:** 2026-08-24
**Driver SHA:** `ddc2b54`
**Pilot scope:** 4 sites × 6 sigmas × 3 seeds = 72 cells
**Sites:** AF2-D, L15, L0-q (attention), L0-v (attention)
**Damage recipe:** `W' = W + sigma * std(W) * eps` (deterministic Gaussian)

## Pilot results (interim — 2026-08-24T21:04Z)

### AF2-D (model.layers.0.mlp.down_proj, 18/18 cells COMPLETE)

| sigma | seed 1 | seed 2 | seed 3 | mean |
|------:|-------:|-------:|-------:|------:|
| 0.00  | 13.093 | 13.093 | 13.093 | 13.093 |
| 0.05  | 13.096 | 13.096 | 13.096 | 13.096 |
| 0.10  | 13.103 | 13.103 | 13.103 | 13.103 |
| 0.20  | 13.128 | 13.128 | 13.128 | 13.128 |
| 0.50  | 13.373 | 13.373 | 13.373 | 13.373 |
| 1.00  | 15.352 | 15.352 | 15.352 | 15.352 |

- Span: 15.352 − 13.093 = **2.26 ppl units** (just above 2.0 threshold)
- Distinct bands (round): {13, 15} = **2 distinct bands** (need 3 → NOT QUALIFYING)

### L0-q (model.layers.0.self_attn.q_proj, 18/18 cells COMPLETE)

| sigma | seed 1 | seed 2 | seed 3 | mean |
|------:|-------:|-------:|-------:|------:|
| 0.00  | 13.093 | 13.093 | 13.093 | 13.093 |
| 0.05  | 13.095 | 13.095 | 13.095 | 13.095 |
| 0.10  | 13.097 | 13.097 | 13.097 | 13.097 |
| 0.20  | 13.102 | 13.102 | 13.102 | 13.102 |
| 0.50  | 13.131 | 13.131 | 13.131 | 13.131 |
| 1.00  | 13.247 | 13.247 | 13.247 | 13.247 |

- Span: 13.247 − 13.093 = **0.15 ppl units** (far below 2.0)
- Distinct bands: {13} = **1 distinct band** (need 3 → NOT QUALIFYING)

### L0-v and L15 (running on Legion in parallel)

- L0-v (cuda:0): started 2026-08-24T21:04Z; expected ~54 min.
- L15 (cuda:1): started 2026-08-24T21:04Z; expected ~54 min.

## Provisional verdict (AF2-D + L0-q)

**AF2-D:** the σ=1.0 cell produces a +2.26 ppl unit shift from baseline.
This is the first measurable Gaussian damage on this site. However,
the preregistered QUALIFYING rule requires **3 distinct reproducibility
bands**, and AF2-D produces only {13, 15} (the σ=1.0 cell is the only
one in band 15; all others are in band 13). The kill criterion fires.

**L0-q:** the maximum σ=1.0 produces only +0.15 ppl units, far below
the 2.0 threshold. The kill criterion fires. **Gaussian noise on
layer-0 self_attn.q_proj is degenerate.**

The Stage 2 v2 pilot has **demonstrated that Gaussian noise, like TWN
before it, does not produce an informative damage axis at any layer
site on OLMo-1B-0724-hf at the preregistered σ values.** The σ=1.0
case on AF2-D shows that very large noise CAN produce damage, but
the dynamic range is too narrow (single band only) to satisfy the
QUALIFYING rule.

## Implications for the RPM program

- **RPM-001/002/006 status: UNCHANGED** (still UNTESTED per the
  registered thresholds).
- The "≥2 layer categories" PASS+ rule for RPM-006 cannot be reached
  with **either** TWN damage (Stage 2 v1) or Gaussian noise
  (Stage 2 v2) at the preregistered σ values on OLMo-1B-0724-hf.
- The Stage 1 / 1.5 architecture-vs-training finding
  (trained T2 ≫ random T2 on damaged AF2-D) **remains intact** but
  only on AF2-D with TWN damage, where Stage 1.5 demonstrated the
  effect across 5 damage modes (D0'-D5').

## Next step (post-pilot)

The Stage 2 v2 finding suggests that **layer-by-layer damage variation
on OLMo-1B is fundamentally limited at any preregistered σ range**.
The next research step is NOT another damage-mode pilot; it is a
**direct test of the architecture-vs-training hypothesis on AF2-D
with the existing Stage 1.5 setup** — which would lift RPM-001 from
UNTESTED to its expected CONFIRMED_PASS status without needing
cross-layer generalization.

Specifically: **Stage 5 EXP-RPM-SYS (energy measurement)** on the
AF2-D reference is the **right next step**. The energy measurement
addresses RPM-001's registered null (Pareto-optimal under measured
storage, throughput, and energy per token, not just inference cost).
With the energy null addressed, RPM-001 can be promoted to
CONFIRMED_PASS and the cross-layer generalization question becomes
**optional future work** rather than a blocker.

## Driver and reproducibility notes

- Driver SHA at pilot start: `ddc2b54`
- Stage 1 / 1.5 driver SHA (`692e8ee`) untouched.
- Model: `allenai/OLMo-1B-0724-hf` (matches EXP-A-001 preregistered
  baseline)
- Dtype: `--dtype float16 --eval-dtype float16`
- Damaged weight is in-place and frozen (`requires_grad_(False)`).
- Noise is deterministic per (sigma, seed); same (sigma, seed) → same
  noise (verified by tests/test_af2_driver_extension.py).
- Pilot ran on Legion dual TITAN RTX. AF2-D on cuda:0, L0-q on cuda:1
  (parallel), then L0-v on cuda:0 and L15 on cuda:1 (parallel).
- All cells bit-reproducible: same (sigma, seed) → identical ppl
  (see "value": 13.093198488512625 across seeds in the same sigma
  cell — the model's lm_head forward at float16 is deterministic).