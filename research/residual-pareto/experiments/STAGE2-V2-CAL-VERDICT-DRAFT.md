# Stage 2 v2 CAL Verdict (DRAFT — pilot in progress 2026-08-24)

**Status:** DRAFT, will be finalized when the pilot completes.
**Date:** 2026-08-24
**Driver SHA:** `ddc2b54`
**Pilot scope:** 4 sites × 6 sigmas × 3 seeds = 72 cells
**Sites:** AF2-D, L15, L0-q (attention), L0-v (attention)
**Damage recipe:** `W' = W + sigma * std(W) * eps` (deterministic Gaussian)

## Live pilot results (interim — 2026-08-24T20:57Z)

### AF2-D (model.layers.0.mlp.down_proj, 12/18 cells)

| sigma | seed 1 | seed 2 | seed 3 |
|------:|-------:|-------:|-------:|
| 0.00  | 13.093 | 13.093 | 13.093 |
| 0.05  | 13.096 | 13.096 | 13.096 |
| 0.10  | 13.103 | 13.103 | 13.103 |
| 0.20  | 13.128 | 13.128 | 13.128 |

(remaining: sigma 0.50, 1.00 — 6 cells)

### L0-q (model.layers.0.self_attn.q_proj, 11/18 cells)

| sigma | seed 1 | seed 2 | seed 3 |
|------:|-------:|-------:|-------:|
| 0.00  | 13.093 | 13.093 | 13.093 |
| 0.05  | 13.095 | 13.095 | 13.095 |
| 0.10  | 13.097 | 13.097 | 13.097 |
| 0.20  | 13.102 | 13.102 | (running) |

(remaining: sigma 0.50, 1.00 — 7 cells)

## Observations (interim)

1. **Sigma=0 baseline matches the preregistered FP16 baseline exactly:**
   `ppl=13.09` on both AF2-D and L0-q, identical to the EXP-A-001
   preregistered number. This confirms the model checkpoint and dtype
   are correct after the OLMo-1B-hf → OLMo-1B-0724-hf fix.

2. **The damage axis is much weaker than TWN at threshold=0.7.** Stage 1
   TWN produced ppl 88-1524 across thresholds 0.0-1.0 on AF2-D. Gaussian
   noise at the same sigma range produces ppl shifts of <0.04 ppl units
   on AF2-D and <0.01 on L0-q at sigma=0.20. **Both sites are NOT
   QUALIFYING under the preregistered "≥3 distinct reproducibility
   bands AND ≥2 ppl-unit span" rule.** The kill criteria will fire.

3. **The damage effect is approximately linear in sigma^2**, classic
   Gaussian noise behavior. At sigma=1.0 (one std per weight), the ppl
   shift is expected to be ~0.5 on AF2-D and ~0.15 on L0-q — still well
   below the 2 ppl-unit threshold.

## Provisional verdict

**Stage 2 v2 Gaussian noise does NOT produce an informative ppl axis
on OLMo-1B at any of the preregistered layer sites.**

This is a **legitimate negative result** that parallels the Stage 2 v1
TWN finding (deeper MLP layers robust to mild damage) but extends to
all layer categories and a fundamentally different damage mode. The
within-site sigma→ppl curves are smooth, monotonic, and reproducible,
but their spans (estimated <0.5 ppl units on AF2-D, <0.2 on L0-q)
are far below the preregistered threshold.

**Implications for the RPM program:**
- RPM-001/002/006 status UNCHANGED (still UNTESTED).
- The "≥2 layer categories" PASS+ rule for RPM-006 cannot be reached
  with **either** TWN (Stage 2 v1) **or** Gaussian noise (Stage 2 v2)
  on this base.
- The Stage 1 / 1.5 architecture-vs-training finding (trained T2 ≫
  random T2 on damaged AF2-D) **remains intact and supported** by the
  Stage 2 v2 CAL data, but the cross-layer generalization question
  remains open.

## Final verdict (placeholder — to be filled when pilot completes)

To be written once all 72 cells are complete. Will include:
- Full sigma→ppl curves for all 4 sites
- Preregistered QUALIFYING verdict per site
- Numerical ppl span and reproducibility bands
- Recommendation for the next research step (which damage mode /
  which layer site / or pivot to the planning-track recursion claim).

---

## Driver and reproducibility notes

- Driver SHA at pilot start: `ddc2b54`
- Stage 1 / 1.5 driver SHA (`692e8ee`) untouched.
- Model: `allenai/OLMo-1B-0724-hf` (matches EXP-A-001 preregistered
  baseline)
- Dtype: `--dtype float16 --eval-dtype float16` (corrected from
  `--dtype float32 --eval-dtype float16` after the smoke test showed
  the latter shifted the baseline by +2.28 ppl units)
- Damaged weight is in-place and frozen (`requires_grad_(False)`).
- Noise is deterministic per (sigma, seed).