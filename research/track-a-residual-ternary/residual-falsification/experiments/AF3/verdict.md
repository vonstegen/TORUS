# Verdict — EXP-AF-003 — AF3 initialization robustness

**Date:** 2026-08-28
**Run:** `runs/a/EXP-AF-003/20260828T142354Z/` on legion
**Code revision:** recorded in provenance.json (`git_sha`, run start)
**Manifest:** `experiments/AF3/manifest.yaml` (thresholds frozen
2026-08-28 at PROPOSE; damaged-base lambada reference corrected to
0.2418 pre-run per smoke measurement; classification thresholds
unchanged)
**Audit:** `runs/a/EXP-AF-003/20260828T142354Z/audit.json`

## Question

Is the AF2-D T2 correction recipe (A-RP-002 CONFIRMED_PASS recipe)
initialization-fragile? Matrix: residual-init σ ∈ {0, 1e-4, 3e-4,
1e-3, 3e-3, 1e-2} × seeds {11, 22, 33} = 18 cells (suite doc §6
verbatim), TWN-damaged frozen primary at
`model.layers.0.mlp.down_proj`, 500 steps matched CE, full 3-task
eval.

## Integrity

18/18 cells complete. σ flag consistent with namespace in every cell.
Deployed bytes uniform across cells (4,199,318 — byte-identical to
AF2-D's t2_ternary; packing is shape-only as constructed). Pre-train
verification (σ=1e-2 invocation, 3 seeds): ppl 425.76 ∈ [400, 460],
arc_easy 0.4891 — damage recipe reproduces AF2-D exactly. No NaN/inf
in any cell. Actual cost ~1.4 GPU-h of the 4 GPU-h budget.

## Results (per level, n=3 seeds each)

| σ      | ppl mean ± std  | arc_easy | lambada | success |
|--------|-----------------|----------|---------|---------|
| 0      | **18.95 ± 0.09** | 0.5989  | 0.5354  | 3/3     |
| 1e-4   | 19.93 ± 0.56    | 0.5899   | 0.5315  | 3/3     |
| 3e-4   | 20.54 ± 0.65    | 0.5950   | 0.5361  | 3/3     |
| 1e-3   | 20.20 ± 0.39    | 0.5967   | 0.5357  | 3/3     |
| 3e-3   | 19.39 ± 1.99    | 0.6009   | 0.5458  | 3/3     |
| 1e-2   | 19.47 ± 1.43    | 0.5999   | 0.5485  | 3/3     |

Every cell far below the frozen success bar (ppl ≤ 100). Spread ratio
(worst level mean / best level mean) = 1.06.

## Classification (frozen rules)

**ROBUST** — all 5 non-zero σ levels succeed at all 3 seeds AND
spread ratio 1.06 ≤ 2. Capability cross-check: no divergent levels
(every level's arc_easy and lambada means exceed the damaged base
0.4891 / 0.2418 by wide margins).

## Findings

1. **The AF2-D T2 recipe has no measurable init-σ sensitivity at this
   site/budget.** The per-row absmax quantizer (threshold = scale/3)
   makes code activation scale-invariant; 500 steps at lr 1e-3 grows
   the correction to useful magnitude from any tested init.
2. **σ=0 SUCCEEDED (ppl 18.95 — the best level mean).** The
   near-dead-start model is **falsified at this site/budget**: at
   r=0 the adapter starts with all-zero codes and a 1e-6 clamped
   scale, yet the STE identity gradient grows the latent out of the
   dead zone fast enough to fully recover. Note this does NOT
   contradict the dead-zone analysis in `torus/train/hf_adapter.py`
   (the torus-ste two-plane path uses a mean-based quantizer with a
   true zero-gradient dead zone at r=0) — it bounds that analysis to
   its own quantizer. The driver's per-row absmax adapter has no such
   dead zone. σ=0 outperforming the non-zero inits (by ~0.5-1.6 ppl)
   suggests the N(0, σ) init noise is mildly counterproductive at
   this site — a noisy correction that must be partially unlearned.
3. **Seed robustness of the CONFIRMED_PASS recipe reproduced.** The
   σ=1e-2 level at fresh seeds {11, 22, 33} (ppl 19.47, arc 0.5999,
   lambada 0.5485) matches AF2-D's seeds 1-3 (ppl 17.32-24.01, arc
   ~0.60, lambada ~0.55) within seed noise.
4. **Scope bound.** ROBUST holds at THIS site (the AF2-D damaged-TWN
   regime), THIS budget (500 steps), THIS recipe. It says nothing
   about other sites (RPM showed the damage axis degenerate at L8/L15
   down_proj and informative at L0-v attention) or other budgets.

## Grade

**A** — confirmation tier, 18/18 cells, frozen classification rules,
byte-level and damage-level integrity checks passed, capability
cross-check clean, one preregistered-model falsification (σ=0)
captured as a finding.

## Decision

**ROBUST.** Per the frozen escalation rule, this classification
annotates the A-RP-002 CONFIRMED_PASS entry (method-level init
robustness + seed robustness); no claim state changes. The surviving
A-RP-002 phenomenon is NOT initialization-fragile at the AF2-D site;
per user steering (2026-08-28), AF6 (context robustness) is now much
more informative and is the next suite item.

## Next permitted experiment

- EXP-AF-006 (AF6 dataset/context robustness) per user steering.
- A σ=0-vs-noise-init follow-up (does zero init dominate at OTHER
  budgets/sites?) is a candidate preregistered experiment but not
  required by any claim.

## Experiments explicitly blocked by this result

- None. (No unlock rules reference AF3.)
