# Verdict — EXP-RPM-SITE-DISCOVERY — Track B condition-4 site-discovery sweep

**Date:** 2026-08-30
**Run:** `runs/r/EXP-RPM-SITE-DISCOVERY/20260830T175339Z/` on legion
**Manifest:** `research/residual-pareto/experiments/EXP-RPM-SITE-DISCOVERY/manifest.yaml`
**Driver SHA:** `dfbd260` (driver `examples/af2_storage_tournament.py` unchanged — Stage 2 v2 CAL protocol)
**Prior runs:** `20260830T174505Z` INVALID (launcher execution defect — concurrent workers per GPU → CUDA OOM on the reference cells; design untouched, recorded in the manifest `prior_runs`).

## Question

Does the AF2-D correction phenomenon have a second site — a
CAL-qualifying layer category under independently calibrated damage
where the damaged base can hold a correction (ppl ≥ 100) — inside
the preregistered evidence-bounded grid? (User steering 2026-08-30:
narrow purpose; CAL may return NO second site; no grid expansion on
a null.)

## Integrity

- Reference cells bit-exact against frozen pilot values:
  ref-gauss-v-L0 σ=0.5 → 439.2520499 (frozen 439.2520) ✓;
  ref-twn-d-L0 thr=0.7 → 429.5512705 (frozen 429.55) ✓.
- 239/239 cells complete; all 13 sites `baseline_ok` (zero-knob cell
  in FP16 band 13) and 18/18 cells per site; no kills; ~2h wall
  (≤ 16 GPU-h cap).
- 22/22 unit tests green (rule reproduces the four frozen pilot
  decisions exactly).

## Results — per-site knob→ppl maps

| site | span | bands | max ppl | qualifying | candidate |
|---|---:|---:|---:|---|---|
| twn-v-L0 | 47.5 | 5 | 74.0 | YES | NO (74 < 100) |
| gauss-v-L2 | 39.1 | 3 | 52.2 | YES | NO |
| gauss-v-L1 | 13.2 | 3 | 26.3 | YES | NO |
| gauss-d-L1 | 13.0 | 3 | 26.1 | YES | NO |
| gauss-v-L12 | 7.6 | 3 | 20.7 | YES | NO |
| gauss-v-L4 | 7.3 | 3 | 20.4 | YES | NO |
| gauss-v-L8 | 6.0 | 3 | 19.1 | YES | NO |
| gauss-v-L15 | 4.8 | 3 | 17.9 | YES | NO |
| gauss-d-L12 | 2.2 | 3 | 15.3 | YES | NO |
| gauss-d-L4 | 1.2 | 2 | 14.3 | NO | NO |
| gauss-d-L8 | 1.1 | 2 | 14.2 | NO | NO |
| twn-v-L4 | 0.3 | 1 | 14.0 | NO | NO |
| twn-v-L15 | 0.3 | 1 | 14.0 | NO | NO |

## Grade

**B** — discovery tier, decisive null, clean integrity, frozen rules
applied by an independent summarizer.

## Decision

**DECIDED NO_SECOND_SITE.** Zero candidates under the frozen rule
(QUALIFYING ∧ max ppl ≥ 100). Recorded without rescue:

- The correction phenomenon requires catastrophic damage (the AF2-D
  TWN band is base ppl 88–697). NO grid site reaches ppl 100 under
  either mechanism; the closest is twn-v-L0 at 74.0 (thr=0.9, a
  non-monotonic TWN×attention map).
- Depth gradient, strongest yet observed: catastrophic sensitivity
  is pinned to layer 0. Gaussian v_proj span collapses 20070 (L0,
  pilot) → 39.1 (L2) → 7.6 (L12); TWN×v_proj collapses 47.5 (L0)
  → 0.3 (L4); down_proj never exceeds 26.1 anywhere.
- TWN at attention projections — the untested mechanism×category
  pairing — is informative but not catastrophic at any depth.

Per the preregistered decision logic: **A-RP-002 is annotated
site-local within the searched space**; RPM-006's layer-category
criterion stays unmet; Track B condition 4 stays blocked; no grid
expansion (exclusions frozen).

## Next permitted experiment

Per the steering order: capability-damaging AF5 rerun (Track B
condition 3 unblock path 1), then the dedicated T1-only test.
