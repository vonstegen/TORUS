# Verdict — EXP-A-H1 — native Hadamard ternary training (small model)

**Date:** 2026-08-30
**Run:** `runs/a/EXP-A-H1/20260830T151743Z/` on legion
**Manifest:** `research/track-a5-hadamard/EXP-A-H1/manifest.yaml`
**Audit:** `runs/a/EXP-A-H1/20260830T151743Z/audit.json`
**Prior runs:** 20260829T201327Z, 20260829T224005Z, 20260830T125059Z —
all INVALID (driver/recipe defects; see manifest `prior_runs`).

## Question

Does native Hadamard-rotated ternary training (rotation geometry built
into the parameterization from the start of quantization-aware
training) beat standard ternary training at matched budget on a small
model? (Track A2 v2 H-NATIVE; A-RP-HAD v1; CP3.2.)

## Design (revised 2026-08-30)

OPT-125M PRETRAINED weights; paired arms (control: plain domain,
hadamard: block-64 Sylvester rotations, W_eff = R_out Q R_in);
AF-proven recipe (per-group absmean scales, group 128, zero
threshold 0.7, detach-trick STE, SGD momentum 0.9, lr 1e-3, clip
1.0, seq 128, batch 4, 1000 steps, openwebtext stream); frozen
bars: hadamard wikitext ppl ≤ 0.97 × control AND arc_easy ≥ control
− 0.03 AND lambada ≥ control − 0.02.

## Integrity and verification

- Parity gate PASS (amended pre-run to 0.6 nats): step-0 gap 0.492 —
  the rotated-base PTQ starts behind by the H-POST admission penalty
  (recorded covariate).
- Materialize cross-check: 0.0001 / 0.0002 nats (≤ 0.1 ✓).
- Both arms: 1000/1000 steps, finite histories, gradients nonzero
  (conditioning proxy 0.0018–0.0025), no kills, full budget (2
  GPU-h cap respected).

## Results

| metric | control | hadamard | bar | verdict |
|--------|---------|----------|-----|---------|
| final train loss | 7.280 | **7.231** | — | reported |
| runtime CE (held-out) | 7.879 | **7.529** | — | reported |
| wikitext ppl (ladder) | **5,491.9** | 5,739.0 | ≤ 0.97× | **FAIL** (1.045×) |
| arc_easy | 0.2820 | **0.2992** | ≥ −0.03 | PASS (+0.017) |
| lambada_openai | 0.0 | 0.0 | ≥ −0.02 | PASS |
| code-flip rate (last window) | 8.1e-9 | 0.0 | — | reported |

The rotated arm closed its 0.49-nat step-0 deficit and ended with a
BETTER train loss (7.231 vs 7.280) and better arc_easy (+0.0172),
but its downstream wikitext ppl is 4.5% WORSE than the control's.
Train-loss improvement with downstream regression on the primary
bar: per OPERATING-PLAN §6 the proxy gain never rescues the bar —
the frozen decision stands.

## Grade

**B** — discovery tier, decisive falsification, clean integrity,
frozen bars applied by an independent auditor.

## Decision

**DECIDED FAIL.** Per CP3.2 ("preregistered advantage without
breaching kill criteria → CONFIRMED_PASS after reproduction; else
FAIL, Hadamard line closes with H-POST"): **the Hadamard line
closes with H-POST.** A-RP-HAD → PROVISIONAL_FAIL (the manifest's
pre-registered transition for a bar miss).

Observation recorded WITHOUT rescue (post-hoc analysis forbidden by
§6): at this site/budget/recipe the rotated geometry improved
training-loss convergence and arc_easy but regressed wikitext ppl —
a train/downstream disagreement on the primary metric. Any future
Hadamard work would need a NEW experiment ID and a preregistered
explanation for this discrepancy before the line could reopen.

## Next permitted experiment

Per the user's steering order: CAL-first second-site discovery
(Track B unblock path 2) → capability-damaging AF5 rerun (unblock
path 3) → dedicated T1-only test.

## Experiments explicitly blocked by this result

- Large-model Hadamard runs (locked until a small-model
  CONFIRMED_PASS — suite doc 10; now unreachable without a new
  preregistered line).
- Any H-POST revival at the tested budget.
