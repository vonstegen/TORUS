# Verdict — EXP-AF-004 — AF4 sequential-vs-joint training (A-RP-003)

**Date:** 2026-08-28
**Run:** `runs/a/EXP-AF-004/20260828T121414Z/` on legion
**Code revision:** `f1df165` (provenance.json; run namespace
`runs/a/EXP-AF-004/20260828T121414Z/`, ARTIFACTS.json: 55 files)
**Manifest:** `experiments/AF4/manifest.yaml` (thresholds frozen
2026-08-28 at PROPOSE; unchanged)

## Hypothesis (A-RP-003 v1)

At matched total training budget (1000 steps × batch 4 × seq 128 =
512k tokens, identical batches, identical SGD settings, identical
next-token CE objective) and matched deployed storage (two ternary
planes on `model.layers.0.mlp.down_proj`), the sequential curriculum
(train primary latent 500 steps @ n_planes=1 → freeze → train residual
latent 500 steps @ n_planes=2) beats matched joint training (both
latents, 1000 steps @ n_planes=2) by >2 standard errors on at least
one capability metric, with no regression beyond 1 standard error on
the others.

## Result summary

9/9 runs complete (3 arms × 3 seeds). Freeze invariant held on every
seq run (primary latent bitwise identical across the stage boundary;
asserted, not assumed). No divergence (no NaN/inf in 372 logged loss
records). Every post-train ppl far above the kill line (untrained PTQ
reference 427.71). Deployed bytes matched for the claim arms
(seq = joint = 8,912,896; t1_only = 4,456,448 by construction).
Actual cost: ~0.93 GPU-h of the 8 GPU-h budget.

Per-arm means ± stderr (n=3; full wikitext test / arc_easy /
lambada_openai, float16 eval):

| metric          | t1_only (1 plane) | seq (2 planes)    | joint (2 planes)  |
|-----------------|-------------------|-------------------|-------------------|
| wikitext ppl    | **19.30 ± 0.98**  | 24.92 ± 0.64      | 21.44 ± 0.20      |
| arc_easy        | **0.6547 ± 0.0006** | 0.5610 ± 0.0039 | 0.5689 ± 0.0079   |
| lambada_openai  | **0.5927 ± 0.0011** | 0.4432 ± 0.0009 | 0.4684 ± 0.0007   |

Claim test (seq − joint, in stderr-of-difference):

| metric          | Δ (seq − joint) | z       | direction        |
|-----------------|-----------------|---------|------------------|
| wikitext ppl    | +3.48           | +5.20   | **joint better** |
| arc_easy        | −0.0079         | −0.90   | not separated    |
| lambada_openai  | −0.0252         | −21.88  | **joint better** |

Secondary context (preregistered as NOT part of the claim test):
t1_only dominates BOTH two-plane arms on every metric (vs joint:
ppl +2.15σ, arc +10.90σ, lambada +91.63σ; vs seq: ppl +4.81σ,
arc +23.81σ, lambada +104.40σ). At this site under matched CE,
engaging the residual plane at all — sequentially or jointly — is
worse than spending the whole budget on the primary latent. This is
the single-site analogue of A-RP-001's CONFIRMED_FAIL and is reported
as context only; it does not enter the A-RP-003 verdict.

## Grade

**A-** — confirmation tier, n=3 seeds, preregistered thresholds,
matched-control design with the freeze invariant machine-checked,
full-task evals, decisive separation. Minus: single site, single
budget point (1000 steps), single LR family; the verdict is about
this regime, not an asymptotic statement.

## Decision

**FAIL** — direction **joint superior**. Manifest fail clause 2 fires:
joint beats seq by >2 stderr-of-difference on wikitext ppl (+5.20σ)
and lambada_openai (+21.88σ). The PASS clause fails on all three
metrics. **A-RP-003 → PROVISIONAL_FAIL** (REPRODUCTION_REQUIRED).
The program's T1 → freeze → T2 curriculum is not merely unnecessary
at this site — it is materially inferior to matched joint training.
Per the claim text, "sequential is then retained only if simpler; the
mechanism claim is abandoned."

## Confidence and reproduction status

Confirmation-tier design (≥3 seeds, preregistered thresholds, matched
controls). Confidence high within the tested regime; the result is a
curriculum statement, not a plane-capacity statement (the residual
plane's regime of utility is the RPM program's axis, unaffected here).
Per OPERATING-PLAN §3/§11, CONFIRMED_FAIL requires an AF8-style clean
reproduction: **EXP-AF-004-R** (new run ID, independent namespace,
frozen SHA `f1df165`, fresh process, independently generated eval
output; identity of aggregate values is the expected outcome, not a
violation — see EXP-AF-001-R precedent).

## Next permitted experiment

- **EXP-AF-004-R** (clean reproduction) — required before
  A-RP-003 → CONFIRMED_FAIL.
- Suite items that remain open and unblocked: EXP-AF-003 (AF3 init
  robustness), EXP-AF-006 (AF6 dataset/context robustness).
- The t1_only > joint > seq ordering is consistent with the RPM
  regime map (residual value is regime-conditional, established at
  damaged-TWN sites); no new RPM claim is registered from this
  context signal alone.

## Experiments explicitly blocked by this result

- None new. Track B stays locked by its existing §5 conditions
  (AF5, ≥2 layer categories, A-RP-002/LRN chain); this verdict does
  not change A-RP-002, A-RP-LRN, or A-RP-TSP.
- The default `--curriculum 1:500,2:500` in `examples/distill_run.py`
  is flagged as evidence-disfavored for this regime; changing program
  defaults is a follow-up decision after EXP-AF-004-R, never an
  in-place edit motivated by a single unreproduced result.
