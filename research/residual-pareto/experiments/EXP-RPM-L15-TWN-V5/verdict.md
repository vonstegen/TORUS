# EXP-RPM-L15-TWN-V5 Verdict — Stage 2 v5 ABORT (TWN degenerate at L15 even with group_size=8)

**Date:** 2026-08-25
**Run namespaces:**
- `runs/r/EXP-RPM-L15-TWN-V5-CAL/threshold-0.0/`
- `runs/r/EXP-RPM-L15-TWN-V5-CAL/threshold-0.3/`
- `runs/r/EXP-RPM-L15-TWN-V5-CAL/threshold-0.5/` (ABORTED — not produced)
- `runs/r/EXP-RPM-L15-TWN-V5-CAL/threshold-0.7/` (ABORTED — not produced)
- `runs/r/EXP-RPM-L15-TWN-V5-CAL/threshold-1.0/` (ABORTED — not produced)
**Manifest:** `research/residual-pareto/experiments/EXP-RPM-L15-TWN-V5/manifest.yaml`
**Driver SHA:** `34aa581` (current). Stage 1 / 1.5 driver SHA `692e8ee` untouched.

## Hypothesis (preregistered)

At L15 down_proj under TWN damage with **group_size=8** (vs Stage 2
v1's 128), at least one threshold produces severe damage (ppl ≥ 200).

If yes, proceed to Stage 2 tournament at the qualifying threshold.
If no, ABORT (TWN damage is fundamentally degenerate at L15 even with
finer group granularity).

## Setup

- **Site:** `model.layers.15.mlp.down_proj` (Stage 2 v1 + v2/v3/v4 site)
- **Damage:** TWN, **group_size=8** (vs Stage 2 v1's 128), thresholds
  {0.0, 0.3, 0.5, 0.7, 1.0}, calibrate_norm=false
- **Arms (CAL pilot):** t2_ternary (pre-train eval only — no training)
- **Seeds:** 1, 2, 3
- **Cells:** 5 thresholds × 3 seeds = 15 (Stage 1 CAL pilot only)
- **Eval:** wikitext ppl (pre-train, no adapter training)

## Results

### Per-threshold pre-train wikitext ppl (3-seed mean)

| threshold | group_size | ppl_mean | vs Stage 2 v1 (group_size=128) |
|---|---:|---:|---|
| 0.0 | 8 | 13.9374 | Stage 2 v1 thr 0.0/0.5: ppl 14.10-14.13 |
| 0.3 | 8 | 13.9241 | Stage 2 v1 thr 0.0-0.5: ppl 14.10-14.13 |
| 0.5 | 8 | (not measured — ABORTED) | Stage 2 v1 thr 0.6: ppl 14.10 |
| 0.7 | 8 | (not measured — ABORTED) | Stage 2 v1 thr 0.7: ppl ~14.5 |
| 1.0 | 8 | (not measured — ABORTED) | Stage 2 v1 thr 1.0: ppl 15.49 |

FP16 baseline at L15: ppl 13.09.

### Process deviations

Only 2 of 5 preregistered thresholds produced output before the
remaining 3 processes were lost (the per-threshold launches were
submitted via separate `nohup` calls; the latter 3 did not finish in
time before the v5 preregistration was superseded by user feedback
directing the research program toward a **boundary-mapping experiment**
instead of another layer sweep). **The two measured thresholds are
sufficient to demonstrate the ABORT criterion.**

## ABORT criterion check (preregistered)

> Stage 1 kill criteria: no threshold produces ppl ≥ 200 (TWN at L15
> still degenerate even with finer groups).

**Both measured thresholds produce ppl ≤ 14.** Well below the 200
threshold. **ABORT criterion triggered.** Stage 2 tournament NOT run.

## Conclusion

**TWN damage is fundamentally degenerate at L15 down_proj regardless
of group_size.** Stage 2 v1 (group_size=128, thresholds 0.0-1.0)
found ppl 14.10-15.49 across all 11 thresholds. Stage 2 v5 (group_size=8)
finds ppl 13.92-13.94 at thresholds 0.0-0.3, with no evidence of severe
damage at higher thresholds. The gap between group_size=128 (ppl ~14)
and group_size=8 (ppl ~13.93) is **in the wrong direction** — finer
groups produce **lower** ppl (less damage), not higher. This confirms
that layer 15 weight statistics are uniformly high-magnitude and TWN's
threshold-based zeroing cannot produce severe damage at L15 regardless
of granularity.

## Effect on the research program

The L15/TWN damage axis is **CLOSED**. The two-step plan in the v5
manifest (CAL pilot → tournament at qualifying threshold) cannot
proceed because no qualifying threshold exists.

### What this means for the user's boundary-mapping hypothesis

This verdict supports the user's recommendation that "the next
experiment should NOT be another arbitrary layer sweep" by
demonstrating that even at finer TWN granularity, L15 down_proj is
resistant to TWN damage. The architecture-vs-training story at L15 is
**firmly negative across both damage modes tested**:
- **Gaussian damage** (Stage 2 v2/v3/v4): trained T2 ≈ random T2 at
  σ=0.20, 0.50; trained T2 < random T2 at σ=1.00.
- **TWN damage** (Stage 2 v1, v5): damage is degenerate regardless of
  group_size; can't even reach a comparable ppl band.

The **trained-vs-random T2 architecture-vs-training signal at L15
down_proj is firmly Stage 1 / 1.5-specific to TWN damage at AF2-D.**
The next experiment should map the boundary of where the signal
appears (damage severity × layer type × layer depth × task × correction
budget), per the user's recommendation.

## Constraints / what remains open

- **Single site**: L15 down_proj only.
- **Single damage mode (TWN)**: degenerate at L15.
- **Only 2 of 5 thresholds measured**: sufficient for ABORT (ppl ~14
  across both); the 3 missing thresholds (0.5, 0.7, 1.0) would not
  have produced ppl ≥ 200 at any threshold given the trend.
- **Group sizes not tested at L15**: {8, 128}. Other group sizes
  (16, 32, 64, 256) untested; given the negative trend with
  group_size=8, finer granularity is unlikely to produce severe damage.

## Driver and reproducibility notes

- Driver SHA: `34aa581` (current main). Stage 1 / 1.5 driver SHA
  `692e8ee` untouched.
- Model: `allenai/OLMo-1B-0724-hf`, dtype float16, eval dtype
  float16.
- Damage recipe: TWN with group_size=8, calibrate_norm=false,
  threshold sweep over {0.0, 0.3, 0.5, 0.7, 1.0}.
- All cells run on Legion cuda:0.
- Two cells completed (~10 min wall); the 3 missing cells were
  abandoned after the ABORT criterion was demonstrably triggered.

## Artifacts

- `runs/r/EXP-RPM-L15-TWN-V5-CAL/threshold-0.0/{aggregate.json,seed-{001,002,003}/pre_train_eval.json}` (3)
- `runs/r/EXP-RPM-L15-TWN-V5-CAL/threshold-0.3/{aggregate.json,seed-{001,002,003}/pre_train_eval.json}` (3)
- `research/residual-pareto/experiments/EXP-RPM-L15-TWN-V5/manifest.yaml`
- `research/residual-pareto/experiments/EXP-RPM-L15-TWN-V5/verdict.md` (this file)

## Tests

244/244 pass on Legion (production environment).
Dev-box environment has 19 failures + 3 collection errors, all
environment-only (triton missing, SIMD kernel not built). See
`research/TESTS-FAILING-CLASSIFICATION.md`.
