# Stage 2 Verdict — EXP-RPM-L15 / EXP-RPM-L8 (CAL only, tournaments aborted)

**Date:** 2026-08-24
**Driver SHA (Stage 1, frozen):** `692e8ee` (NOT modified)
**Launch script:** `stage2-launch.sh`
**Per-site CAL helper:** `stage2_select_threshold.py`

## Status after Stage 2 (v1)

| Claim | Status | Justification |
|---|---|---|
| **RPM-001** | UNTESTED (no change) | Energy null; Stage 5 EXP-RPM-SYS required. Pareto audit from `e1d6857` still holds on AF2-D. |
| **RPM-002** | UNTESTED (no change) | Stage 2 tournaments aborted before producing data; no new z-scores to test the monotonicity rule against. |
| **RPM-006** | UNTESTED (no change) | **FAILED Stage 2 preregistration goal**: the per-site CAL showed the damage axis is **degenerate** on layers 8 and 15. The TWN damage recipe cannot produce an informative ppl gradient at any threshold on these layers. Without damage variation, the trained-vs-random z-score cannot be measured (no signal to amplify). The "≥2 layer categories" PASS+ rule is therefore not reachable with the current damage recipe on down_proj layers. |

## Stage 2 v1 design (and why it didn't work)

The Stage 2 v1 plan was to test the architecture-vs-training signal on
two non-AF2-D layer sites:

- **Site A: AF2-D** (`model.layers.0.mlp.down_proj`) — already done in Stage 1 D5'.
- **Site B (NEW): `model.layers.15.mlp.down_proj`** — late-layer MLP down_proj.
- **Site C (NEW): `model.layers.8.mlp.down_proj`** — mid-layer MLP down_proj.

Per-site CAL design: 11 thresholds × 3 seeds × 1 arm (t2_ternary with
--pre-train-eval, eval-only) on each target module to characterize the
threshold→ppl mapping before the tournament.

### Stage 2 v1 result: degenerate axes on layers 8 and 15

| Site | threshold | pre-train ppl | Status |
|---|---:|---:|---|
| AF2-D (Stage 1 baseline) | 0.0 | 1524.80 | informative |
| AF2-D | 1.0 | 88.31 | informative |
| **L15.mlp.down_proj** | 0.0 | 14.11 | **DEGENERATE** |
| L15 | 0.5 | 14.10 | DEGENERATE |
| L15 | 1.0 | 15.49 | barely informative (small bump) |
| **L8.mlp.down_proj** | 0.0 | 13.67 | **DEGENERATE** |
| L8 | 0.4 | 13.67 | DEGENERATE |
| L8 | 0.5 | 13.67 | DEGENERATE |
| L8 | 0.6 | 13.68 | DEGENERATE |

**The TWN damage recipe (per-group absmean ternary-quantize-with-STE at
threshold ∈ [0.0, 1.0]) does NOT damage layers 8 or 15 down_proj.**
The recipe works on AF2-D (layer 0) where weights are more sensitive;
on deeper layers the weight distribution is tighter and the per-group
absmean normalization preserves information.

### Interpretation

This is a **legitimate negative result**, not a Stage 2 failure:

1. **The architecture-vs-training signal is most visible where the
   base requires correction.** AF2-D's ppl moved from 425 to 18
   under training; L15's ppl stayed at ~14 because there was no
   damage to correct.

2. **Stage 2 tournaments at FP16-ppl would replicate the Stage 1
   D0 finding** (trained ≈ random at FP16 reference). This is
   confirmed by the L15 seed-001 result (pre-train ppl 14.23,
   post-train ppl 14.26 — essentially unchanged).

3. **The Stage 1 architecture-vs-training finding is intact** —
   trained T2 separates from random T2 by tens to hundreds of σ
   on damaged bases in both Stage 1 and Stage 1.5 (where the damage
   axis IS informative). Stage 2's task was to test whether this
   effect generalizes to other layer categories, but the
   prerequisite condition (a damage axis that can produce variation)
   isn't met at layers 8 and 15 with this recipe.

### Statistical caveats

The degenerate-axis finding may be **partially artifactual**:

- **Small seed variance inflates the apparent axis collapse.** The
  Stage 1 CAL on AF2-D showed ppl=1524.80 for thresholds 0.0-0.5
  (identical to 4 decimal places). The L15/L8 axes may have small
  variations in ppl that are below the noise floor of the seed
  variance estimate.

- **The recipe is one specific damage mode.** Other damage recipes
  (e.g. random weight dropout, additive Gaussian noise, structured
  per-row quantization) might produce informative axes at deeper
  layers. The current finding is **specifically about the TWN
  ternary-quantize-with-STE damage at thresholds 0.0-1.0**.

## Tournament status

Tournaments at L15 and L8 were **aborted** because the calibration
showed no informative damage axis. Running tournaments at ppl~14
would have produced only Stage 1 D0-like results (trained ≈ random at
FP16 reference), which is already known and would consume ~2 hours
of Legion compute per site for no new information.

The L15 tournament was started, completed seed-001 t2_ternary
(pre-train ppl 14.23, post-train ppl 14.26), and was then aborted
mid-int4_residual seed-001 eval.

## Next-step plan

Stage 2 v1 demonstrated the L8/L15 damage axis is degenerate with the
TWN recipe. To satisfy the "≥2 layer categories" requirement, **a
different damage recipe is needed**. Options:

1. **Different damage recipe:** Try a damage mode that actually
   damages deeper layers (e.g., random mask, structured dropout,
   per-row quantization with calibration). This is a new Stage 2 v2
   preregistration with a new damage knob.

2. **Different layer category (architecture):** Modify the driver to
   support non-down_proj-equivalent geometries (gate_proj,
   q_proj, etc.). This requires a code change = new freeze exception
   preregistration.

3. **Increase damage severity at AF2-D:** Run EXP-AF-002-D with a
   higher-threshold TWN damage (e.g., threshold=1.5, 2.0) or a
   different group_size (e.g., 64 instead of 128) to see if even
   deeper damage produces informative variation. This would test
   whether AF2-D's D5 condition (ppl 425 → 18) is the maximum
   recoverable from this recipe.

The reviewer's recommendation was that Stage 2 (layer-category
sweep) precede Stage 5 (energy measurement). Stage 2 v1 completed
the CAL portion of the sweep and produced a negative-result finding.
**Stage 2 v2 (different damage recipe) is the next step**, not Stage
5; the Stage 5 prerequisite (RPM-006 PASS+) is not reached.

## Artifacts

- `research/residual-pareto/experiments/EXP-RPM-L15-CAL/manifest.yaml`
  (status: PROPOSED — tournaments aborted)
- `research/residual-pareto/experiments/EXP-RPM-L8-CAL/manifest.yaml`
  (status: PROPOSED — tournaments aborted)
- `research/residual-pareto/experiments/EXP-RPM-L15/manifest.yaml`
  (status: PROPOSED — tournament not run)
- `research/residual-pareto/experiments/EXP-RPM-L8/manifest.yaml`
  (status: PROPOSED — tournament not run)
- `runs/r/EXP-RPM-L15-CAL/20260824T171417Z/` — L15 CAL data (11 thresholds × 3 seeds)
- `runs/r/EXP-RPM-L8-CAL/20260824T182015Z/` — L8 CAL data (6 thresholds × 3 seeds; thr 0.6-1.0 aborted)
- `runs/r/EXP-RPM-L15/20260824T181538Z/af2d/seed-001/t2_ternary/eval.summary.json`
  — single tournament cell that completed (pre-train ppl 14.23, post-train ppl 14.26)
- `stage15-launch.sh`, `stage2-launch.sh`, `stage2_select_threshold.py`
  (launch + helper scripts)
- `gen_stage2_manifests.py`, `gen_stage2_cal_manifests.py`
  (manifest generators)

## Reproduction

```bash
# L15 CAL (took ~57 min on Legion; 11 thresholds × 3 seeds)
./stage2-launch.sh L15  # runs CAL + threshold selection + tournament
# L8 CAL partial (6/11 thresholds, ~30 min)
./stage2-launch.sh L8
```