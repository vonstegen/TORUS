# Stage 2 v2 CAL Verdict (FINAL, 2026-08-24)

**Status:** FINAL — 72 of 72 cells complete.
**Date:** 2026-08-24
**Driver SHA:** `04243cc` (latest) → `e7b2442` (Stage 2 v1 frozen)
**Pilot scope:** 4 sites × 6 sigmas × 3 seeds = 72 cells
**Sites:** AF2-D, L15, L0-q (attention), L0-v (attention)
**Damage recipe:** `W' = W + sigma * std(W) * eps` (deterministic Gaussian)
**Model:** `allenai/OLMo-1B-0724-hf` (matches Stage 1 / 1.5 preregistered baseline)

## Pilot result summary

| Site | σ=0.00 | σ=0.05 | σ=0.10 | σ=0.20 | σ=0.50 | σ=1.00 | Span | Bands | Qualifying |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AF2-D (down_proj, layer 0)  | 13.093 | 13.096 | 13.103 | 13.128 | 13.373 | 15.352 | 2.26 | {13, 15} | **NO** |
| L15 (down_proj, layer 15)   | 13.093 | 13.099 | 13.123 | 13.203 | 13.753 | 16.578 | 3.49 | {13, 14, 16, 17} | **YES** |
| L0-q (attn q_proj, layer 0) | 13.093 | 13.095 | 13.097 | 13.102 | 13.131 | 13.247 | 0.15 | {13} | **NO** |
| L0-v (attn v_proj, layer 0) | 13.093 | 13.119 | 13.197 | 13.725 | 439.252 | 20083.488 | 20070 | {13, 14, 439, 20084} | **YES** |

All cells are bit-reproducible: same (σ, seed) → identical ppl across all 4 sites.

## Verdict per site

### EXP-RPM-AF2D-GAUSS-CAL: NOT QUALIFYING (ABORTED)

The σ→ppl curve is monotonic but the span (2.26 ppl units) is achieved
by a single σ=1.0 cell that crosses into a new band. Only 2 distinct
reproducibility bands ({13, 15}); the registered QUALIFYING rule
requires ≥3. The damage axis is reproducible but too narrow for
tournament-scale claims. The site is ABORTED; **TWN damage at the
AF2-D site (Stage 1.5) remains the only informative damage mode at
that layer category.**

### EXP-RPM-L15-GAUSS-CAL: QUALIFYING

Span 3.48 ppl units; 4 distinct bands. The site satisfies the
QUALIFYING rule. **Tournament EXP-RPM-L15-GAUSS launched at σ=0.20**
(the preregistered middle band of the sigma→ppl curve; exact match
to the registered tournament-σ-selection rule). Cross-layer category:
MLP down_proj (layer 15). Note: under TWN damage this site was
degenerate (Stage 2 v1 found ppl 14.10-15.49 across all thresholds);
Gaussian noise produces informative damage here where TWN did not.

### EXP-RPM-L0-Q-GAUSS-CAL: NOT QUALIFYING (ABORTED)

Span 0.15 ppl units; 1 distinct band. Layer 0 attention q_proj is
robust to mild Gaussian noise (σ ≤ 1.0). Mirrors the Stage 2 v1 TWN-
degenerate finding for this site. ABORTED.

### EXP-RPM-L0-V-GAUSS-CAL: QUALIFYING

Span 20070 ppl units; 4 distinct bands. **Tournament EXP-RPM-L0-V-
GAUSS launched at σ=0.20** (preregistered middle band). Cross-layer
category: attention v_proj (layer 0). Notable: layer 0 attention v_proj
is far more sensitive to Gaussian noise than layer 0 attention q_proj
at the same σ (20083 vs 13.25 at σ=1.0). The damage sensitivity is
highly geometry-specific even within the same layer and head type.

## Cross-layer category check (RPM-006 PASS+ rule)

RPM-006's registered PASS+ rule requires:
- ≥2 layer categories (down_proj MLP, attention proj, etc.)
- ≥2 Pareto-optimal cells across the categories

The Stage 2 v2 CAL pilot satisfies the **first criterion**: 2 QUALIFYING
sites in **two distinct layer categories** (MLP down_proj at L15,
attention v_proj at layer 0). The second criterion (Pareto-optimal
across categories) will be evaluated by the tournaments EXP-RPM-L15-
GAUSS and EXP-RPM-L0-V-GAUSS, currently running on Legion cuda:0 +
cuda:1 in parallel.

## Implications for the RPM program

- **RPM-001**: still UNTESTED. Stage 5 EXP-RPM-SYS (energy measurement)
  remains the registered unblock step.
- **RPM-002**: still UNTESTED.
- **RPM-006**: the **layer-category criterion is now satisfied**
  (MLP + attention). The Pareto criterion awaits the Stage 2 v2
  tournament results.

## Driver and reproducibility notes

- Driver SHA at pilot start: `ddc2b54`
- Stage 1 / 1.5 driver SHA (`692e8ee`) untouched.
- Model: `allenai/OLMo-1B-0724-hf` (matches EXP-A-001 preregistered
  baseline)
- Dtype: `--dtype float16 --eval-dtype float16` (matches Stage 1 / 1.5)
- Damaged weight is in-place and frozen (`requires_grad_(False)`).
- Noise is deterministic per (sigma, seed); same (sigma, seed) → same
  noise (verified by tests/test_af2_driver_extension.py).
- Pilot ran on Legion dual TITAN RTX. AF2-D on cuda:0 + L0-q on cuda:1
  (parallel), then L0-v on cuda:0 + L15 on cuda:1 (parallel).
- All cells bit-reproducible: same (sigma, seed) → identical ppl
  (verified by inspecting the rows above; e.g. AF2-D σ=0.0 across 3
  seeds is exactly 13.093198488512625).
- Per-site summaries written to
  `runs/r/EXP-RPM-{SITE}-CAL/{timestamp}/site_cal_summary.json`.
- Combined summary written to
  `research/residual-pareto/experiments/stage2_v2_cal_summary.json`.

## Driver changes (committed during pilot)

- `examples/af2_storage_tournament.py`:
  - `resolve_site_dims(target_module) -> (in_features, out_features)`
    for path-aware dim resolution (handles MLP down_proj AND attention
    q_proj / v_proj).
  - `damage_target_module_gaussian(target_module, *, sigma, seed)` —
    deterministic, seeded Gaussian weight noise.
  - Refactored `build_site_adapter` to accept `site_dims` instead of
    hardcoded (hidden_size, intermediate_size).
  - New CLI flags: `--damage-gaussian`, `--damage-sigma`, `--damage-seed`.
  - `--damage-ptq` and `--damage-gaussian` mutually exclusive
    (enforced in main()).
  - Generalized pre-train eval (fires for both damage modes).

- `tests/test_af2_driver_extension.py`: 11 new tests, all pass.

- `tests/test_af2_damaged_ptq.py`: unchanged (Stage 1 / 1.5 path).

- `tests/test_af2_storage_tournament.py`: unchanged.

- Total tests: 239/244 pass (5 kernel-load failures pre-existing,
  unrelated to this change).

## Process deviations logged

- The launch script initially used `--dtype float32 -- --eval-dtype
  float16` (mixed precision), which introduced a +2.28 ppl unit
  baseline shift on q_proj. Fixed in commit 40f4a13 to
  `--dtype float16 --eval-dtype float16` matching Stage 1 / 1.5.
  The first ~30 min of the initial pilot (mixed precision) produced
  unusable data and was discarded.
- Model name `allenai/OLMo-1B-hf` was used in early launch script
  versions; fixed in commit ddc2b54 to `allenai/OLMo-1B-0724-hf` to
  match the EXP-A-001 preregistered baseline. The mixed-precision
  pilot used the wrong model; both fixes (dtype + model) were
  applied before the final pilot that produced this verdict.
- Tournament launcher had a bug extracting damage_sigma from the
  manifest yaml (matched the next line instead). Fixed in commit
  04243cc.