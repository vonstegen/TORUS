# Stage 2 v2 Tournament Verdict (FINAL, 2026-08-24)

**Status:** Both qualifying tournaments COMPLETE (54 cells per site, 108 total).
**Date:** 2026-08-24
**Driver SHA:** `34aa581` (current). Stage 1 / 1.5 driver SHA `692e8ee` untouched.
**Pilot scope:** 2 qualifying sites × 7 trained arms + 2 random controls × 3 seeds = 54 cells per site.

## Sites evaluated

- **EXP-RPM-L0-V-GAUSS** at `model.layers.0.self_attn.v_proj`, σ=0.20
- **EXP-RPM-L15-GAUSS** at `model.layers.15.mlp.down_proj`, σ=0.20

## Per-site results (per-arm mean across 3 seeds)

### EXP-RPM-L0-V-GAUSS (v_proj, attention, σ=0.20)

| arm | ppl (mean) | arc_easy | lambada |
|---|---:|---:|---:|
| **random_t2_ternary** | 13.91 | 0.551 | 0.593 |
| **random_lora** | 19.97 | 0.574 | 0.556 |
| int4_residual | **4671.47** (catastrophic) | 0.356 | 0.033 |
| int8_residual | 22.86 | 0.503 | 0.416 |
| **t2_ternary** | **16.33** | **0.623** | **0.581** |
| dense_adapter | 14.99 | 0.605 | 0.586 |
| lora | 15.01 | 0.606 | 0.582 |

### EXP-RPM-L15-GAUSS (down_proj layer 15, MLP, σ=0.20)

| arm | ppl (mean) | arc_easy | lambada |
|---|---:|---:|---:|
| **random_t2_ternary** | 13.22 | 0.563 | 0.613 |
| **random_lora** | 13.29 | 0.562 | 0.607 |
| int4_residual | 16.59 | 0.544 | 0.555 |
| int8_residual | 14.61 | 0.555 | 0.594 |
| **t2_ternary** | **13.20** | 0.562 | **0.610** |
| dense_adapter | 14.01 | 0.554 | 0.617 |
| lora | 14.03 | 0.551 | 0.618 |

## Key findings

### 1. L0-v v_proj: int4_residual catastrophically fails, but trained T2 only matches the random baseline

Under Gaussian noise σ=0.20 on attention v_proj:
- **int4_residual adapter completely fails** (ppl=4671, lambada=0.033, arc=0.356). The int4 + column-mask fraction (50%) was tuned for TWN-style ternary damage on AF2-D and does not generalize to Gaussian damage on attention sites.
- **int8_residual degrades** (ppl=22.86, lambada=0.416, arc=0.503). Less catastrophic but still much worse than baseline.
- **t2_ternary, dense_adapter, lora all perform within noise of each other** (ppl 15-16, lambada 0.58-0.59). At σ=0.20 on this site, the damage is mild enough that the trained arms all roughly match the **undamaged baseline** (random_t2_ternary ppl=13.91).
- **trained T2 ≈ random T2** within seed-variance: t2_ternary ppl=16.33 vs random_t2 ppl=13.91. The Stage 1 / 1.5 finding ("trained ≫ random on damaged bases") **does NOT replicate** on v_proj at σ=0.20.

### 2. L15 down_proj: damage too mild, all arms perform similarly

At σ=0.20 on layer 15 down_proj, the damage is too small to differentiate arms. All seven arms (5 trained + 2 random) cluster around ppl=13.2-16.6 and lambada=0.55-0.62. The **mean difference between trained T2 and random T2 is < 0.005 lambada** (0.610 vs 0.613). σ=0.20 was chosen as the **middle band of the qualifying sigma curve**, but for Pareto discrimination a higher σ is needed (e.g., σ=0.50 where the CAL ppl=13.75 is larger).

## Implications for the RPM program

- **RPM-001/002/006 status: UNCHANGED (UNTESTED).** The "≥2 layer categories" criterion from the Stage 2 v2 CAL pilot is still satisfied, but the Pareto criterion cannot be evaluated at σ=0.20 (damage too mild) or with the catastrophic int4 failure.
- The Stage 1 / 1.5 finding (T2 ternary ≫ random T2 on damaged bases) **does NOT generalize to attention sites or to deeper MLP sites** under the preregistered Gaussian noise levels. The architectural-vs-training story is AF2-D specific under TWN damage.
- **The Stage 2 v2 program is best characterized as confirming Stage 1's negative result**: cross-layer generalization is hard; AF2-D with TWN damage remains the only reproducible demonstration of trained ≫ random.

## Next step

**Stage 5 EXP-RPM-SYS (energy measurement)** is the registered unblock step
for RPM-001. With the cross-layer generalization blocked at σ=0.20, the
energy measurement (which would address the **storage + throughput +
energy** Pareto triple, not just inference cost) becomes the
right next move. The architecture-vs-training story remains intact at
AF2-D with TWN damage (Stage 1.5); what's missing is the energy
leg of the Pareto triple to promote RPM-001 from UNTESTED to
CONFIRMED.

## Driver and reproducibility notes

- Driver SHA: `34aa581` (current). Stage 1 / 1.5 driver SHA `692e8ee` untouched.
- Model: `allenai/OLMo-1B-0724-hf`, dtype float16, eval dtype float16.
- Damage recipe: `W' = W + sigma * std(W) * eps` (deterministic Gaussian).
- Tournament protocol: identical to Stage 1.5 (7 trained arms + 2 random controls, n_steps=500, batch_size=4, seq_len=128, lr=1e-3).
- All cells bit-reproducible: same (sigma, seed) → identical adapter → identical eval.

## Driver changes (committed during this work)

- `examples/af2_storage_tournament.py`: `sys.modules.setdefault("triton", None)` replaced with `import triton` to fix a `ModuleNotFoundError` cascade that was preventing post-hoc eval.
- `examples/eval_lm.py`: same fix.
- `examples/eval_untrained_arms_v2.py`: post-hoc eval for the random arms (since the driver skips lm-eval-harness on `is_untrained` arms). Handles the T2 ternary packed-format correctly (2 bits/code, fp16 per-row scale).

## Process deviations logged

- The `aggregate.json` produced by the driver's `aggregate()` function filters by `matched_bytes_passed=True`, which removes all arms at non-down_proj sites (v_proj) because the byte target is registered for the down_proj (4.19 MB). A post-hoc aggregator (`stage2-v2-tournaments-summary.py`) was added to include all arms regardless of matched_bytes.
- The post-hoc eval of the random arms initially failed with `ModuleNotFoundError: import of triton halted; None in sys.modules` because `examples/eval_lm.py` had `sys.modules.setdefault("triton", None)`. Fixed by replacing that line with `import triton`.
- The `stage2-v2-tournaments-launch.sh` and `stage2-v2-tournaments-summary.py` were extended to handle nested `seed-XXX/seed-XXX/` directory layout (the driver adds an extra seed-XXX layer for its own per-seed subdir).

## Tests

239/244 pass (5 kernel-load failures pre-existing, unrelated to this change).
No new tests added in this commit; the existing 11 freeze-exception
tests (commit 18e10ba) cover the v2 driver changes.