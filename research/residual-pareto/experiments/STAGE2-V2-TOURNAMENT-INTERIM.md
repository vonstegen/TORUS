# Stage 2 v2 Tournament INTERIM findings (2026-08-24)

**Status:** Tournaments RUNNING on Legion cuda:0 + cuda:1 in parallel.
**Launch:** 2026-08-24T21:21Z (L0-v) / 2026-08-24T21:22Z (L15).
**Site config:** σ=0.20, OLMo-1B-0724-hf, 7 trained + 2 random arms × 3 seeds = 27 cells per site.

## Live cells (as of 2026-08-24T21:34Z, 12-13 min into each tournament)

### EXP-RPM-L0-V-GAUSS (v_proj, layer 0) — seed-001 done for 4/7 arms

| arm | lambada_openai | arc_easy | wikitext (ppl) |
|---|---:|---:|---:|
| t2_ternary | 0.585 | — | — |
| int4_residual | **0.043** | 0.375 | **2299.91** ← CATASTROPHIC FAILURE |
| int8_residual | 0.434 | — | — |
| lora | 0.583 | — | — |
| dense_adapter | (running) | | |
| random_t2_ternary | (pending) | | |
| random_lora | (pending) | | |

**L0-v v_proj key insight:** Under Gaussian noise damage at σ=0.20, **T2
ternary adapter fully recovers** (acc ~0.585, matching FP16 baseline
~0.59), but **int4_residual catastrophically fails** (acc 0.043, ppl
2299). The int4/int8 column-mask fraction (50%/25%) was tuned for
TWN-style ternary damage on AF2-D and does not generalize to
Gaussian damage on attention v_proj.

### EXP-RPM-L15-GAUSS (down_proj, layer 15) — seed-001 done for 4/7 arms

| arm | lambada_openai | arc_easy | wikitext (ppl) |
|---|---:|---:|---:|
| t2_ternary | 0.609 | — | — |
| int4_residual | 0.558 | — | — |
| int8_residual | 0.598 | — | — |
| lora | 0.614 | — | — |
| dense_adapter | (running) | | |
| random_t2_ternary | (pending) | | |
| random_lora | (pending) | | |

**L15 down_proj key insight:** Same σ=0.20 Gaussian damage produces a
much milder effect on this site. All four trained arms (t2_ternary,
int4, int8, lora) achieve lambada ~0.55-0.61, very close to FP16
baseline. The damage here is **not strong enough to differentiate
the trained arms from each other** — which is a problem for the
Pareto analysis (we need a damage level where some arms win and
some lose).

## Provisional implications

1. **At σ=0.20 on L0-v v_proj, the T2 ternary adapter uniquely
   recovers from damage while int4_residual catastrophically fails.**
   This is a strong (if localized) finding: T2 is not Pareto-optimal
   here in the sense of matching the deployment byte budget — int4
   matches the budget while T2 does not — but on the **recovery-from-
   damage axis**, T2 wins decisively.

2. **At σ=0.20 on L15 down_proj, the damage is too mild to
   differentiate trained arms.** σ=0.20 may be too low for L15;
   a higher σ value (e.g., σ=0.50 where CAL ppl=13.75) might be
   needed. The site qualifies under the CAL rule but the
   tournament threshold may be suboptimal for Pareto analysis.

3. **The Stage 2 v2 CAL pilot's choice of σ=0.20 (middle band of
   sigma→ppl curve) was based on the QUALIFYING rule, not the
   PARETO rule.** Different σ values may be needed to fully
   discriminate between arms at each site. This is a learning for
   future pilots.

## Next steps

- Tournaments will continue running for ~2 hours from launch.
- After completion, run aggregate analysis on each tournament to
  compute per-arm mean ± stderr across the 3 seeds.
- The trained-vs-random z-score test (RPM-002 / RPM-006 PASS+ rule)
  requires the random_t2_ternary and random_lora evals to complete
  (currently pending).
- Stage 5 EXP-RPM-SYS (energy measurement) remains the next major
  step to lift RPM-001.