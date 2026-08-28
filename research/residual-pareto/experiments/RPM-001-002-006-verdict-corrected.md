# RPM-001 / RPM-002 / RPM-006 Verdict — Stage 1 + Stage 1.5 post-hoc eval

**Date:** 2026-08-24
**Driver SHA (Stage 1, frozen):** `692e8ee`
**Post-hoc eval scripts:** `examples/eval_untrained_arms.py`,
`research/residual-pareto/experiments/fix_metric.py`
**Analysis:** `research/residual-pareto/experiments/analyze_stage15.py`
**Pareto audit:** `research/residual-pareto/experiments/pareto_audit.py`
**Inputs:** per-(regime, seed, arm) `eval.summary.json` from Stage 1
(EXP-RPM-D{0..5}) and Stage 1.5 (EXP-RPM-D{0..5}p) runs + post-hoc
random-arm evals.

## Status after review (CORRECTED)

| Claim | Status | Justification |
|---|---|---|
| **RPM-001** | UNTESTED (tentative PASS) | Energy-per-token (E) null across all arms; verdict becomes CONFIRMED only when E is measured (Stage 5 EXP-RPM-SYS). T2 IS NOT dominated on the joint (3 cap × 5 cost B/F/O/M/L) vector at any regime (Pareto audit); but this is necessary not sufficient for the registered PASS threshold (also requires ≥1 (regime, layer, task, budget) point + global-comparator-rule satisfied). |
| **RPM-002** | **UNTESTED** (was incorrectly marked DECIDED PASS in earlier verdict) | The registered PASS threshold requires ≥3 consecutive damage regimes (in regime order) where the trained-vs-random effect size is non-decreasing. Computed z-score sequences (Stage 1 arc_easy: 116, 59, 66, 22, 64; Stage 1 lambada: 164, 253, 169, 78, 237; Stage 1.5 arc: 19, 48, 40, 20, 28; Stage 1.5 lambada: 155, 79, 262, 113, 62) contain NO 3-consecutive non-decreasing subsequence in regime order. The FAIL clause (non-increasing across all consecutive pairs) is also not met. The correct status is UNTESTED. **The narrower empirical finding (trained T2 separates from random T2 at every damaged regime in both stages) IS supported but does not satisfy the registered monotonicity rule.** |
| **RPM-006** | **UNTESTED (provisional; layer-sweep + boundary-reproduction evidence required)** | The registered PASS rule requires (a) ≥2σ separation at an identified activation boundary D*, (b) no material regression on other registered metrics, (c) reproduction under clean reruns, (d) ≥2 layer categories. We have (a) at every damaged regime (no specific D* identified) but lack (c) clean reruns reproducing the boundary and (d) layer-category generalization. The healthy-base "indistinguishability" clause is also violated in the data: D0 arc_easy is -5.43σ, D0' lambada is -3.40σ (small absolute differences amplified by low seed variance — but literal indistinguishability does not hold). |

## Empirical finding that IS supported

**Trained T2 decisively separates from random T2 on every tested damaged regime in both Stage 1 (threshold axis) and Stage 1.5 (CAL-calibrated observed-ppl axis).**

| Stage | Regime | arc_easy z | lambada z | wikitext ppl z |
|---|---|---|---|---|
| Stage 1 | D1 (threshold=0.0) | +116 | +164 | -1094 |
| Stage 1 | D2 (threshold=0.3) | +59 | +253 | -1683 |
| Stage 1 | D3 (threshold=0.5) | +66 | +169 | -580 |
| Stage 1 | D4 (threshold=0.6) | +22 | +78 | -248 |
| Stage 1 | D5 (threshold=0.7) | +64 | +237 | -1079 |
| Stage 1.5 | D1' (threshold=1.0) | +19 | +155 | -465 |
| Stage 1.5 | D2' (threshold=0.9) | +48 | +79 | -380 |
| Stage 1.5 | D3' (threshold=0.8) | +40 | +262 | -2450 |
| Stage 1.5 | D4' (threshold=0.7) | +20 | +113 | -356 |
| Stage 1.5 | D5' (threshold=0.6) | +28 | +62 | -589 |

All 10 damaged regimes show trained > random by ≥19σ on arc_easy and
≥62σ on lambada_openai. Wikitext ppl is reduced from ~600-1500 (random)
to ~17-25 (trained).

**Qualitative finding:** on every tested damaged base, the trained T2
ternary correction plane carries substantial learnable information that
the architecture alone (random T2) does not.

## Statistical caveats

- **Small seed variance amplifies ratios.** With 3 seeds and
  near-deterministic eval, the trained-vs-random standard error can
  be tiny (e.g. 0.0009 ppl on arc_easy for D1's trained arm). The
  reported z-scores are computed under the assumption that the
  seed-to-seed variance estimates the relevant uncertainty, but with
  n=3 the distribution of the test statistic is not Gaussian.
  The qualitative finding (trained ≫ random by tens to hundreds of
  effect-size units) is robust to this caveat.

- **No multiple-comparison treatment applied.** The 10 damaged-regime
  × 3-metric tests have not been adjusted. The qualitative finding
  is consistent across all 30 cells (all positive in the trained >
  random direction).

- **Stage 1 and Stage 1.5 are not fully independent replications.**
  They reuse the same model (OLMo-1B), target layer (AF2-D),
  training recipe, driver, evaluation suite, and related calibration
  evidence. The damage-axis parameterization differs (threshold knob
  vs CAL-calibrated ppl), but the underlying experimental setup is
  shared.

## Pareto dominance audit (RPM-001 evidence)

`pareto_audit.py` checks every arm's 6-dim cost vector (B/F/O/M/L/E)
against T2's. Result across all 12 regimes (D0..D5 + D0'..D5'):

| Arm | deployed_bytes | Notes |
|---|---|---|
| T2 (t2_ternary) | 4,199,318 | reference |
| int4_residual | 4,197,020 | slightly smaller bytes |
| int8_residual | 4,195,994 | slightly smaller bytes |
| lora | 4,424,265 | larger bytes |
| dense_adapter | 3,932,771 | **smallest bytes** |

**T2 is NOT dominated on the joint (3 cap × 5 cost) vector in any
regime.** Despite dense_adapter having ~6% fewer deployed_bytes than T2,
its ppl is much worse (e.g. dense D1 ppl 43.97 vs T2 D1 ppl 24.14) and
its capability metrics are not consistently better than T2's. T2's
relative advantage: capability (arc_easy, lambada_openai) per byte
exceeds dense's. No arm achieves better-or-equal on every dimension.

**Correction to previous verdict:** "T2 wins on storage" was misleading.
T2 has the **second-smallest** deployed_bytes; dense_adapter is
smallest. T2 is on the Pareto frontier of (3 cap × 5 cost B/F/O/M/L)
because no comparator simultaneously beats T2 on capability AND cost.

## What is required for next-claim promotion

### RPM-002 (registered)
- Either: identify 3 consecutive damage regimes (in regime order) with
  non-decreasing effect size on any single registered metric (arc_easy,
  lambada_openai, OR ppl with appropriate sign convention).
- Or: revise the claim definition. Per OPERATING-PLAN §3, claim
  definitions are NOT altered to fit available evidence.
- The empirical pattern (large trained-vs-random separation that does
  not grow monotonically with damage severity) is **incompatible with
  the registered monotonicity PASS rule as written**.

### RPM-006 (registered)
- Identify a specific activation boundary D* (claimed to be in D1-D3).
  The current data does not point to a single transition threshold;
  trained > random at every damaged regime and approximately equal to
  random at the FP16 reference.
- Reproduce the boundary under clean reruns (AF8 governance).
- Demonstrate the same activation in ≥2 layer categories (Stage 2
  EXP-RPM-Lxx preregistration).
- Address the healthy-base "indistinguishability" clause — D0 arc_easy
  -5.43σ and D0' lambada -3.40σ are statistically distinguishable
  from zero (small absolute but ≠ 0).

### RPM-001 (registered)
- Stage 5 EXP-RPM-SYS: measure per-token energy on Legion. Required
  before RPM-001 can become CONFIRMED (not tentative).
- Stage 2 EXP-RPM-Lxx: layer-category evidence. Required for PASS+
  (which lifts the "core TORUS representation" status).
- Stage 3 EXP-RPM-B1..B5: budget-sweep evidence. Required for PASS+.
- Stage 4 EXP-RPM-Txx: task-robustness evidence. Required for PASS+.

## Next-step plan

Per the reviewer recommendation, the preregistration order should be:

1. **Stage 2 EXP-RPM-Lxx** — ≥2 genuinely distinct layer categories
   (currently only AF2-D / down_proj is tested). This is the immediate
   scientific gate because:
   - Required evidence for RPM-006 PASS+ (≥2 layer categories).
   - Required evidence for Track B B1 unlock (per A-RP-002's
     unlock_rules_affected).
   - Distinguishes a broadly architectural effect from one localized
     to a favorable layer.

2. **Stage 5 EXP-RPM-SYS** — measure per-token energy; required for
   RPM-001 CONFIRMED.

3. **Stage 3 EXP-RPM-B1..B5 + Stage 4 EXP-RPM-Txx** — budget and task
   robustness, required for RPM-001 PASS+.

Before launching Stage 2, the Stage 2 manifest should:

- Specify ≥2 genuinely distinct layer categories (e.g. `down_proj`
  + `attention_k_proj` + `gate_proj`).
- Freeze the metric keys (`acc_norm,none` for arc_easy, etc.) BEFORE
  running.
- Specify the ppl sign convention explicitly (lower ppl = better effect
  = |z| on ppl axis is what should be tested).
- Specify the registered monotonicity test (which metric, which sign
  convention, what ε for "non-decreasing").
- Document values before any post-hoc corrections.

## Artifacts

- `research/residual-pareto/experiments/RPM-001-002-006-analysis-combined.md`
- `research/residual-pareto/experiments/RPM-001-002-006-analysis-combined.json`
- `research/residual-pareto/experiments/RPM-001-002-006-analysis-15.md`
- `research/residual-pareto/experiments/RPM-001-002-006-analysis-15.json`
- `research/residual-pareto/experiments/RPM-001-002-006-analysis.md`
- `research/residual-pareto/experiments/RPM-001-002-006-analysis.json`
- `research/residual-pareto/experiments/RPM-001-002-006-verdict.md`
  (original; superseded by this corrected verdict)
- `research/residual-pareto/experiments/RPM-001-002-006-verdict-15.md`
  (Stage 1.5 verdict; superseded by this corrected verdict)
- `research/residual-pareto/experiments/analyze_stage1.py`
- `research/residual-pareto/experiments/analyze_stage15.py`
- `research/residual-pareto/experiments/fix_metric.py`
- `research/residual-pareto/experiments/pareto_audit.py`
- `research/residual-pareto/experiments/rpm002_registered_test.py`
- `examples/eval_untrained_arms.py`
- `examples/eval_untrained_arms.py` post-fix metric picker override.

## Reproduction

```bash
# Stage 1 + Stage 1.5 launch (already executed; data on Legion)
./rpm-d-launch.sh         # Stage 1: EXP-RPM-D{0..5}
./stage15-launch.sh      # Stage 1.5: EXP-RPM-D{0..5}p

# Post-hoc random-arm eval (already executed; data on Legion)
PYTHONPATH=. .venv/bin/python examples/eval_untrained_arms.py \
    --regimes 0,1,2,3,4,5,0p,1p,2p,3p,4p,5p \
    --arms random_t2_ternary,random_lora \
    --tasks wikitext,arc_easy,lambada_openai

# Metric fix (re-pick acc_norm,none for arc_easy)
.venv/bin/python research/residual-pareto/experiments/fix_metric.py

# Analysis (Stage 1 + Stage 1.5 + combined)
PYTHONPATH=. .venv/bin/python research/residual-pareto/experiments/analyze_stage15.py

# Registered RPM-002 test (3-consecutive non-decreasing subsequence search)
.venv/bin/python research/residual-pareto/experiments/rpm002_registered_test.py

# Pareto dominance audit (capability × cost B/F/O/M/L/E)
.venv/bin/python research/residual-pareto/experiments/pareto_audit.py
```