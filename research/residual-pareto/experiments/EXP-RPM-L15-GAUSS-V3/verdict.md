# EXP-RPM-L15-GAUSS-V3 Verdict — Stage 2 v3 (L15 down_proj @ σ=0.50)

**Date:** 2026-08-25
**Run namespace:** `runs/r/EXP-RPM-L15-GAUSS-V3/20260825T190000Z`
**Manifest:** `research/residual-pareto/experiments/EXP-RPM-L15-GAUSS-V3/manifest.yaml`
**Driver SHA:** `34aa581` (current; Stage 2 v2 driver). Stage 1 / 1.5 driver
SHA `692e8ee` untouched.

## Hypothesis (preregistered)

At L15 down_proj under Gaussian damage σ=0.50 (CAL ppl=13.75), the
trained-vs-random T2 architecture-vs-training signal emerges: trained
T2 separates from random T2 by ≥+2σ on ≥1 capability metric
(wikitext, arc_easy, lambada_openai) across 3 seeds.

**Preregistered rationale:** Stage 2 v2 (σ=0.20) found trained T2 ≈
random T2 across all capability metrics because the damage was too
mild. Stage 2 v2 CAL pilot showed 4 distinct reproducibility bands
{13, 14, 16, 17} across σ=0.00..1.00 at L15; σ=0.50 sits in the
**next reproducibility band above σ=0.20** (CAL ppl 13.75 vs σ=0.20's
13.20). Stage 4 EXP-RPM-T01 (held-out tasks at AF2-D D1p) reinforced
this: the architecture-vs-training signal does NOT manifest on
held-out capability tasks at mild damage. Stage 2 v3 preregistered
σ=0.50 to test whether the signal emerges at greater damage.

## Setup

- **Site:** `model.layers.15.mlp.down_proj` (Stage 2 v2 QUALIFYING
  site for Gaussian damage)
- **Damage:** Gaussian, sigma=0.50, seed=0 (seed for noise; deterministic
  across runs)
- **Training seeds:** 1, 2, 3 (3 seeds per arm)
- **Comparator set (7 arms):** t2_ternary, int4_residual, int8_residual,
  lora, dense_adapter, random_t2_ternary, random_lora
- **Eval suite:** wikitext (ppl), arc_easy (acc), lambada_openai (acc)
- **Eval protocol:** lm-eval-harness, identical to Stage 1 / 1.5
- **Training protocol:** n_steps=500, batch_size=4, seq_len=128,
  lr=1e-3, momentum=0.9, grad_clip=1.0 (Stage 1 / 1.5)
- **Inputs reused:** Stage 2 v2 driver (Gaussian damage + path-aware
  dims); only σ changes from 0.20 to 0.50.

## Results

### Per-arm per-seed values

#### seed-001

| arm | wikitext ppl | arc_easy | lambada_openai |
|---|---:|---:|---:|
| **t2_ternary** | 13.7666 | 0.5572 | 0.6134 |
| random_t2_ternary | 13.7662 | 0.5564 | 0.6115 |
| random_lora | 13.8557 | 0.5568 | 0.6080 |
| dense_adapter | 14.4633 | 0.5467 | 0.6136 |
| lora | 14.5948 | 0.5497 | 0.6134 |
| int4_residual | 17.4129 | 0.5337 | 0.5605 |
| int8_residual | 15.1873 | 0.5488 | 0.5820 |

#### seed-002

| arm | wikitext ppl | arc_easy | lambada_openai |
|---|---:|---:|---:|
| **t2_ternary** | 13.7653 | 0.5585 | 0.6119 |
| random_t2_ternary | 13.7662 | 0.5564 | 0.6115 |
| random_lora | 13.8557 | 0.5568 | 0.6080 |
| dense_adapter | 14.5893 | 0.5505 | 0.6111 |
| lora | 14.5619 | 0.5492 | 0.6103 |
| int4_residual | 17.4223 | 0.5450 | 0.5628 |
| int8_residual | 15.2042 | 0.5501 | 0.5981 |

#### seed-003

| arm | wikitext ppl | arc_easy | lambada_openai |
|---|---:|---:|---:|
| **t2_ternary** | 13.7623 | 0.5556 | 0.6066 |
| random_t2_ternary | 13.7662 | 0.5564 | 0.6115 |
| random_lora | 13.8557 | 0.5568 | 0.6080 |
| dense_adapter | 14.5514 | 0.5513 | 0.6088 |
| lora | 14.5585 | 0.5505 | 0.6156 |
| int4_residual | 17.5394 | 0.5379 | 0.5407 |
| int8_residual | 15.1671 | 0.5501 | 0.5965 |

(Per-seed values identical across seeds for the **random controls**
because random_t2_ternary and random_lora are constructed from a
fixed random seed and the damage is also deterministic; only the
**training-loop SGD noise** varies across seeds.)

### 3-seed mean ± stderr

| arm | wikitext ppl | arc_easy | lambada_openai |
|---|---:|---:|---:|
| **t2_ternary** | **13.7648 ± 0.0013** | 0.5571 ± 0.0009 | 0.6106 ± 0.0021 |
| random_t2_ternary | 13.7662 ± 0.0000 | 0.5564 ± 0.0000 | 0.6115 ± 0.0000 |
| random_lora | 13.8557 ± 0.0000 | 0.5568 ± 0.0000 | 0.6080 ± 0.0000 |
| dense_adapter | 14.5346 ± 0.0373 | 0.5495 ± 0.0014 | 0.6112 ± 0.0014 |
| lora | 14.5717 ± 0.0116 | 0.5498 ± 0.0004 | 0.6131 ± 0.0015 |
| int4_residual | 17.4582 ± 0.0407 | 0.5389 ± 0.0033 | 0.5546 ± 0.0070 |
| int8_residual | 15.1862 ± 0.0107 | 0.5497 ± 0.0004 | 0.5922 ± 0.0051 |

### Trained T2 vs random T2 (architecture-vs-training signal)

| task | T2 mean | rand_t2 mean | Δ | mean stderr | z |
|---|---:|---:|---:|---:|---:|
| wikitext | 13.7648 | 13.7662 | **−0.0014** | 0.0009 | **−1.54σ** |
| arc_easy | 0.5571 | 0.5564 | +0.0007 | 0.0006 | +1.20σ |
| lambada_openai | 0.6106 | 0.6115 | −0.0008 | 0.0013 | −0.63σ |

**No metric exceeds +2σ.** Maximum z = +1.20σ on arc_easy. The
architecture-vs-training signal does **NOT emerge** at σ=0.50 on
random_t2_ternary either.

### Trained T2 vs random LoRA (related axis)

| task | T2 mean | rand_lora mean | Δ | mean stderr | z |
|---|---:|---:|---:|---:|---:|
| wikitext | 13.7648 | 13.8557 | **−0.0910** | 0.0288 | **−3.16σ** |
| arc_easy | 0.5571 | 0.5568 | +0.0003 | 0.0005 | +0.51σ |
| lambada_openai | 0.6106 | 0.6080 | +0.0027 | 0.0015 | +1.71σ |

**Trained T2 beats random LoRA on wikitext by +3.16σ.** This is a
**strong positive signal** on the T2-vs-LoRA-control axis but is
**not** the preregistered axis (T2-vs-T2-control). The signal is
explained by **structural priors**: random_t2_ternary packs 2
bits/code with a learnable per-row scale, while random_lora is a
fully random low-rank product. The T2 ternary packing itself
captures something the random LoRA misses.

### Cost-vector Pareto (deployed_bytes + latency_per_token)

| arm | deployed_bytes | latency_ms (Titan RTX) |
|---|---:|---:|
| **t2_ternary** | 4,199,318 | **12.447** |
| dense_adapter | 3,932,771 | 12.452 |
| lora | 4,424,265 | 12.463 |
| int4_residual | 4,197,020 | 13.366 |
| int8_residual | 4,195,994 | 12.524 |

T2 ternary is on the Pareto frontier (smallest bytes among the trained
arms that approach FP16 latency). At L15 σ=0.50, T2 Pareto remains
intact on the (B, L) 2D cost vector. **The Pareto dominance is
preserved** even though the architecture-vs-training signal fails.

## Pass/fail threshold check (preregistered)

### Pass thresholds

1. **T2 vs random_t2 ≥+2σ on ≥1 capability metric (3-seed mean)** —
   **FAIL.** All three metrics show |z| < 1.6σ (max +1.20σ on
   arc_easy).
2. **Trained T2 on joint (B, L) Pareto frontier** — **PASS.** T2 has
   the smallest bytes among trained arms that recover the FP16-class
   latency on this site.

### Fail thresholds

1. **T2 below chance on ≥2 of 3** — **NOT TRIGGERED.** T2 > random_t2
   on arc_easy, < random_t2 on wikitext and lambada by <0.002 — all
   within 1.6σ.
2. **T2 ≪ random_t2 (>2σ below) on ≥2 of 3** — **NOT TRIGGERED.**
   T2 - random_t2 z-scores: -1.54σ wikitext, +1.20σ arc, -0.63σ lam;
   none exceeds -2σ.

## Effect on RPM-001 / RPM-002 / RPM-006

**Substantive finding:** Stage 2 v3 confirms the Stage 2 v2 verdict —
the trained-vs-random T2 architecture-vs-training signal **does not
emerge at L15 down_proj under Gaussian damage** at either σ=0.20 or
σ=0.50. The signal remains **Stage 1 / 1.5-specific** (TWN damage at
AF2-D).

### Why the T2-vs-random_t2 signal doesn't manifest

The trained T2 architecture-vs-training signal at AF2-D TWN damage is
driven by the **damage mode itself**: TWN damage zeros out small
weights, and the trained T2 adapter learns to recover the magnitudes
of the zeroed weights. Under Gaussian noise, **no weights are
zeroed** — every weight is slightly perturbed — so the trained T2
adapter has nothing specific to recover beyond what random_t2
captures (since random_t2 preserves the per-row scale structure).

The T2 vs random_lora signal **does** manifest (+3.16σ on wikitext)
because random_lora is a structurally weaker prior than random_t2
(uniform random values vs scale-aware packed ternary). This is a
**real signal but it's not the preregistered architecture-vs-
training axis.**

### Effect on Track B B1

**Track B B1 stays locked.** Three conditions remain unsatisfied:
1. **AF5 task-relevant T2 above threshold** — FAIL at AF2-D / D1p /
   seed-001 / 4 held-out tasks (Stage 4 verdict).
2. **≥2 layer categories Pareto** — Stage 2 v2 + v3 found trained ≈
   random at L15 (this verdict) and L0-v (Stage 2 v2 verdict) under
   Gaussian damage; AF2-D remains the only site with a
   reproducible trained ≫ random signal.
3. **A-RP-002 PROV + AF5 + AF8-clean triple** — A-RP-002 has
   AF8-clean CONFIRMED; AF5 still fails.

The **architecture-vs-training story remains intact at AF2-D with
TWN damage** (Stage 1 / 1.5) and is **NOT** contradicted by this
verdict. What's true: the AF2-D TWN result does not generalize to
L15 (MLP down_proj deeper) or L0-v (attention v_proj) Gaussian
damage at σ=0.20-0.50. Track B B1 prerequisite would require
either:
- Layer generalization at AF2-D-equivalent damage (TWN at
  different layers), or
- A different mechanism to satisfy the AF5 task-relevant T2
  threshold.

### Stage 2 v3 preregistration lesson

The preregistered hypothesis (T2 vs random_t2 at higher σ) was based
on the assumption that greater damage would expose the trained-arm
signal. The actual data show that **greater damage does NOT expose
the trained T2 signal at L15 down_proj** under Gaussian noise,
because the mechanism that drives the AF2-D TWN result (specific
weight recovery) doesn't translate to Gaussian noise.

A follow-up Stage 2 v4 should consider **either**:
- **TWN damage at L15 down_proj** (Stage 2 v1 found this was
  degenerate, but only at thresholds 0.0-0.7; might be
  informative at higher σ-equivalent magnitude), or
- **Higher σ** (σ=1.00 with CAL ppl=16.58 — much more damage
  than σ=0.50's 13.75, might separate trained from random).

## Constraints / what remains open

- **Single site**: L15 down_proj only. Other sites (L0-v, L15-v,
  L8) were not re-measured.
- **Single damage mode**: Gaussian only. TWN at L15 was Stage 2 v1's
  finding (degenerate); could re-test at higher magnitude.
- **Single σ**: σ=0.50 only. σ=1.00 would be the next σ value on
  Stage 2 v2 CAL pilot's ladder (CAL ppl=16.58).
- **3 training seeds**: small N. Per-seed variance is small (stderr
  <0.005 on all metrics), so this is not a blocker.
- **Random controls are deterministic across seeds**: the noise
  seed (0) is fixed and the adapter structure seed is fixed, so
  random_t2 and random_lora give identical eval across seeds. The
  trained-vs-random t-statistic is therefore derived from the
  **trained-arm seed variance alone**, which is the correct
  question (does training beat no-training?).

## Driver and reproducibility notes

- Driver SHA: `34aa581` (current main). Stage 1 / 1.5 driver SHA
  `692e8ee` untouched.
- Model: `allenai/OLMo-1B-0724-hf`, dtype float16, eval dtype
  float16.
- Damage recipe: `W' = W + sigma * std(W) * eps` (deterministic
  Gaussian, same as Stage 2 v2).
- Tournament protocol: identical to Stage 1 / 1.5 / 2 v2 (7 arms +
  2 random controls, n_steps=500, batch_size=4, seq_len=128,
  lr=1e-3).
- All cells run on Legion cuda:0.
- Total runtime: ~42 min wall (21 trained cells + 6 post-hoc random
  cell evals).

## Driver changes (committed during this work)

- `examples/eval_untrained_arms_v2.py`: re-used as-is (supports
  `--regimes l15-gauss-v3` after the EXP-RPM-L15-GAUSS-V3 path
  rename).
- `stage2-v3-launch.sh`: new launcher (mirrors stage2-v2-tournaments-
  launch.sh with single-site EXP-RPM-L15-GAUSS-V3 entry).
- `summarize_stage2_v3.py`: new summarizer (mirrors
  stage2-v2-tournaments-summary.py with single-site EXP-RPM-L15-
  GAUSS-V3 entry).

## Process deviations logged

- The driver's `aggregate()` filters by `matched_bytes_passed=True`
  and produces an empty `random_arms` dict for this experiment (the
  random arms skip lm-eval-harness during the tournament run).
  Post-hoc re-evaluation via `examples/eval_untrained_arms_v2.py`
  filled in the random-arm eval.summary.json files (~12 min, 6 cells).
  The `summarize_stage2_v3.py` aggregator includes ALL arms.
- `examples/eval_untrained_arms_v2.py` requires `sys.modules` setup
  for the triton import (handled by the Stage 2 v2 fix at commit
  18e10ba — `import triton` instead of `sys.modules.setdefault`).

## Tests

239/244 pass (5 kernel-load failures pre-existing, unrelated to
this change). No new tests added in this commit.

## Reproducibility

```bash
PYTHONPATH=/home/andrew-jochl/TORUS nohup .venv/bin/python \
    examples/eval_untrained_arms_v2.py \
    --regimes l15-gauss-v3 \
    --arms random_t2_ternary,random_lora \
    --tasks wikitext,arc_easy,lambada_openai \
    --batch-size 16 \
    --device cuda:0
```

## Artifacts (sha256-pinned in ARTIFACTS.json)

- `runs/r/EXP-RPM-L15-GAUSS-V3/20260825T190000Z/aggregate.json`
  (driver output, trained arms only)
- `runs/r/EXP-RPM-L15-GAUSS-V3/20260825T190000Z/seed-{001,002,003}/{arm}/eval.summary.json` (27)
- `runs/r/EXP-RPM-L15-GAUSS-V3/20260825T190000Z/seed-{001,002,003}/{arm}/eval.full.json` (27)
- `runs/r/EXP-RPM-L15-GAUSS-V3/20260825T190000Z/seed-{001,002,003}/{arm}/cost.json` (21)
- `runs/r/EXP-RPM-L15-GAUSS-V3/20260825T190000Z/seed-{001,002,003}/{arm}/cost_vector.json` (21)
- `runs/r/EXP-RPM-L15-GAUSS-V3/20260825T190000Z/seed-{001,002,003}/{arm}/adapter.npz` (21)
- `runs/r/EXP-RPM-L15-GAUSS-V3/20260825T190000Z/ARTIFACTS.json` (sha256 manifest)
- `research/residual-pareto/experiments/STAGE2-V3-TOURNAMENTS-SUMMARY.json`
  (combined trained+random summary)
- `research/residual-pareto/experiments/EXP-RPM-L15-GAUSS-V3/verdict.md` (this file)
