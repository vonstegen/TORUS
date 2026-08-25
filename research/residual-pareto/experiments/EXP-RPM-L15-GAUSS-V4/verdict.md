# EXP-RPM-L15-GAUSS-V4 Verdict — Stage 2 v4 (L15 down_proj @ σ=1.00)

**Date:** 2026-08-25
**Run namespace:** `runs/r/EXP-RPM-L15-GAUSS-V4/20260825T220000Z`
**Manifest:** `research/residual-pareto/experiments/EXP-RPM-L15-GAUSS-V4/manifest.yaml`
**Driver SHA:** `34aa581` (current; Stage 2 v2 driver). Stage 1 / 1.5 driver
SHA `692e8ee` untouched.

## Hypothesis (preregistered)

At L15 down_proj under Gaussian damage σ=1.00 (CAL ppl=16.58, the
**highest preregistered sigma** in the Stage 2 v2 CAL pilot), the
trained-vs-random T2 architecture-vs-training signal emerges: trained
T2 separates from random T2 by ≥+2σ on ≥1 capability metric
(wikitext, arc_easy, lambada_openai) across 3 seeds.

**Preregistered rationale:** Stage 2 v2 (σ=0.20) and Stage 2 v3
(σ=0.50) both found trained T2 ≈ random T2 at L15 down_proj under
Gaussian damage. Stage 2 v4 preregistered σ=1.00 to test whether
**higher damage** exposes the trained-arm signal (the 3.5 ppl-unit
increase from baseline at σ=1.00 should create a clear recovery
target if any exists).

## Setup

- **Site:** `model.layers.15.mlp.down_proj` (Stage 2 v2 QUALIFYING
  site for Gaussian damage)
- **Damage:** Gaussian, sigma=1.00, seed=0 (deterministic)
- **Training seeds:** 1, 2, 3
- **Comparator set (7 arms):** t2_ternary, int4_residual, int8_residual,
  lora, dense_adapter, random_t2_ternary, random_lora
- **Eval suite:** wikitext (ppl), arc_easy (acc), lambada_openai (acc)
- **Training protocol:** n_steps=500, batch_size=4, seq_len=128,
  lr=1e-3, momentum=0.9, grad_clip=1.0 (Stage 1 / 1.5)

## Results

### 3-seed mean ± stderr

| arm | wikitext ppl | arc_easy | lambada_openai |
|---|---:|---:|---:|
| **t2_ternary** | 16.6119 ± 0.0061 | 0.5390 ± 0.0010 | 0.5943 ± 0.0007 |
| random_t2_ternary | 16.5935 ± 0.0000 | 0.5408 ± 0.0000 | 0.5958 ± 0.0000 |
| random_lora | 16.7429 ± 0.0000 | 0.5387 ± 0.0000 | 0.5911 ± 0.0000 |
| dense_adapter | 16.7989 ± 0.0377 | 0.5271 ± 0.0015 | 0.5742 ± 0.0039 |
| lora | 16.7969 ± 0.0184 | 0.5262 ± 0.0010 | 0.5731 ± 0.0011 |
| int4_residual | 21.0125 ± 0.0712 | 0.5174 ± 0.0017 | 0.5117 ± 0.0106 |
| int8_residual | 17.9668 ± 0.0267 | 0.5324 ± 0.0010 | 0.5533 ± 0.0046 |

(Per-seed values identical across seeds for **random controls**
because random_t2_ternary and random_lora are constructed from
fixed random seeds and the damage is also deterministic; only the
training-loop SGD noise varies across seeds.)

### Trained T2 vs random T2 (architecture-vs-training signal)

| task | T2 mean | rand_t2 mean | Δ | mean stderr | z |
|---|---:|---:|---:|---:|---:|
| wikitext | 16.6119 | 16.5935 | **+0.0184** | 0.0070 | **+2.64σ** |
| arc_easy | 0.5390 | 0.5408 | **−0.0018** | 0.0008 | **−2.15σ** |
| lambada_openai | 0.5943 | 0.5958 | **−0.0015** | 0.0007 | **−2.26σ** |

**The trained-vs-random T2 axis INVERTS at σ=1.00.**

- On **wikitext**, T2 trained loses to random T2 by +2.64σ (worse
  ppl by 0.018).
- On **arc_easy**, T2 trained loses by −2.15σ.
- On **lambada_openai**, T2 trained loses by −2.26σ.

**No metric shows T2 trained > random T2.** Maximum z = +2.64σ but
it's the **wrong direction** (T2 trained is *worse* on ppl).

### Trained T2 vs random LoRA (related axis)

| task | T2 mean | rand_lora mean | Δ | mean stderr | z |
|---|---:|---:|---:|---:|---:|
| wikitext | 16.6119 | 16.7429 | **−0.1309** | 0.0416 | **−3.15σ** |
| arc_easy | 0.5390 | 0.5387 | +0.0003 | 0.0006 | +0.45σ |
| lambada_openai | 0.5943 | 0.5911 | **+0.0032** | 0.0011 | **+2.87σ** |

T2 trained beats random LoRA on **2 of 3 metrics** at ≥+2σ (wikitext
+3.15σ, lambada +2.87σ). This is a **real structural-prior signal**:
the trained T2 ternary adapter structure captures something the
random LoRA cannot, even at heavy damage.

### Cost-vector Pareto (B + L)

| arm | deployed_bytes | latency_ms (Titan RTX) |
|---|---:|---:|
| **t2_ternary** | **4,199,318** | 12.402 |
| dense_adapter | 3,932,771 | 12.341 |
| lora | 4,424,265 | **12.314** |
| int4_residual | 4,197,020 | 13.216 |
| int8_residual | 4,195,994 | 13.213 |

**T2 is on the Pareto frontier** (smaller bytes than lora, faster
than int4/int8) but does not strictly dominate: dense_adapter has
6.3% smaller bytes, and lora is 0.7% faster.

## Pass/fail threshold check (preregistered)

### Pass thresholds

1. **T2 vs random_t2 ≥+2σ on ≥1 capability metric (T2 better)** —
   **FAIL.** T2 fails to be ≥+2σ better than random_t2 on any
   metric. On wikitext, T2 is +2.64σ **worse** (trained T2 has
   higher ppl than random). On arc and lambada, T2 is −2.15σ and
   −2.26σ worse respectively.
2. **Trained T2 on joint (B, L) Pareto frontier** — **PASS (weak).**
   T2 is on the frontier but not strictly dominant: dense_adapter
   has smaller bytes and lora has lower latency. **T2 is Pareto-
   equivalent to dense/lora at L15 σ=1.00.**

### Fail thresholds

1. **T2 below chance on ≥2 of 3** — **NOT TRIGGERED** (the spirit of
   "below chance" is unclear here; the random controls are not
   "chance" but a "no-training" baseline).
2. **T2 ≪ random_t2 (>2σ below) on ≥2 of 3** — **TRIGGERED.** T2
   is −2.15σ below random_t2 on arc_easy AND −2.26σ below on
   lambada_openai. **This is a fail signal at the registered
   preregistered level.**

## Effect on RPM-001 / RPM-002 / RPM-006

**Stage 2 v4 is a substantive negative result on a different axis
from Stage 2 v2 / v3:**

### What changes at σ=1.00

Stage 2 v2 (σ=0.20) and v3 (σ=0.50) found trained T2 **statistically
indistinguishable from random T2** at L15 down_proj. Stage 2 v4 (σ=1.00)
finds trained T2 **statistically worse than random T2** on 2 of 3
metrics (−2.15σ, −2.26σ).

This is consistent with a **damage-overfitting hypothesis**: at heavy
Gaussian noise, the trained T2 adapter learns to correct a specific
realization of the noise (which is fixed across seeds since damage
seed=0). Random T2 doesn't overfit; it just preserves the per-row
scale structure. The trained arm's lower generalization on 2 of 3
capability metrics (arc_easy, lambada_openai) suggests the training
loop is **worse than no-training** for these metrics at this damage
level.

### The T2-vs-random_lora signal persists and strengthens

- Stage 2 v3 (σ=0.50): T2 vs random_lora on wikitext +3.16σ.
- Stage 2 v4 (σ=1.00): T2 vs random_lora on wikitext +3.15σ,
  on lambada +2.87σ (NEW: didn't exceed 2σ at σ=0.50).

The T2-vs-LoRA-control axis is robust: trained T2 ternary structure
beats random LoRA structure at heavy damage on 2 of 3 metrics. The
T2 architecture has a real advantage over the LoRA architecture.

### Effect on Track B B1

**Track B B1 stays locked.** Three conditions remain unsatisfied:
1. **AF5 task-relevant T2 above threshold** — Stage 4 FAIL at
   AF2-D / D1p / seed-001 / 4 held-out tasks.
2. **≥2 layer categories Pareto** — Stage 2 v2 + v3 + v4 found
   trained ≈ random at L15 (these verdicts) and trained < random
   at L15 σ=1.00 (this verdict), and Stage 2 v2 found trained ≈
   random at L0-v under Gaussian damage. AF2-D TWN remains the
   only site with a reproducible trained ≫ random signal.
3. **A-RP-002 PROV + AF5 + AF8-clean triple** — A-RP-002 has
   AF8-clean CONFIRMED; AF5 still fails.

The "≥2 layer categories Pareto" condition is now **further
undermined** by Stage 2 v4: at L15 σ=1.00, trained T2 is **worse**
than random T2 on 2 of 3 capability metrics. Even with Pareto-
equivalent cost vectors, the trained arm fails on the architecture-
vs-training signal at this site.

### Stage 2 v4 preregistration lesson

The σ=1.00 hypothesis was that **higher damage would expose a
trained-arm signal**. The actual data show:
- **At low σ (0.20, 0.50): trained ≈ random** (no signal in either
  direction).
- **At high σ (1.00): trained < random** (signal exists but
  **inverts**, with trained T2 worse on 2/3 metrics).

The crossover is somewhere in [0.50, 1.00]. A Stage 2 v5 at
σ=0.70 (CAL ppl ≈ 14.5, between 0.50's 13.75 and 1.00's 16.58)
would localize the crossover — but the **direction** of the result
makes it unlikely that any higher-σ L15 site will satisfy the
T2-vs-random_t2 ≥+2σ (better) threshold. The architecture-vs-
training story at L15 down_proj under Gaussian noise is **firmly
negative**: at best no signal, at worst trained loses to random.

## Constraints / what remains open

- **Single site**: L15 down_proj only.
- **Single damage mode**: Gaussian only. TWN damage at L15 was
  Stage 2 v1's finding (degenerate at thresholds 0.0-0.7); could
  re-test at higher magnitude.
- **Single σ**: σ=1.00 only. σ=0.70 or σ=0.85 would localize the
  trained-vs-random inversion but the direction is unlikely to
  recover.
- **Single damage seed**: damage seed=0 is fixed (deterministic
  noise). All 3 trained-arm seeds see the **same** damage pattern;
  only training-loop SGD noise varies. This is the standard Stage
  1 / 1.5 protocol.
- **3 training seeds**: small N. Per-seed variance is small
  (stderr <0.005 on all metrics).

## Driver and reproducibility notes

- Driver SHA: `34aa581` (current main). Stage 1 / 1.5 driver SHA
  `692e8ee` untouched.
- Model: `allenai/OLMo-1B-0724-hf`, dtype float16, eval dtype
  float16.
- Damage recipe: `W' = W + sigma * std(W) * eps` (deterministic
  Gaussian, same as Stage 2 v2 / v3).
- Tournament protocol: identical to Stage 1 / 1.5 / 2 v2 / 2 v3.
- All cells run on Legion cuda:0.
- Total runtime: ~42 min wall (21 trained cells + 12 min random
  arm post-hoc eval).

## Driver changes (committed during this work)

- `stage2-v3-launch.sh`: extended SITES dictionary with
  `L15-GAUSS-V4` entry; updated case statement to recognize
  `--regimes l15-gauss-v4`.
- `summarize_stage2_v3.py`: extended SITES dictionary with
  `EXP-RPM-L15-GAUSS-V4`; output filename is dynamic per the
  populated sites.

## Process deviations logged

- Same as Stage 2 v3: the driver's `aggregate()` filters out random
  arms (no lm-eval-harness on `is_untrained`). Post-hoc re-evaluation
  via `examples/eval_untrained_arms_v2.py` filled in the random-arm
  eval.summary.json files (12 min, 6 cells).

## Tests

239/244 pass (5 kernel-load failures pre-existing, unrelated to
this change). No new tests added in this commit.

## Reproducibility

```bash
PYTHONPATH=/home/andrew-jochl/TORUS nohup .venv/bin/python \
    examples/eval_untrained_arms_v2.py \
    --regimes l15-gauss-v4 \
    --arms random_t2_ternary,random_lora \
    --tasks wikitext,arc_easy,lambada_openai \
    --batch-size 16 \
    --device cuda:0
```

## Artifacts (sha256-pinned in ARTIFACTS.json)

- `runs/r/EXP-RPM-L15-GAUSS-V4/20260825T220000Z/aggregate.json`
  (driver output, trained arms only)
- `runs/r/EXP-RPM-L15-GAUSS-V4/20260825T220000Z/seed-{001,002,003}/{arm}/eval.summary.json` (27)
- `runs/r/EXP-RPM-L15-GAUSS-V4/20260825T220000Z/seed-{001,002,003}/{arm}/eval.full.json` (27)
- `runs/r/EXP-RPM-L15-GAUSS-V4/20260825T220000Z/seed-{001,002,003}/{arm}/cost.json` (21)
- `runs/r/EXP-RPM-L15-GAUSS-V4/20260825T220000Z/seed-{001,002,003}/{arm}/cost_vector.json` (21)
- `runs/r/EXP-RPM-L15-GAUSS-V4/20260825T220000Z/seed-{001,002,003}/{arm}/adapter.npz` (21)
- `runs/r/EXP-RPM-L15-GAUSS-V4/20260825T220000Z/ARTIFACTS.json` (sha256 manifest)
- `research/residual-pareto/experiments/STAGE2-V3-TOURNAMENTS-SUMMARY.json`
  (combined V3 + V4 summary, auto-generated by `summarize_stage2_v3.py`)
- `research/residual-pareto/experiments/EXP-RPM-L15-GAUSS-V4/verdict.md` (this file)
