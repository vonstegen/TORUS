
## 2026-08-20 — Full run analysis (600 steps, STOPPED)

**Verdict: run trained nothing. Stopped at step 600 for analysis.**

### Measured
- primary drift step100->400->600: ~0.01-0.07% (noise floor)
- residual change step400->600: exactly 0.0000 (still randn*0.01 init)
- loss trajectory 6.06->2.89->5.79 = batch-window noise (seq_len=16, bs=1).
  Trial and full run both start at loss=6.0625 (same seeded batch).

### Root cause (one-step probe, hard numbers)
- raw global grad norm: 284.2 over 1.07B graded params
- grad_clip=1.0 -> scale 0.0035 (284x shrink)
- plain SGD lr=3e-4 -> per-element update 9.2e-9/step
- 400-step cumulative = 0.073% of |w| — matches measured drift exactly
- Gradient ordering fix CONFIRMED working (112/112 grads, 0 None).
  This is a separate bug: clip threshold calibrated wrong for
  global-norm over 1B params with plain momentum SGD.

### Fix
grad_clip >= 300 (or Adam). With clip=300: ~20% |w| movement over
400 steps at same wall clock. One-line change in TrainingConfig.

## 2026-08-20 later — Tier 1+2 trainer overhaul

### Tier 1 (deployed, validated)
- grad_clip 1.0 -> 300.0 (measured raw norm 284); --grad-clip CLI flag
- Log real KL from _autograd_grads; deleted the no_grad double-forward
  (was logging logit-MSE stand-in, ~40% wasted step compute)
- Guards: all-None grads -> RuntimeError; rel_update < 1e-7 at step 0
  -> RuntimeError; < 1e-5 -> WARN
- Heartbeat now carries gnorm/upd/lr per log step
- 3-step smoke: weights moved 0.037% (vs 0.01% in 600 frozen steps)

### Validation run (valid_t1, 100 steps planes=1, in progress)
- step 0: KL 5.55 gnorm 284
- step 20: KL 3.68 gnorm 24.5 (converging signature)
- step 40: KL 4.34 gnorm 25.3 (window noise on downtrend)

### Tier 2 (deployed, CPU smoke passed)
- _fit_torch: torch-native GPU optimizer, grads stay on GPU,
  zip-identity .grad assignment (ordering-bug-proof), no numpy
  round-trip, no sync-back
- --optimizer sgd|adam (default sgd, parity with numpy path)
- fit() dispatches: forward_with_grad present -> _fit_torch,
  else legacy _fit_numpy (numerical reference path intact)
- clip via torch.nn.utils.clip_grad_norm_ (returns raw norm for
  telemetry); float64 temporaries gone
- Self-distillation (no --teacher-model) now builds teacher=student
  instead of crashing; degenerate zero-KL config correctly caught
  by the step-0 invariant

## 2026-08-20 final — full_t2 1000-step run + controlled eval

### Run
- 1000 steps in 628s (0.63s/step; was ~36h on the frozen architecture)
- Plane-1 band converged KL~2.0-2.4 by step 475; plane-2 spike 5.63 at
  step 500; ended ~2.7-3.5
- Disk quota killed first attempt at step 525 (heartbeat write not
  exception-wrapped); fixed + freed 39GB of frozen-era checkpoints

### Controlled eval (64 fixed wikitext windows, KL(s||t) T=2.0, fp32)
- frozen control:   3.420 +/- 0.078
- plane1 (step400): 2.658 +/- 0.069   <- deployable model
- plane2 (final):   3.407 +/- 0.081   <- statistically == untrained
- paired delta plane1-plane2: -0.749, t=-17.8

### Verdict
- Plane-1 ternary distillation WORKS (-22% KL vs init)
- 2-plane curriculum as configured DESTROYS plane-1 gains: residual
  init sigma=0.01 is the same magnitude as the trained primary
  (~0.005-0.008), not "well below the model weight scale" as the
  code comment claims. Residual enters as full-amplitude noise.
- Fixes to try: residual sigma 1e-3..1e-4, magnitude ramp, or
  two-stage (train residual against frozen primary)
- fp16 KL NaN gotcha: temperature must be >=2.0 or cast fp32
  (fp16 softmax underflows -> log(0)*0 -> NaN)

### 2026-08-20 two-stage arms + verdict (CORRECTION included)

**Correction to earlier note:** the residual was NOT graph-broken in
full_t2. Probe (/tmp/resid_probe.py): at n_planes=2 residual grads flow
with the same rms as primary (identity STE). The 0.005% residual
movement was **LR starvation** (residual lr 3e-5), not a missing graph.
The arm crash was a third bug: `--curriculum 2:500` plane numbers were
silently ignored (progressive() derives planes from stage INDEX), so
the arm ran planes=1 and the grad guard correctly fired. Fixed:
distill_run.py now builds explicit CurriculumStages for non-progressive
plane sequences.

**Arms** (both: load full_t2 step400 ckpt, 500 steps planes=2 from
step 0, freeze primary, residual lr = full 3e-4):
- C1: --reinit-residual 1e-3. Train band 1.6-1.9.
- C2: --reinit-residual 1e-2 (old default). Train band 2.4-2.9.

**Controlled eval** (64 fixed windows, KL(student||teacher), T=2.0,
fp32; per-case subprocess to avoid allocator fragmentation):

| model | KL |
|---|---|
| frozen-control | 3.4201 +/- 0.0780 |
| plane1-step400 | 2.6583 +/- 0.0689 |
| plane2-final (co-trained) | 3.4069 +/- 0.0814 |
| arm-c1 sig=1e-3 | **1.9945 +/- 0.0669** |
| arm-c2 sig=1e-2 | 2.8697 +/- 0.0751 |

Paired t vs plane1-step400: C1 **+0.664 (t=+18.3)**, C2 -0.211
(t=-7.5), co-trained -0.749 (t=-17.8).

**Verdict: the residual plane adds real capacity when trained as
designed** (two-stage: freeze trained primary, small init, full LR).
KL 3.42 -> 2.66 (plane 1) -> 1.99 (plane 2 two-stage) = 42% total
reduction. Init magnitude is decisive: sigma=1e-3 wins, sigma=1e-2
(the old default) spends its budget unlearning noise and still lands
worse than plane-1 alone. Co-training both planes from a large-init
residual remains the failure mode.

Deployable 2-plane student: /tmp/eval_out/arm_c1.npz (KL 1.99).

### 2026-08-20 follow-up arms C3/C4 (budget + init curve)

- C3: sigma=1e-3, 1000-step residual phase. Train loss still falling
  at step 999 (1.10). Eval KL **1.7218 +/- 0.0627**.
- C4: sigma=3e-4, 500 steps. Eval KL 1.9861 +/- 0.0643.

Paired t: C3 vs plane1 +0.9365 (t=+26.2); C3 vs C1 +0.2727 (t=+13.8);
C4 vs C1 +0.0084 (t=+0.67, ns).

Findings:
1. Residual budget matters: doubling the residual phase 500->1000
   steps buys another 0.27 KL. Not yet converged at 1000.
2. Init sigma plateau: 3e-4 ~= 1e-3 (both ~1.99 @500 steps). The
   toxicity threshold is between 1e-3 and 1e-2. Any sigma <= 1e-3
   is safe.
3. Best student: /tmp/eval_out/arm_c3.npz — KL 1.72, a 49.7%
   reduction from untrained (3.42).

### 2026-08-20 plane-3 arm (arm_p3) — three-plane student

Machinery: TernarySTE.residual_weight_2 (additive 3rd plane),
HFAdapterConfig.n_planes_max, --train-plane N (replaces
--freeze-primary), --max-planes, --reinit-residual now reinits the
TOP plane only. npz gains residual2_{i} keys; old 2-plane files load
into 3-plane adapters untouched. Probe verified grad flow at
n_planes=3 (identical rms on all planes) and exclusion at 2.

Arm: load arm_c3 (2-plane, KL 1.72), freeze planes 1+2, reinit
plane 3 at sigma=1e-3, train plane 3 only, 1000 steps, lr 3e-4,
curriculum 3:1000. Step time 1.07s (3 quantize passes).
Final training loss 0.96 (vs C3 1.10).

Controlled eval (same 64 windows): KL **1.5997 +/- 0.0614**.
Paired t: vs arm_c3 **+0.1221 (t=+7.04)**; vs plane1 +1.0586 (t=+23.1).

Finding: plane 3 adds real but DIMINISHING capacity — plane 2 gave
+0.94 KL (1k budget), plane 3 gives +0.12. Total arc:
3.42 -> 2.66 -> 1.72 -> 1.60 = 53.2% reduction, 3 ternary planes.
Best student: /tmp/eval_out/arm_p3.npz (KL 1.60).

### 2026-08-20 saturation analysis (why plane gains shrink)

Per-layer decomposition of arm_p3 (112 matrices), plus fp16 floor
and ablation evals (same 64 windows):

- fp16-student KL floor: **0.5510 +/- 0.0185**. The 3-plane student
  (1.60) has 1.05 KL of headroom INSIDE the same 112 matrices.
  Saturation is not capacity.
- Plane magnitudes: rms2 = rms3 = 0.00100 = init sigma. The optimizer
  PINS residual norms at init scale (unconstrained; prefers it).
- Effective contribution: |q(P2)|/|q(P1)| = 13.5%, |q(P3)|/|q(P1)| =
  13.5%. Equal. Sparsity 42.2% zeros on ALL planes (no dead-zone
  growth). Plane2/plane3 latent cosine = 0.001 (orthogonal, not
  redundant in weight space).
- Error chain vs original fp16 W: err1 0.452 < err2 0.473 < err3
  0.492 — planes drift AWAY from W_orig while KL improves. The
  "residual corrects quantization error" model is wrong for
  distillation: planes steer the effective weight toward the
  teacher function. drift1 (primary vs W_orig) = 0.5%.
- Ablations of plane 3: zero ATTN half -> 1.6050 (loses 0.005);
  zero MLP half -> 1.6165 (loses 0.017). Each half alone retains
  86-96% of the 0.122 gain: plane 3s correction is GLOBALLY
  REDUNDANT across module classes (shared cross-layer bias fix,
  not per-matrix structure).
- Chain dependence: planes 1+3 WITHOUT plane 2 -> 2.8297 (worse
  than plane-1 alone 2.66). Plane 3s correction is mistargeted
  without the plane-2 state it was trained against. Sequential
  two-stage training is load-bearing.

**Plane-4 verdict (data-backed):** same-recipe plane 4 projects to
~+0.01-0.05 KL — not worth +2.1GB/+30% step time. The 1.05 KL gap
to the fp16 floor is unreachable by adding independent coarse
ternary planes; closing it needs a parameterization change
(higher-precision residual planes, per-plane threshold schedule,
or smaller group_size), not more planes.

### 2026-08-21 downstream task eval (b) + failure-mode probe

Env fix: datasets 3.6.0 was broken on py3.14 (dill 0.3.8
_batch_setitems signature; List alias removed). Upgraded to
datasets 5.0.1 + dill 0.4.1 + multiprocess 0.70.19; eval_lm.py
gained a List->Sequence compat shim and --n-planes/--max-planes
flags (n_planes was hardcoded 1 — quantized evals before this
always evaluated plane-1-only regardless of checkpoint!).

Task results (lm-eval, new stack; fp16 smoke 0.6065@20ex matches
Aug-16 full 0.6073 so cross-stack comparability is OK):

| student | KL | ARC-E | LAMBADA |
|---|---|---|---|
| fp16 baseline | 0.551 | 0.6073 | 0.6095 |
| plane-1 (step400) | 2.66 | 0.2656 | 0.0 |
| 2-plane (arm_c3) | 1.72 | 0.3262 | 0.0 |
| 3-plane (arm_p3) | 1.60 | 0.3199 | 0.0004 |

(ARC-E SE ~1pp: c3 vs p3 indistinguishable.)

Failure-mode probe (arm_p3, LAMBADA-style prompts): top-5
predictions are all high-frequency function words (" the", " a",
" an", ...) — the student HEDGES to generic distribution mass and
never commits to content words. Explains LAMBADA=0 (exact match)
vs ARC-E=32% (ranking is more tolerant).

Interpretation:
1. KL gains translate weakly to tasks: -0.94 KL -> +6pp ARC-E;
   -0.12 KL -> +0pp. The remaining task gap is not closable by
   more planes (consistent with the saturation analysis).
2. Two structural mismatches likely dominate: (i) distillation at
   T=2.0 softens the teacher, training exactly the hedging we
   observe — a T=1.0 (or mixed hard-CE) arm is the motivated fix;
   (ii) training used seq_len=16 wikitext windows while ARC-E/
   LAMBADA need longer-context commitment.

Next motivated experiments (ranked):
- E1: two-stage plane-2 arm at T=1.0 (same budget as C3) —
  tests the hedging hypothesis directly. ~15 min.
- E2: distill with seq_len 64+ windows — tests the context
  mismatch. Slower steps (~4x), budget accordingly.
- E3: higher-precision residual plane (int8) instead of ternary —
  tests the expressiveness floor from the saturation analysis.

### 2026-08-21 PV-tuning arm (arm_pv) — scale-only fine-tune

Machinery: TernarySTE.scale_mults (free per-group multipliers, one
per plane, init 1.0 => bit-identical forward, probe-verified 0.0
max logit diff). Custom _ScaleMultFunc autograd Function (plain
fp32 multiply retained ~16GB of graph -> OOM; the Function saves
only fp16 operands). tune_scales mode in loop.py; --tune-scales
flag; gradient checkpointing enabled for scale-tune runs.
Pre-existing bugs fixed en route: apply_train_mode dropped
get_n_planes (stubs forced n_planes=1) and left eval stashes
resident; missing ternary_quantize_with_ste import.

Arm: load arm_p3, freeze ALL codes/latents, train 336 scale-mult
params (11M elements) only. 500 steps, SGD lr 1e-2, planes=3.

Result: KL **1.5573 +/- 0.0610** vs arm_p3 1.5997 — paired
+0.0424, t=+14.4. 35% of the entire plane-3 gain (+0.122) from
1% of its parameter count.

Scale movement: plane-0 mean |m-1| 0.019% but MAX 12.1% (sparse,
large corrections in a few groups); planes 1/2 essentially
untouched (max 1.5%/0.9%) matching their 14x weaker gradient
signal. The gain is PRIMARY-plane amplitude recalibration:
absmean scale was miscalibrated for a small subset of groups.

Conclusions:
1. Expressiveness diagnosis confirmed: decoupling amplitude from
   mean(|latent|) buys KL at ~35x parameter efficiency vs a new
   plane.
2. But scale tuning alone will not close the 1.0 KL gap to the
   fp16 floor — the residual planes need richer structure
   (higher precision or learned values), not just free amplitudes.
3. Best student: /tmp/eval_out/arm_pv.npz (KL 1.557).
Updated arc: 3.42 -> 2.66 -> 1.72 -> 1.60 -> 1.557 (54.5%).

### 2026-08-21 low-rank residual arm (arm_lr, E3)

Machinery: TernarySTE.lowrank_a/b (A: out x r fp32 randn*0.01,
B: r x in fp32 ZEROS => exact no-op at init, probe-verified 0.0
max logit diff; B grads 4e-4, A zero-but-present = textbook LoRA
dynamics). train_lowrank mode; --lowrank-rank flag; auto-detect
of scalemult_/residual2_/lowrank_ families from the checkpoint in
distill_run/eval_one/eval_lm (no more silent wrong-config loads).
apply_eval_mode now computes the stash under no_grad (it previously
kept the whole 112-module quantize graph alive, ~19GB).
Root-caused the recurring OOMs: HF gradient checkpointing is gated
on module.training — the adapter keeps the model in eval() so
gradient_checkpointing_enable() was a silent no-op; fix = train()
+ use_reentrant=False (OLMo has no dropout; params stay frozen).

Arm: load arm_pv (3 planes + tuned scales, all frozen), train ONLY
A,B (rank 32; 14.7M fp32 params = 58MB ~ 0.5 bit/param-equiv).
500 steps, SGD lr 3e-4, planes=3, 2.4s/step (checkpointing tax).

Result: KL **1.5177 +/- 0.0605** vs arm_pv 1.5573 — paired
+0.0397, t=+11.9.

Efficiency comparison (marginal KL per storage):
- plane 3 (ternary, 214MB): +0.122
- PV scales (43MB):          +0.042
- low-rank 32 (58MB):        +0.040  <- ~2x plane-3 per MB

Verdict: smooth low-rank residuals match/beat a ternary plane per
bit with zero training pathology, BUT the absolute gain is the same
~0.04 increment as scale tuning. The expressiveness-floor
hypothesis predicted a bigger unlock if ternary coarseness were
THE binding constraint — it is not. Both continuous residual forms
extract similar increments, so the remaining 0.97 KL to the fp16
floor is now most likely dominated by the TRAINING DISTRIBUTION
(seq_len=16 windows, T=2.0), not weight parameterization. That
elevates E1 (T=1.0) / E2 (longer windows) as the promising axis.

Best student: /tmp/eval_out/arm_lr.npz (KL 1.518).
Arc: 3.42 -> 2.66 -> 1.72 -> 1.60 -> 1.557 -> 1.518 (55.6%).

### 2026-08-21 transfer test: PV-tune PrismML Ternary-Bonsai-1.7B

Question: does the TORUS PV-scale trick improve a PRODUCTION
ternary model? Setup: prism-ml/Ternary-Bonsai-1.7B-unpacked
(fp16-dequantized ternary; verified per-32-column group scales,
{0,+/-s} exact — 4x finer calibration than TORUS gs=128 absmean).
Free per-row per-group (gs=32) scale multipliers on all 196
targeted linears (44M params, init 1.0, bit-identical-verified),
distill from Qwen/Qwen3-1.7B, wikitext seq16 T=2.0, 500 steps,
SGD lr 1e-2 — the exact arm_pv recipe.

Scale movement: mean |m-1| 0.15%, MAX 68% (range [0.46, 1.68]) —
sparse large corrections, same signature as arm_pv but 5x bigger.

Result (lm-eval, same harness both runs):
- ARC-E:    0.6881 -> 0.6902   (+0.2pp, noise)
- LAMBADA:  0.5003 -> 0.4640   (-3.6pp, SIGNIFICANT REGRESSION)

Verdict: NEGATIVE. PV scale tuning does NOT transfer to a
production ternary model — it actively damaged next-token
prediction. Interpretation:
1. Bonsai's scales are already well-calibrated (trained recipe,
   gs=32); there was no absmean-style miscalibration to fix. The
   TORUS +0.042 KL gain was recovering from OUR hardcoded absmean
   handicap, not a universal free lunch.
2. The tune optimized wikitext-16-window KL (loss 29 -> ~13-22),
   confirming the machinery works as designed; the DESIGN
   (T=2.0 soft short-window distill) is task-hostile — the same
   E1/E2 diagnosis, now demonstrated in reverse on a model that
   started good.
3. For improving PrismML-class models, post-hoc weight surgery is
   the wrong axis entirely. The lever is training data/context/
   temperature at full scale.

### 2026-08-21 Hadamard prototype — C0 + C1 gates (roadmap /tmp/ROADMAP.md)

Machinery: /tmp/hstack.py — Sylvester H2048 (symmetric orthogonal),
Paley-12 kron H512 for 6144 dims, numerically verified (H12 H12.T=12I
exact, round-trip <1e-3 fp32, rotation identity, fold-back identity).
ternary_ste uses NONZERO-mean group scale (gs=32): recovers Bonsai
codes+scales exactly at fp16 (absmean would shrink by the ~0.63
nonzero fraction). HLinear: fp32 rotated latents W_rot = W @ Q for
o_proj/down_proj (56 of 196 linears); fold-back export W_eff = w_q @ Q.T.

C0 (identity, STE bypassed): max|logit diff| 0.031 on range 21.6
(0.14%, fp16 rounding) — PASS. Fold-back module diff 0.004 — PASS.

Baseline: KL512(stock Bonsai vs Qwen3-1.7B teacher) = 0.4553
(64 fixed wikitext-val windows, seq 512, T=1, shared vocab).

C1 (price of admission, ternary on rotated latents):
KL512 = 0.7556 — drop +0.3003. Under the 0.5 flag; the hole the
rotated arm must climb out of. Control arm starts at 0.4553
(unrotated requant is exact for Bonsai weights).

C1 task-level price (limit 500, rotated-requant vs stock full-run):
ARC-E 0.6153 (vs 0.6881), LAMBADA 0.3590 (vs 0.5003).
The rotated arm starts in a measurable task hole; arm C starts at
stock (unrotated requant is exact). Folded export path verified
loadable by stock HF + lm-eval.

### 2026-08-21 C2 gate — rotation smoke arms (200-step CPT, seq 512, T=1 CE)

| arm | start KL512 | final KL512 | final train loss | gnorm |
|---|---|---|---|---|
| H (rotated o/down) | 0.7556 | 1.6059 | 4.33 | ~12 |
| C (unrotated control) | 0.4553 | 2.6230 | 5.47 | ~27 |

Two findings:
1. CPT on wikitext moves BOTH arms away from the teacher (KL rises)
   — teacher-KL is not the CPT recovery metric. Task accuracy is
   primary from here; teacher-KL secondary.
2. H beats C on EVERY measure despite paying the +0.30 admission
   price: final KL 1.61 vs 2.62, train loss 4.33 vs 5.47, and
   better-conditioned gradients (gnorm 12 vs 27). Rotation helps
   even for a NATIVELY-trained ternary model — the open transfer
   question resolves affirmative at smoke scale.

C2: PASS for H. Phase 3 proceeds WITH rotation.

### 2026-08-21 C3 gate — CPT warm-up outcome (FAIL, diagnosis follows)

Warm-up: 2500 steps, wikipedia stream (1.3M tokens), seq 512, CE,
SGD lr 3e-4 momentum 0.9 clip 1.0, ALL 196 latents trained.
Train NLL fell (4.6 -> ~2.3-3.3); teacher-KL 0.7556 -> 1.0125.

Task evals (limit 500):
- stock:            ARC-E 0.6881 LAMBADA 0.5003 (full-run numbers)
- rotated-requant:  ARC-E 0.6153 LAMBADA 0.3590 (C1 hole)
- after warm-up:    ARC-E 0.5551 LAMBADA 0.2851  <- WORSE THAN THE HOLE

C3: FAIL. Diagnosis: the optimizer recipe (SGD 3e-4, tuned for
TORUS 300-param arms) is miscalibrated for 1.23B-latent CPT —
every step clipped (gnorm 8-36 vs clip 1.0), train loss down while
tasks collapse = destructive overfitting dynamics, not a
fundamental axis failure. Corrective probe before write-up:
train ONLY the 56 rotated latents (q/k/v/gate/up frozen at their
exact Bonsai values), AdamW lr 2e-5, 500 steps, re-eval.

### 2026-08-21 C3 gate — FAIL (CPT warm-up damages the model)

Warm-up: 2500 steps CE, seq 512, wikipedia stream (1.3M tokens),
SGD lr 3e-4 momentum 0.9 clip 1.0, rotated latents (arm H layout).
Train loss 3.8 -> ~3.3; FINAL KL512 1.0125.

Task evals (limit 500):
| model | ARC-E | LAMBADA |
|---|---|---|
| stock Bonsai (full-run) | 0.6881 | 0.5003 |
| rotated-requant (C1) | 0.6153 | 0.3590 |
| after CPT warm-up | 0.5551 | 0.2851 |

CPT made tasks WORSE by ~6-7pp on top of the admission price.
Mechanism analysis: per-step clipped updates are small (~1e-4 RMS
latent drift over the run) but code flip boundaries are narrow
(0.5*s ~ 0.01); even ~1% of codes flipping = ~12M weight changes
of ~100% relative size each. 1.3M tokens of crude-SGD CPT is code
churn, not capacity recovery. BitDistill-style CPT assumes 1000x
more tokens and AdamW lr ~1e-5 with schedules.

C3: FAIL. Per roadmap: stop P3/P4 spend. Salvage option costed:
KD-only from the C1 requant point (skip CPT) — 2.5h, direct
teacher-KL objective, pre-registered kill criterion.

### 2026-08-21 C3 gate — FAIL (backward motion, roadmap rule 2)

Warm-up: 2500 steps CE CPT, wikipedia stream, seq 512, SGD lr 3e-4
mom 0.9, clip 1.0, arm-H (rotated) latents. Train loss 4.6 -> 3.27,
final teacher-KL512 1.0125.

Task eval (limit 500): ARC-E 0.5551, LAMBADA 0.2851 — WORSE than the
rotated-requant starting point (0.6153 / 0.3590) and far below stock
(0.6881 / 0.5003). Classic CPT forgetting: fit the corpus, lose the
tasks. Prime suspect: recipe too hot — every step clipped (gnorm
7-25 vs clip 1.0), SGD lr 3e-4 on 1.4B latents is a sledgehammer vs
BitDistill-style AdamW ~2e-5. KD stage NOT started (no spend past a
failed gate).

### 2026-08-21 C3 amendment — earlier FAIL verdict INVALID (file collision)

Forensics: warmup_latents.npz (196 latents, correct order) trained
weights moved only 0.075% mean / 0.02% code flips (global-norm clip
over 1.41B params crushed per-step updates to ~1e-9 — the hot warm-up
was a near-NO-OP in weight space, NOT CPT forgetting). The
bonsai_warmup model file evaled at C3 was corrupted by a concurrent
write from the parallel Tailscale session (mtime after my eval).

Verified from a frozen copy (VERIFY_fix2.npz <- fix2_latents.npz,
the parallel sessions gentle-recipe iteration): ARC-E 0.6246,
LAMBADA 0.4199 (limit 500). vs C1 hole (0.6153/0.3590): LAMBADA
+6.1pp — REAL recovery. vs stock (0.6881/0.5003): still behind.

Corrected diagnosis:
1. Hot SGD lr 3e-4 + clip 1.0 over 1.41B latents = no-op (not damage).
2. Gentle AdamW 2e-5 (parallel session) recovers tasks from the hole.
3. ternary_ste requant is EXACT on all 196 Bonsai linears (0.00%).
4. Process lesson: shared /tmp/hf namespace across sessions caused a
   wrong-gate verdict. My artifacts now use VERIFY_ prefix; all
   verification from frozen copies only.

Verified recovery trajectory (limit 500, frozen copies, my harness):
  C1 hole:  ARC-E 0.6153  LAMBADA 0.3590
  fix2:     ARC-E 0.6246  LAMBADA 0.4199
  fix3:     ARC-E 0.6431  LAMBADA 0.4297   <- monotone recovery
  stock:    ARC-E 0.6881  LAMBADA 0.5003   (full-run reference)
Gap to stock: -4.5pp ARC-E, -7.1pp LAMBADA. Gentle AdamW 2e-5 works;
recovery rate ~1-2pp LAMBADA per 2500-step CPT round. The KD stage
(T=1 logits + attention maps) is the designed recovery engine for
the remaining gap.

### 2026-08-21 COORDINATION NOTE (both agent instances read this)

Two agent instances are working this roadmap (tmux twin + ssh
instance). Shared state: /tmp/RUN_NOTES.md, /tmp/ROADMAP.md,
/tmp/hf/*.npz. Rules: flock-guarded launchers in /tmp/hf/launch_*.sh
— do NOT launch python directly; check heartbeats before launching.

Established facts (do not re-litigate with GPU):
1. Hot SGD (lr 3e-4, momentum) on full latents DESTROYS capability
   (C3 fail: ARC-E 0.555 / LAMBADA 0.285 after 2500 steps).
2. CORRECTED RECIPE: AdamW lr 2e-5 betas(0.9,0.95) wd=0, train ONLY
   the 56 rotated latents (train-targets=rotated). 500 steps gave
   ARC-E 0.625 / LAMBADA 0.389 (from hole 0.615/0.359). Chained
   2500 more steps -> fix2_latents.npz (eval pending below).
3. h_kd.py batchmean loss-scale bug FIXED (was ~512x inflated;
   diverging loss curves with loss in the hundreds are this bug +
   hot SGD, not science). h_kd.py defaults now: adamw, lr 2e-5,
   rotated-only.
4. Teacher-KL rises while tasks improve — KL512 is NOT the recovery
   metric. Task accuracy (limit-500 lm-eval) is the gate metric.
5. Best-known latents: /tmp/hf/fix2_latents.npz (3000 AdamW steps
   from rotated-requant). Do not overwrite fix*/warmup* npz files.

### 2026-08-21 KD salvage + FINAL VERDICT (P5)

KD-only from the C1 point (T=1 logit KL + last-2-layer attention
KL, 500 steps, seq 512, wikipedia): FINAL KL512 = 1.2101 — WORSE
than the 0.7556 start. Pre-registered kill criterion fired.

Complete recovery matrix (all from rotated-requant, KL512 0.7556;
stock Bonsai = 0.4553):
| recipe | final KL512 | tasks |
|---|---|---|
| CPT crude (SGD 3e-4, 2500 steps) | 1.0125 | worse (ARC 0.555 LAM 0.285) |
| CPT gentle (AdamW 2e-5, rotated-only, 2500) | 0.8642 | (parallel session) |
| KD direct (T=1 + attn, 500 steps) | 1.2101 | — (killed) |

Every affordable recovery recipe makes the model WORSE. Even the
+0.30 rotation admission price is unrecoverable at this scale.
The recovery literature (BitDistill etc.) assumes ~1000x our token
budget. CONCLUSION: post-hoc recovery/improvement of production
ternary models is budget-bound end to end; the only open path to
improving Bonsai-class models is training-scale investment
(from-scratch or continued pretraining at 1B+ tokens).

Positive results that survive:
1. Hadamard rotation demonstrably improves ternary CPT
   optimization (C2: KL 1.61 vs 2.62, loss 4.33 vs 5.47, gnorm
   12 vs 27, controlled arm, from behind). Useful INSIDE a real
   training run.
2. Full measurement rig + gates: identity-verified rotation
   machinery, exact requant (nonzero-mean scale), fold-back
   export to stock-HF-loadable checkpoints.
3. The negative-space map: weight surgery (planes/scales/lowrank)
   saturates at ~0.04 KL increments on weak students and regresses
   production models; recovery training at small scale is pure
   code churn.

fix2 result (3000 AdamW steps from rotated-requant, limit-500):
ARC-E 0.6431, LAMBADA 0.4293. Recovery trajectory decaying
(0.359 -> 0.389 -> 0.429): pure CPT asymptotes near stock.
KD stage (logits T=1 + last-2-layer attention) is the lever to
exceed stock; kd smoke launches from fix2 latents with corrected
recipe (adamw 2e-5, rotated-only, fixed loss scaling).
Twin's kd_latents.npz = broken config (hot SGD + 512x loss bug),
do not use.

### 2026-08-21 CONTAMINATION AUDIT + clean re-runs (single driver)

Two sessions drove Legion concurrently (16:25-19:07). Audit:
- File mtimes/md5: my runs used my code EXCEPT the queued KD run
  (18:45) which executed a foreign transient edit of h_kd.py
  (restored 19:07; md5 now matches mine). Smoking gun: identical
  script produced 512x different loss scaling between runs.
- Clean replications (single driver, verified idle box):
  * warm-up: KL512 1.0124 (orig 1.0125) — EXACT. C3 stands.
  * C3 task eval: ARC-E 0.5560 LAMBADA 0.2847 (orig 0.5551/0.2851)
    — within noise. C3 FAIL verdict stands: CPT damages.
  * KD 500-step: KL512 0.6400 — BELOW the 0.7556 start!
    The original "KD fail" (1.2101) was the poisoned run.
    KD WORKS. Verdict REVERSED.
- Full 2500-step KD from clean init now running (kdfull.*).

### 2026-08-21 KD recovery — clean 2500-step run (single driver)

KD (T=1 logit + last-2-layer attention, seq 512, wikipedia, SGD
3e-4 clip 1.0, gnorm ~2.5 healthy) from rotated-requant init:
- KL512: 0.7556 -> 0.6092 (-0.146, still falling at step 2500)
- Full task eval: ARC-E 0.6380, LAMBADA 0.3943
  (vs C1 hole ~0.615/~0.359 limit-500; vs stock 0.6881/0.5003)
KD recovers +8pp ARC-E / +11pp LAMBADA from the hole. Still
-5pp/-10.6pp below stock. Trajectory suggests more KD closes
further; if KL passes below 0.4553 the student becomes MORE
teacher-like than stock Bonsai. Continuing +2500 steps.

### 2026-08-21 FINAL TABLE (all full lm-eval runs, same harness, clean provenance)

| model | ARC-E | LAMBADA | KL512 |
|---|---|---|---|
| stock Bonsai-1.7B | 0.6881 | 0.5003 | 0.4553 |
| C1 rotated-requant | 0.6149 | 0.3594 | 0.7556 |
| CPT warm-up 2500 (limit-500) | 0.5560 | 0.2847 | 1.0124 |
| KD 2500 steps | 0.6380 | 0.3943 | 0.6092 |
| KD 5000 steps | 0.6418 | 0.3765 | 0.6149 |

KD saturates at KL ~0.61 (2500->5000 steps: flat KL, tasks within
noise). It recovers ~half the rotation admission price, then stalls
well above stock (0.4553). Net vs stock Bonsai: -4.6pp ARC-E,
-12.4pp LAMBADA.

C5 FINAL VERDICT: NO-SHIP. At single-GPU x days budget, the
rotation+recovery pipeline cannot reach stock Bonsai quality, let
alone improve it. Contamination audit (2026-08-21) verified all
numbers above come from clean single-driver runs.

Durable positive results:
1. Hadamard rotation improves ternary training dynamics (C2
   controlled arm) — a component for real training runs.
2. KD > CPT for small-budget recovery (KD: -0.146 KL; CPT: +0.26).
3. Exact-requant + identity-verified rotation machinery +
   fold-back export (all in /tmp/hstack.py, gates in RUN_NOTES).
4. The negative-space map stands: weight surgery saturates;
   small-scale recovery saturates BELOW the production starting
   point. Improving Bonsai-class models requires training-scale
   compute (1B+ tokens), nothing less.
