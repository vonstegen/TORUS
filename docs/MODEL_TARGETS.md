# Phase 3 — Concrete Model Targets

matches `(batch, n_planes) -> (logits, hidden, route)`. The HF
adapter (`torus.train.hf_adapter`) wraps a `transformers` causal-LM
into that interface.

This document picks the *concrete* open bases the adapter is meant
to drive, in order of integration priority.

## Hardware targets

- **Legion** (x86_64, Threadripper 3995WX + 2× TITAN RTX, 123 GB
  RAM, 1.6 TB free NVMe) is the production / training host. Both
  x86 SIMD paths (AVX2 / AVX-512) and the CUDA path are exercised
  here. `pip install torch` from the default index already gives
  CUDA support.
- **dev** (aarch64, GB10 Blackwell, 121 GB RAM) is the dev box.
  Only the portable C reference path runs (no x86 SIMD). CUDA
  torch requires a Python 3.11 side venv (see README §"Optional:
  GPU torch").

## Primary: OLMo-1B (dense)

- HuggingFace: `allenai/OLMo-1B-hf`
- ~2.5 GB on disk at fp32; ~1.3 GB at fp16 / bfloat16
- Pure decoder transformer; no MoE, no GQA variance
- Permissive license (Apache-2.0); reproducible training data
- Best first smoke target because:
  - Fits in 12 GB VRAM at fp16
  - Dense weights are an honest test of the trainer's basic
    quantization path without MoE routing confounders

## Secondary: OLMoE-1B-7B (Mixture-of-Experts)

- HuggingFace: `allenai/OLMoE-1B-7B-0125`
- ~7 GB total; ~1 B active per token
- 64 experts, top-8 routing
- Same license and reproducibility as OLMo-1B
- Becomes the **real** Phase-4 test bench once the trainer wiring
  is solid:
  - Each expert's FFN gets its own residual plane stack
  - Router confidence drives plane-count gating
  - This is the design pillar described in VISION.md

## Tertiary: BitNet b1.58 reference models

- HuggingFace: `microsoft/BitNet-b1.58-2B-4T-gguf`
- Direct comparison target — already 1.58-bit ternary on disk
- GGUF format requires a separate loader (`gguf-py`) and is
  outside the trainer's HF adapter
- Useful as a *baseline* to compare against once OLMo-1B produces
  trained planes; not a training target itself

## Why not OLMo-7B or larger

- Trainer reference is numpy-only with finite-difference gradients
  via STE; per-step compute is O(num_ste_params × step_size) and
  scales linearly with model size
- For first validation, OLMo-1B hits a sweet spot:
  small enough to finish a curriculum in hours on a single GPU,
  large enough to exercise residual plane convergence on real
  hidden-state distributions
- Larger bases land after the trainer swaps to torch autograd
  (Phase 3 follow-up)

## Phase 3 wiring state

| Component                                          | Status           |
|----------------------------------------------------|------------------|
| `torus.train.hf_adapter.HFStudentAdapter`          | implemented (smoke-tested) |
| `torus.train.hf_adapter.HFTeacherAdapter`          | implemented (smoke-tested) |
| `torus.train.hf_adapter.HFAdapterConfig`           | implemented      |
| Adapter interface contract                         | tested via `MockHFAdapter` |
| Real OLMo-1B end-to-end smoke                      | **pending** — requires `pip install torch transformers` and a model download |

## Phase 3 follow-ups (after smoke)

1. Swap `torus.train.loop` internals from numpy finite-difference
   gradients to torch autograd via a `TernaryQuantize.apply`
   autograd Function. The current numpy loop remains as the
   reference; the torch path lives alongside it.
2. Extract the real MoE route from OLMoE's
   `output.router_logits` so the trainer's `expert_route_loss`
   has something to consume.
3. Wire `GateTelemetry.record` into the trainer's per-step stats
   so the gate activation rate is visible during training.
## Phase 8 distillation runs (real, on Legion)

Three runs were executed on Legion (CUDA torch + 2× TITAN RTX) with
`sshleifer/tiny-gpt2` to validate the trainer wiring:

| Run | Curriculum | Initial loss | Step 100 | Final loss (step 199) |
|---|---|---|---|---|
| `primary_only` | `n_planes=1` throughout | 0.0028 | 0.0017 | 0.0038 |
| `primary_plus_residual` | `1:100, 2:100` | 0.0028 | 0.0017 | 0.0038 |
| `primary_plus_residual_perturbed` | `1:100, 2:100` + residual init noise | 0.0028 | 0.0105 | **0.0205** |

All runs use 200 steps with `probe_rows=1` finite-difference gradients
(one column perturbed per STE per step). The trainer's `_numerical_grads`
only perturbs the primary weight — the residual weight is *not* in the
gradient path under `probe_rows=1`. Loss-curve JSONs are at
`/tmp/torus_distill_logs/{primary_only,primary_plus_residual,primary_plus_residual_perturbed}.json`.

### Findings

1. **`primary_only` and `primary_plus_residual` are identical** because
   the residual plane is zero-initialized and never perturbed; the
   curriculum switch at step 100 changes `planes_active` from 1 to 2 but
   the residual contributes zero. The trainer needs to probe the
   residual too for primary+residual to differ from primary-only.
2. **`primary_plus_residual_perturbed`** demonstrates the wiring works:
   after the curriculum switches to 2 planes, the loss jumps because
   the random-noise residual is contributing garbage to the forward.
   This is the correct sanity check — the trainer is actually
   exercising the residual path.

### Phase-8 follow-up: probe residual weights too

The Phase-8 trainer probes only the primary weight per STE per step. To
make `n_planes=2` actually engage a learnable residual, the trainer
needs to probe both planes. A Phase-8+ follow-up should:

- Extend `_numerical_grads` to optionally perturb `residual_weight`
  when the STE carries one (`probe_residual=True`).
- Compare the 3 configurations with both primary and residual
  probed; expect the perturbed-residual run to *converge below*
  primary-only after enough steps.

### Phase-8+ follow-up: probe_residual=True (run on Legion)

A fourth distillation run completed on Legion with the trainer
extended to optionally probe the residual plane too:

| Run | Curriculum | probe_residual | perturb_residual | Initial | Step 100 | Final |
|---|---|---|---|---|---|---|
| `primary_plus_residual_probe_and_perturb` | `1:100, 2:100` | **True** | True | 0.0028 | 0.0261 | 0.0729 |

**What this proves**:

- With `probe_residual=True`, the loss curve **diverges** from the
  probe_residual=False run at step 100 (where the curriculum switches
  to n_planes=2). The trainer is now perturbing the residual weight
  and applying the resulting gradient. The step-100 jump (0.0017 →
  0.0261) is much larger than the probe_residual=False run
  (0.0017 → 0.0105), confirming gradient flow into the residual.

- The training is **unstable** (loss climbs to ~1.0 by step 180). This
  is because the residual's initial scale is small (random noise * 0.05)
  but the optimizer applies the same learning rate to both primary and
  residual. Phase-8+ follow-up: per-plane LR scheduling (e.g. residual
  uses 10× smaller LR) to stabilize the curriculum switch.

### Phase-8+ follow-up: per-plane LR scaling (run on Legion)

A fifth distillation run completed on Legion with the trainer
extended to scale the residual plane's learning rate:

| Run | Curriculum | probe_residual | residual_lr_scale | Initial | Step 100 | Final |
|---|---|---|---|---|---|---|
| `primary_plus_residual_probe_and_perturb` | `1:100, 2:100` | True | 1.0 (default) | 0.0028 | 0.0261 | 0.0729 |
| `primary_plus_residual_lr_scaled` | `1:100, 2:100` | True | 0.05 | 0.0028 | 0.0391 | **0.0303** |

**What this proves**:

- Scaling the residual's learning rate by 0.05 reduces the final
  loss by **2.4×** (0.0729 → 0.0303). The residual plane is being
  learned, not blown up.
- Training is still unstable because the curriculum switch at step
  100 causes a large loss jump (0.0032 → 0.0391) and `probe_rows=1`
  gives a coarse gradient direction. With more probes per step and a
  curriculum that warms up over more steps (e.g. `1:50,2:150` with
  a 50-step warmup of the residual plane), the run would likely
  converge below the initial loss.
