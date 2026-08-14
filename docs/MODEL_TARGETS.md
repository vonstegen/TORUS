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