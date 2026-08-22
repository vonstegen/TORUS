# Roadmap

> **SUPERSEDED — historical record only.** As of 2026-08-22, TORUS operates
> under the gated research program in `research/ROADMAP.md` +
> `research/OPERATING-PLAN.md`, per the review package in
> `docs/TORUS-feedback/`. The phase plan below documents completed engineering;
> results it references predate the experimental contract and are
> **engineering-validation evidence only** (see `research/OPERATING-PLAN.md`
> §4). Note: this document's claim of AVX-512 on the Threadripper PRO 3995WX
> is incorrect (Zen 2 has no AVX-512) — correction tracked as roadmap item 0.3.

TORUS is built in five phases. Each phase ends with a runnable
artifact and an evaluable deliverable — no half-finished layers.

## Phase 1 — Reference Implementation (this repo)

**Deliverable**: pure-Python, framework-free reference impls of the
core primitives with passing tests.

- [x] `torus.quant.ternary` — single-plane ternary quantization
- [x] `torus.quant.residual` — residual-plane stack + compose
- [x] `torus.core.gate` — adaptive residual gate (ALWAYS/NEVER/ADAPTIVE)
- [x] `torus.core.residual_linear` — reference drop-in layer
- [x] `torus.moe.expert_bank` + `router` — MoE scaffolding
- [x] `torus.rlm.context` + `repl` — recursive context-as-variable
- [x] pytest suite covering quantization, gating, layering, MoE, RLM
- [x] ARCHITECTURE / VISION / ROADMAP / KERNELS docs

**Status**: Phase 1 complete.

## Phase 2 — Hardware-aware Kernels (in progress)

**Deliverable**: accelerated ternary GEMM kernels for Legion (x86-64
Threadripper + 2× TITAN RTX) and the dev box (aarch64 GB10), keeping
the public API stable.
- [x] CPU reference kernels with op counts
      (`torus.core.kernels`: dense / sparse / unrolled)
- [x] Memory-hierarchy policy (`torus.core.memory`, `place_planes`)
- [x] Gate telemetry (`torus.core.GateTelemetry`)
- [x] Kernel spec (CUDA + AVX-512) (`docs/KERNELS.md`)
- [x] CUDA implementation of `docs/KERNELS.md` §3
      (`torus.kernels.cuda`, numba-compiled, graceful fallback)
- [x] AVX-512 / AVX2 implementation of `docs/KERNELS.md` §4
      (`torus.kernels.csrc.torus_kernel.c`, portable + SIMD dispatch
      via `torus.kernels.build`; portable C reference also covers
      AArch64 SVE)


## Phase 3 — Training & Distillation (scaffolding complete)

**Deliverable**: a real ternary model trained from an open base on
modest hardware.

- [x] Capability-aware distillation loss that targets *intermediate*
      residual errors, not just final logits (`torus.train.losses`)
- [x] Straight-through estimator for ternary weights (`torus.train.ste`)
- [x] Progressive residual-plane curriculum (`torus.train.curriculum`)
- [x] QAT / distillation training loop (`torus.train.loop`)
- [ ] Realistic model wiring: integrate the trainer with a concrete
      open base (OLMoE ~7B / ~1B active is the first target)
- [ ] Larger targets: 7B–13B dense + Qwen-style 27B
- [ ] Joint gate learning (replacing the heuristic with a small
      head trained alongside residual planes)
- [ ] Eval: perplexity + downstream benchmarks vs full-precision
      baseline
- [ ] Open release of weights + training scripts

**Acceptance**: a ternary model sized for a single workstation reaches
≥ 90% of the FP16 baseline on a representative LM eval harness.

## Phase 4 — Runtime & Local Serving

**Deliverable**: a serving path that takes a TORUS-trained checkpoint
and makes it usable locally, end-to-end.

- [ ] GGUF exporter (for llama.cpp / Ollama)
- [ ] Residual-plane-aware runtime on top of an existing inference
      engine (so existing tooling stays usable)
- [ ] Adapter for RLM-style context engine (the third pillar)
- [ ] CLI: `torus serve path/to/checkpoint --residual on|auto|off`
- [ ] Recipes: long-context summarization, code review, multi-doc QA

**Acceptance**: a 70B-class ternary model runs on the GB10 at
interactive rates with the gate keeping total compute under 1.5x
the primary-plane floor.

## Phase 5 — Research: Custom Silicon (long horizon)

**Deliverable**: a research-grade ASIC or FPGA exploration showing
that residual ternary planes compose cleanly into a hardware-friendly
fabric.


the primary-plane floor.

interactive rates with the gate keeping total compute under 1.5x
the primary-plane floor.

**Deliverable**: a research-grade ASIC or FPGA exploration showing
that residual ternary planes compose cleanly into a hardware-friendly
fabric.

- [ ] Extend existing open ternary accelerator designs (TernaryCore,
      LUT-based ternary arrays) with a second gateable residual plane
- [ ] Power gate the residual lanes when the gate is low (silicon
      energy proportional to gate activation rate)
- [ ] MoE-aware prefetch plane for residual experts
- [ ] Compare against: same chip area in pure ternary, same area in
      INT8
- [ ] Publish findings; do not gate Phase 1–4 progress on this phase

**Acceptance**: a hardware study that quantifies the energy / latency
benefit of treating residual ternary planes as a first-class
hardware resource.

---

## Cross-phase principles

- **The math is the contract.** Each phase's math is fixed by the
  Phase 1 types and tests. Phase 2+ optimizes behind that contract.
- **No heavy dependencies in core.** Phase 1 uses only numpy. Phase 2
 wraps the math in optional `torch` / `numba` integrations.
- **No silent precision drift.** Every ternary plane uses
  `ternary_quantize_with_ste` or `ternary_quantize`. No silent
  casting.
- **Open by default.** All code, docs, and (when possible) weights go
  under permissive licenses.

---

## Hardware targets

The project targets two named environments. Kernel paths are
identical across both — the C reference works on any ISA, the
AVX2 path activates on Legion, and the numba CUDA path activates
on any CUDA-capable GPU.

### `legion` — production / training host

| Component   | Spec                                                |
|-------------|-----------------------------------------------------|
| CPU         | AMD Ryzen Threadripper PRO 3995WX, 64c/128t        |
| RAM         | 123 GB                                              |
| GPU         | 2× NVIDIA TITAN RTX (Turing, sm_75), NVLink         |
| Storage     | 1.8 TB NVMe (1.6 TB free)                           |
| ISA         | x86-64; AVX2 (Zen 2 — **no AVX-512**)                 |
| Driver      | CUDA 13.0 (580.173.02), compute capability 7.5      |

### `dev` — this dev box (where the GB10 lives)

| Component   | Spec                                                  |
|-------------|-------------------------------------------------------|
| CPU         | ARM Cortex-X925, 10c/10t, 3.9 GHz boost               |
| RAM         | 121 GB (114 GB available)                             |
| GPU         | 1× NVIDIA GB10 (Blackwell, sm_120), CUDA 13.0 driver |
| Storage     | 916 GB NVMe (579 GB free)                            |
| ISA         | aarch64; portable C reference path (no AVX)           |
| Driver      | CUDA 13.0 (580.159.03), compute capability 12.1       |

### Notes

- **Legion** is the original P620-style dev box the docs assume.
  The AVX2 kernel path is exercised there; the 3995WX is Zen 2 and
  has no AVX-512, so the AVX-512 dispatch path has never run on this
  hardware.
- **GB10** is the Jetson-Thor-style ARM dev box the recent
  benchmarking was done on. Only the portable C reference path
  runs (no x86 SIMD).
- **Phase 1–2 + 4** are sized to fit comfortably on Legion.
  Phase 3 may need additional compute for larger students.

### Notes on CUDA torch

`torch` with CUDA support is *not* available for Python 3.12 +
aarch64 at the time of writing (PyPI only ships cp310/cp111 CUDA
wheels for ARM). The TORUS CUDA kernel uses `numba` instead and
runs on either GPU out of the box. To get a CUDA-enabled torch on
the dev box (so the HF adapter can drive a real model on the GPU),
use a Python 3.11 side venv; see README §"Optional: GPU torch".
On Legion (x86_64) `pip install torch` from the default index
already gives CUDA support.
