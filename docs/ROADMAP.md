# Roadmap

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

**Deliverable**: accelerated ternary GEMM kernels for the dev box
(P620, Threadripper 3995WX + 2× TITAN RTX 24 GB + 128 GB ECC), keeping
the public API stable.

- [x] Packed 2-bit / 1.6-bit weight layout (`torus.quant.packing`)
- [x] CPU reference kernels with op counts
      (`torus.core.kernels`: dense / sparse / unrolled)
- [x] Memory-hierarchy policy (`torus.core.memory`, `place_planes`)
- [x] Gate telemetry (`torus.core.GateTelemetry`)
- [x] Kernel spec (CUDA + AVX-512) (`docs/KERNELS.md`)
- [ ] CUDA implementation of `docs/KERNELS.md` §3
- [ ] AVX-512 / AVX2 implementation of `docs/KERNELS.md` §4
- [ ] Hardware-side gate activation tracking
- [ ] End-to-end benchmark vs `bitnet.cpp` baseline


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

**Acceptance**: a 70B-class ternary model runs on the P620 at
interactive speed with the gate active; toggling the gate measurably
trades speed for quality on a held-out task.

## Phase 5 — Research: Custom Silicon (long horizon)

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
  may add torch / MLX / custom runtimes behind the same interfaces.
- **Test before you ship.** Every phase ends with a green test suite
  and at least one measurable deliverable.
- **Open by default.** All code, docs, and (when possible) weights go
  under permissive licenses.

## Hardware target

| Component            | Spec                                |
|----------------------|-------------------------------------|
| CPU                  | Threadripper 3995WX, 64c/128t       |
| RAM                  | 128 GB ECC                          |
| GPU                  | 2× NVIDIA TITAN RTX 24 GB, NVLink   |
| Storage              | 2 TB NVMe                           |
| Network              | 10 GbE                              |

This machine is a useful single-workstation testbed, not a frontier
training rig. Phase 3 may need additional compute for larger
students; Phase 1–2 + 4 are sized to fit comfortably on this box.
