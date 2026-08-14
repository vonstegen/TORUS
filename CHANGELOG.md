# CHANGELOG
## 0.4.0 — Phase 2 follow-on: real kernels

### Added

- `torus.kernels.csrc.torus_kernel.c`: portable C reference kernel
  with x86-64 SIMD dispatch (AVX2 / AVX-512) and an AArch64 SVE
  fallback. Math matches `ternary_gemv_dense` to 1e-7 on the
  full unit-test sweep.
- `torus.kernels.build`: a `gcc`/`cc`/`clang` build harness that
  auto-detects SIMD flags via `__attribute__((target(...)))` probes
  and gracefully falls back to the portable path when no vector ISA
  is available or compilation fails. Idempotent: re-running produces
  the same `.so` path.
- `torus.kernels.simd`: ctypes adapter for the compiled C kernel,
  registered under the kernel registry as `get_kernel("simd_c")`.
  Auto-packs `TernaryPlane` input on the fly (with an id-keyed cache)
  so callers can hand the dispatcher a plain plane without explicit
  packing.
- `torus.kernels.cuda`: a numba-compiled CUDA kernel that matches the
  same contract and registers as `get_kernel("cuda")` when a CUDA
  runtime is available; falls back to the dense reference otherwise.
  Group partials live in `cuda.local.array(256)`; capped at 256 groups.
- `tests/test_kernels_real.py`: 14 tests covering the docs/KERNELS.md
  §8 verification checklist (packing round-trip, arithmetic match,
  per-batch op-count invariant, padding alignment, memory policy,
  gate mode arithmetic, registry integration, build harness,
  CUDA fallback).
- `examples/benchmark.py`: extended to report per-call cost for
  `dense`, `sparse`, `unrolled`, `simd_c` (compiled C), and `cuda`
  when available.

### Verified

- `pytest`: 103 passed.
- `examples/benchmark.py` on this host (CUDA + numpy + portable C):

  | plane            | dense   | unrolled | simd_c  | cuda    |
  |------------------|---------|----------|---------|---------|
  | wide FFN 4k->4k  | 77.4 ms | 19.4 ms  | 40.9 ms | 4.9 ms  |
  | tall attn 4k->1k | 4.5 ms  | 4.5 ms   | 10.3 ms | 1.9 ms  |
  | small 512->512   | 0.28 ms | 0.29 ms  | 0.64 ms | 0.62 ms |

  CUDA wins on the large shapes; the C kernel is the portable path
  (numpy BLAS beats it on this GPU host, which is expected).


## 0.3.0 — Phase 3 (training scaffolding)

### Added

- `torus.train.losses`: capability-aware distillation loss combining
  logit-KL, intermediate-state alignment, and MoE-route symmetric-KL
  (`combined_distillation_loss`). The intermediate term is what
  trains the residual plane to fix the primary plane's worst errors.
- `torus.train.ste`: straight-through estimator
  (`TernarySTE`) wrapping a learnable full-precision weight with a
  ternary quantization forward pass and an identity backward pass.
  Reference SGD-friendly gradient is computed via finite differences.
- `torus.train.curriculum`: `CurriculumSchedule` with progressive
  stages that grow `n_planes_active` from 1 to N, lock per-stage
  thresholds, and decide active plane count by training step.
- `torus.train.loop`: end-to-end `DistillationTrainer` with step
  loop, eval hooks, training stats, grad-clip, momentum-SGD on the
  latent weight, and an `on_log` callback. Phase-3 trainer is a
  pure-numpy reference; autograd swaps in behind the same
  interface.
- `examples/qat_smoke.py`: 10-step smoke run that demonstrates the
  curriculum handing off from plane 1 to plane 2 at the configured
  step boundary.
- `tests/test_training.py`: 24 new tests covering distillation
  losses, STE, curriculum, and trainer smoke / curriculum /
  grad-clip / data-exhaustion paths.

### Verified

- `pytest`: 89 passed in 0.12 s.
- `examples/qat_smoke.py`: curriculum handoff at step 4 reported.

## 0.2.0 — Phase 2 (in progress)

### Added

- `torus.quant.packing`: 2-bit packed weight layout with round-trip
  encoding, exposed as `pack_plane(plane) -> PackedTernaryPlane`.
- `torus.core.kernels`: three reference CPU kernels (`dense`,
  `sparse`, `unrolled`) with a uniform `(x, plane) -> (y, OpCount)`
  contract and a registry for adding Phase-3 hardware kernels
- `torus.core.memory`: declarative placement policy for residual
  planes across `VRAM`, `RAM`, `NVME` tiers, plus a
  `p620_default_budget()` helper for the P620 target machine.
- `torus.core.telemetry`: `GateTelemetry` accumulates per-layer gate
  activation rates, trends, and recorded op counts; supports
  `flagged_layers()` and `top_layers_by_activation()` queries.
- `torus.core.residual_linear`: extended with `kernel=` selection and
  optional `telemetry=` recording; Phase-1 callers and tests are
  unchanged.
- `docs/KERNELS.md`: CUDA / AVX-512 kernel spec giving the exact
  contract (weight layout, GEMM semantics, op counts) future
  hardware kernels must satisfy.
- `examples/benchmark.py`: end-to-end microbenchmark with telemetry
  dump and memory-tier placement exercise.
- `tests/test_packing_and_kernels.py`: 21 new tests covering packing,
  kernel-correctness, kernel-equivalence-with-dense, op-count
  invariants, memory policy, and telemetry.

### Changed

- `torus.core.residual_linear` returns `(y, decision)` as before;
  additionally feeds ops into `telemetry` when provided.
- `torus.core.gate.ResidualGate` decision arrays are now reliably
  bool-dtyped via `.astype(bool)`.

### Verified

- `pytest`: 65 passed in 0.08 s.
- `examples/quickstart.py`: clean end-to-end run.
- `examples/benchmark.py`: clean end-to-end run with real timing +
  memory placement numbers.

## 0.1.0 — Phase 1 (initial release)

### Added

- `torus.quant`: ternary + residual-plane quantization math
  (`ternary_quantize`, `residual_quantize`, `compose_planes`).
- `torus.core`: `ResidualGate` and `ResidualTernaryLinear`.
- `torus.moe`: `ExpertBank` and `TopKRouter` scaffolding.
- `torus.rlm`: `RecursiveContext` and `ContextREPL` primitives.
- `docs/`: VISION, ARCHITECTURE, ROADMAP.
- `examples/quickstart.py`: end-to-end smoke run.
- 44 tests across the primitives.
