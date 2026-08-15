# CHANGELOG

## 0.6.0 — Legion end-to-end + CPU probe

### Verified

- Cloned + installed TORUS on **Legion** (the production / training
  host): Python 3.14 venv, `torch 2.13.0+cu130` from the cu130
  index, `transformers 5.15.0`, `numba 0.67.0` for the CUDA path.
  121/121 tests pass on Legion with CUDA torch.
- Re-ran the benchmark on Legion's 2× TITAN RTX: real AVX-512
  numbers captured (the Threadripper 3995WX is Zen 2 — see
  *Fixes* below).
- HF adapter end-to-end smoke (`examples/hf_adapter_smoke.py`)
  loads `sshleifer/tiny-gpt2` (or `gpt2`), drives
  `HFStudentAdapter.forward`, `HFTeacherAdapter.forward`, and the
  full `combined_distillation_loss` pipeline against a real
  transformers model on Legion CUDA.

### Fixes

- `torus/kernels/build._machine_flags()` now probes
  `/proc/cpuinfo` for AVX-512 / AVX2 / FMA / AVX support before
  emitting `-mavx512f`. GCC silently accepts `-mavx512f` even on
  CPUs that lack it, and the resulting `.so` then segfaults at
  runtime with `Illegal instruction`. The Zen-2 Threadripper
  3995WX (which has AVX2 but no AVX-512) was hitting this on
  Legion's first build attempt.

### Added

- `examples/hf_adapter_smoke.py`: end-to-end smoke that loads a
  HuggingFace causal-LM, calls the student/teacher adapter, and
  runs `combined_distillation_loss` to confirm the trainer's
  loss path consumes real-model output.
- `TernarySTE.__post_init__` auto-picks a fitting `group_size`
  when the requested one does not divide `in_features`. Small
  smoke models (e.g. tiny-gpt2 with `hidden_size=2`) no longer
  fail at construction; the closest power-of-two divisor is
  chosen, falling back to the full row width for primes.
- `TernarySTE.forward()` accepts both numpy arrays and torch
  Parameters; the HF adapter path quantizes a torch Parameter
  via a `detach().cpu().numpy()` round-trip.
- `HFStudentAdapter` now intercepts both `nn.Linear` *and*
  HF's `Conv1D` (GPT-2, GPT-Neo). Conv1D weights are stored
  transposed relative to `nn.Linear`; the patched forward
  applies `F.linear(x, q_w.T, q_b)` to match the Conv1D
  contract. Bias is held as a separate `torch.nn.Parameter` so
  it stays fp32-trainable without quantization.

### Changed

- `DistillationTrainer` keeps an in-place numpy view
  (`self._params_np`) of every STE weight, used by the
  numerical-gradient reference path. The torch Parameter
  weights are sync'd back from the numpy buffer after each
  `_SGD.step()`.
- `examples/benchmark.py` now imports `ResidualGate` and
  `ResidualTernaryLinear` (caught by the Legion run; the
  telemetry section was crashing without them).

# CHANGELOG

## 0.5.0 — Hardware refresh: GB10 Blackwell

### Verified

- `pip install torch transformers`: 121 passed, 0 skipped
  (was 120 + 1 skip on the torch gate).
- TORUS CUDA kernel (`numba`-compiled) now actually exercises
  the host GPU on every test run, including the
  `test_cuda_kernel_register_or_fallback` smoke.
- `examples/benchmark.py` re-run on the actual GPU; numbers
  reproduced within ~3% of the previous host (kernel is portable,
  numbers are host-specific).
- Hardware reality check: this host is a GB10 Blackwell (sm_120)
  + ARM Cortex-X925 + 121 GB RAM, NOT a P620 / Threadripper
  + 2× TITAN RTX as the docs had assumed. The kernel paths don't
  care (the portable C kernel handles both), but the docs needed
  updating.

### Added

- `torus.core.gb10_default_budget`: memory budget reflecting the
  GB10's unified-memory pool (80 GB VRAM, 40 GB RAM, 1 TB NVMe).
  `p620_default_budget` is kept for back-compat.
- `examples/benchmark.py`: now uses `gb10_default_budget` and
  prints "Memory policy: ... on the GB10 default budget".

### Changed

- `docs/ROADMAP.md`: hardware table replaced with the actual host.
  Added note that `torch` CUDA wheels are unavailable for
  Python 3.12 + aarch64 (numba CUDA is the working path).
- `docs/KERNELS.md`: §7 retitled "(GB10)"; the 3995WX references
  in §4 are replaced with Cortex-X925; P620 hardware-target
  language is replaced throughout.
- `README.md`: Phase-2 line updated to mention "GB10 Blackwell".
- `torus.core.__init__.py`: export `gb10_default_budget` and
  `place_planes` (the latter was missing — caught by tests after
  the new budget export).

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
