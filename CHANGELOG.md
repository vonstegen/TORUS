# CHANGELOG

## 0.11.0 — Phase 8 distillation on Legion + trainer probe-rows

### Verified

- Three distillation runs completed on Legion (CUDA torch +
  2× TITAN RTX) with `sshleifer/tiny-gpt2`, 200 steps each:
  - `primary_only` (curriculum `1:200`): initial loss 0.0028,
    final loss 0.0038, ~103 s wall time.
  - `primary_plus_residual` (curriculum `1:100,2:100`, residual
    zero-init): initial 0.0028, final 0.0038 — *identical to
    primary_only* because the trainer only probes the primary weight
    and the residual plane contributes nothing when zero-init.
  - `primary_plus_residual_perturbed` (same curriculum, residual
    weights initialized with N(0, 0.05) noise): initial 0.0028,
    step-100 (curriculum switch) 0.0105, final **0.0205**. Loss
    goes up because the random-noise residual contributes garbage
    that the trainer then has to undo.
- 155/155 tests still passing on both dev and Legion.

### Added

- `examples/distill_run.py`: end-to-end distillation runner that
  loads a HF model via `HFStudentAdapter` + `HFTeacherAdapter`,
  drives `DistillationTrainer` with a curriculum, and logs the
  loss curve + final stats to a JSON file. Supports
  `--n-steps`, `--probe-rows`, `--curriculum`, `--batch-size`,
  `--seq-len`, `--label`, and `--perturb-residual`.
- `TrainingConfig.probe_rows`: per-module finite-difference probe
  budget. Default 1 (one column per STE per step); set higher for
  less noisy gradients at the cost of more forward passes.
- `_numerical_grads` now samples `probe_rows` random rows per STE
  instead of probing every row. With `probe_rows=1` and 6 STEs in
  tiny-gpt2, each step costs ~12 forward passes instead of ~9k.

### Found

- The current trainer only probes `weight`, not `residual_weight`.
  When the residual plane is zero-init, the curriculum switch from
  `n_planes=1` to `n_planes=2` has no effect on the loss curve.
  Phase-8 follow-up: add `--probe-residual` (default off) and have
  `_numerical_grads` perturb `residual_weight` when the STE carries
  one.

# CHANGELOG

## 0.10.0 — Phase 7 multi-expert wiring

### Verified

- 155/155 tests passing on dev (.venv, Py3.12 + CPU torch +
  Blackwell via numba) and Legion (.venv-py311, Py3.14 + CUDA
  torch + 2× TITAN RTX).
- `examples/multi_expert_demo.py`: 16-expert × 4-plane bank, 32
  tokens, top_k=2. Adaptive router-confidence-driven plane
  engagement saves ~17% of plane activations vs. always-4.

### Added

- `torus.moe.multi_expert.MultiExpertRouter`: composes
  `TopKRouter` + `ExpertBank` + `GatePolicy` for production-shape
  multi-expert routing. Each (token, expert) decision emits a
  `PerCallDecision(token_idx, expert_id, weight, confidence,
  n_planes)`.
- `torus.moe.multi_expert.GatePolicy`: linear-interpolation policy
  that maps router confidence to plane-count engagement. Defaults
  calibrated against the Phase-1 random router.
- `torus.moe.multi_expert.MultiExpertResult` and
  `PerCallDecision`: data classes for the per-call decision list.
- `torus.moe.router.RouteResult.raw_mass`: pre-renormalization
  top-k prob mass. `TopKRouter.confidence()` now returns
  `raw_mass` (the meaningful "what fraction of prob mass landed in
  top-k" signal) instead of the normalized weight sum, which is
  always 1.0 by construction.
- `MultiExpertRouter.decision_table(features)`: human-readable
  rendering of the per-call decisions (used by the demo).
- `examples/multi_expert_demo.py`: 16-expert bank + 32-token batch
  demo reporting plane-activation savings and the n_planes
  distribution.
- 8 new tests in `tests/test_moe.py` covering basic routing,
  policy threshold behavior, 100-expert scaling, shared-primary
  composition, the on_decision callback, and graceful handling
  of unknown experts.

# CHANGELOG

## 0.9.0 — SandboxedContextREPL + REPL injection

### Verified

- 147/147 tests passing on dev (.venv, Py3.12 + CPU torch +
  Blackwell via numba) and Legion (.venv-py311, Py3.14 + CUDA
  torch + 2× TITAN RTX).
- `examples/sandbox_demo.py`: a stub model emits three dangerous
  snippets (`import os`, `exec(...)`, `open('/etc/passwd').read()`)
  and one safe snippet. The first three are rejected at AST
  level with `SandboxError` surfaced as stdout; the safe
  `context.grep + context.slice` snippet runs and produces the
  final answer.

### Added

- `torus.rlm.sandbox.SandboxedContextREPL`: drop-in replacement
  for `ContextREPL` that enforces an AST-level whitelist (no
  imports, no `exec`/`eval`/`open`/`getattr`, attribute access
  only on `context`, subscript only on local Names), restricts
  `__builtins__` to a safe subset, and caps per-call resources
  (lines, output size, recursion depth, wall-clock timeout).
- `torus.rlm.sandbox.SandboxPolicy`: per-REPL config object
  (max_lines, max_output, max_recursion_depth, timeout_seconds,
  extra_allowed_call_names).
- `torus.rlm.sandbox.SandboxError`: raised by the AST check
  when model output violates the policy.
- `PrimeAgentLoop(repl=...)` parameter: accept a custom REPL
  (`ContextREPL` for the default untrusted flow; pass
  `SandboxedContextREPL` for production). `run()` now catches
  exceptions from `self.repl.run(code)` and surfaces them as
  stdout so the model can recover on the next step.

### Fixed

- The Phase-2 docs called out REPL execution as an open security
  risk for the entire session. Phase 2 (security sub-phase) is
  now resolved: any production deployment uses
  `SandboxedContextREPL` (see Phase 2 section in ARCHITECTURE.md).

# CHANGELOG

## 0.8.0 — Phase 9 inverted index for PersistentContext

### Verified

- 130/130 tests passing on both dev (.venv, Py3.12 + CPU torch +
  Blackwell via numba) and Legion (.venv-py311, Py3.14 + CUDA
  torch + 2× TITAN RTX).
- `examples/persistent_grep_demo.py`: on a 2000-chunk context with
  unique-needle queries, indexed grep is **0.041 ms warm** vs
  linear's 49.8 ms (~1200× speedup). The first indexed call pays
  the one-time index-build cost (~1.2 s for 2000 chunks); after
  that, every grep is constant-time-ish.

### Added

- `torus.rlm.index.PersistentContextIndex`: append-only inverted
  index over a `PersistentContext` directory. Token → sorted
  list of chunk indices, stored at `<root>/index.json`. Two
  variants (case-sensitive, case-insensitive) live in separate
  files.
- `PersistentContext.grep`: now consults the index first; falls
  back to a linear scan when the pattern has no `\w+` tokens or
  when `use_index=False`.
- `PersistentContext.flush_index()`: force a flush of pending
  index updates to disk before exit.
- `PersistentContext(use_index=False)` opt-out for tests that
  want pure linear-scan semantics.
- `examples/persistent_grep_demo.py`: comparison runner that
  builds an N-chunk context and reports the indexed-vs-linear
  speedup.

### Changed

- `PersistentContextIndex.add_chunk` debounces disk writes to
  every 64 appends (in-memory updates are still immediate, so
  grep calls between appends see fresh data).

# CHANGELOG

## 0.7.0 — n_planes plumbing + comparison

### Verified

- Extended `HFStudentAdapter` and `HFTeacherAdapter` to honor the
  trainer's `n_planes` parameter end-to-end against a real model on
  Legion (CUDA torch). Both venvs green: 121/121 tests pass on
  dev (CPU torch + Blackwell via numba) and Legion (CUDA torch +
  TITAN RTX).
- New `examples/n_planes_compare.py` runs the same model with
  primary-only and primary+residual quantization, perturbing the
  residual weights with random noise to confirm the residual plane
  contributes to the forward (verified:
  `||y_n_planes=1 - y_n_planes=2|| > 0` after perturbation).

### Added

- `TernarySTE.residual_weight`: optional second learnable
  parameter. When set, `forward(n_planes=2)` returns the sum of
  two independently quantized ternary weights (primary +
  residual). `forward(n_planes=1)` is the original primary-only
  behavior.
- `HFStudentAdapter.residual_params`: list of
  `torch.nn.Parameter` — one per STE. The adapter creates a
  single shared tensor between the STE's `residual_weight` and
  this list, so mutating either reference updates both.
- `HFAdapterConfig.target_modules`: now matched by suffix; the
  default covers GPT-2/GPT-Neo's `c_attn` and `c_proj` Conv1Ds.
- `examples/n_planes_compare.py`: comparison runner that loads a
  HF model, perturbs the residual weights, and demonstrates
  n_planes=1 vs n_planes=2 produce different outputs.
- `examples/hf_adapter_smoke.py --n-planes {1,2}`: CLI flag for
  the existing smoke to exercise primary-only vs primary+residual.

### Fixed

- The residual-weight tensor was being constructed twice (once
  passed to `TernarySTE`, once appended to `_residual_params`)
  with `torch.zeros_like(weight)` returning *separate* tensors.
  Mutating one didn't affect the other, so perturbing
  `adapter.residual_params` never reached the STE. Now both
  references share a single `zero_param`, and perturbation
  propagates as expected.

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
