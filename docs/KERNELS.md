# Kernel Spec (Phase 2→3)

This document is the **contract** between the Phase-1 reference math
and the Phase-2/3 hardware kernels. A CUDA or AVX-512 implementation
that satisfies this spec drops in behind the existing
`ResidualTernaryLinear` and `get_kernel(...)` registry without any
changes to callers.

The contract is small on purpose: the math has been frozen, and the
runtime decisions (gate, memory tier, kernel selection) live in Python
so they can evolve without touching the kernels.

## 1. Weight Layout

The layout that all kernels must consume is:

```c
// uint8 packed_codes[OUT_FEATURES][IN_PACKED];
// IN_PACKED = ceil(IN_FEATURES / 4)
// Each byte encodes 4 codes little-endian:
//   bits 0..1 = code at position 4i + 0
//   bits 2..3 = code at position 4i + 1
//   bits 4..5 = code at position 4i + 2
//   bits 6..7 = code at position 4i + 3
//
//   0b00 -> 0 (zero weight, skip)
//   0b01 -> +1
//   0b10 -> -1
//   0b11 -> reserved (decode as 0)
//
// fp32 scales[OUT_FEATURES][N_GROUPS];
// N_GROUPS = IN_FEATURES / GROUP_SIZE
// Per-group scale; broadcast across the GROUP_SIZE columns belonging
// to the group.
```

This layout is produced by `torus.quant.pack_plane(plane)` and verified
by `tests/test_packing_and_kernels.py::test_packing_round_trip`.

## 2. GEMM Contract

Each kernel implements:

```c
// Compute y = x @ (T * s)^T, where:
//   x : float32[batch][in_features]
//   T : int8 ternary per the packed layout above
//   s : per-group fp32 scales
//   y : float32[batch][out_features]
//
// Side-channel: an OpCount record (`adds`, `subs`, `skips`,
// `n_rows`, `n_cols`, `elems_loaded`) reports the actual arithmetic
// performed. `adds + subs + skips == n_rows * n_cols`.
```

Three observations matter:

1. The arithmetic **must** match the dense reference
   `y = x @ (T * s)^T` bit-for-bit when run on full-precision float.
   Tests in `tests/test_packing_and_kernels.py` enforce this with
   `np.testing.assert_allclose(rtol=1e-5, atol=1e-6)`.

2. The op count **must** reflect only the work actually performed.
   A CUDA kernel that ignores zeros still counts them as skips in the
   op record; a kernel that loops over all codes still counts zeros
   as skips conceptually, because "no add/sub" equals a skip. This is
   how `ternary_gemv_sparse` reports its `skips` count and the same
   model applies to the hardware kernels.

3. `elems_loaded` records the bytes of weight data the kernel read
   from the tier it ran on. This feeds the memory-policy telemetry
   (warm vs cold planes).

## 3. CUDA Kernel

Pseudocode for `ternary_gemm_cuda(x, packed_plane, scales, out)`:

```cpp
// One CUDA block per output row (OUT_FEATURES).
// Each thread computes one (batch, row) output element.
__global__ void ternary_gemm(
    const float* x,            // [batch, in_features]
    const uint8_t* packed,     // [out_features, in_packed]
    const float* scales,       // [out_features, n_groups]
    float* y,                  // [batch, out_features]
    int batch,
    int in_features,
    int in_packed,
    int n_groups,
    int group_size,
    bool activate_residual     // gate signal: 1 if residual plane runs
) {
    int r = blockIdx.x;
    int b = threadIdx.x;
    if (b >= batch) return;

    // Accumulator lives in a register. One register per (b, r).
    float acc = 0.0f;

    // Inner loop: 4-element vectorized unpack + predicated add/sub.
    for (int p = 0; p < in_packed; ++p) {
        uint8_t byte = packed[r * in_packed + p];

        // Bit-pair extraction, little-endian.
        int s0 = (byte >> 0) & 0x3;
        int s1 = (byte >> 2) & 0x3;
        int s2 = (byte >> 4) & 0x3;
        int s3 = (byte >> 6) & 0x3;

        int k = p * 4;
        float x0 = x[b * in_features + k + 0];
        float x1 = x[b * in_features + k + 1];
        float x2 = x[b * in_features + k + 2];
        float x3 = x[b * in_features + k + 3];

        // Predicated adds / subs; zeros are free.
        if (s0 == 0x1) acc += x0; else if (s0 == 0x2) acc -= x0;
        if (s1 == 0x1) acc += x1; else if (s1 == 0x2) acc -= x1;
        if (s2 == 0x1) acc += x2; else if (s2 == 0x2) acc -= x2;
        if (s3 == 0x1) acc += x3; else if (s3 == 0x2) acc -= x3;

        // Apply the per-group scale when we cross a boundary.
        // (group_size is usually 64 or 128; we amortize the FMA cost.)
        if ((k + 4) % group_size == 0) {
            int g = (k + 3) / group_size;
            acc *= scales[r * n_groups + g];
        }
    }
    y[b * blockDim.x /*=out_features*/ + r] = acc;  // writer uses coalesced layout
}
```

Notes for the implementation:

- The block-dim is `batch`, so for typical `batch=1` (decode step)
  we'd switch to a 1D grid where each thread handles one row across
  multiple groups; the pattern above generalizes cleanly.
- The CUDA kernel can be **fused with the residual plane** by passing
  `activate_residual = true` and reading a *second* `packed` buffer
  with the residual codes; the second pass adds into `acc`.
- The gate is the **kernel-launch parameter** `activate_residual`.
  The runtime sets it from `GateTelemetry` / `ResidualGate` on the
  Python side and dispatches a single GPU call. There is no CUDA-side
  branching on the gate.

## 4. AVX-512 Kernel

Pseudocode for `ternary_gemv_avx512(x, packed_plane, scales, out)`:

```c
// Single-threaded; vectorized via AVX-512 (16 floats per ZMM).
// Outer loop: groups of 16 codes (4 bytes at a time).
void ternary_gemv_avx512(
    const float* x, const uint8_t* packed, const float* scales,
    float* y, int in_features, int n_groups, int group_size)
{
    __m512 acc = _mm512_setzero_ps();

    for (int p = 0; p < in_features / 64; ++p) {
        // Load 16 bytes (= 64 codes) packed along last axis.
        __m128i bytes = _mm_loadu_si128((__m128i*)(packed + p * 16));

        // Unpack to 64 int8 codes via shuffle / shift cascades.
        // (Standard bitnet.cpp idiom; commit a small lookup-table
        // version if shift cascades are too slow on Skylake-X.)
        __m512i codes = expand_4bit_to_8bit(bytes);  // int8 {-1, 0, +1}

        // Load 64 activations.
        __m512 acts = _mm512_loadu_ps(x + p * 64);

        // Multiply-and-accumulate: codes in {0, +1, -1} -> mul adds/sub.
        acc = _mm512_add_ps(acc, _mm512_mul_ps(_mm512_cvtepi8_ps(codes), acts));
    }
    // Apply group scales by chunking into 64-wide groups.
    // Final reduce.
    y[0] = _mm512_reduce_add_ps(acc);
}
```

Notes:

- The CPU kernel can parallelize at the row or batch dim using OpenMP
  or a simple work-stealing loop on the `64-core 3995WX`.
- The gate signal here is even cheaper: if `activate_residual == 0`,
  the runtime skips the second matmul entirely (no kernel launch).
- AVX-512 is available on `3995WX`-class Threadripper Pros but
  down-clocks aggressively when used; large models need the OpenMP
  fan-out to amortize that. A pure AVX2 fallback is the practical
  default for the P620.

## 5. Memory Hierarchy

The runtime keeps three weight tiers available (from `memory.py`):

- `VRAM` — TITAN RTX card(s); access from CUDA kernels.
- `RAM`  — system memory; access from CPU via AVX kernels.
- `NVME` — local SSD; loaded into RAM/VRAM on demand.

The placement policy is declarative: `place_planes([...], budget)`
returns a `Placement` whose `.tiers[i]` says where plane i should
live. A kernel invocation receives the actual byte pointer; if the
plane is cold (not in fast tier) the runtime transparently stages it.

The **gate activation rate** measured by `GateTelemetry` drives plane
movement:

- If a plane's activation rate trends upward, it gets prefetched
  into the faster tier on next iteration.
- If a plane's activation rate trends toward zero, it gets demoted.

This is the Phase 3 directive.

## 6. Op Counters (for telemetry)

Each kernel reports an `OpCount`:

```python
@dataclass
class OpCount:
    adds: int          # counts every +1 code processed
    subs: int          # counts every -1 code processed
    skips: int         # counts every 0 code skipped
    n_rows: int        # output rows (= out_features)
    n_cols: int        # input cols (= in_features)
    elems_loaded: int  # bytes read from this tier
```

Reporting `adds + subs + skips == n_rows * n_cols` is a hard
constraint (checked by tests):

```python
def test_op_count_invariant():
    plane = ternary_quantize(...)
    _, ops = kernel(x, plane)
    assert ops.adds + ops.subs + ops.skips == ops.n_rows * ops.n_cols
```

If your kernel batches multiple planes per call, sum the per-plane
`OpCount`s before reporting.

## 7. Performance Targets (P620)

| Kernel         | Metric                                  | Target                      |
|----------------|-----------------------------------------|------------------------------|
| `ternary_gemv` | TFLOPS-equivalent on fp32 activations  | ≥ 0.5x llama.cpp Q4 baseline on residual OFF |
| `ternary_gemv` | TFLOPS-equivalent on fp32 activations  | ≥ 0.4x llama.cpp Q4 baseline on residual ON  |
| Gate toggle    | round-trip latency to switch            | < 5 µs                       |
| NVMe→RAM stage | end-to-end plane transfer              | < 50 ms per 100 MB plane     |

(These targets are illustrative; final numbers come after Phase-2
measurements.)


## 8. Verification Checklist

Before claiming a kernel is Phase-2-complete:

- [x] Round-trip packing with `pack_plane` / `unpack`
      (`tests/test_kernels_real.py::test_packing_round_trip_via_simd_path`)
- [x] Arithmetic matches `ternary_gemv_dense` within `1e-5`
      (`test_simd_kernel_matches_dense_arithmetic`,
      `test_simd_kernel_padding_alignment_arithmetic`,
      `test_cuda_kernel_register_or_fallback`)
- [x] `OpCount.adds + subs + skips == batch * n_rows * n_cols`
      (per-batch invariant; `test_simd_kernel_op_count_invariant`)
- [x] Memory policy (`place_planes`) places primary plane in VRAM
      under default budget (`test_memory_policy_primary_vram`)
- [x] Gate `NEVER` / `ALWAYS` produce the same y as the matching
      dense reference (`test_gate_always_matches_two_plane_dense`,
      `test_gate_never_matches_primary_only_dense`)
- [x] `GateTelemetry.record` reflects the kernel's reported ops
      (covered by `tests/test_packing_and_kernels.py` Phase 2 suite)
- [x] End-to-end benchmark showing real per-call cost across
      `dense` / `sparse` / `unrolled` / `simd_c` / `cuda`
      (`examples/benchmark.py`)

The test harness in `tests/test_packing_and_kernels.py` exercises
items 1-5, and `tests/test_kernels_real.py` exercises items 1-7
against the compiled C kernel and the CUDA kernel.

Item 8 (end-to-end benchmark) lives in `examples/benchmark.py`
and prints real per-call cost across all five kernels on the
current host.
