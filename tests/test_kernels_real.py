"""Tests for the Phase-2-followon compiled kernels.

Verifies the `docs/KERNELS.md` §8 checklist:

1. Round-trip packing with `pack_plane` / `unpack`
2. Arithmetic matches `ternary_gemv_dense` within `1e-5`
3. `OpCount.adds + subs + skips == batch * n_rows * n_cols` (per-batch)
4. Memory policy places primary plane in VRAM under default budget
5. Gate `NEVER` / `ALWAYS` produce the same y as a single-plane or
   multi-plane reference.
"""
from __future__ import annotations

import numpy as np
import pytest

from torus.core import (
    Budget,
    GateMode,
    MemoryTier,
    PlaneSize,
    ResidualGate,
    ResidualTernaryLinear,
    p620_default_budget,
    place_planes,
)
from torus.core.kernels import OpCount, ternary_gemv_dense
from torus.core.residual_linear import residual_ternary_matmul
from torus.quant import (
    compose_planes,
    pack_plane,
    residual_quantize,
    ternary_quantize,
)


# --------------------------------------------------------------------------
# Build / load helpers
# --------------------------------------------------------------------------


def _ensure_simd_lib():
    from torus.kernels import simd
    if simd.find_lib() is None:
        try:
            from torus.kernels.build import build_shared_object
            return build_shared_object(quiet=True)
        except Exception:
            return None
    return simd.find_lib()


# --------------------------------------------------------------------------
# KERNELS.md §8 items 1-3: packing, arithmetic, op count
# --------------------------------------------------------------------------


def test_simd_lib_builds() -> None:
    path = _ensure_simd_lib()
    assert path is not None and path.exists()
    assert path.suffix == ".so"


def test_packing_round_trip_via_simd_path() -> None:
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((32, 512)) * 0.1).astype(np.float32)
    plane = ternary_quantize(w, group_size=128)
    packed = pack_plane(plane)
    recovered = packed.unpack()
    np.testing.assert_array_equal(recovered.codes, plane.codes)
    np.testing.assert_array_equal(recovered.scales, plane.scales)


def test_simd_kernel_matches_dense_arithmetic() -> None:
    if _ensure_simd_lib() is None:
        pytest.skip("simd lib not available")
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((128, 512)) * 0.05).astype(np.float32)
    plane = ternary_quantize(w, group_size=128)
    packed = pack_plane(plane)
    x = rng.standard_normal((3, 512)).astype(np.float32)
    y_dense, _ = ternary_gemv_dense(x, plane)
    from torus.kernels import simd
    y_simd, _ = simd.ternary_gemm_simd(x, packed)
    np.testing.assert_allclose(y_dense, y_simd, rtol=1e-5, atol=1e-5)


def test_simd_kernel_op_count_invariant() -> None:
    if _ensure_simd_lib() is None:
        pytest.skip("simd lib not available")
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((64, 256)) * 0.05).astype(np.float32)
    plane = ternary_quantize(w, group_size=128)
    packed = pack_plane(plane)
    x = rng.standard_normal((4, 256)).astype(np.float32)
    from torus.kernels import simd
    _, ops = simd.ternary_gemm_simd(x, packed)
    expected = x.shape[0] * ops.n_rows * ops.n_cols
    actual = ops.adds + ops.subs + ops.skips
    assert actual == expected, (
        f"adds+subs+skips={actual} != batch*n_rows*n_cols={expected}"
    )


def test_simd_kernel_padding_alignment_arithmetic() -> None:
    if _ensure_simd_lib() is None:
        pytest.skip("simd lib not available")
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((32, 257)) * 0.05).astype(np.float32)
    plane = ternary_quantize(w, group_size=257)
    packed = pack_plane(plane)
    x = rng.standard_normal((2, 257)).astype(np.float32)
    y_dense, _ = ternary_gemv_dense(x, plane)
    from torus.kernels import simd
    y_simd, _ = simd.ternary_gemm_simd(x, packed)
    np.testing.assert_allclose(y_dense, y_simd, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# KERNELS.md §8 item 4: memory policy
# --------------------------------------------------------------------------


def test_memory_policy_primary_vram() -> None:
    sizes = [PlaneSize.from_estimate(1024, 8) for _ in range(3)]
    budget = p620_default_budget()
    placement = place_planes(sizes, budget)
    assert placement.tiers[0] is MemoryTier.VRAM


# --------------------------------------------------------------------------
# KERNELS.md §8 item 5: gate modes preserve arithmetic vs dense
# --------------------------------------------------------------------------


def test_gate_always_matches_two_plane_dense() -> None:
    rng = np.random.default_rng(4)
    w = (rng.standard_normal((32, 128)) * 0.05).astype(np.float32)
    x = rng.standard_normal((2, 128)).astype(np.float32)
    planes = residual_quantize(w, num_planes=2, group_size=128)
    layer = ResidualTernaryLinear(
        planes=planes,
        gate=ResidualGate(mode=GateMode.ALWAYS),
        kernel="dense",
    )
    y_layer, _ = layer.forward(x)
    # Reference: sum of two dense ternary GEMMs.
    y_p1, _ = ternary_gemv_dense(x, planes.plane(0))
    y_p2, _ = ternary_gemv_dense(x, planes.plane(1))
    y_ref = y_p1 + y_p2
    np.testing.assert_allclose(y_ref, y_layer, rtol=1e-5, atol=1e-5)


def test_gate_never_matches_primary_only_dense() -> None:
    rng = np.random.default_rng(5)
    w = (rng.standard_normal((32, 128)) * 0.05).astype(np.float32)
    plane = ternary_quantize(w, group_size=128)
    x = rng.standard_normal((2, 128)).astype(np.float32)
    planes = residual_quantize(w, num_planes=3, group_size=128)
    layer = ResidualTernaryLinear(
        planes=planes,
        gate=ResidualGate(mode=GateMode.NEVER),
        kernel="dense",
    )
    y_layer, _ = layer.forward(x)
    y_dense, _ = ternary_gemv_dense(x, plane)
    np.testing.assert_allclose(y_dense, y_layer, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# Registry integration: simd registers, dispatcher agrees with dense
# --------------------------------------------------------------------------


def test_simd_kernel_registers_under_simd_c_name() -> None:
    if _ensure_simd_lib() is None:
        pytest.skip("simd lib not available")
    from torus.kernels import simd
    name = simd.register()
    if name is not None:
        registry = __import__(
            "torus.core.kernels", fromlist=["_KERNEL_REGISTRY"]
        )._KERNEL_REGISTRY
        assert name in registry


def test_kernel_dispatch_via_simd_path() -> None:
    """`residual_ternary_matmul(..., kernel="dense")` and `simd_c` agree."""
    if _ensure_simd_lib() is None:
        pytest.skip("simd lib not available")
    from torus.kernels import simd
    if simd.register() is None:
        pytest.skip("simd kernel not registered")
    rng = np.random.default_rng(6)
    w = (rng.standard_normal((64, 256)) * 0.05).astype(np.float32)
    planes = residual_quantize(w, num_planes=2, group_size=128)
    x = rng.standard_normal((2, 256)).astype(np.float32)
    y_dense, _ = residual_ternary_matmul(x, planes, active_planes=2, kernel="dense")
    y_simd, _ = residual_ternary_matmul(x, planes, active_planes=2, kernel="simd_c")
    np.testing.assert_allclose(y_dense, y_simd, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# CUDA: graceful fallback when no CUDA runtime, real compare when present
# --------------------------------------------------------------------------


def test_cuda_module_loads() -> None:
    from torus.kernels import cuda
    assert hasattr(cuda, "ternary_gemm_cuda")


def test_cuda_kernel_register_or_fallback() -> None:
    from torus.kernels import cuda as cuda_mod
    from torus.kernels.cuda import _cuda_available, register_cuda_kernel
    if not _cuda_available():
        assert register_cuda_kernel() is None
    else:
        name = register_cuda_kernel()
        assert name == "cuda"
        from torus.quant import pack_plane
        rng = np.random.default_rng(7)
        w = (rng.standard_normal((128, 256)) * 0.05).astype(np.float32)
        plane = ternary_quantize(w, group_size=128)
        packed = pack_plane(plane)
        x = rng.standard_normal((2, 256)).astype(np.float32)
        out = cuda_mod.ternary_gemm_cuda(x, packed)
        assert out is not None
        y_cuda, _ops_cuda = out
        y_ref, _ = ternary_gemv_dense(x, plane)
        np.testing.assert_allclose(y_ref, y_cuda, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------
# Build harness smoke
# --------------------------------------------------------------------------


def test_build_harness_idempotent() -> None:
    from torus.kernels.build import build_shared_object
    p1 = build_shared_object(quiet=True)
    p2 = build_shared_object(quiet=True)
    assert p1 == p2


def test_build_module_imports_cleanly() -> None:
    from torus.kernels import build as build_mod
    assert hasattr(build_mod, "build_shared_object")
