"""Tests for Phase-2 primitives: packing, kernels, memory policy, telemetry."""
from __future__ import annotations

import numpy as np
import pytest

from torus.core import (
    Budget,
    GateTelemetry,
    MemoryTier,
    OpCount,
    PlaneSize,
    get_kernel,
    p620_default_budget,
    place_planes,
    register_kernel,
    ternary_gemv_dense,
    ternary_gemv_sparse,
    ternary_gemv_unrolled,
)
from torus.core.gate import GateDecision, GateMode, ResidualGate
from torus.core.residual_linear import (
    ResidualTernaryLinear,
    residual_ternary_matmul,
)
from torus.quant import (
    PackedTernaryPlane,
    TernaryPlane,
    compose_planes,
    pack_plane,
    residual_quantize,
    ternary_quantize,
)


# --- Packed layout ---------------------------------------------------------


def test_packing_round_trip() -> None:
    rng = np.random.default_rng(0)
    weight = (rng.standard_normal((32, 512)) * 0.1).astype(np.float32)
    plane = ternary_quantize(weight, group_size=128)
    packed = pack_plane(plane)
    assert isinstance(packed, PackedTernaryPlane)
    assert packed.packed_codes.shape == (32, 512 // 4)
    assert packed.packed_codes.dtype == np.uint8
    recovered = packed.unpack()
    assert np.array_equal(recovered.codes, plane.codes)
    assert np.array_equal(recovered.scales, plane.scales)
    assert recovered.group_size == plane.group_size


def test_packing_bits_per_weight() -> None:
    rng = np.random.default_rng(1)
    weight = (rng.standard_normal((64, 4096)) * 0.05).astype(np.float32)
    plane = ternary_quantize(weight, group_size=128)
    packed = pack_plane(plane)
    assert 2.0 <= packed.bits_per_weight <= 2.2


def test_packing_padding_alignment() -> None:
    rng = np.random.default_rng(2)
    weight = (rng.standard_normal((8, 257)) * 0.05).astype(np.float32)
    plane = ternary_quantize(weight, group_size=257)
    packed = pack_plane(plane)
    assert packed.packed_codes.shape == (8, (257 + 3) // 4)
    recovered = packed.unpack()
    assert np.array_equal(
        recovered.codes[:, :257],
        plane.codes[:, :257],
    )


def test_pack_rejects_non_int8_codes() -> None:
    bad = TernaryPlane(
        codes=np.zeros((4, 4), dtype=np.int16),
        scales=np.ones((4, 1), dtype=np.float32),
        group_size=4,
    )
    with pytest.raises(TypeError):
        pack_plane(bad)


# --- Kernels ---------------------------------------------------------------


def _plane64x256() -> TernaryPlane:
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((64, 256)) * 0.05).astype(np.float32)
    return ternary_quantize(w, group_size=128)


def test_dense_and_sparse_agree() -> None:
    plane = _plane64x256()
    rng = np.random.default_rng(4)
    x = rng.standard_normal((4, 256)).astype(np.float32)
    y_dense, ops_dense = ternary_gemv_dense(x, plane)
    y_sparse, ops_sparse = ternary_gemv_sparse(x, plane)
    np.testing.assert_allclose(y_dense, y_sparse, rtol=1e-5, atol=1e-6)
    assert ops_dense.skips == 0
    assert ops_sparse.skips > 0
    assert ops_sparse.nonzero == ops_dense.nonzero


def test_unrolled_agrees_with_dense() -> None:
    plane = _plane64x256()
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 256)).astype(np.float32)
    y_dense, _ = ternary_gemv_dense(x, plane)
    y_unrolled, ops_unrolled = ternary_gemv_unrolled(x, plane)
    np.testing.assert_allclose(y_dense, y_unrolled, rtol=1e-5, atol=1e-6)
    expected = int(np.sum(plane.codes == 1) + np.sum(plane.codes == -1))
    assert ops_unrolled.nonzero == expected


def test_unrolled_group_count_matches_plane() -> None:
    plane = _plane64x256()
    _, ops = ternary_gemv_unrolled(np.zeros((1, 256), dtype=np.float32), plane)
    assert ops.n_rows == 64
    assert ops.n_cols == 256


def test_kernel_registry_lookup() -> None:
    fn = get_kernel("unrolled")
    assert callable(fn)
    with pytest.raises(KeyError):
        get_kernel("does-not-exist")


def test_kernel_registry_double_register() -> None:
    def f(x, p):
        return x, None
    with pytest.raises(ValueError):
        register_kernel("unrolled", f)


def test_residual_ternary_matmul_kernel_dispatch() -> None:
    rng = np.random.default_rng(6)
    w = (rng.standard_normal((32, 256)) * 0.05).astype(np.float32)
    planes = residual_quantize(w, num_planes=3, group_size=128)
    x = rng.standard_normal((2, 256)).astype(np.float32)

    y_dense, ops_dense = residual_ternary_matmul(x, planes, active_planes=2, kernel="dense")
    y_sparse, ops_sparse = residual_ternary_matmul(x, planes, active_planes=2, kernel="sparse")
    np.testing.assert_allclose(y_dense, y_sparse, rtol=1e-5, atol=1e-6)
    assert len(ops_dense) == 2
    assert len(ops_sparse) == 2


def test_residual_ternary_matmul_active_range() -> None:
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((8, 256)) * 0.05).astype(np.float32)
    planes = residual_quantize(w, num_planes=3, group_size=128)
    x = rng.standard_normal((1, 256), dtype=np.float32)
    with pytest.raises(ValueError):
        residual_ternary_matmul(x, planes, active_planes=0)
    with pytest.raises(ValueError):
        residual_ternary_matmul(x, planes, active_planes=4)


def test_layer_kernel_sparse_matches_dense() -> None:
    rng = np.random.default_rng(8)
    w = (rng.standard_normal((32, 256)) * 0.05).astype(np.float32)
    planes = residual_quantize(w, num_planes=2, group_size=128)
    layer_d = ResidualTernaryLinear(
        planes=planes, gate=ResidualGate(mode=GateMode.ALWAYS), kernel="dense",
    )
    layer_s = ResidualTernaryLinear(
        planes=planes, gate=ResidualGate(mode=GateMode.ALWAYS), kernel="sparse",
    )
    x = rng.standard_normal((3, 256), dtype=np.float32)
    y_d, _ = layer_d.forward(x)
    y_s, _ = layer_s.forward(x)
    np.testing.assert_allclose(y_d, y_s, rtol=1e-5, atol=1e-6)


# --- Memory policy --------------------------------------------------------


def test_plane_size_from_estimate() -> None:
    p = PlaneSize.from_estimate(num_weights=4096 * 4096, num_scales=4096 * 32)
    assert p.weight_bytes == (4096 * 4096 + 3) // 4
    assert p.scale_bytes == 2 * 4096 * 32
    assert p.total_bytes == p.weight_bytes + p.scale_bytes


def test_place_planes_vram_first() -> None:
    sizes = [PlaneSize.from_estimate(1024, 8) for _ in range(3)]
    budget = Budget(vram_bytes=10**9, ram_bytes=10**9, nvme_bytes=10**9)
    placement = place_planes(sizes, budget)
    assert placement.n_planes == 3
    assert placement.tiers[0] is MemoryTier.VRAM


def test_place_planes_overflow_demotes() -> None:
    big = PlaneSize(
        weight_bytes=500 * 1024 ** 2,
        scale_bytes=1024,
        total_bytes=500 * 1024 ** 2,
    )
    budget = Budget(vram_bytes=600 * 1024 ** 2, ram_bytes=2 * 1024 ** 3, nvme_bytes=10 ** 9)
    placement = place_planes([big, big, big], budget)
    assert placement.tiers[0] is MemoryTier.VRAM
    assert placement.tiers[1] in (MemoryTier.RAM, MemoryTier.NVME)


def test_place_planes_empty() -> None:
    placement = place_planes([], Budget(0, 0, 0))
    assert placement.n_planes == 0
    assert placement.tiers == ()


def test_p620_default_budget() -> None:
    b = p620_default_budget()
    assert b.vram_bytes < b.ram_bytes < b.nvme_bytes
    assert b.vram_bytes == 48 * 1024 ** 3
    assert b.nvme_bytes == 2 * 1024 ** 4
    assert b.ram_bytes == 128 * 1024 ** 3


# --- Telemetry -------------------------------------------------------------


def test_telemetry_records_calls() -> None:
    rng = np.random.default_rng(10)
    w = (rng.standard_normal((32, 256)) * 0.05).astype(np.float32)
    planes = residual_quantize(w, num_planes=2, group_size=128)
    tele = GateTelemetry()
    layer = ResidualTernaryLinear(
        planes=planes,
        gate=ResidualGate(mode=GateMode.ADAPTIVE, threshold=0.5),
        kernel="sparse",
        layer_id=0,
        telemetry=tele,
    )
    tele.begin_layer(0)
    for _ in range(8):
        x = rng.standard_normal((1, 256), dtype=np.float32)
        layer.forward(x, residual_relative_magnitude=0.6, depth=0.6)
    summary = tele.summary()
    assert summary["layers"][0]["n_calls"] == 8
    assert summary["layers"][0]["activation_rate"] == 1.0
    assert summary["average_activation"] == 1.0


def test_telemetry_distinguishes_layers() -> None:
    rng = np.random.default_rng(11)
    w1 = (rng.standard_normal((16, 256)) * 0.05).astype(np.float32)
    w2 = (rng.standard_normal((16, 256)) * 0.05).astype(np.float32)
    p1 = residual_quantize(w1, num_planes=2, group_size=128)
    p2 = residual_quantize(w2, num_planes=2, group_size=128)
    tele = GateTelemetry()
    layer1 = ResidualTernaryLinear(
        planes=p1, gate=ResidualGate(mode=GateMode.ALWAYS), layer_id=0, telemetry=tele,
    )
    layer2 = ResidualTernaryLinear(
        planes=p2, gate=ResidualGate(mode=GateMode.NEVER), layer_id=1, telemetry=tele,
    )
    tele.begin_layer(0)
    for _ in range(4):
        layer1.forward(rng.standard_normal((1, 256), dtype=np.float32))
    tele.begin_layer(1)
    for _ in range(4):
        layer2.forward(rng.standard_normal((1, 256), dtype=np.float32))
    summary = tele.summary()
    rates = {s["layer_id"]: s["activation_rate"] for s in summary["layers"]}
    assert rates[0] == 1.0
    assert rates[1] == 0.0


def test_telemetry_flagged_layers() -> None:
    tele = GateTelemetry()
    tele.begin_layer(99)
    decision = GateDecision(
        activate=np.array([True], dtype=bool),
        score=np.array([1.0], dtype=np.float32),
    )
    op = OpCount(adds=10, subs=5, skips=85)
    tele.record(decision, [op])
    flagged = tele.flagged_layers()
    assert any(f["layer_id"] == 99 for f in flagged)


def test_telemetry_topk() -> None:
    tele = GateTelemetry()
    rates = [0.1, 0.9, 0.4, 0.2]
    for lid, rate in enumerate(rates):
        tele.begin_layer(lid)
        n = 100
        n_active = int(rate * n)
        for _ in range(n_active):
            tele.record(
                GateDecision(
                    activate=np.array([True], dtype=bool),
                    score=np.array([1.0], dtype=np.float32),
                ),
                [OpCount(adds=1, subs=0, skips=0)],
            )
        for _ in range(n - n_active):
            tele.record(
                GateDecision(
                    activate=np.array([False], dtype=bool),
                    score=np.array([0.0], dtype=np.float32),
                ),
                [OpCount(adds=0, subs=0, skips=1)],
            )
    top = tele.top_layers_by_activation(k=2)
    assert top[0]["layer_id"] == 1
    assert top[1]["layer_id"] in (0, 2)
