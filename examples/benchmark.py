"""Phase-2 microbenchmark + telemetry dump.

Runs the three reference kernels (dense / sparse / unrolled) on a
batch of representative activations, reports wall-clock time and the
recorded op count, and exercises the memory-tier placement policy
on a realistic 70B-ish plane size budget.

Run with:

    python examples/benchmark.py
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from torus.core import (
    Budget,
    GateMode,
    GateTelemetry,
    MemoryTier,
    PlaneSize,
    ResidualGate,
    ResidualTernaryLinear,
    p620_default_budget,
    place_planes,
    ternary_gemv_dense,
    ternary_gemv_sparse,
    ternary_gemv_unrolled,
)
from torus.quant import (
    compose_planes,
    pack_plane,
    residual_quantize,
    ternary_quantize,
)


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def _bench(
    name: str,
    fn: Callable,
    x: np.ndarray,
    plane,
    runs: int = 25,
) -> tuple[float, object]:
    # Warmup
    fn(x, plane)
    t0 = time.perf_counter()
    last_ops = None
    for _ in range(runs):
        _, ops = fn(x, plane)
        last_ops = ops
    t1 = time.perf_counter()
    ms = (t1 - t0) / runs * 1000
    print(f"  {name:>8}: {ms:7.3f} ms / call   "
          f"(adds={last_ops.adds:>6}, subs={last_ops.subs:>6}, "
          f"skips={last_ops.skips:>6}, density={last_ops.density():.3f})")
    return ms, last_ops


def main() -> None:
    rng = np.random.default_rng(0)

    # Two realistic plane shapes:
    #   1) a "wide" FFN plane (4096 -> 4096) at group_size=128
    #   2) a "tall" attention plane (1024 -> 4096) at group_size=128
    sizes = [
        ("wide FFN  4096 -> 4096", (4096, 4096), 128),
        ("tall attn 1024 -> 4096", (4096, 1024), 128),
        ("small attn 512 ->  512",  (512, 512),   128),
    ]

    for label, (out_f, in_f), gs in sizes:
        print()
        print("=" * 72)
        print(label)
        print("=" * 72)
        w = (rng.standard_normal((out_f, in_f)) * 0.02).astype(np.float32)
        plane = ternary_quantize(w, group_size=gs)
        x = rng.standard_normal((1, in_f)).astype(np.float32)
        _bench("dense", ternary_gemv_dense, x, plane)
        _bench("sparse", ternary_gemv_sparse, x, plane)
        _bench("unrolled", ternary_gemv_unrolled, x, plane)

        # Packed layout size
        packed = pack_plane(plane)
        print(f"  packed bytes: {_fmt_bytes(packed.packed_codes.nbytes)}"
              f"   scales bytes: {_fmt_bytes(packed.scales.nbytes)}")

        # Reconstruction error with residual planes
        planes = residual_quantize(w, num_planes=3, group_size=gs)
        for k in range(1, planes.num_planes + 1):
            wh = compose_planes(planes, active=k)
            err = float(np.linalg.norm(w - wh) / np.linalg.norm(w))
            print(f"  residual planes used = {k}: relative L2 error = {err:.4f}")

    print()
    print("=" * 72)
    print("Telemetry: gate activation rates on a small network")
    print("=" * 72)
    rng = np.random.default_rng(2)
    tele = GateTelemetry()
    layers = []
    for lid in range(4):
        w = (rng.standard_normal((256, 512)) * 0.02).astype(np.float32)
        planes = residual_quantize(w, num_planes=2, group_size=128)
        gate = ResidualGate(
            mode=GateMode.ADAPTIVE,
            threshold=0.4,
            magnitude_bias=-0.5 + 0.25 * lid,  # vary the policy per layer
        )
        layer = ResidualTernaryLinear(
            planes=planes,
            gate=gate,
            kernel="sparse",
            layer_id=lid,
            telemetry=tele,
        )
        layers.append(layer)
    for _ in range(16):
        for layer in layers:
            mag = float(np.random.default_rng(layer.layer_id + 7).uniform(-1, 1))
            x = rng.standard_normal((1, layer.in_features), dtype=np.float32)
            layer.forward(x, residual_relative_magnitude=mag, depth=mag)
    summary = tele.summary()
    print(f"  average activation rate: {summary['average_activation']:.2f}")
    for layer in summary["layers"]:
        print(
            f"  layer {layer['layer_id']}: rate={layer['activation_rate']:.2f}"
            f" trend={layer['trend']:+.2f} n={layer['n_calls']}"
            f" density={layer['density']:.3f}"
        )

    print()
    print("=" * 72)
    print("Memory policy: placing 3 residual planes on the P620 default budget")
    print("=" * 72)
    budget = p620_default_budget()
    print(f"  budget: {_fmt_bytes(budget.vram_bytes)} VRAM, "
          f"{_fmt_bytes(budget.ram_bytes)} RAM, "
          f"{_fmt_bytes(budget.nvme_bytes)} NVMe")
    plane_sizes = [
        # A 70B-style layer packed, per residual plane:
        # 4096 * 4096 weights -> 4 MB packed + scales
        PlaneSize.from_estimate(num_weights=4096 * 4096, num_scales=4096 * 32),
        PlaneSize.from_estimate(num_weights=4096 * 4096, num_scales=4096 * 32),
        PlaneSize.from_estimate(num_weights=4096 * 4096, num_scales=4096 * 32),
    ]
    placement = place_planes(plane_sizes, budget)
    for idx, tier in enumerate(placement.tiers):
        sz = plane_sizes[idx]
        print(f"  plane {idx}: {tier.value:>5}  ({_fmt_bytes(sz.total_bytes)})")


if __name__ == "__main__":
    main()
