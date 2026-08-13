"""Ternary + residual-plane quantization primitives.

The math here intentionally has no ML framework dependencies (numpy only) so
that kernels can later be written against the same semantics in whatever
backend (CUDA, AVX-512, MLX, custom ASIC). Each routine returns plain
arrays; torch adapters live in `torus.adapters.torch_quant` (Phase 2).
"""
from torus.quant.ternary import (
    TernaryPlane,
    ternary_quantize,
)
from torus.quant.residual import (
    ResidualTernaryPlanes,
    residual_quantize,
    compose_planes,
)

__all__ = [
    "TernaryPlane",
    "ResidualTernaryPlanes",
    "ternary_quantize",
    "residual_quantize",
    "compose_planes",
]
