"""Compiled (real) kernels.

Phase-2 follow-on: `simd` (C/AVX/AVX-512/SVE) and `cuda` (numba)
implementations behind the same `(x, plane) -> (y, OpCount)`
contract as `torus.core.kernels`.
"""
from torus.kernels import build, cuda, simd

__all__ = ["build", "cuda", "simd"]
