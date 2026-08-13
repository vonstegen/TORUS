"""CPU reference kernels for ternary GEMM.

The Phase-1 `core.residual_linear.ternary_matmul` builds the dense
float weight `(T * s)` and falls back to BLAS for `x @ W.T`. That's a
great *correctness* reference but it wastes the structure: a true
ternary kernel performs at most one add or subtract per non-zero code
and skips the zeros entirely.

This module provides CPU kernels whose *op-count* matches a real
hardware kernel, while keeping arithmetic identical to the Phase-1
reference. Both drop in behind the same `(x, plane) -> (y, OpCount)`
contract used by `ResidualTernaryLinear`.

Op-count model used for telemetry:
    adds  += num_+1 codes      (one add per +1)
    subs  += num_-1 codes      (one sub per -1)
    skips += num_zero codes    (no work for zeros)

The `dense` kernel counts every nonzero code as work but zero skips.
The `sparse` and `unrolled` kernels count zeros as skipped because
that's what a real ternary datapath does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torus.quant.ternary import TernaryPlane


@dataclass
class OpCount:
    """Counted arithmetic operations for a kernel call."""
    adds: int = 0
    subs: int = 0
    skips: int = 0
    n_rows: int = 0
    n_cols: int = 0
    elems_loaded: int = 0

    @property
    def nonzero(self) -> int:
        return self.adds + self.subs

    @property
    def total(self) -> int:
        return self.adds + self.subs + self.skips

    def density(self) -> float:
        return self.nonzero / self.total if self.total else 0.0


def _dense_op_count(plane: TernaryPlane) -> OpCount:
    """Op count for a dense reference matmul: no skipping."""
    codes = plane.codes
    adds = int(np.sum(codes == 1))
    subs = int(np.sum(codes == -1))
    return OpCount(
        adds=adds,
        subs=subs,
        skips=0,
        n_rows=codes.shape[0],
        n_cols=codes.shape[1],
        elems_loaded=codes.size,
    )


def _sparse_op_count(plane: TernaryPlane) -> OpCount:
    """Op count for a sparse ternary kernel: zeros are skipped."""
    codes = plane.codes
    adds = int(np.sum(codes == 1))
    subs = int(np.sum(codes == -1))
    return OpCount(
        adds=adds,
        subs=subs,
        skips=int(codes.size - adds - subs),
        n_rows=codes.shape[0],
        n_cols=codes.shape[1],
        elems_loaded=codes.size,
    )


def ternary_gemv_dense(
    x: np.ndarray, plane: TernaryPlane
) -> tuple[np.ndarray, OpCount]:
    """Reference `y = x @ (T * s)^T`.

    Counts *every* nonzero code as an add/sub and counts no skips.
    Matches `core.residual_linear.ternary_matmul` arithmetically.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.shape}")
    if x.shape[1] != plane.codes.shape[1]:
        raise ValueError(
            f"x.shape[1]={x.shape[1]} != in_features={plane.codes.shape[1]}"
        )
    s_full = np.repeat(plane.scales, plane.group_size, axis=-1)
    weight = plane.codes.astype(np.float32) * s_full
    return x @ weight.T, _dense_op_count(plane)


def ternary_gemv_sparse(
    x: np.ndarray, plane: TernaryPlane
) -> tuple[np.ndarray, OpCount]:
    """Sparse-op kernel: arithmetic identical to dense, op count is sparse.

    The recorded `OpCount` reflects what a real ternary kernel would
    perform: only nonzeros do an add/sub; zeros are skipped entirely.
    The arithmetic is identical to `ternary_gemv_dense` so callers can
    swap kernels without changing predictions.

    Args:
        x: float32 activations of shape (batch, in_features).
        plane: a single `TernaryPlane`.

    Returns:
        (y, op_count) where `y` has shape (batch, out_features).
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.shape}")
    if x.shape[1] != plane.codes.shape[1]:
        raise ValueError(
            f"x.shape[1]={x.shape[1]} != in_features={plane.codes.shape[1]}"
        )
    s_full = np.repeat(plane.scales, plane.group_size, axis=-1)
    t_scaled = plane.codes.astype(np.float32) * s_full
    return x @ t_scaled.T, _sparse_op_count(plane)


def ternary_gemv_unrolled(
    x: np.ndarray, plane: TernaryPlane
) -> tuple[np.ndarray, OpCount]:
    """Per-group unrolled kernel: same op count, smaller intermediates.

    Builds the per-group `T * s` for one group at a time and accumulates
    into the output. The largest intermediate is `(batch, out_f)` per
    group rather than a full `(out_f, in_f)` `T * s` matrix.

    Args:
        x: float32 activations of shape (batch, in_features).
        plane: a single `TernaryPlane`.

    Returns:
        (y, op_count) where `y` has shape (batch, out_features).
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got {x.shape}")
    codes = plane.codes
    out_f, in_f = codes.shape
    if x.shape[1] != in_f:
        raise ValueError(f"x.shape[1]={x.shape[1]} != in_features={in_f}")
    g_size = plane.group_size
    if in_f % g_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={g_size}")
    n_groups = in_f // g_size
    y = np.zeros((x.shape[0], out_f), dtype=np.float32)
    counts = _sparse_op_count(plane)
    for g in range(n_groups):
        a = g * g_size
        b = a + g_size
        codes_slice = codes[:, a:b]            # (out_f, g_size)
        s_slice = plane.scales[:, g][:, None]  # (out_f, 1)
        t_scaled = codes_slice.astype(np.float32) * s_slice
        y += x[:, a:b] @ t_scaled.T
    return y, counts


# Registry so future kernel work (CUDA, AVX-512) can register handlers
# without changing callers.
_KERNEL_REGISTRY: dict[str, callable] = {}


def register_kernel(name: str, fn: callable) -> None:
    if name in _KERNEL_REGISTRY:
        raise ValueError(f"kernel {name!r} already registered")
    _KERNEL_REGISTRY[name] = fn


def get_kernel(name: str) -> callable:
    if name not in _KERNEL_REGISTRY:
        raise KeyError(
            f"unknown kernel {name!r}; registered: {sorted(_KERNEL_REGISTRY)}"
        )
    return _KERNEL_REGISTRY[name]


# Register the built-in kernels.
register_kernel("dense", ternary_gemv_dense)
register_kernel("sparse", ternary_gemv_sparse)
register_kernel("unrolled", ternary_gemv_unrolled)
