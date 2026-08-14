"""CUDA ternary GEMM kernel (numba) with graceful fallback.

The kernel follows `docs/KERNELS.md` §3:

- One block per output row.
- BLOCK_SIZE threads per block; each thread handles one
  (batch, row) output element.

If no CUDA runtime is present, register_cuda_kernel() returns None
and the get_kernel registry keeps using the dense/sparse/unrolled
fallback.
"""
from __future__ import annotations

import numpy as np

from torus.core.kernels import OpCount, register_kernel
from torus.quant.packing import PackedTernaryPlane


def _cuda_available() -> bool:
    try:
        from numba import cuda  # type: ignore
    except Exception:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


_KERNELS: dict = {}


_KERNEL_SOURCE = """
from numba import cuda


@cuda.jit
def ternary_gemm_kernel(
    x,
    packed,
    scales,
    y,
    batch,
    in_features,
    in_packed,
    n_groups,
    group_size,
    activate_residual,
    atomic_adds,
    atomic_subs,
    atomic_skips,
):
    r = cuda.blockIdx.x
    b = cuda.threadIdx.x
    if r >= y.shape[0]:
        return
    if b >= batch:
        return
    partials = cuda.local.array(256, np.float32)
    for g in range(256):
        partials[g] = 0.0
    if n_groups < 256:
        active_groups = n_groups
    else:
        active_groups = 256
    local_adds = 0
    local_subs = 0
    local_skips = 0
    for p in range(in_packed):
        byte = packed[r, p]
        k0 = p * 4
        for slot in range(4):
            k = k0 + slot
            if k >= in_features:
                break
            pair = (byte >> (slot * 2)) & 0x3
            g = k // group_size
            if pair == 0x1:
                partials[g] += x[b, k]
                local_adds += 1
            elif pair == 0x2:
                partials[g] -= x[b, k]
                local_subs += 1
            else:
                local_skips += 1
    acc = 0.0
    for g in range(active_groups):
        acc += partials[g] * scales[r, g]
    if activate_residual:
        y[r, b] = acc
    cuda.atomic.add(atomic_adds, 0, local_adds)
    cuda.atomic.add(atomic_subs, 0, local_subs)
    cuda.atomic.add(atomic_skips, 0, local_skips)
"""


def _compile():
    from numba import cuda  # type: ignore
    ns: dict = {"__name__": "__torus_cuda__"}
    ns["np"] = np
    exec(_KERNEL_SOURCE, ns)
    return ns["ternary_gemm_kernel"], cuda


def _get_kernel():
    if not _cuda_available():
        return None, None
    if "kernel" not in _KERNELS:
        try:
            kernel, cuda = _compile()
            _KERNELS["kernel"] = kernel
            _KERNELS["cuda"] = cuda
        except Exception:
            return None, None
    return _KERNELS.get("kernel"), _KERNELS.get("cuda")


def ternary_gemm_cuda(x, plane):
    """Run the CUDA ternary GEMM; return None if CUDA isn't available."""
    kernel, cuda = _get_kernel()
    if kernel is None or cuda is None:
        return None

    if x.ndim != 2 or x.dtype != np.float32:
        raise ValueError(
            "x must be float32 (batch, in_features), got %s %s" % (x.shape, x.dtype)
        )
    out_f = plane.out_features
    in_f = plane.in_features
    n_groups = plane.scales.shape[-1]
    in_packed = plane.packed_codes.shape[-1]
    if x.shape[1] != in_f:
        raise ValueError("x.shape[1]=%d != in_features=%d" % (x.shape[1], in_f))
    if n_groups > 256:
        raise ValueError("n_groups=%d exceeds CUDA kernel cap of 256" % n_groups)

    batch = x.shape[0]
    y_dev = cuda.device_array((out_f, batch), dtype=np.float32)
    atomic_adds = cuda.to_device(np.zeros(1, dtype=np.int64))
    atomic_subs = cuda.to_device(np.zeros(1, dtype=np.int64))
    atomic_skips = cuda.to_device(np.zeros(1, dtype=np.int64))

    BLOCK = 32
    kernel[(out_f, 1, 1), (BLOCK, 1, 1)](
        cuda.to_device(x),
        cuda.to_device(plane.packed_codes),
        cuda.to_device(plane.scales),
        y_dev,
        batch,
        in_f,
        in_packed,
        n_groups,
        plane.group_size,
        1,
        atomic_adds,
        atomic_subs,
        atomic_skips,
    )
    y_host = y_dev.copy_to_host().T
    adds = int(atomic_adds.copy_to_host()[0])
    subs = int(atomic_subs.copy_to_host()[0])
    skips = int(atomic_skips.copy_to_host()[0])
    ops = OpCount(
        adds=adds,
        subs=subs,
        skips=skips,
        n_rows=out_f,
        n_cols=in_f,
        elems_loaded=in_packed * 4 * out_f,
    )
    return y_host, ops


def register_cuda_kernel():
    """Register the CUDA path under get_kernel('cuda')."""
    name = "cuda"
    registry = __import__(
        "torus.core.kernels", fromlist=["_KERNEL_REGISTRY"]
    )._KERNEL_REGISTRY
    if name in registry:
        return name

    def dispatch(x, plane):
        out = ternary_gemm_cuda(x, plane)
        if out is None:
            from torus.core.kernels import ternary_gemv_dense
            return ternary_gemv_dense(x, plane)
        return out

    if _cuda_available():
        register_kernel(name, dispatch)
        return name
    return None