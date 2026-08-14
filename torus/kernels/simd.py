"""ctypes loader + Python adapter for the compiled C kernels.

When the build step produced a shared object (see
`torus.kernels.build`), this module makes it usable behind the same
`(x, plane) -> (y, OpCount)` contract as `torus.core.kernels`.

The contract:

    kernel(packed_codes, scales, x) -> (y, OpCount)

`packed_codes` is the uint8 `PackedTernaryPlane.packed_codes` array.
`scales` is the float32 `PackedTernaryPlane.scales` array.
`x` is the (batch, in_features) float32 activations.

`OpCount` semantics: `adds + subs + skips == batch * n_rows * n_cols`
for a batched call (this matches `docs/KERNELS.md` §6 when batch > 1
is counted across all calls).
"""
from __future__ import annotations

import ctypes
import sysconfig
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from torus.core.kernels import OpCount, register_kernel
from torus.quant.packing import PackedTernaryPlane, pack_plane
from torus.quant.ternary import TernaryPlane


# Per-instance cache: id(TernaryPlane) -> PackedTernaryPlane.
# The dispatcher may call us with the same plane many times; pack once.
_PACK_CACHE: dict[int, PackedTernaryPlane] = {}

# Compiled .so candidates.
_THIS_DIR = Path(__file__).resolve().parent
_BUILD_DIR = _THIS_DIR / "build"
_LIB_CANDIDATES: list[Path] = [_BUILD_DIR / "libtorus_kernel.so"]
_SITE_DATA = Path(sysconfig.get_paths()["purelib"]) / "torus" / "kernels" / "build"
_LIB_CANDIDATES.append(_SITE_DATA / "libtorus_kernel.so")


# Op-count C struct layout, must match torus_kernel.c.
class _CTOpCount(ctypes.Structure):
    _fields_ = [
        ("adds", ctypes.c_int64),
        ("subs", ctypes.c_int64),
        ("skips", ctypes.c_int64),
        ("n_rows", ctypes.c_int64),
        ("n_cols", ctypes.c_int64),
        ("elems_loaded", ctypes.c_int64),
    ]


def _signature(lib) -> None:
    fn = lib.ternary_gemm
    fn.restype = None
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(_CTOpCount),
    ]


_LIB_LOCK = threading.Lock()
_LIB: Optional[ctypes.CDLL] = None
_LIB_PATH: Optional[Path] = None


def find_lib() -> Optional[Path]:
    for p in _LIB_CANDIDATES:
        if p.exists():
            return p
    return None


def load_lib(force_reload: bool = False) -> Optional[ctypes.CDLL]:
    global _LIB, _LIB_PATH
    with _LIB_LOCK:
        if _LIB is not None and not force_reload:
            return _LIB
        path = find_lib()
        if path is None:
            return None
        lib = ctypes.CDLL(str(path))
        _signature(lib)
        _LIB = lib
        _LIB_PATH = path
        return lib


def _try_build_now() -> Optional[ctypes.CDLL]:
    try:
        from torus.kernels.build import build_shared_object
        path = build_shared_object(quiet=True)
    except Exception:
        return None
    if path is None or not path.exists():
        return None
    lib = ctypes.CDLL(str(path))
    _signature(lib)
    return lib


def ternary_gemm_simd(
    x: np.ndarray, plane
) -> tuple[np.ndarray, OpCount]:
    """Run the compiled C kernel against the given plane.

    Falls back to `ternary_gemv_dense` if no compiled library exists
    on this machine. `plane` may be a `PackedTernaryPlane` (preferred)
    or a plain `TernaryPlane` (auto-packed and cached).
    """
    if x.ndim != 2 or x.dtype != np.float32:
        raise ValueError(f"x must be float32 (batch, in_features), got {x.shape} {x.dtype}")

    if isinstance(plane, TernaryPlane) and not isinstance(plane, PackedTernaryPlane):
        key = id(plane)
        cached = _PACK_CACHE.get(key)
        if cached is None:
            cached = pack_plane(plane)
            _PACK_CACHE[key] = cached
        plane = cached

    out_f, in_f = plane.out_features, plane.in_features
    n_groups = plane.scales.shape[-1]
    in_packed = plane.packed_codes.shape[-1]
    if x.shape[1] != in_f:
        raise ValueError(f"x.shape[1]={x.shape[1]} != in_features={in_f}")

    lib = load_lib()
    if lib is None:
        lib = _try_build_now()
    if lib is None:
        from torus.core.kernels import ternary_gemv_dense
        return ternary_gemv_dense(x, plane)

    batch = x.shape[0]
    y = np.zeros((batch, out_f), dtype=np.float32)
    ops_c = _CTOpCount()
    ops_c.n_rows = out_f
    ops_c.n_cols = in_f

    x_c = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    packed_c = plane.packed_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    scales_c = plane.scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    y_c = y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.ternary_gemm(
        x_c, packed_c, scales_c, y_c,
        batch, in_f, in_packed, n_groups, plane.group_size,
        1,
        ctypes.byref(ops_c),
    )

    ops = OpCount(
        adds=int(ops_c.adds),
        subs=int(ops_c.subs),
        skips=int(ops_c.skips),
        n_rows=int(ops_c.n_rows),
        n_cols=int(ops_c.n_cols),
        elems_loaded=int(ops_c.elems_loaded),
    )
    return y, ops


def register() -> Optional[str]:
    """Register `ternary_gemm_simd` under `get_kernel("simd_c")`.

    Returns the registered name on success, or None if no compiled
    library could be loaded.
    """
    lib = load_lib()
    if lib is None:
        lib = _try_build_now()
    if lib is None:
        return None
    name = "simd_c"
    registry = __import__(
        "torus.core.kernels", fromlist=["_KERNEL_REGISTRY"]
    )._KERNEL_REGISTRY
    if name not in registry:
        register_kernel(name, lambda x, p: ternary_gemm_simd(x, p))
    return name


def lib_path() -> Optional[Path]:
    """Return the .so path if loaded, else None."""
    load_lib()
    return _LIB_PATH
