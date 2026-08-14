"""Build the compiled C kernel (`libtorus_kernel.so`) on demand.

Called either from `pip install` (the entry point
`build_py_libtorus_kernel`), from the runtime fallback in
`torus.kernels.simd._try_build_now`, or from CI. The build is
incremental: if the .so exists and matches the source mtime, it is
reused.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_CSRC = _THIS_DIR / "csrc" / "torus_kernel.c"
_BUILD = _THIS_DIR / "build"
_OUTPUT = _BUILD / "libtorus_kernel.so"


def _compiler() -> list[str]:
    """Pick the best available C compiler."""
    for name in ("cc", "gcc", "clang"):
        path = shutil.which(name)
        if path:
            return [path]
    raise RuntimeError("no C compiler (cc / gcc / clang) found on $PATH")


def _machine_flags() -> list[str]:
    """Return SIMD flags the host supports."""
    host = sysconfig.get_config_var("HOST_SYSTEM") or ""
    machine = os.uname().machine.lower()
    out: list[str] = []
    if machine.startswith("x86") or "x86" in host:
        # x86_64: try AVX-512 first, fall back to AVX2.
        out += ["-mavx512f", "-mavx512vl", "-mavx512bw"]
        out += ["-mavx2"]
        out += ["-mfma"]
    elif machine in ("aarch64", "arm64"):
        out += ["-march=armv8.2-a+sve+fp16"]
    return out


def build_shared_object(
    quiet: bool = False,
    force: bool = False,
) -> Path:
    """Compile `csrc/torus_kernel.c` and return the output path."""
    if not _CSRC.exists():
        raise FileNotFoundError(_CSRC)
    _BUILD.mkdir(parents=True, exist_ok=True)
    if (
        _OUTPUT.exists()
        and not force
        and _OUTPUT.stat().st_mtime >= _CSRC.stat().st_mtime
    ):
        return _OUTPUT

    compiler = _compiler()
    cmd = (
        compiler
        + ["-O3", "-fPIC", "-shared", "-fopenmp"]
        + _machine_flags()
        + [
            "-std=c11",
            "-DTORUS_KERNEL_BUILD=1",
            str(_CSRC),
            "-o", str(_OUTPUT),
            "-lm",
        ]
    )
    if not quiet:
        print(f"[torus] building C kernel: {' '.join(cmd)}", file=sys.stderr)
    res = subprocess.run(
        cmd, capture_output=True, text=True,
    )
    if res.returncode != 0:
        # Fall back: drop SIMD flags, retry with portable reference.
        if not quiet:
            print("[torus] SIMD build failed; retrying with portable reference",
                  file=sys.stderr)
            print(res.stderr, file=sys.stderr)
        cmd = (
            compiler
            + ["-O3", "-fPIC", "-shared"]
            + ["-std=c11", str(_CSRC), "-o", str(_OUTPUT), "-lm"]
        )
        res2 = subprocess.run(cmd, capture_output=True, text=True)
        if res2.returncode != 0:
            raise RuntimeError(
                f"C kernel build failed: {res2.stderr or res2.stdout}"
            )
    return _OUTPUT


def main_entry_point() -> None:
    """Entry point for `python -m build_kernel`."""
    out = build_shared_object(quiet=False, force=True)
    print(out)


if __name__ == "__main__":
    main_entry_point()
