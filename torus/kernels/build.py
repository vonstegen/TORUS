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


def _cpuinfo_flags() -> set[str]:
    """Read the set of CPU flags exposed by `/proc/cpuinfo`.

    Returns an empty set if the file is missing or unreadable.
    Used to gate SIMD flags so we don't emit AVX-512 instructions
    for a CPU that doesn't actually support them.
    """
    try:
        with open("/proc/cpuinfo", "r") as f:
            text = f.read()
    except OSError:
        return set()
    flags: set[str] = set()
    for line in text.splitlines():
        if line.startswith("flags") or line.startswith("Features"):
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            for tok in parts[1].split():
                flags.add(tok)
            break
    return flags


def _machine_flags() -> list[str]:
    """Return SIMD flags the host's CPU actually supports.

    GCC will silently accept `-mavx512f` even on a CPU that lacks
    AVX-512 (defining `__AVX512F__` in the process), and the
    resulting binary crashes at runtime with `Illegal instruction`.
    Probe `/proc/cpuinfo` first to avoid that.
    """
    host = sysconfig.get_config_var("HOST_SYSTEM") or ""
    machine = os.uname().machine.lower()
    flags = _cpuinfo_flags()
    out: list[str] = []
    if machine.startswith("x86") or "x86" in host:
        # Order matters: AVX-512 implies AVX2 + FMA.
        if "avx512f" in flags:
            out += ["-mavx512f", "-mavx512vl", "-mavx512bw"]
            out += ["-mavx2", "-mfma"]
        elif "avx2" in flags:
            out += ["-mavx2", "-mfma"]
        elif "avx" in flags:
            out += ["-mavx"]
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