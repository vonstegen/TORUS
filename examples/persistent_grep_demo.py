"""Phase 9 demo: the inverted-index speedup over linear scan.

Builds a `PersistentContext` with N chunks, each containing a
unique needle, then runs `grep` against the indexed and non-indexed
paths. The first indexed call also pays the one-time index build
cost (rebuilding the inverted index from the on-disk chunks);
subsequent calls only pay the per-query cost.

The point is to show that grep is *constant-time-ish* once the
index is built, vs O(n) for the linear path on a context that
doesn't fit in cache.

Run with:

    python examples/persistent_grep_demo.py
    python examples/persistent_grep_demo.py --n-chunks 5000
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

from torus.rlm import PersistentContext


DEFAULT_N_CHUNKS = 2000
WORDS_PER_CHUNK = 5000
WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa "
    "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi"
).split()


def _make_chunk(i: int) -> str:
    """Construct a chunk with a *unique* needle that no other chunk has."""
    needle = f"NEEDLE-{i:08d}-UNIQUE"
    body = " ".join(WORDS[(i + j) % len(WORDS)] for j in range(WORDS_PER_CHUNK))
    return f"chunk {i}: needle={needle}; body: {body}"


def main(n_chunks: int = DEFAULT_N_CHUNKS) -> None:
    cache_size = 16  # small so cache evictions force real disk reads
    root = Path(tempfile.mkdtemp(prefix="torus_grep_demo_"))
    print(f"  root: {root}")
    print(f"  chunks: {n_chunks}")
    print(f"  cache_size: {cache_size} (forces disk reads during grep)")
    try:
        # Step 1: build the indexed context.
        t0 = time.perf_counter()
        ctx = PersistentContext(root, cache_size=cache_size)
        for i in range(n_chunks):
            ctx.add_chunk(_make_chunk(i))
        ctx.flush_index()  # ensure pending index writes hit disk
        t_append = time.perf_counter() - t0
        print(f"  append {n_chunks} chunks (indexed): {t_append * 1000:.1f} ms")

        # Step 2: indexed grep against a single unique needle.
        # First call pays the index-build cost; second call is pure
        # query cost.
        query = f"NEEDLE-{n_chunks // 2:08d}-UNIQUE"
        t0 = time.perf_counter()
        hits1 = ctx.grep(query)
        t_first = time.perf_counter() - t0
        print(
            f"  indexed grep (first call, includes index build): "
            f"{len(hits1)} hits in {t_first * 1000:.3f} ms"
        )

        t0 = time.perf_counter()
        hits2 = ctx.grep(query)
        t_second = time.perf_counter() - t0
        print(
            f"  indexed grep (second call, pure query): "
            f"{len(hits2)} hits in {t_second * 1000:.3f} ms"
        )

        # Step 3: linear scan with the same query.
        root2 = Path(tempfile.mkdtemp(prefix="torus_grep_demo_noindex_"))
        try:
            ctx2 = PersistentContext(
                root2, cache_size=cache_size, use_index=False
            )
            t0 = time.perf_counter()
            for i in range(n_chunks):
                ctx2.add_chunk(_make_chunk(i))
            t_append_lin = time.perf_counter() - t0
            print(
                f"  append {n_chunks} chunks (no index): "
                f"{t_append_lin * 1000:.1f} ms"
            )
            t0 = time.perf_counter()
            hits3 = ctx2.grep(query)
            t_linear = time.perf_counter() - t0
            print(
                f"  linear grep: {len(hits3)} hits in "
                f"{t_linear * 1000:.3f} ms"
            )
            # Compare second-call indexed vs linear (same work).
            speedup = t_linear / max(t_second, 1e-9)
            print(f"  indexed (warm) vs linear speedup: {speedup:.1f}x")
        finally:
            shutil.rmtree(root2)

        print(
            f"[grep_demo] OK: indexed grep {speedup:.1f}x faster "
            "than linear on a needle that matches one chunk"
        )
    finally:
        shutil.rmtree(root)


def parse_args(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-chunks",
        type=int,
        default=DEFAULT_N_CHUNKS,
        help=f"number of chunks to append (default: {DEFAULT_N_CHUNKS})",
    )
    a = p.parse_args(argv)
    return a.n_chunks


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))