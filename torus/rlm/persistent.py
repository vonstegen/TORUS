"""Persistent (NVMe-backed) `RecursiveContext`.

Each chunk is a separate file under a directory, named `NNNNN.txt`.
A `manifest.json` records the count and aggregate stats. Chunks
are loaded lazily and held in an LRU cache of bounded size.

The class exposes the same public surface as `RecursiveContext`
(`slice`, `grep`, `chunk`, `ask`, `recurse_on`, `total`, `__len__`,
`chunks`, `as_iterable`) so the REPL and `PrimeAgentLoop` don't
care which backing store is in use.

Notes
-----
- Writes are append-only. `add_chunk(text)` is the only way to grow
  the context. There is no `remove_chunk` or `edit_chunk`.
- The directory is not safe for concurrent writers; the lock
  (`_LOCK_FILENAME`) serializes appends across processes on the
  same machine via `fcntl`.
- This is a single-process, single-host implementation. Cross-host
  shards are a Phase-9 concern.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from torus.rlm.context import ContextSlice


MANIFEST_FILENAME = "manifest.json"
_LOCK_FILENAME = ".lock"
CHUNK_FORMAT = "{:08d}.txt"


def _chunk_path(root: Path, idx: int) -> Path:
    return root / CHUNK_FORMAT.format(idx)


@dataclass
class _CacheEntry:
    text: str
    size: int


class PersistentContext:
    """NVMe-backed `RecursiveContext` replacement.

    Args:
        root: directory holding the chunk files and `manifest.json`.
            Created if missing. If non-empty, the directory is read
            (the existing chunks become the context).
        cache_size: maximum number of chunks kept in memory.
        ask_callable: optional stub for `RecursiveContext.ask`.
    """

    def __init__(
        self,
        root: Path | str,
        cache_size: int = 64,
        ask_callable: Callable[[str], str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.cache_size = max(1, cache_size)
        self._ask = ask_callable or (lambda s: f"[stub-answer-for:{s[:32]}...]")
        self._cache: OrderedDict[int, _CacheEntry] = OrderedDict()
        self._lock_path = self.root / _LOCK_FILENAME
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / MANIFEST_FILENAME
        if self._manifest_path.exists():
            with self._manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            self._total = int(manifest.get("total", 0))
        else:
            self._total = 0
            # Count any pre-existing chunk files (recovery path).
            chunk_files = sorted(
                (p for p in self.root.glob("*.txt") if p.stem.isdigit()),
                key=lambda p: int(p.stem),
            )
            if chunk_files:
                last_idx = int(chunk_files[-1].stem)
                self._total = last_idx + 1
                self._write_manifest()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _with_lock(self, fn: Callable[[], None]) -> None:
        """Run `fn` while holding the directory's append lock."""
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("w") as lockf:
            try:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            except OSError:
                # Fallback for non-POSIX (Windows): best-effort, no real lock.
                pass
            try:
                fn()
            finally:
                try:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass

    def _write_manifest(self) -> None:
        # Atomic write so concurrent readers don't see partial JSON.
        with tempfile.NamedTemporaryFile(
            "w",
            dir=self.root,
            delete=False,
            prefix=".manifest.",
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp:
            json.dump({"total": self._total}, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(self._manifest_path)

    # ------------------------------------------------------------------
    # Read path (lazy + cached)
    # ------------------------------------------------------------------

    def _read_chunk(self, idx: int) -> str:
        if idx < 0 or idx >= self._total:
            raise IndexError(f"chunk index {idx} out of range [0, {self._total})")
        cached = self._cache.get(idx)
        if cached is not None:
            self._cache.move_to_end(idx)
            return cached.text
        path = _chunk_path(self.root, idx)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise IndexError(f"chunk file {path} missing on disk") from None
        self._cache[idx] = _CacheEntry(text=text, size=len(text))
        self._cache.move_to_end(idx)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return text

    # ------------------------------------------------------------------
    # Public API: drop-in for `RecursiveContext`
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        return self._total

    def __len__(self) -> int:
        return self._total

    def slice(self, s: ContextSlice) -> str:
        if s.stop > self._total:
            raise IndexError(
                f"slice {s} out of range [0, {self._total})"
            )
        # Join via newline; matches RecursiveContext.slice exactly.
        return "\n".join(self._read_chunk(i) for i in range(s.start, s.stop))

    def grep(self, pattern: str, ignore_case: bool = False) -> list[ContextSlice]:
        if ignore_case:
            pattern = pattern.lower()
        hits: list[ContextSlice] = []
        # Iterate every chunk on disk; this is the slowest path but
        # the only way to support grep on a context that's mostly
        # evicted. Phase-9 could add an inverted index.
        for i in range(self._total):
            text = self._read_chunk(i)
            hay = text.lower() if ignore_case else text
            if pattern in hay:
                hits.append(ContextSlice(i, i + 1))
        return hits

    def chunk(self, chunk_size: int) -> list[ContextSlice]:
        return ContextSlice(0, self._total).split(chunk_size)

    def ask(self, s: ContextSlice | str) -> str:
        text = s if isinstance(s, str) else self.slice(s)
        return self._ask(text)

    def recurse_on(
        self,
        s: ContextSlice,
        chunk_size: int,
        aggregator: Callable[[list[str]], str],
    ) -> str:
        sub = s.split(chunk_size)
        pieces = [self.ask(sub_i) for sub_i in sub]
        return aggregator(pieces)

    def as_iterable(self) -> Iterable[tuple[int, str]]:
        for i in range(self._total):
            yield i, self._read_chunk(i)

    def chunks(self) -> list[str]:
        return [self._read_chunk(i) for i in range(self._total)]

    # ------------------------------------------------------------------
    # Write path (append-only)
    # ------------------------------------------------------------------

    def add_chunk(self, text: str) -> int:
        """Append a new chunk; return its index.

        Atomically:
          1. acquire the directory lock,
          2. write the chunk file under a temp name then rename,
          3. bump `total` and rewrite the manifest.
        """
        if not isinstance(text, str):
            raise TypeError(f"chunk text must be str, got {type(text).__name__}")

        def _do() -> None:
            idx = self._total
            target = _chunk_path(self.root, idx)
            tmp = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            try:
                tmp.replace(target)
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    raise MemoryError("NVMe out of space while writing chunk") from e
                raise
            self._total += 1
            self._write_manifest()

        self._with_lock(_do)
        return self._total - 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def cache_info(self) -> dict:
        """Return a snapshot of cache state (for telemetry / tests)."""
        return {
            "cached_chunks": len(self._cache),
            "cache_size": self.cache_size,
            "total_chunks": self._total,
            "root": str(self.root),
        }

    def storage_bytes(self) -> int:
        """Total bytes used by chunk files on disk."""
        total = 0
        for p in self.root.glob("*.txt"):
            if p.stem.isdigit():
                total += p.stat().st_size
        return total