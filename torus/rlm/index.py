"""Append-only inverted index for `PersistentContext`.

`PersistentContext.grep` previously scanned every chunk on disk
(linear in the number of chunks). For a context measured in
millions of tokens, that's a real bottleneck — and the RLM
`grep → slice → ask` pattern calls grep repeatedly.

This module provides a small JSON-backed inverted index:
- One entry per token (or lowercased token, for `ignore_case=True`).
- Each entry is a sorted list of chunk indices that contain it.
- The index is append-only: as new chunks are added, only the
  new chunk's tokens are merged in. No chunk is ever re-tokenized.
- The index lives at `<root>/index.json`. If the file is missing
  or stale (chunk count mismatch), the index is rebuilt on first
  read. Rebuilding is O(n_chunks * token_count_per_chunk) and only
  happens once per directory.

The index does not change `PersistentContext`'s public surface;
it transparently speeds up `grep` when present.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable


INDEX_FILENAME = "index.json"
# Tokens are sequences of word characters; lowercase normalization is
# applied only when the caller asks for ignore_case.
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str, ignore_case: bool) -> Iterable[str]:
    """Yield tokens from `text`. Lowercases when `ignore_case=True`."""
    for tok in _TOKEN_RE.findall(text):
        if ignore_case:
            tok = tok.lower()
        yield tok


def _merge_chunk_into_index(
    index: dict[str, list[int]], chunk_idx: int, text: str, ignore_case: bool
) -> None:
    """Add `chunk_idx` to every token entry drawn from `text`."""
    seen: set[str] = set()
    for tok in _tokenize(text, ignore_case):
        if tok in seen:
            continue
        seen.add(tok)
        index.setdefault(tok, []).append(chunk_idx)


class PersistentContextIndex:
    """In-memory inverted index over a `PersistentContext` directory.

    Args:
        root: directory containing chunk files (managed by
            `PersistentContext`). The index itself lives at
            `<root>/index.json`.
        ignore_case: whether tokens were lowercased on insertion.
            Two indices with different `ignore_case` cannot be merged.
    """

    def __init__(self, root: Path | str, ignore_case: bool = False) -> None:
        self.root = Path(root)
        self.ignore_case = ignore_case
        self._path = self.root / INDEX_FILENAME
        # token -> sorted list of chunk indices.
        self._index: dict[str, list[int]] = {}
        self._chunks_indexed: int = 0  # how many chunks the index knows about
        self._load_or_rebuild()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_or_rebuild(self) -> None:
        if not self._path.exists():
            self._rebuild()
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._rebuild()
            return
        if (
            payload.get("ignore_case") != self.ignore_case
            or payload.get("chunks_indexed", 0) > self._count_chunks_on_disk()
        ):
            # Mismatch: ignore_case differs, or on-disk has fewer chunks
            # than the index claims (some were deleted). Rebuild.
            self._rebuild()
            return
        self._index = {
            tok: sorted(set(indices))
            for tok, indices in payload.get("index", {}).items()
        }
        self._chunks_indexed = int(payload.get("chunks_indexed", 0))

    def _count_chunks_on_disk(self) -> int:
        """Count chunk files on disk (filenames matching `NNNNNNNN.txt`)."""
        n = 0
        for p in self.root.glob("*.txt"):
            if p.stem.isdigit():
                n = max(n, int(p.stem) + 1)
        return n

    def _rebuild(self) -> None:
        """Walk every chunk on disk and rebuild the index from scratch."""
        self._index = {}
        self._chunks_indexed = 0
        chunk_paths = sorted(
            (p for p in self.root.glob("*.txt") if p.stem.isdigit()),
            key=lambda p: int(p.stem),
        )
        for chunk_path in chunk_paths:
            idx = int(chunk_path.stem)
            try:
                text = chunk_path.read_text(encoding="utf-8")
            except OSError:
                continue
            _merge_chunk_into_index(self._index, idx, text, self.ignore_case)
            self._chunks_indexed = max(self._chunks_indexed, idx + 1)
        self._save()

    def _save(self) -> None:
        """Atomically write the index to disk."""
        payload = {
            "ignore_case": self.ignore_case,
            "chunks_indexed": self._chunks_indexed,
            "index": self._index,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to a temp file in the same directory, then
        # rename. This avoids leaving a partial index behind if the
        # process is killed mid-write.
        with tempfile.NamedTemporaryFile(
            "w",
            dir=self.root,
            delete=False,
            prefix=".index.",
            suffix=".tmp",
            encoding="utf-8",
        ) as tmp:
            json.dump(payload, tmp)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        tmp_path.replace(self._path)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    # Append N chunks between full index writes. Each chunk still
    # extends the in-memory index immediately, so grep calls are
    # correct without a save in between. We flush every
    # `_save_every` chunks and on `flush()`.
    _SAVE_EVERY = 64

    def add_chunk(self, chunk_idx: int, text: str) -> None:
        """Index the tokens of a newly appended chunk.

        Caller is responsible for writing the chunk file and
        updating `PersistentContext._total` BEFORE calling this
        method. `chunk_idx` is the index assigned by
        `PersistentContext`.
        """
        _merge_chunk_into_index(self._index, chunk_idx, text, self.ignore_case)
        self._chunks_indexed = max(self._chunks_indexed, chunk_idx + 1)
        if (chunk_idx + 1) % self._SAVE_EVERY == 0:
            self._save()

    def flush(self) -> None:
        """Force a save of the in-memory index to disk."""
        self._save()
    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def candidates_for(self, pattern: str) -> list[int] | None:
        """Return chunk indices that *might* contain `pattern`.

        Returns `None` if the index can't help (i.e. the pattern is
        a substring that wouldn't appear in any token). In that
        case the caller should fall back to a linear scan.

        Otherwise returns a sorted list of chunk indices. Each
        chunk is guaranteed to contain at least one token from the
        pattern; the caller must still verify the pattern is a
        substring of the chunk (the index is token-level, not
        substring-level).
        """
        tokens = list(_tokenize(pattern, self.ignore_case))
        if not tokens:
            return None
        candidate_sets: list[set[int]] = []
        for tok in tokens:
            hits = self._index.get(tok)
            if hits is None:
                # Token isn't in any chunk -> no possible match.
                return []
            candidate_sets.append(set(hits))
        # Intersection: chunks that contain ALL pattern tokens.
        if not candidate_sets:
            return []
        result = candidate_sets[0]
        for s in candidate_sets[1:]:
            result &= s
        return sorted(result)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Snapshot of index state (for telemetry / tests)."""
        return {
            "tokens": len(self._index),
            "chunks_indexed": self._chunks_indexed,
            "ignore_case": self.ignore_case,
            "path": str(self._path),
        }