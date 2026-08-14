"""RecursiveContext: the substrate that turns the prompt into a variable.

The model never loads the entire long context into its own working
window. Instead the prompt lives here, and the model reads from it by
issuing structured queries (`slice`, `grep`, `chunk`, etc.). The model
can also spawn recursive `ask(slice)` calls that go back through the
same context primitive.

In Phase 1 this is a plain Python class wrapping an in-memory list of
string chunks. Phase 2 will add a persistent-backed implementation
(NVMe-backed on the P620) and adaptive prefetching.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class ContextSlice:
    """A lightweight handle into the context: indices into the chunk list."""
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise ValueError(
                f"invalid slice ({self.start}, {self.stop}); need start >= 0 and stop >= start"
            )

    @property
    def length(self) -> int:
        return self.stop - self.start

    def split(self, chunk_size: int) -> list["ContextSlice"]:
        """Split this slice into pieces of `chunk_size` (last may be smaller)."""
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        out: list[ContextSlice] = []
        cur = self.start
        while cur < self.stop:
            nxt = min(cur + chunk_size, self.stop)
            out.append(ContextSlice(cur, nxt))
            cur = nxt
        return out

    def __len__(self) -> int:
        return self.length


class RecursiveContext:
    """The context-as-variable substrate for an RLM-style loop.

    The context is an ordered sequence of string "chunks" (lines, passages,
    tokens — whatever the caller decides). Helpers expose:

        total           : total length
        slice(s)        : materialize a ContextSlice -> str
        grep(pat)       : find chunks containing a pattern, as slices
        chunk(size)     : split the whole context into fixed-size chunks
        recurse_on(s)   : recursively call `ask` on each sub-slice of `s`

    The `ask_callable` is whatever produces an answer for a single chunk;
    it can be a real model call (`model.generate(slice_text)`) or a stub.
    """

    def __init__(
        self,
        chunks: Sequence[str],
        ask_callable: Callable[[str], str] | None = None,
    ) -> None:
        if any(not isinstance(c, str) for c in chunks):
            raise TypeError("chunks must be an ordered sequence of str")
        self._chunks: list[str] = list(chunks)
        self._ask = ask_callable or (lambda s: f"[stub-answer-for:{s[:32]}...]")

    @property
    def total(self) -> int:
        return len(self._chunks)

    def __len__(self) -> int:
        return self.total

    def slice(self, s: ContextSlice) -> str:
        if s.stop > len(self._chunks):
            raise IndexError(
                f"slice end {s.stop} exceeds context length {len(self._chunks)}"
            )
        return "\n".join(self._chunks[s.start:s.stop])

    def grep(self, pattern: str, ignore_case: bool = False) -> list[ContextSlice]:
        """Return ContextSlice handles for each chunk containing `pattern`.

        Args:
            pattern: substring to search for.
            ignore_case: when True, match case-insensitively. Defaults
                to False (case-sensitive).
        """
        if ignore_case:
            pattern = pattern.lower()
        hits: list[ContextSlice] = []
        for i, c in enumerate(self._chunks):
            hay = c.lower() if ignore_case else c
            if pattern in hay:
                hits.append(ContextSlice(i, i + 1))
        return hits

    def chunk(self, chunk_size: int) -> list[ContextSlice]:
        return ContextSlice(0, self.total).split(chunk_size)

    def ask(self, s: ContextSlice | str) -> str:
        text = s if isinstance(s, str) else self.slice(s)
        return self._ask(text)

    def recurse_on(
        self,
        s: ContextSlice,
        chunk_size: int,
        aggregator: Callable[[list[str]], str] | None = None,
    ) -> str:
        """Recursively `ask` each sub-slice of `s` and aggregate the answers.

        Args:
            s: the slice to recurse over.
            chunk_size: sub-slice size passed through `split`.
            aggregator: how to combine sub-answers. Defaults to
                `"\n".join` for non-empty results.

        Returns:
            The aggregated answer string.
        """
        pieces = [self.ask(c) for c in s.split(chunk_size)]
        if aggregator is None:
            non_empty = [p for p in pieces if p]
            return "\n".join(non_empty)
        return aggregator(pieces)

    def as_iterable(self) -> Iterable[tuple[int, str]]:
        return enumerate(self._chunks)

    def chunks(self) -> list[str]:
        return list(self._chunks)
