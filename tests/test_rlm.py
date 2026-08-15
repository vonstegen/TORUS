"""Tests for the RLM context-as-variable primitive."""
from __future__ import annotations

import pytest

from torus.rlm import ContextREPL, ContextSlice, RecursiveContext


SAMPLE = [
    "TORUS is a research project.",
    "It uses residual ternary planes.",
    "The recursive context handles long prompts.",
    "Adaptive gates control the residual plane.",
]


def test_context_length() -> None:
    ctx = RecursiveContext(SAMPLE)
    assert ctx.total == len(SAMPLE)
    assert len(ctx) == len(SAMPLE)


def test_slice_returns_joined_chunks() -> None:
    ctx = RecursiveContext(SAMPLE)
    text = ctx.slice(ContextSlice(0, 2))
    assert "TORUS" in text and "ternary" in text


def test_grep_finds_pattern() -> None:
    ctx = RecursiveContext(SAMPLE)
    hits = ctx.grep("residual")
    assert len(hits) >= 2
    assert all(isinstance(h, ContextSlice) for h in hits)


def test_grep_ignore_case() -> None:
    ctx = RecursiveContext(["TORUS uses ternary planes.", "Other text."])
    # Default (case-sensitive) misses:
    assert ctx.grep("torus") == []
    # ignore_case=True finds:
    hits = ctx.grep("torus", ignore_case=True)
    assert len(hits) == 1



def test_split_arbitrary_size() -> None:
    s = ContextSlice(0, 7)
    pieces = s.split(3)
    assert [p.length for p in pieces] == [3, 3, 1]


def test_split_invalid_size() -> None:
    with pytest.raises(ValueError):
        ContextSlice(0, 3).split(0)


def test_ask_uses_callable() -> None:
    def fake_ask(text: str) -> str:
        return f"echo({len(text)})"

    ctx = RecursiveContext(SAMPLE, ask_callable=fake_ask)
    out = ctx.ask(ContextSlice(0, 1))
    assert out.startswith("echo(")


def test_recurse_on_aggregates() -> None:
    def fake_ask(text: str) -> str:
        return f"len={len(text)}"

    ctx = RecursiveContext(SAMPLE, ask_callable=fake_ask)
    pieces = ctx.recurse_on(
        ContextSlice(0, 3),
        chunk_size=1,
        aggregator=lambda parts: "|".join(parts),
    )
    assert "|" in pieces


def test_repl_run_simple_expression() -> None:
    ctx = RecursiveContext(SAMPLE)
    repl = ContextREPL(ctx)
    stdout, last = repl.run("len(context)")
    assert stdout.strip() == str(len(SAMPLE))
    assert last == str(len(SAMPLE))


def test_repl_reports_syntax_errors() -> None:
    """`def (` is an unparseable statement that raises SyntaxError."""
    ctx = RecursiveContext(SAMPLE)
    repl = ContextREPL(ctx)
    stdout, last = repl.run("def (")
    assert "SyntaxError" in stdout
    assert last == ""


def test_repl_reserves_names() -> None:
    ctx = RecursiveContext(SAMPLE)
    repl = ContextREPL(ctx)
    with pytest.raises(ValueError):
        repl.install("context", lambda: None)


# --------------------------------------------------------------------------
# PrimeAgentLoop (Phase 5)
# --------------------------------------------------------------------------


SAMPLE_LONG = [
    "The ternary GEMV kernel lives in torus/kernels/simd.py.",
    "The CUDA kernel is in torus/kernels/cuda.py.",
    "Context-as-variable lives in torus/rlm/.",
    "The trainer's HF adapter is torus/train/hf_adapter.py.",
    "Adaptive residual gating is in torus/core/gate.py.",
]


def test_agent_loop_single_step_done() -> None:
    """A model that emits a one-liner + sentinel stops the loop after 1 step."""
    from torus.rlm import DONE_SENTINEL, PrimeAgentLoop

    def stub(prompt: str) -> str:
        return f'print("hello")\n{DONE_SENTINEL}\n"my-answer"'

    ctx = RecursiveContext(SAMPLE_LONG)
    agent = PrimeAgentLoop(ctx, model_fn=stub, max_steps=4)
    result = agent.run(goal="say hi")
    assert len(result.steps) == 1
    assert result.answer == "'my-answer'"


def test_agent_loop_multi_step_search() -> None:
    """Model emits grep + slice across two steps before declaring done."""
    from torus.rlm import DONE_SENTINEL, PrimeAgentLoop

    state = {"calls": 0}

    def stub(prompt: str) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            return 'hits = context.grep("GEMV")\nprint(len(hits))'
        if state["calls"] == 2:
            # Sentinel AFTER the answer expression so REPL picks it
            # up as the last value.
            return f'context.slice(hits[0])\n{DONE_SENTINEL}'

    ctx = RecursiveContext(SAMPLE_LONG)
    agent = PrimeAgentLoop(ctx, model_fn=stub, max_steps=4)
    result = agent.run(goal="find the GEMV module location")
    assert state["calls"] == 2
    assert "simd.py" in result.answer


def test_agent_loop_max_steps_fallback() -> None:
    """When the model never emits DONE, the last stdout is the fallback."""
    from torus.rlm import PrimeAgentLoop

    def stub(prompt: str) -> str:
        return 'print("still thinking")'

    ctx = RecursiveContext(SAMPLE_LONG)
    agent = PrimeAgentLoop(ctx, model_fn=stub, max_steps=3)
    result = agent.run(goal="find stuff")
    assert len(result.steps) == 3
    assert "still thinking" in result.final_stdout


def test_agent_loop_custom_prompt_builder() -> None:
    """A custom prompt_builder can rewrite the prompt shape entirely."""
    from torus.rlm import DONE_SENTINEL, PrimeAgentLoop

    def builder(goal, repl, history, context_summary):
        return f"G:{goal}; H:{len(history)}; C:{context_summary[:10]}"

    seen: list[str] = []

    def stub(prompt: str) -> str:
        seen.append(prompt)
        return DONE_SENTINEL + "\n42"

    ctx = RecursiveContext(["chunk1"])
    agent = PrimeAgentLoop(ctx, model_fn=stub, max_steps=2, prompt_builder=builder)
    result = agent.run(goal="x")
    assert len(seen) == 1
    assert seen[0].startswith("G:x; H:0; C:")
    assert result.answer == "42"


def test_agent_loop_inherits_repl_state_across_steps() -> None:
    """Variables set in one step are visible to the next step's code."""
    from torus.rlm import DONE_SENTINEL, PrimeAgentLoop

    state = {"calls": 0}

    def stub(prompt: str) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            # Store a slice of the simd-matching chunk in `value`.
            return "value = context.slice(context.grep('simd')[0])"
        return f"value[:16]\n{DONE_SENTINEL}"

    ctx = RecursiveContext(SAMPLE_LONG)
    agent = PrimeAgentLoop(ctx, model_fn=stub, max_steps=3)
    result = agent.run(goal="x")
    assert state["calls"] == 2
    # The second step's expression is `value[:8]` -> first 8 chars of the simd chunk.
    assert "ternary" in result.answer.lower()

# --------------------------------------------------------------------------
# PersistentContext (Phase 8)
# --------------------------------------------------------------------------


import shutil
import tempfile
from pathlib import Path


def _tmp_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="torus_rlm_persist_"))


def test_persistent_context_append_and_read() -> None:
    """Append three chunks, slice them back, verify content."""
    from torus.rlm import PersistentContext
    root = _tmp_root()
    try:
        ctx = PersistentContext(root, cache_size=4)
        assert ctx.total == 0
        ctx.add_chunk("alpha")
        ctx.add_chunk("beta")
        ctx.add_chunk("gamma")
        assert ctx.total == 3
        assert ctx.slice(ContextSlice(0, 2)) == "alpha\nbeta"
        assert ctx.slice(ContextSlice(1, 3)) == "beta\ngamma"
    finally:
        shutil.rmtree(root)


def test_persistent_context_reopen_recovers_state() -> None:
    """Open the same directory twice -> second open sees prior chunks."""
    from torus.rlm import PersistentContext
    root = _tmp_root()
    try:
        a = PersistentContext(root, cache_size=2)
        a.add_chunk("one")
        a.add_chunk("two")
        del a
        b = PersistentContext(root, cache_size=2)
        assert b.total == 2
        assert b.slice(ContextSlice(0, 2)) == "one\ntwo"
    finally:
        shutil.rmtree(root)


def test_persistent_context_grep_ignore_case() -> None:
    """grep with ignore_case=True is case-insensitive."""
    from torus.rlm import PersistentContext
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("First")
        ctx.add_chunk("second")
        ctx.add_chunk("THIRD")
        assert [s.start for s in ctx.grep("first")] == []
        assert [s.start for s in ctx.grep("first", ignore_case=True)] == [0]
        assert [s.start for s in ctx.grep("second", ignore_case=True)] == [1]
    finally:
        shutil.rmtree(root)


def test_persistent_context_cache_eviction() -> None:
    """LRU eviction: with cache_size=2, accessing chunk 0 then 2 evicts chunk 1."""
    from torus.rlm import PersistentContext
    root = _tmp_root()
    try:
        ctx = PersistentContext(root, cache_size=2)
        for i in range(3):
            ctx.add_chunk(f"chunk-{i}")
        # Touch chunks 0, 1, 2 in order. After touching 2, the LRU evicts 0.
        ctx.slice(ContextSlice(0, 1))
        ctx.slice(ContextSlice(1, 2))
        ctx.slice(ContextSlice(2, 3))
        info = ctx.cache_info()
        assert info["cached_chunks"] == 2
        # 0 was evicted; 1 and 2 remain. Touching 0 again must not crash.
        assert ctx.slice(ContextSlice(0, 1)) == "chunk-0"
    finally:
        shutil.rmtree(root)


def test_persistent_context_out_of_range() -> None:
    """`slice` and `grep`-adjacent reads raise IndexError on out-of-range."""
    from torus.rlm import PersistentContext
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("only")
        with pytest.raises(IndexError):
            ctx.slice(ContextSlice(0, 2))
        # Internal read with negative index.
        with pytest.raises(IndexError):
            ctx._read_chunk(-1)
        with pytest.raises(IndexError):
            ctx._read_chunk(5)
    finally:
        shutil.rmtree(root)


def test_persistent_context_atomic_manifest_write() -> None:
    """Manifest write is atomic; no half-written JSON on disk."""
    from torus.rlm import PersistentContext
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("a")
        ctx.add_chunk("b")
        manifest = (root / "manifest.json").read_text(encoding="utf-8")
        # Must parse cleanly.
        import json as _json
        parsed = _json.loads(manifest)
        assert parsed["total"] == 2
        # No leftover temp files (atomic rename).
        leftovers = list(root.glob(".manifest.*.tmp"))
        assert leftovers == []
    finally:
        shutil.rmtree(root)


def test_persistent_context_storage_bytes_grows() -> None:
    """storage_bytes() reflects the chunk files on disk."""
    from torus.rlm import PersistentContext
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("a" * 1000)
        ctx.add_chunk("b" * 500)
        assert ctx.storage_bytes() >= 1500
    finally:
        shutil.rmtree(root)


def test_repl_accepts_persistent_context() -> None:
    """ContextREPL works with PersistentContext (duck-typed)."""
    from torus.rlm import ContextREPL, PersistentContext
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("hello world")
        ctx.add_chunk("goodbye world")
        repl = ContextREPL(ctx)
        stdout, last = repl.run("len(context)")
        assert stdout.strip() == "2"
        assert last == "2"
        # Grep via REPL.
        stdout, last = repl.run("context.grep('hello', ignore_case=True)")
        assert "ContextSlice" in last
    finally:
        shutil.rmtree(root)


def test_prime_agent_loop_works_with_persistent_context() -> None:
    """End-to-end: PrimeAgentLoop drives a PersistentContext-backed REPL."""
    from torus.rlm import (
        DONE_SENTINEL,
        PersistentContext,
        PrimeAgentLoop,
    )
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("The ternary GEMV kernel lives in simd.py.")
        ctx.add_chunk("The CUDA kernel lives in cuda.py.")
        ctx.add_chunk("Other unrelated text.")

        def stub(prompt: str) -> str:
            # Always emit a 3-step plan: grep -> slice -> answer.
            return (
                "hits = context.grep('GEMV', ignore_case=True)\n"
                f"{DONE_SENTINEL}\n"
                "context.slice(hits[0])"
            )

        agent = PrimeAgentLoop(ctx, model_fn=stub, max_steps=4)
        result = agent.run(goal="where is GEMV")
        assert "simd.py" in result.answer
    finally:
        shutil.rmtree(root)


# --------------------------------------------------------------------------
# PersistentContextIndex (Phase 9)
# --------------------------------------------------------------------------


def test_index_built_on_first_read() -> None:
    """Opening a context with existing chunks builds the index lazily."""
    from torus.rlm import PersistentContext, PersistentContextIndex
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("alpha beta gamma")
        ctx.add_chunk("delta epsilon")
        idx = PersistentContextIndex(root, ignore_case=False)
        s = idx.stats()
        assert s["chunks_indexed"] == 2
        # Tokens: alpha, beta, gamma, delta, epsilon (5).
        assert s["tokens"] == 5
    finally:
        shutil.rmtree(root)


def test_index_reopened_recovers_state() -> None:
    """Closing + reopening preserves the index on disk."""
    from torus.rlm import PersistentContext, PersistentContextIndex
    root = _tmp_root()
    try:
        a = PersistentContext(root)
        a.add_chunk("foo bar")
        a.add_chunk("baz qux")
        # Force index to be on disk by creating one explicitly:
        PersistentContextIndex(root, ignore_case=False)
        del a
        b = PersistentContextIndex(root, ignore_case=False)
        assert b.stats()["chunks_indexed"] == 2
        # foo is in chunk 0, qux is in chunk 1.
        cands_foo = b.candidates_for("foo")
        assert cands_foo == [0]
        cands_qux = b.candidates_for("qux")
        assert cands_qux == [1]
    finally:
        shutil.rmtree(root)


def test_index_ignore_case_separate() -> None:
    """Case-sensitive and case-insensitive indices are stored separately."""
    from torus.rlm import PersistentContext, PersistentContextIndex
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("HELLO World")
        cs = PersistentContextIndex(root, ignore_case=False)
        ci = PersistentContextIndex(root, ignore_case=True)
        # Case-sensitive: only "World" (lowercase) isn't in the file;
        # "hello" doesn't match "HELLO".
        assert cs.candidates_for("hello") == []
        assert cs.candidates_for("World") == [0]
        # Case-insensitive: both match.
        assert ci.candidates_for("hello") == [0]
        assert ci.candidates_for("HELLO") == [0]
    finally:
        shutil.rmtree(root)


def test_index_pattern_with_multiple_tokens() -> None:
    """Multi-token patterns require ALL tokens to be present."""
    from torus.rlm import PersistentContext, PersistentContextIndex
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("alpha beta gamma")  # chunk 0: alpha, beta
        ctx.add_chunk("alpha gamma delta")  # chunk 1: alpha, gamma
        idx = PersistentContextIndex(root, ignore_case=False)
        # "alpha beta": only chunk 0 has both.
        assert idx.candidates_for("alpha beta") == [0]
        # "alpha gamma": both chunks have both.
        assert idx.candidates_for("alpha gamma") == [0, 1]
        # "alpha epsilon": chunk 1 has alpha, neither has epsilon.
        # candidates_for returns [] because epsilon isn't indexed.
        assert idx.candidates_for("alpha epsilon") == []
    finally:
        shutil.rmtree(root)


def test_index_pattern_with_no_tokens() -> None:
    """A pattern made entirely of non-word chars returns None.

    Note: "a.b" DOES get tokenized (\w+ matches "a" and "b"
    separately). We use "---" instead, which has no \w+ tokens.
    """
    from torus.rlm import PersistentContextIndex
    root = _tmp_root()
    try:
        idx = PersistentContextIndex(root, ignore_case=False)
        assert idx.candidates_for("---") is None
        assert idx.candidates_for("...") is None
    finally:
        shutil.rmtree(root)


def test_persistent_context_grep_uses_index() -> None:
    """`PersistentContext.grep` consults the index and returns correct hits."""
    from torus.rlm import PersistentContext
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("the quick brown fox")  # 0
        ctx.add_chunk("jumps over the lazy dog")  # 1
        ctx.add_chunk("nothing matches here")  # 2
        # First grep forces index build; second is fast-path.
        hits_cs = ctx.grep("the", ignore_case=False)
        assert sorted([h.start for h in hits_cs]) == [0, 1]
        hits_ci = ctx.grep("THE", ignore_case=True)
        assert sorted([h.start for h in hits_ci]) == [0, 1]
        # Token that's only in one chunk.
        hits_fox = ctx.grep("fox", ignore_case=False)
        assert [h.start for h in hits_fox] == [0]
    finally:
        shutil.rmtree(root)


def test_persistent_context_index_extends_on_add() -> None:
    """`add_chunk` extends the index for any variant that's already loaded."""
    from torus.rlm import PersistentContext, PersistentContextIndex
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("first chunk")  # chunk 0
        # Build the case-sensitive index; the case-insensitive one is
        # not yet built.
        idx_cs = PersistentContextIndex(root, ignore_case=False)
        # Now run a case-sensitive grep to force the adapter to load
        # the case-sensitive index.
        assert ctx.grep("first") == [ContextSlice(0, 1)]
        # Append another chunk; the case-sensitive index should pick
        # up the new chunk automatically.
        ctx.add_chunk("second chunk")
        ctx.flush_index()  # force the in-memory index to disk
        # Re-fetch the case-sensitive index from disk to verify.
        idx_cs2 = PersistentContextIndex(root, ignore_case=False)
        cands = idx_cs2.candidates_for("second")
        assert cands == [1]
    finally:
        shutil.rmtree(root)


def test_persistent_context_use_index_false_falls_back_to_linear() -> None:
    """Setting `use_index=False` skips the index entirely."""
    from torus.rlm import PersistentContext
    root = _tmp_root()
    try:
        ctx = PersistentContext(root, use_index=False)
        ctx.add_chunk("alpha")
        ctx.add_chunk("beta")
        # grep still works; index is never built.
        assert [h.start for h in ctx.grep("alpha")] == [0]
        info = ctx.cache_info()
        assert info["use_index"] is False
        assert "index" not in info  # no index loaded
    finally:
        shutil.rmtree(root)


def test_index_rebuilds_on_missing_or_stale_file() -> None:
    """Index rebuilds from chunk files when the index file is missing
    or its chunks_indexed count is greater than what's on disk."""
    from torus.rlm import PersistentContext, PersistentContextIndex
    from torus.rlm.index import INDEX_FILENAME
    root = _tmp_root()
    try:
        ctx = PersistentContext(root)
        ctx.add_chunk("hello world")
        ctx.add_chunk("foo bar")
        # Force index creation by running a grep.
        ctx.grep("hello")
        # Delete the index file directly; next access rebuilds.
        (root / INDEX_FILENAME).unlink()
        idx = PersistentContextIndex(root, ignore_case=False)
        # After rebuild, all chunks are re-indexed.
        assert idx.stats()["chunks_indexed"] == 2
        assert idx.candidates_for("hello") == [0]
    finally:
        shutil.rmtree(root)
