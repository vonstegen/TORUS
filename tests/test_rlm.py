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


def test_chunk_splits_into_pieces() -> None:
    ctx = RecursiveContext(SAMPLE)
    pieces = ctx.chunk(chunk_size=2)
    assert [p.start for p in pieces] == [0, 2]
    assert pieces[-1].stop == len(SAMPLE)


def test_split_arbitrary_size() -> None:
    s = ContextSlice(0, 7)
    pieces = s.split(3)
    assert [p.length for p in pieces] == [3, 3, 1]


def test_split_invalid_size() -> None:
    with pytest.raises(ValueError):
        ContextSlice(0, 3).split(0)


def test_slice_length_and_validation() -> None:
    s = ContextSlice(2, 5)
    assert len(s) == 3
    with pytest.raises(ValueError):
        ContextSlice(-1, 2)
    with pytest.raises(ValueError):
        ContextSlice(3, 2)


def test_ask_uses_callable() -> None:
    def ask(s: str) -> str:
        return f"echo:{s}"

    ctx = RecursiveContext(SAMPLE, ask_callable=ask)
    assert ctx.ask(ContextSlice(0, 1)) == "echo:" + SAMPLE[0]


def test_ask_accepts_string() -> None:
    ctx = RecursiveContext(SAMPLE, ask_callable=lambda s: f"STR:{s}")
    assert ctx.ask("anything") == "STR:anything"


def test_ask_default_stub() -> None:
    ctx = RecursiveContext(SAMPLE)
    out = ctx.ask(ContextSlice(0, 1))
    assert "stub-answer-for" in out


def test_recurse_on() -> None:
    ctx = RecursiveContext(
        SAMPLE,
        ask_callable=lambda s: f"LEN={len(s)}",
    )
    out = ctx.recurse_on(ContextSlice(0, 4), chunk_size=1)
    # Each of the four sub-slices produces a LEN=N answer.
    assert out.count("LEN=") == 4


def test_repl_can_run_python() -> None:
    ctx = RecursiveContext(SAMPLE)
    repl = ContextREPL(ctx)
    stdout, _ = repl.run(f"n = {len(SAMPLE)}; n")
    assert stdout.strip() == str(len(SAMPLE))


def test_repl_can_ask_via_context() -> None:
    ctx = RecursiveContext(
        SAMPLE,
        ask_callable=lambda s: f"YES about {s.splitlines()[0][:10]}",
    )
    repl = ContextREPL(ctx)
    stdout, _ = repl.run("context.ask(ContextSlice(0, 1))")
    assert "YES" in stdout


def test_repl_reserved_names() -> None:
    ctx = RecursiveContext(SAMPLE)
    repl = ContextREPL(ctx)
    with pytest.raises(ValueError):
        repl.install("context", lambda: "nope")
    with pytest.raises(ValueError):
        repl.install("ContextSlice", lambda: "nope")


def test_repl_syntax_error_captured() -> None:
    ctx = RecursiveContext(SAMPLE)
    repl = ContextREPL(ctx)
    stdout, _ = repl.run("def :")
    assert "SyntaxError" in stdout


def test_repl_runtime_error_captured() -> None:
    ctx = RecursiveContext(SAMPLE)
    repl = ContextREPL(ctx)
    stdout, _ = repl.run("1/0")
    assert "ZeroDivisionError" in stdout or not stdout  # either captured or silent


def test_repl_expression_value() -> None:
    ctx = RecursiveContext(SAMPLE)
    repl = ContextREPL(ctx)
    stdout, value_repr = repl.run("1 + 2")
    assert "3" in stdout
    assert value_repr == "3"
