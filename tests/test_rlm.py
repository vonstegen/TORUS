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