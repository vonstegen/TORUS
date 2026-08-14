"""Phase 5 demo: a stub-model Prime Agent drives a ContextREPL.

This shows the RLM pattern end-to-end on a long fake "paper" without
the model ever seeing the full prompt. The stub `model_fn` follows a
deterministic policy: it greps for the goal keywords, slices the
relevant chunks, and emits the answer.

In a real deployment the model is the trained ternary model; this
demo's stub stands in until torch + a downloaded base is available.

Run with:

    python examples/prime_agent_demo.py
"""
from __future__ import annotations

import re
from typing import Iterable

from torus.rlm import DONE_SENTINEL, ContextSlice, PrimeAgentLoop, RecursiveContext


# A long fake paper prompt, chunked into 16 paragraphs. The stub model
# will be asked a question that requires grep + slice to answer.
PAPER = [
    "TORUS introduces residual ternary planes as the core weight "
    "representation for efficient local inference.",
    "Each weight is decomposed into a primary plane plus optional "
    "residual planes that fire only on hard tokens.",
    "An adaptive gate decides per call site whether to engage a "
    "residual plane; the gate is the runtime quality-efficiency dial.",
    "The training recipe combines straight-through estimation with "
    "capability-aware distillation over intermediate hidden states.",
    "Hardware-aware kernels include a portable C reference, an AVX-512 "
    "specialization, and a CUDA kernel registered as get_kernel('cuda').",
    "Memory-hierarchy placement moves residual planes between VRAM, "
    "RAM, and NVMe based on a declared Budget and the gate's recent "
    "activation rate.",
    "MoE-aware specialization gives each expert its own residual "
    "stack with an optional shared primary plane.",
    "The TopKRouter exposes a per-token confidence signal; low "
    "confidence biases the gate toward engaging the residual plane.",
    "RecursiveContext exposes slice, grep, chunk, and ask primitives "
    "so the model never loads the whole long prompt into its own "
    "working window.",
    "ContextREPL is a tiny Python environment that binds `context` "
    "as a variable; the model emits Python snippets and reads the "
    "resulting stdout / last value.",
    "PrimeAgentLoop drives the REPL iteratively with a model_fn that "
    "maps prompt -> Python code, until the model emits a DONE sentinel.",
    "Phase 1 shipped the primitives; Phase 2 the kernels; Phase 3 the "
    "HF adapter; Phase 4 the MoE wiring; Phase 5 the Prime Agent loop.",
    "The Phase 5 milestone demonstrates context-as-variable on long "
    "prompts that would otherwise exceed the model's working window.",
    "Phase 6 will benchmark the trained ternary model against a "
    "vanilla fp32 baseline and a hypothetical bitnet.cpp install.",
    "Phase 7 will scale the residual-plane stack to multi-expert "
    "production models with shared primary planes.",
    "Phase 8 will close the loop with a persistent NVMe-backed "
    "RecursiveContext for prompts measured in millions of tokens.",
]


# Keyword extraction is a deterministic stand-in for whatever the
# real model's intent-extractor would do.
def _keywords(goal: str) -> list[str]:
    """Return lowercase keywords from a goal sentence."""
    # Strip punctuation, keep word tokens length >= 3.
    words = re.findall(r"[A-Za-z]{3,}", goal.lower())
    # Drop a few common stopwords.
    stop = {"the", "and", "for", "are", "how", "what", "does", "with",
            "this", "that", "use", "uses", "from", "into"}
    return [w for w in words if w not in stop]


def stub_model_fn_factory(goal: str):
    """Build a deterministic stub model that answers `goal`.

    Step 1: emit a `grep` over the keywords.
    Step 2: emit a `slice(hits[0])` to dump the first match.
    Step 3: emit the answer with the DONE sentinel.
    """
    keywords = _keywords(goal)
    state = {"calls": 0}

    def fn(prompt: str) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            return (
                f"hits = context.grep({keywords[0]!r}, ignore_case=True)\n"
                "print('found', len(hits), 'hits')"
            )
        if state["calls"] == 2:
            return (
                "text = context.slice(hits[0])[:120]\n"
                "print(text)"
            )
        joined = " | ".join(keywords)
        return f'"answer involves: {joined}"\n{DONE_SENTINEL}'

    return fn, state


def main() -> None:
    goals = [
        "How does TORUS decide when to engage a residual plane?",
        "What does the TopKRouter expose for the gate?",
        "How are hardware-aware kernels registered?",
    ]
    for goal in goals:
        print()
        print("=" * 72)
        print(f"GOAL: {goal}")
        print("=" * 72)
        ctx = RecursiveContext(PAPER)
        model_fn, state = stub_model_fn_factory(goal)
        agent = PrimeAgentLoop(ctx, model_fn=model_fn, max_steps=4)
        result = agent.run(goal=goal)
        print(f"  steps used: {len(result.steps)} (model calls: {state['calls']})")
        for s in result.steps:
            print(f"  --- step {s.step} ---")
            print(f"  code:\n    " + s.code.replace("\n", "\n    "))
            if s.stdout.strip():
                print(f"  stdout: {s.stdout.strip()}")
            if s.last_value:
                print(f"  last:   {s.last_value}")
        print()
        print(f"  FINAL ANSWER: {result.answer}")


if __name__ == "__main__":
    main()