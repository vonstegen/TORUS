"""Phase 8 demo: a PrimeAgentLoop driven against a PersistentContext.

This shows the same RLM pattern as `prime_agent_demo.py`, but with
the long prompt chunked into NVMe-backed files instead of an
in-memory list. The agent and REPL are unchanged.

Run with:

    python examples/persistent_context_demo.py /tmp/torus_ctx
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from torus.rlm import (
    DONE_SENTINEL,
    PersistentContext,
    PrimeAgentLoop,
)


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
    "Phase 8 adds PersistentContext: chunk files on disk, LRU cache, "
    "atomic manifest writes, and append-only growth.",
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
    "Phase 8 closes the loop with a persistent NVMe-backed "
    "RecursiveContext for prompts measured in millions of tokens.",
]


GOAL = "What is Phase 8 about?"


def stub_model_fn(prompt: str) -> str:
    """Stub: grep the word 'Phase 8' and emit the matched chunk."""
    return (
        "hits = context.grep('Phase 8', ignore_case=True)\n"
        f"{DONE_SENTINEL}\n"
        "context.slice(hits[0])"
    )


def main(argv: list[str]) -> None:
    """Drive the loop against a PersistentContext rooted at argv[1] (or a tempdir)."""
    root = Path(argv[1]) if len(argv) > 1 else Path(tempfile.mkdtemp(prefix="torus_pctx_"))
    cleanup = len(argv) <= 1
    try:
        ctx = PersistentContext(root, cache_size=4)
        # Cold start: append each chunk to disk.
        for chunk in PAPER:
            ctx.add_chunk(chunk)
        print(f"  root: {root}")
        print(f"  chunks on disk: {ctx.total}")
        print(f"  storage bytes: {ctx.storage_bytes()}")

        agent = PrimeAgentLoop(ctx, model_fn=stub_model_fn, max_steps=4)
        result = agent.run(goal=GOAL)
        print(f"  steps: {len(result.steps)}")
        for s in result.steps:
            print(f"  --- step {s.step} ---")
            print("    code:")
            print("    " + s.code.replace("\n", "\n    "))
        print()
        print(f"  FINAL ANSWER: {result.answer}")
    finally:
        if cleanup and root.exists():
            shutil.rmtree(root)


if __name__ == "__main__":
    main(sys.argv)