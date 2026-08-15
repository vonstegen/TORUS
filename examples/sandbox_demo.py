"""Phase-2 sandbox demo.

Drives `SandboxedContextREPL` with a stub model that emits both
safe and dangerous code. Demonstrates the AST-level whitelist in
action: dangerous snippets are rejected *before* any code runs;
the safe snippet executes and produces an answer.

Run with:

    python examples/sandbox_demo.py
"""
from __future__ import annotations

from torus.rlm import (
    DONE_SENTINEL,
    PrimeAgentLoop,
    RecursiveContext,
    SandboxedContextREPL,
)


PAPER = [
    "TORUS introduces residual ternary planes for efficient inference.",
    "Each weight decomposes into a primary plane plus optional residual planes.",
    "The adaptive gate decides whether to engage the residual plane per call.",
    "SandboxedContextREPL enforces an AST whitelist before any code runs.",
    "PrimeAgentLoop drives the REPL iteratively with a model callable.",
]


def make_stub_model() -> tuple:
    """Return a deterministic stub model: 3 dangerous + 1 safe."""
    state = {"calls": 0}

    def fn(_prompt: str) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            return "import os; os.system('echo PWNED')"
        if state["calls"] == 2:
            return "exec('print(1)')"
        if state["calls"] == 3:
            return "open('/etc/passwd').read()"
        # Safe: actually useful work.
        return (
            "hits = context.grep('gate', ignore_case=True)\n"
            f"{DONE_SENTINEL}\n"
            "context.slice(hits[0])"
        )

    return fn, state


def main() -> None:
    ctx = RecursiveContext(PAPER)
    repl = SandboxedContextREPL(ctx)
    fn, state = make_stub_model()
    agent = PrimeAgentLoop(
        ctx, model_fn=fn, max_steps=6, repl=repl,
    )
    result = agent.run(goal="find the section about the gate")

    print(f"steps taken: {len(result.steps)}")
    for i, s in enumerate(result.steps):
        kind = "dangerous" if i < 3 else "safe"
        print(f"  [{kind}] step {s.step}: code = {s.code!r}")
        # First 80 chars of stdout; dangerous steps show SandboxError,
        # safe steps show the actual answer.
        out = s.stdout.strip()
        if len(out) > 80:
            out = out[:80] + "..."
        print(f"           stdout: {out}")
    print()
    print(f"final answer: {result.answer!r}")
    print()
    print(
        "Note: the first three steps were rejected at AST level"
        " (SandboxError surfaced in stdout). Only the safe"
        " grep+slice in step 4 executed and produced an answer."
    )


if __name__ == "__main__":
    main()