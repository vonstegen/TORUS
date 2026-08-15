"""PrimeAgentLoop: a model-driven REPL loop over a `RecursiveContext`.

The "Prime Agent" pattern (à la RLM):

- The model emits Python that reads `context` (a `RecursiveContext`).
- The `ContextREPL` runs the code, captures stdout + last value.
- The loop feeds the result back to the model.
- After `max_steps`, the last successful answer is returned.

This module is *pure-torus* (no torch, no transformers). It only
depends on the model's `Callable[[str], str]` interface — anything
that maps a prompt to a string works, including a stub.

The loop's job is to be the *substrate* a model-driven RLM agent
sits on top of. The trainer's `hf_adapter` plugs into this on the
"model_fn" side; the agent's history becomes part of the trainer's
hidden-state distillation target (Phase 5 follow-up).
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Callable, Sequence

from torus.rlm.context import RecursiveContext
from torus.rlm.repl import ContextREPL


# Sentinel the model emits to signal "I'm done, here's my answer".
DONE_SENTINEL = "### DONE ###"


@dataclass
class AgentStep:
    """One iteration of the loop."""
    step: int
    code: str                # what the model emitted
    stdout: str              # REPL stdout for this step
    last_value: str          # repr() of the last expression (or "")


@dataclass
class AgentResult:
    """Final output of a `PrimeAgentLoop.run()`."""
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    final_stdout: str = ""


def default_prompt_builder(
    goal: str,
    repl: ContextREPL,
    history: Sequence[AgentStep],
    context_summary: str,
) -> str:
    """Build the prompt shown to the model at each step.

    The model sees:
      - the original goal,
      - the context summary (chunk count + first/last chunk),
      - the history so far,
      - the available Python API,
      - the sentinel rule.
    """
    history_text = "\n\n".join(
        f"### Step {s.step}\n```python\n{s.code}\n```\n"
        f"stdout: {s.stdout.strip() or '(empty)'}\n"
        f"last_value: {s.last_value or '(none)'}"
        for s in history
    ) or "(no history yet)"

    return (
        f"You have a long context stored as `context` (a RecursiveContext).\n"
        f"Goal: {goal}\n\n"
        f"Context summary: {context_summary}\n\n"
        f"History so far:\n{history_text}\n\n"
        f"Available API:\n"
        f"  context.slice(ContextSlice(start, stop)) -> str\n"
        f"  context.grep(pattern) -> list[ContextSlice]\n"
        f"  context.chunk(chunk_size) -> list[ContextSlice]\n"
        f"  context.ask(slice_or_text) -> str\n"
        f"  context.recurse_on(slice, aggregator) -> str\n\n"
        f"Emit a Python snippet using only the API above. To finish, emit\n"
        f"a final line and on its own line: {DONE_SENTINEL}\n"
    )


class PrimeAgentLoop:
    """Drive `ContextREPL` iteratively with a model callable.

    Args:
        context: the long prompt, chunked.
        model_fn: maps a prompt string to Python code (or a final
            answer ending with the `DONE_SENTINEL` line).
        max_steps: hard cap on iterations.
        prompt_builder: optional override for how the prompt is shaped.
    """
    def __init__(
        self,
        context: RecursiveContext,
        model_fn: Callable[[str], str],
        max_steps: int = 8,
        prompt_builder: Callable[..., str] = default_prompt_builder,
        repl=None,
    ) -> None:
        # `repl` is a duck-typed object with `.run(code) -> (stdout, last)`.
        # Defaults to `ContextREPL` for backward compatibility; pass
        # `SandboxedContextREPL` (or any other `run`-shaped object) for
        # production use.
        self.context = context
        self.repl = repl if repl is not None else ContextREPL(context)
        self.model_fn = model_fn
        self.max_steps = max_steps
        self.prompt_builder = prompt_builder
    def _context_summary(self) -> str:
        if self.context.total == 0:
            return "(empty context)"
        first = self.context.chunks()[0] if self.context.total > 0 else ""
        last_chunk = self.context.chunks()[-1] if self.context.total > 0 else ""
        return (
            f"{self.context.total} chunks; first={first[:40]!r}; "
            f"last={last_chunk[:40]!r}"
        )

    def run(self, goal: str) -> AgentResult:
        """Drive the loop until `DONE_SENTINEL`, `max_steps`, or exhaustion.

        Sandbox errors and runtime errors are caught and surfaced
        as stdout so the model can recover on the next step.
        """
        history: list[AgentStep] = []
        answer = ""
        for step in range(self.max_steps):
            prompt = self.prompt_builder(
                goal=goal,
                repl=self.repl,
                history=history,
                context_summary=self._context_summary(),
            )
            code = self.model_fn(prompt)
            try:
                stdout, last_value = self.repl.run(code)
            except Exception as e:  # noqa: BLE001
                # Surface sandbox / parse / runtime errors as stdout
                # so the model can recover on the next step.
                stdout = f"{type(e).__name__}: {e}"
                last_value = ""
            history.append(AgentStep(
                step=step, code=code,
                stdout=stdout, last_value=last_value,
            ))
            if DONE_SENTINEL in code:
                # The last-value line is the model's final answer.
                answer = last_value or stdout.strip()
                break

        if not answer and history:
            # No sentinel emitted — fall back to the last stdout/last_value.
            answer = history[-1].last_value or history[-1].stdout.strip()

        final_stdout = history[-1].stdout if history else ""
        return AgentResult(answer=answer, steps=history, final_stdout=final_stdout)