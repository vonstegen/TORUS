"""ContextREPL: a tiny persistent Python environment that exposes
`RecursiveContext` to the model as a `context` variable.

The REPL pattern (à la RLM / Prime Agent) is what allows the model to
treat long context as a variable. Phase-1 ships a deterministic local
REPL backed by the standard library; Phase 8 makes it work over any
duck-typed context (PersistentContext, RecursiveContext, mocks).
"""
from __future__ import annotations

import ast
import io
import sys
from contextlib import redirect_stdout
from typing import Callable

from torus.rlm.context import ContextSlice, RecursiveContext


class ContextREPL:
    """Tiny Python REPL with one bound variable: `context`.

    The model writes Python that reads/manipulates the context, e.g.:

        >>> relevant = [c for c in context.grep("tokenizers")]
        >>> answer = context.ask(relevant[0])
        >>> answer

    The REPL is intentionally minimal: it's an execution environment, not
    a full sandbox. Phase 2 will add a secured variant for production use.
    """

    def __init__(self, context) -> None:
        # `context` is duck-typed: any object exposing
        # `slice(ContextSlice)`, `grep`, `chunk`, `ask`, and
        # `recurse_on` works. `RecursiveContext` and
        # `PersistentContext` both satisfy this Protocol.
        self.context = context
        self._env: dict = {
            "context": context,
            "RecursiveContext": RecursiveContext,
            "ContextSlice": ContextSlice,
        }

    def run(self, code: str) -> tuple[str, str]:
        """Execute `code` against this REPL.

        Returns:
            (stdout, repr_of_last_value). `repr_of_last_value` is the
            `repr()` of the last expression if the input ended with one,
            else an empty string. SyntaxErrors are reported in `stdout`
            so callers can see them.
        """
        out = io.StringIO()
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            out.write(f"SyntaxError: {e}")
            return out.getvalue(), ""

        with redirect_stdout(out):
            value_repr = self._exec(tree)

        return out.getvalue(), value_repr

    def _exec(self, tree: ast.Module) -> str:
        body = list(tree.body)
        value_repr = ""
        if body and isinstance(body[-1], ast.Expr):
            last_expr = body.pop()
            stmts_module = ast.Module(body=body, type_ignores=[])
            try:
                exec(compile(stmts_module, "<repl>", "exec"), self._env)
            except Exception as e:  # noqa: BLE001 - surface eval errors
                print(f"{type(e).__name__}: {e}", file=sys.__stdout__)
                return ""
            try:
                value = eval(compile(
                    ast.Expression(body=last_expr.value),
                    "<repl>", "eval",
                ), self._env)
            except Exception as e:  # noqa: BLE001
                print(f"{type(e).__name__}: {e}", file=sys.__stdout__)
                return ""
            if value is not None:
                value_repr = repr(value)
                print(value_repr)
        else:
            try:
                exec(compile(tree, "<repl>", "exec"), self._env)
            except Exception as e:  # noqa: BLE001
                print(f"{type(e).__name__}: {e}", file=sys.__stdout__)
                return ""
        return value_repr

    # Convenience for adapters that need to pre-load sub-routines.
    def install(self, name: str, fn: Callable[..., object]) -> None:
        if name in {"context", "RecursiveContext", "ContextSlice"}:
            raise ValueError(f"{name!r} is reserved")
        self._env[name] = fn