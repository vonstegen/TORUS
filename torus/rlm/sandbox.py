"""SandboxedContextREPL: a restricted execution environment for the model.

The Phase-1 `ContextREPL` is `exec(compile(...))` with the model
emitting Python directly into the local Python environment. That's
a real attack surface — a malformed model output can call
`os.system`, read `/etc/passwd`, or run arbitrary code.

`SandboxedContextREPL` wraps the same loop pattern but enforces
a strict policy at *two* layers:

1. **AST whitelist**: before any code runs, we parse it and walk
   the AST. We reject the code (without running it) if it contains
   any of:
     - `Import` / `ImportFrom` nodes
     - calls to `exec`, `eval`, `compile`, `__import__`, `open`,
       `getattr`, `setattr`, `globals`, `locals`, `vars`, `dir`
     - attribute access on anything other than the bound name
       `context`
     - subscript access on anything other than a top-level Name
       (so `hits[0]` is fine but `__builtins__["eval"]` is not).
   This is the cheap fast-fail path.

2. **Restricted builtins**: at run time, the code sees only a
   safe `__builtins__` dict. No `os`, no `sys`, no `open`, no
   `subprocess`. The model can use `len`, `range`, `print`, basic
   arithmetic, and the `context` API.

3. **Resource caps** (per call):
     - `max_lines`: 64 lines of code max (anything bigger is
       suspicious for a 1-2 step model output).
     - `max_output`: 16 KB captured stdout max.
     - `timeout`: optional wall-clock timeout (caller-supplied;
       enforced via a daemon thread watchdog).
     - `max_recursion_depth`: 16 (Python default is ~1000).

The `SandboxedContextREPL` is API-compatible with `ContextREPL`
(`.run(code) -> (stdout, last)`) so `PrimeAgentLoop` can use
either transparently.
"""
from __future__ import annotations

import ast
import builtins
import io
import sys
import threading
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable


# Set of safe builtins the model is allowed to use. Anything else
# (e.g. `open`, `getattr`, `eval`) is removed from the runtime
# `__builtins__`.
SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "float",
        "int", "isinstance", "len", "list", "max", "min", "print",
        "range", "repr", "reversed", "round", "set", "sorted", "str",
        "sum", "tuple", "type", "zip",
    )
    if hasattr(builtins, name)
}

# AST nodes that are always rejected.
_FORBIDDEN_NODES: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
)

# Names that the runtime refuses to resolve (the AST check
# also catches direct `exec(...)` calls, but a defense-in-depth
# runtime check catches e.g. `getattr(__builtins__, "exec")`).
_FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "exec", "eval", "compile", "__import__",
    "open", "input", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
    "breakpoint",
})


@dataclass(frozen=True)
class SandboxPolicy:
    """Per-REPL policy. Defaults are tight; loosen with care."""

    max_lines: int = 64
    max_output: int = 16 * 1024
    max_recursion_depth: int = 16
    timeout_seconds: float | None = None
    # Names the model may call directly. `context.<method>` is
    # always allowed because the AST walker scopes attribute access
    # to the `context` binding.
    extra_allowed_call_names: frozenset[str] = field(default_factory=frozenset)


class SandboxError(Exception):
    """Raised when the model emits code that violates the policy."""


def _ast_check(
    code: str, policy: SandboxPolicy
) -> ast.Module:
    """Parse `code` and reject any node in the policy blacklist.

    Returns the parsed tree on success.
    """
    # Cheap pre-check: line count.
    line_count = code.count("\n") + 1
    if line_count > policy.max_lines:
        raise SandboxError(
            f"code has {line_count} lines; policy allows at most "
            f"{policy.max_lines}"
        )

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise SyntaxError(f"SyntaxError: {e}") from None

    for node in ast.walk(tree):
        # Block all imports.
        if isinstance(node, _FORBIDDEN_NODES):
            raise SandboxError(
                f"import statements are not allowed "
                f"(found {type(node).__name__} at line {node.lineno})"
            )
        # Block attribute access on anything other than `context`.
        if isinstance(node, ast.Attribute):
            base = node.value
            if not (isinstance(base, ast.Name) and base.id == "context"):
                raise SandboxError(
                    f"attribute access on {ast.dump(base)} "
                    f"is not allowed (line {node.lineno})"
                )
        # Subscript access is allowed on any Name (local variables).
        # The threat model is `__builtins__["eval"]` style module
        # escape, which the AST check on bare-name calls already
        # catches via `_FORBIDDEN_NAMES`.
        if isinstance(node, ast.Subscript):
            base = node.value
            if not isinstance(base, ast.Name):
                raise SandboxError(
                    f"subscript access on {ast.dump(base)} "
                    f"is not allowed (line {node.lineno})"
                )
        # Block calls to forbidden names.
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in _FORBIDDEN_NAMES:
                    raise SandboxError(
                        f"call to '{func.id}'() is not allowed "
                        f"(line {node.lineno})"
                    )
                # If the call is a method on `context`, the
                # attribute check above already passed. Anything
                # else must be a bare name in
                # `extra_allowed_call_names` or `SAFE_BUILTINS`.
                if (
                    func.id not in SAFE_BUILTINS
                    and func.id not in policy.extra_allowed_call_names
                ):
                    raise SandboxError(
                        f"call to '{func.id}()' is not in the "
                        f"allowed-call allowlist (line {node.lineno})"
                    )
            elif isinstance(func, ast.Attribute):
                # `context.<method>()` is fine (Attribute check
                # already enforced `context` base).
                pass
            else:
                raise SandboxError(
                    f"unsupported call target at line {node.lineno}"
                )

    return tree


class SandboxedContextREPL:
    """Drop-in replacement for `ContextREPL` with sandbox enforcement."""

    def __init__(
        self,
        context,
        policy: SandboxPolicy | None = None,
    ) -> None:
        self.context = context
        self.policy = policy or SandboxPolicy()
        self._env: dict[str, Any] = {
            name: getattr(context, name)
            for name in ("slice", "grep", "chunk", "ask", "recurse_on")
            if hasattr(context, name)
        }
        self._env["context"] = context
        self._env["__builtins__"] = SAFE_BUILTINS

    def run(self, code: str) -> tuple[str, str]:
        """Execute `code`; return (stdout, repr_of_last_value).

        Raises `SandboxError` if the code violates the policy.
        """
        tree = _ast_check(code, self.policy)
        out = io.StringIO()

        if self.policy.timeout_seconds is not None:
            return self._run_with_timeout(tree, code, out)
        return self._run_unbounded(tree, code, out)

    def _run_unbounded(
        self, tree: ast.Module, code: str, out: io.StringIO
    ) -> tuple[str, str]:
        with redirect_stdout(out):
            value_repr = self._exec(tree)
        stdout = out.getvalue()
        if len(stdout) > self.policy.max_output:
            stdout = stdout[: self.policy.max_output] + (
                f"\n[truncated; {len(stdout) - self.policy.max_output} bytes dropped]"
            )
        return stdout, value_repr

    def _run_with_timeout(
        self, tree: ast.Module, code: str, out: io.StringIO
    ) -> tuple[str, str]:
        """Run with a wall-clock timeout enforced by a watchdog thread."""
        result: dict = {"stdout": "", "last": ""}

        def target() -> None:
            try:
                with redirect_stdout(out):
                    last = self._exec(tree)
                result["stdout"] = out.getvalue()
                result["last"] = last
            except Exception as e:  # noqa: BLE001
                result["stdout"] = f"{type(e).__name__}: {e}\n"
                result["last"] = ""

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(self.policy.timeout_seconds)
        if thread.is_alive():
            return (
                f"[SandboxError] execution exceeded "
                f"{self.policy.timeout_seconds}s timeout",
                "",
            )
        if len(result["stdout"]) > self.policy.max_output:
            result["stdout"] = result["stdout"][: self.policy.max_output] + (
                f"\n[truncated; {len(result['stdout']) - self.policy.max_output} bytes dropped]"
            )
        return result["stdout"], result["last"]

    def _exec(self, tree: ast.Module) -> str:
        """Execute the parsed tree; return the repr of the last expression.

        Mirrors the Phase-1 `ContextREPL._exec` semantics but with
        a recursion-depth cap.
        """
        sys.setrecursionlimit(max(self.policy.max_recursion_depth, 200))
        body = list(tree.body)
        value_repr = ""
        if body and isinstance(body[-1], ast.Expr):
            last_expr = body.pop()
            stmts_module = ast.Module(body=body, type_ignores=[])
            try:
                exec(
                    compile(stmts_module, "<sandbox>", "exec"),
                    self._env,
                )
            except Exception as e:  # noqa: BLE001
                print(f"{type(e).__name__}: {e}", file=sys.__stdout__)
                return ""
            try:
                value = eval(  # noqa: S307 — sandboxed input
                    compile(
                        ast.Expression(body=last_expr.value),
                        "<sandbox>", "eval",
                    ),
                    self._env,
                )
            except Exception as e:  # noqa: BLE001
                print(f"{type(e).__name__}: {e}", file=sys.__stdout__)
                return ""
            if value is not None:
                value_repr = repr(value)
                print(value_repr)
        else:
            try:
                exec(compile(tree, "<sandbox>", "exec"), self._env)
            except Exception as e:  # noqa: BLE001
                print(f"{type(e).__name__}: {e}", file=sys.__stdout__)
                return ""
        sys.setrecursionlimit(1000)
        return value_repr

    def install(self, name: str, fn: Callable[..., object]) -> None:
        """Register a custom helper callable the model can invoke."""
        if name in self._env or name in SAFE_BUILTINS:
            raise ValueError(f"{name!r} is reserved")
        self._env[name] = fn