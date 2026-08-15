"""Recursive Language Model primitive: context-as-variable in a REPL.

Phase 1: small in-memory REPL + RecursiveContext (slice / grep / chunk / ask).
Phase 5: PrimeAgentLoop drives the REPL iteratively with a model callable.
Phase 8: PersistentContext is an NVMe-backed RecursiveContext drop-in.
Phase 9: PersistentContextIndex gives grep O(log n + matches).
"""
from torus.rlm.agent import (
    DONE_SENTINEL,
    AgentResult,
    AgentStep,
    PrimeAgentLoop,
)
from torus.rlm.context import ContextSlice, RecursiveContext
from torus.rlm.index import PersistentContextIndex
from torus.rlm.persistent import PersistentContext
from torus.rlm.repl import ContextREPL
from torus.rlm.sandbox import SandboxError, SandboxPolicy, SandboxedContextREPL

__all__ = [
    "RecursiveContext",
    "ContextSlice",
    "ContextREPL",
    "SandboxedContextREPL",
    "SandboxError",
    "SandboxPolicy",
    "PersistentContext",
    "PersistentContextIndex",
    "PrimeAgentLoop",
    "AgentResult",
    "AgentStep",
    "DONE_SENTINEL",
]
