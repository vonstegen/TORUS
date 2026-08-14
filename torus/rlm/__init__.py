"""Recursive Language Model primitive: context-as-variable in a REPL.

Phase 1: small in-memory REPL + RecursiveContext (slice / grep / chunk / ask).
Phase 5: PrimeAgentLoop drives the REPL iteratively with a model callable.
Phase 8: PersistentContext is an NVMe-backed RecursiveContext drop-in.
"""
from torus.rlm.agent import (
    DONE_SENTINEL,
    AgentResult,
    AgentStep,
    PrimeAgentLoop,
)
from torus.rlm.context import ContextSlice, RecursiveContext
from torus.rlm.persistent import PersistentContext
from torus.rlm.repl import ContextREPL

__all__ = [
    "RecursiveContext",
    "ContextSlice",
    "ContextREPL",
    "PersistentContext",
    "PrimeAgentLoop",
    "AgentResult",
    "AgentStep",
    "DONE_SENTINEL",
]