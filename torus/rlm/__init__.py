"""Recursive Language Model primitive: context-as-variable in a REPL.

Phase 1: small in-memory REPL + RecursiveContext (slice / grep / chunk / ask).
Phase 5: PrimeAgentLoop drives the REPL iteratively with a model callable.
"""
from torus.rlm.agent import (
    DONE_SENTINEL,
    AgentResult,
    AgentStep,
    PrimeAgentLoop,
)
from torus.rlm.context import ContextSlice, RecursiveContext
from torus.rlm.repl import ContextREPL

__all__ = [
    "RecursiveContext",
    "ContextSlice",
    "ContextREPL",
    "PrimeAgentLoop",
    "AgentResult",
    "AgentStep",
    "DONE_SENTINEL",
]