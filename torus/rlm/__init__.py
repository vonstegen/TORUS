"""Recursive Language Model primitive: context-as-variable in a REPL.

Phase 1 provides a small in-memory REPL and a RecursiveContext helper that
exposes slicing/inspection/calling primitives. This is the substrate that
later fuses with the ternary model itself.
"""
from torus.rlm.context import RecursiveContext, ContextSlice
from torus.rlm.repl import ContextREPL

__all__ = ["RecursiveContext", "ContextSlice", "ContextREPL"]
