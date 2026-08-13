"""TORUS - Ternary Optimized Recursive Unified System.

A reference implementation of:
- Residual ternary planes (W = s1*T1 + s2*T2 + ...)
- Adaptive residual gating (per-token / per-layer / per-expert)
- Recursive context-as-variable (RLM-style REPL primitive)
"""

from torus import core, moe, quant, rlm

__version__ = "0.1.0"

__all__ = ["core", "moe", "quant", "rlm", "__version__"]
