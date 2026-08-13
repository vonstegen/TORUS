"""Core TORUS primitives: residual plane container + adaptive gate."""
from torus.core.gate import ResidualGate, GateMode, GateDecision
from torus.core.residual_linear import ResidualTernaryLinear

__all__ = [
    "ResidualGate",
    "GateMode",
    "GateDecision",
    "ResidualTernaryLinear",
]
