"""Core TORUS primitives: residual plane container + adaptive gate."""
from torus.core.gate import GateDecision, GateMode, ResidualGate
from torus.core.kernels import (
    OpCount,
    get_kernel,
    register_kernel,
    ternary_gemv_dense,
    ternary_gemv_sparse,
    ternary_gemv_unrolled,
)
from torus.core.memory import (
    Budget,
    MemoryTier,
    Placement,
    PlaneSize,
    p620_default_budget,
    gb10_default_budget,
    place_planes,
)
from torus.core.residual_linear import (
    ResidualTernaryLinear,
    residual_ternary_matmul,
    ternary_matmul,
)
from torus.core.telemetry import GateTelemetry, LayerStats

__all__ = [
    "GateDecision",
    "GateMode",
    "ResidualGate",
    "ResidualTernaryLinear",
    "residual_ternary_matmul",
    "ternary_matmul",
    "OpCount",
    "get_kernel",
    "register_kernel",
    "ternary_gemv_dense",
    "ternary_gemv_sparse",
    "ternary_gemv_unrolled",
    "Budget",
    "MemoryTier",
    "PlaneSize",
    "place_planes",
    "p620_default_budget",
    "gb10_default_budget",
    "GateTelemetry",
    "LayerStats",
]
