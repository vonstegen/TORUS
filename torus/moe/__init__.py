"""Mixture-of-Experts aware scaffolding for residual ternary planes.

Phase 1 scaffold: expert bank + router-aware gate.
Phase 4: per-expert residual stacks with optional shared primary;
router confidence drives plane-count gate activation.
Phase 7: MultiExpertRouter composes TopKRouter + ExpertBank +
GatePolicy for production-shape multi-expert routing.
"""
from torus.moe.expert_bank import ExpertBank, ExpertResidualStack
from torus.moe.multi_expert import (
    GatePolicy,
    MultiExpertResult,
    MultiExpertRouter,
    PerCallDecision,
)
from torus.moe.router import TopKRouter

__all__ = [
    "ExpertBank",
    "ExpertResidualStack",
    "GatePolicy",
    "MultiExpertResult",
    "MultiExpertRouter",
    "PerCallDecision",
    "TopKRouter",
]
