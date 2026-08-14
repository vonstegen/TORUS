"""Mixture-of-Experts aware scaffolding for residual ternary planes.

Phase 1 scaffold: expert bank + router-aware gate.
Phase 4: per-expert residual stacks with optional shared primary;
router confidence drives plane-count gate activation.
"""
from torus.moe.expert_bank import ExpertBank, ExpertResidualStack
from torus.moe.router import TopKRouter

__all__ = ["ExpertBank", "ExpertResidualStack", "TopKRouter"]