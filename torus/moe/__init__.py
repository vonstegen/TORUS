"""Mixture-of-Experts aware scaffolding for residual ternary planes.

Phase 1 scaffold: expert bank + router-aware gate. Full training recipes
are Phase 3.
"""
from torus.moe.expert_bank import ExpertBank
from torus.moe.router import TopKRouter

__all__ = ["ExpertBank", "TopKRouter"]
