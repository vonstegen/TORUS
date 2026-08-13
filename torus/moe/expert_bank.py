"""Expert bank: maps expert ids to per-expert residual ternary layers.

The bank itself is just a dict-shaped container. What makes it useful for
TORUS is the *sharing* rule for the primary plane:

- In Phase 1 we store a primary plane *per expert* (the canonical case);
  sharing across experts is a future optimization that Phase 3 will
  explore with capability-aware distillation.

- Each expert keeps its own residual plane(s). When the gate engages,
  only the selected expert's residual plane fires.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from torus.core.residual_linear import ResidualTernaryLinear


@dataclass
class ExpertBank:
    """Holds per-expert `ResidualTernaryLinear` layers, keyed by expert id."""
    experts: dict[int, ResidualTernaryLinear] = field(default_factory=dict)

    def add(self, expert_id: int, layer: ResidualTernaryLinear) -> None:
        self.experts[expert_id] = layer

    def get(self, expert_id: int) -> ResidualTernaryLinear:
        if expert_id not in self.experts:
            raise KeyError(f"expert id {expert_id} not in bank")
        return self.experts[expert_id]

    def __len__(self) -> int:
        return len(self.experts)

    def __contains__(self, expert_id: int) -> bool:
        return expert_id in self.experts
