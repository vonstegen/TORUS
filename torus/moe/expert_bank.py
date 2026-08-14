"""Expert bank with explicit shared-primary / per-expert residual planes.

Phase 4 specialization:

- The bank stores one `ExpertResidualStack` per expert id.
- Each expert's stack has an optional *shared* primary plane (one
  `ResidualTernaryPlanes` reused across experts) plus a *unique*
  residual plane(s) stack.
- Primary sharing is opt-in: when set, multiple experts reuse the
  same primary plane and only the residual stack is per-expert.
  This is the memory-saving direction Phase 4 explores.

The bank is dict-shaped (`__getitem__`, `__contains__`, `__len__`)
so existing call sites continue to work; the new accessors are
additive.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from torus.core.residual_linear import ResidualTernaryLinear
from torus.quant.residual import ResidualTernaryPlanes


@dataclass
class ExpertResidualStack:
    """One expert's residual stack.

    Attributes:
        expert_id: the integer expert id this stack belongs to.
        residual_planes: per-expert residual `ResidualTernaryPlanes`.
        shared_primary: optional shared primary planes used by this
            expert. When `None`, the expert's own `residual_planes`
            is the primary (legacy Phase-1 behaviour).
    """
    expert_id: int
    residual_planes: ResidualTernaryPlanes
    shared_primary: ResidualTernaryPlanes | None = None

    @property
    def primary(self) -> ResidualTernaryPlanes:
        """Return the primary plane stack used by this expert."""
        if self.shared_primary is not None:
            return self.shared_primary
        return self.residual_planes

    @property
    def num_residual_planes(self) -> int:
        return self.residual_planes.num_planes

    def to_layer(
        self, gate=None, kernel: str = "dense", layer_id: int = 0
    ) -> ResidualTernaryLinear:
        """Build a `ResidualTernaryLinear` view of this stack."""
        from torus.core.gate import ResidualGate, GateMode
        return ResidualTernaryLinear(
            planes=self.residual_planes,
            gate=gate or ResidualGate(mode=GateMode.NEVER),
            kernel=kernel,
            layer_id=layer_id,
        )


@dataclass
class ExpertBank:
    """Holds per-expert `ExpertResidualStack`s, keyed by expert id."""
    experts: dict[int, ExpertResidualStack] = field(default_factory=dict)
    _shared_primary: ResidualTernaryPlanes | None = None

    def set_shared_primary(self, planes: ResidualTernaryPlanes) -> None:
        """Set the primary plane shared across all experts added later."""
        self._shared_primary = planes

    @property
    def shared_primary(self) -> ResidualTernaryPlanes | None:
        return self._shared_primary

    def add(
        self,
        expert_id: int,
        stack: ExpertResidualStack,
    ) -> None:
        """Register an expert's residual stack."""
        if stack.shared_primary is None and self._shared_primary is not None:
            stack.shared_primary = self._shared_primary
        self.experts[expert_id] = stack

    def add_residual(
        self,
        expert_id: int,
        residual_planes: ResidualTernaryPlanes,
    ) -> ExpertResidualStack:
        """Convenience: register a stack and return it."""
        stack = ExpertResidualStack(
            expert_id=expert_id,
            residual_planes=residual_planes,
            shared_primary=self._shared_primary,
        )
        self.add(expert_id, stack)
        return stack

    def get(self, expert_id: int) -> ExpertResidualStack:
        if expert_id not in self.experts:
            raise KeyError(f"expert id {expert_id} not in bank")
        return self.experts[expert_id]

    def layer(self, expert_id: int, **kw) -> ResidualTernaryLinear:
        """Return a `ResidualTernaryLinear` view of the expert's stack."""
        return self.get(expert_id).to_layer(**kw)

    def __len__(self) -> int:
        return len(self.experts)

    def __contains__(self, expert_id: int) -> bool:
        return expert_id in self.experts

    def __iter__(self):
        return iter(self.experts)

    def __getitem__(self, expert_id: int) -> ExpertResidualStack:
        return self.get(expert_id)

    def experts_sharing_primary(self) -> list[int]:
        """Return ids of experts whose primary plane is the shared one."""
        if self._shared_primary is None:
            return []
        return [
            eid for eid, st in self.experts.items()
            if st.shared_primary is self._shared_primary
        ]