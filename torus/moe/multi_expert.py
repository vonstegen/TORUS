"""MultiExpertRouter: scale the per-expert residual stacks to production-shape models.

Phase 4 introduced `ExpertBank` (per-expert residual stacks) and
`TopKRouter` (token-to-expert routing) but didn't tie them together
end-to-end. Phase 7 closes that loop:

- Each token is routed to its top-k experts via `TopKRouter.route()`.
- The router emits a per-token confidence signal (top-k prob mass).
- For each (token, expert) pair, a `GatePolicy` decides how many
  residual planes to engage:
  - Low router confidence → engage more planes (the residual
    captures the nuance the router is wavering over).
  - High router confidence → primary plane only (the router is
    sure; extra compute isn't needed).
- The decision is logged per-call for telemetry.

This is the production-shape wiring: many experts × many planes ×
many tokens, with per-call adaptive compute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from torus.moe.expert_bank import ExpertBank
from torus.moe.router import RouteResult, TopKRouter


@dataclass(frozen=True)
class PerCallDecision:
    """One (token, expert) decision from `MultiExpertRouter.route`."""
    token_idx: int
    expert_id: int
    weight: float                 # router weight for this expert
    confidence: float             # top-k prob mass for this token
    n_planes: int                 # how many residual planes to engage


@dataclass(frozen=True)
class MultiExpertResult:
    """Result of one `MultiExpertRouter.route` call."""
    decisions: list[PerCallDecision] = field(default_factory=list)


@dataclass
class GatePolicy:
    """How router confidence maps to plane-count engagement.

    Linear policy: when `confidence < confidence_low`, use
    `n_planes_high` planes; when `confidence > confidence_high`,
    use `n_planes_low` planes; interpolate linearly in between.

    Default values: low confidence (0.5) → 4 planes; high
    confidence (0.9) → 1 plane. This is the natural curve for
    the Phase-4 design: unsure tokens use the full residual stack,
    sure tokens stick to the primary plane.
    """
    confidence_low: float = 0.5
    confidence_high: float = 0.9
    n_planes_low: int = 1
    n_planes_high: int = 4

    def n_planes_for(self, confidence: float, max_planes: int) -> int:
        """Map a per-token confidence to a per-call plane count."""
        if max_planes <= self.n_planes_low:
            return max(1, max_planes)
        ceiling = min(self.n_planes_high, max_planes)
        if confidence <= self.confidence_low:
            return ceiling
        if confidence >= self.confidence_high:
            return self.n_planes_low
        # Linear interpolation between the two thresholds.
        span = self.confidence_high - self.confidence_low
        frac = (confidence - self.confidence_low) / span
        # Map frac in [0, 1] from n_planes_high -> n_planes_low.
        planes = ceiling - frac * (ceiling - self.n_planes_low)
        # Round to nearest int, clamp to [1, ceiling].
        return max(1, min(ceiling, int(round(planes))))


class MultiExpertRouter:
    """Compose `TopKRouter` + `ExpertBank` + `GatePolicy`."""

    def __init__(
        self,
        router: TopKRouter,
        bank: ExpertBank,
        policy: GatePolicy | None = None,
        on_decision: Callable[[PerCallDecision], None] | None = None,
    ) -> None:
        self.router = router
        self.bank = bank
        self.policy = policy or GatePolicy()
        # Optional telemetry hook: callers can attach a function that
        # receives each (token, expert) decision. Used by the demo.
        self.on_decision = on_decision

    def route(
        self,
        token_features: np.ndarray,
        token_values: np.ndarray | None = None,
    ) -> MultiExpertResult:
        """Route each token to its top-k experts with adaptive plane-count.

        Args:
            token_features: 1D float32 array of length batch_size;
                each entry is a per-token routing feature (Phase 1
                uses scalars; richer features land in later phases).
            token_values: optional 1D array of integer token ids;
                only used for telemetry / debugging. When None, the
                token idx is just its position in `token_features`.

        Returns:
            `MultiExpertResult` with one `PerCallDecision` per
            (token, expert) pair. Each decision includes the
            router's weight, the per-token confidence, and the
            per-call `n_planes` the gate decided to engage.
        """
        route: RouteResult = self.router.route(token_features)
        confs: np.ndarray = self.router.confidence(route)

        decisions: list[PerCallDecision] = []
        for tok_idx in range(token_features.shape[0]):
            conf = float(confs[tok_idx])
            for k in range(self.router.top_k):
                eid = int(route.indices[tok_idx, k])
                weight = float(route.weights[tok_idx, k])
                if eid not in self.bank:
                    # Router picked an expert not in the bank. Skip
                    # rather than raise — the production wiring may
                    # route tokens to inactive experts and we want
                    # the call to continue.
                    continue
                stack = self.bank[eid]
                n_planes = self.policy.n_planes_for(conf, stack.num_residual_planes)
                d = PerCallDecision(
                    token_idx=tok_idx,
                    expert_id=eid,
                    weight=weight,
                    confidence=conf,
                    n_planes=n_planes,
                )
                decisions.append(d)
                if self.on_decision is not None:
                    self.on_decision(d)

        return MultiExpertResult(decisions=decisions)

    def decision_table(self, token_features: np.ndarray) -> str:
        """Render the per-call decisions as a human-readable table.

        Useful for the demo and for debugging; not part of the
        hot path.
        """
        result = self.route(token_features)
        if not result.decisions:
            return "(no decisions)"
        lines = [
            "  token  expert  weight  conf    planes",
            "  -----  ------  ------  ------  ------",
        ]
        for d in result.decisions:
            lines.append(
                f"  {d.token_idx:>5}  {d.expert_id:>6}  {d.weight:>.3f}  "
                f"{d.confidence:>.3f}  {d.n_planes:>5}"
            )
        return "\n".join(lines)