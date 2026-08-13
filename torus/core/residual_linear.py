"""A reference `nn.Linear`-equivalent using residual ternary planes.

This is the unit the kernel work targets. The phase-1 implementation is
pure numpy: it dispatches to `compose_planes(..., active=1)` or
`compose_planes(..., active=k)` based on the gate decision, and then
performs the matrix multiplication. The replacement of this matmul
with a specialized kernel is Phase 2; the surface area stays the same.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torus.core.gate import GateDecision, GateMode, ResidualGate
from torus.quant.residual import ResidualTernaryPlanes, compose_planes
from torus.quant.ternary import TernaryPlane


def _column_scales(plane: TernaryPlane) -> np.ndarray:
    """Group scale repeated across the in_features axis (rows broadcast)."""
    s = plane.scales
    s_full = np.repeat(s, plane.group_size, axis=-1)
    # scale shape is (rows, groups); broadcast to (rows, cols) by repeating each group's
    # scale across group_size columns. Each row has the same per-column scale.
    return s_full.astype(np.float32)


def ternary_matmul(x: np.ndarray, plane: TernaryPlane) -> np.ndarray:
    """Compute y = x @ (T * s_group)^T.

    Args:
        x: float32 activations of shape (batch, in_features).
        plane: a single TernaryPlane.

    Returns:
        float32 outputs of shape (batch, out_features).
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (batch, in_features), got {x.shape}")
    if x.shape[1] != plane.codes.shape[1]:
        raise ValueError(
            f"x.shape[1]={x.shape[1]} != in_features={plane.codes.shape[1]}"
        )
    # Replace multiply with add/subtract: T scaled by per-row scale (broadcast).
    s_full = _column_scales(plane)            # (out, in)
    weight = plane.codes.astype(np.float32) * s_full
    return x @ weight.T


def residual_ternary_matmul(
    x: np.ndarray,
    planes: ResidualTernaryPlanes,
    active_planes: int | np.ndarray,
) -> np.ndarray:
    """Matmul using only the first `active_planes` ternary planes.

    Args:
        x: float32 activations of shape (batch, in_features).
        planes: the stacked planes.
        active_planes: int (uniform) or int array of shape (batch,) for
            per-row activation decisions. In production the per-row case
            is the common one (gate decides per token).

    Returns:
        float32 outputs of shape (batch, out_features).
    """
    if isinstance(active_planes, int):
        if active_planes < 1 or active_planes > planes.num_planes:
            raise ValueError(
                f"active_planes={active_planes} out of range [1,{planes.num_planes}]"
            )
        # Uniform: build W once, matmul once.
        weight = compose_planes(planes, active=active_planes)
        return x @ weight.T
    raise NotImplementedError(
        "per-row activation requires kernel support; coming in Phase 2."
    )


@dataclass
class ResidualTernaryLinear:
    """Drop-in reference layer: holds residual ternary planes + a gate.

    `forward(x, gate_signals=...)` dispatches to the appropriate number of
    ternary GEMMs based on the gate decision. The default gate uses
    `GateMode.ALWAYS` so behavior is deterministic for first-time users;
    switching to `GateMode.ADAPTIVE` enables the runtime quality dial.
    """
    planes: ResidualTernaryPlanes
    gate: ResidualGate = None  # type: ignore[assignment]
    bias: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.gate is None:
            self.gate = ResidualGate(mode=GateMode.ALWAYS)

    @property
    def in_features(self) -> int:
        return self.planes.shape[1]

    @property
    def out_features(self) -> int:
        return self.planes.shape[0]

    def forward(
        self,
        x: np.ndarray,
        residual_relative_magnitude: float | np.ndarray | None = None,
        depth: int | float = 0,
    ) -> tuple[np.ndarray, GateDecision]:
        """Run the layer; return (output, gate_decision).

        If `gate.mode is ADAPTIVE`, callers should supply a magnitude
        estimate (e.g. the relative residual energy for this layer) and
        a depth. With `GateMode.ALWAYS` both are ignored.
        """
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"x has in_features={x.shape[1]} but layer expects {self.in_features}"
            )
        if self.gate.mode is GateMode.ADAPTIVE:
            if residual_relative_magnitude is None:
                raise ValueError(
                    "ADAPTIVE gate requires residual_relative_magnitude"
                )
            decision = self.gate.decide(
                residual_relative_magnitude=residual_relative_magnitude,
                depth=depth,
            )
            active_planes = self.planes.num_planes if bool(decision.activate.any()) else 1
        else:
            active_planes = self.planes.num_planes if self.gate.mode is GateMode.ALWAYS else 1
            decision = GateDecision(
                activate=np.array([active_planes > 1], dtype=bool),
                score=np.array([float(active_planes > 1)], dtype=np.float32),
            )
        y = residual_ternary_matmul(x, self.planes, active_planes)
        if self.bias is not None:
            y = y + self.bias
        return y, decision
