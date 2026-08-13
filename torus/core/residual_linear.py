"""A reference `nn.Linear`-equivalent using residual ternary planes.

This is the unit the kernel work targets. Phase 1 dispatched to the
dense numpy matmul. Phase 2 adds:

- A `kernel` argument selecting among registered CPU kernels
  (`dense`, `sparse`, `unrolled`). Future phases add `cuda` and
  `avx512` entries.

- Optional `telemetry` to record per-layer activation rates and op
  counts; used by the memory-tier policy and by evaluation tooling.

The public method signature is unchanged so existing tests, callers,
and the RLM primitive keep working.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torus.core.gate import GateDecision, GateMode, ResidualGate
from torus.core.kernels import (
    OpCount,
    get_kernel,
    ternary_gemv_dense,
)
from torus.core.telemetry import GateTelemetry
from torus.quant.residual import ResidualTernaryPlanes, compose_planes
from torus.quant.ternary import TernaryPlane


def _column_scales(plane: TernaryPlane) -> np.ndarray:
    s = plane.scales
    return np.repeat(s, plane.group_size, axis=-1).astype(np.float32)


def ternary_matmul(x: np.ndarray, plane: TernaryPlane) -> np.ndarray:
    """Compute y = x @ (T * s_group)^T. Pure-numpy reference.

    Kept for backward compatibility with Phase 1 callers. New code
    should use the kernel registry (`get_kernel("dense" | "sparse" | ...)`).
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (batch, in_features), got {x.shape}")
    if x.shape[1] != plane.codes.shape[1]:
        raise ValueError(
            f"x.shape[1]={x.shape[1]} != in_features={plane.codes.shape[1]}"
        )
    weight = plane.codes.astype(np.float32) * _column_scales(plane)
    return x @ weight.T


def residual_ternary_matmul(
    x: np.ndarray,
    planes: ResidualTernaryPlanes,
    active_planes: int | np.ndarray,
    kernel: str = "dense",
) -> tuple[np.ndarray, list[OpCount]]:
    """Matmul using only the first `active_planes` ternary planes.

    Args:
        x: float32 activations of shape (batch, in_features).
        planes: the stacked planes.
        active_planes: int (uniform) for now; per-row variants come in
            Phase 3.
        kernel: kernel name from `get_kernel`. The dense kernel (default)
            matches the Phase-1 behavior exactly. `sparse` and `unrolled`
            match the op count a real ternary kernel performs.

    Returns:
        (y, op_counts) where `y` has shape (batch, out_features) and
        `op_counts` has one entry per *invoked* plane.
    """
    if isinstance(active_planes, int):
        if active_planes < 1 or active_planes > planes.num_planes:
            raise ValueError(
                f"active_planes={active_planes} out of range [1,{planes.num_planes}]"
            )
        # The kernel API is per-plane; compose the result manually.
        impl = get_kernel(kernel)
        out: np.ndarray | None = None
        ops: list[OpCount] = []
        for i in range(active_planes):
            y_i, op = impl(x, planes.plane(i))
            if out is None:
                out = y_i.copy()
            else:
                out += y_i
            ops.append(op)
        if out is None:  # unreachable; active_planes >= 1
            raise RuntimeError("no planes invoked")
        return out, ops
    raise NotImplementedError(
        "per-row activation requires kernel support; coming in Phase 3."
    )


@dataclass
class ResidualTernaryLinear:
    """Drop-in reference layer: holds residual ternary planes + a gate."""
    planes: ResidualTernaryPlanes
    gate: ResidualGate = None  # type: ignore[assignment]
    bias: np.ndarray | None = None
    kernel: str = "dense"
    layer_id: int = -1
    telemetry: GateTelemetry | None = None

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
        active_planes = self._decide_active_planes(
            residual_relative_magnitude=residual_relative_magnitude,
            depth=depth,
        )
        y, ops = residual_ternary_matmul(
            x, self.planes, active_planes, kernel=self.kernel,
        )
        if self.bias is not None:
            y = y + self.bias
        decision = self._make_decision(active_planes)
        if self.telemetry is not None:
            if self.layer_id >= 0:
                self.telemetry.begin_layer(self.layer_id)
            self.telemetry.record(decision, ops)
        return y, decision

    def _decide_active_planes(
        self,
        residual_relative_magnitude: float | np.ndarray | None,
        depth: int | float,
    ) -> int:
        if self.gate.mode is GateMode.ADAPTIVE:
            if residual_relative_magnitude is None:
                raise ValueError(
                    "ADAPTIVE gate requires residual_relative_magnitude"
                )
            decision = self.gate.decide(
                residual_relative_magnitude=residual_relative_magnitude,
                depth=depth,
            )
            return self.planes.num_planes if bool(decision.activate.any()) else 1
        if self.gate.mode is GateMode.ALWAYS:
            return self.planes.num_planes
        return 1  # NEVER

    def _make_decision(self, active_planes: int) -> GateDecision:
        activated = active_planes > 1
        return GateDecision(
            activate=np.array([activated], dtype=bool),
            score=np.array([float(activated)], dtype=np.float32),
        )
