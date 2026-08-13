"""Multi-plane residual ternary quantization.

Each plane is a TernaryPlane. The composite is W = sum_i (T_i * s_i).
Plane 0 is always present (the primary plane). Planes 1..k-1 are residual
planes: plane i is trained to capture the residual left after planes
0..i-1 are subtracted. Residual planes can be activated or skipped by the
adaptive gate at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torus.quant.ternary import TernaryPlane, ternary_quantize


@dataclass(frozen=True)
class ResidualTernaryPlanes:
    """Stack of ternary planes that together approximate one weight matrix."""
    planes: tuple[TernaryPlane, ...]

    @property
    def num_planes(self) -> int:
        return len(self.planes)

    @property
    def shape(self) -> tuple[int, int]:
        return self.planes[0].shape

    def plane(self, idx: int) -> TernaryPlane:
        return self.planes[idx]

    def effective_bits_per_weight(self) -> float:
        """Approximate bits/weight summed across all planes and their scales."""
        out_f, in_f = self.shape
        per_plane = sum(p.effective_bits_per_weight() for p in self.planes)
        return per_plane  # already summed per-element above

    def effective_bits_for_active_planes(self, active: int) -> float:
        """Bits/weight when only the first `active` planes are used."""
        if active < 1:
            raise ValueError(f"active must be >= 1, got {active}")
        if active > self.num_planes:
            raise ValueError(
                f"active={active} exceeds num_planes={self.num_planes}"
            )
        out_f, in_f = self.shape
        weight_bits = 2 * out_f * in_f * active
        total_groups = self.planes[0].n_groups
        scale_bits = 16 * out_f * total_groups * active
        return (weight_bits + scale_bits) / (out_f * in_f)


def residual_quantize(
    weight: np.ndarray,
    num_planes: int = 2,
    group_size: int = 128,
    threshold: float = 0.7,
) -> ResidualTernaryPlanes:
    """Build a residual stack of ternary planes approximating `weight`.

    The residual after plane i is fed into plane i+1. The last plane
    always captures whatever the previous planes did not.

    Args:
        weight: float32 array of shape (out_features, in_features).
        num_planes: how many ternary planes to stack. Must be >= 1.
        group_size: per-group scale width.
        threshold: ternary quantization magnitude threshold (see ternary_quantize).

    Returns:
        ResidualTernaryPlanes holding the stacked planes.
    """
    if num_planes < 1:
        raise ValueError(f"num_planes must be >= 1, got {num_planes}")
    w = weight.astype(np.float32, copy=False)
    planes: list[TernaryPlane] = []
    residual = w
    for _ in range(num_planes):
        plane = ternary_quantize(residual, group_size=group_size, threshold=threshold)
        w_hat = plane.reconstruct()
        residual = residual - w_hat
        planes.append(plane)
    return ResidualTernaryPlanes(planes=tuple(planes))


def compose_planes(planes: ResidualTernaryPlanes, active: int | None = None) -> np.ndarray:
    """Reconstruct W_hat from the first `active` planes (all by default).

    Args:
        planes: a ResidualTernaryPlanes container.
        active: how many planes to include. None means all.

    Returns:
        float32 array with the same shape as the underlying weight matrix.
    """
    if active is None:
        active = planes.num_planes
    if active < 1 or active > planes.num_planes:
        raise ValueError(f"active must be in [1, {planes.num_planes}], got {active}")
    out_f, in_f = planes.shape
    acc = np.zeros((out_f, in_f), dtype=np.float32)
    for i in range(active):
        acc += planes.plane(i).reconstruct()
    return acc
