"""Straight-through estimator (STE) for ternary quantization.

Training a network whose weights are ternary requires gradients to
*flow through* the hard quantizer. The classical trick is the
straight-through estimator: in the forward pass, quantize; in the
backward pass, pretend the quantizer is the identity (and optionally
clip the gradient).

This module is pure numpy. It doesn't implement backprop; it gives
you the *function* (`ternary_quantize_with_ste`) and a small
`TernarySTE` object you can attach to a hidden weight. The actual
optimizer is left to the trainer, which treats this as a deterministic
forward-only quantization with a backward hook.

Phase-3 trainer compatibility:
- `TernarySTE` carries an optional `residual_weight`. When set,
  `forward(n_planes=2)` returns the sum of two independently
  quantized ternary weights (primary + residual). When
  `n_planes=1`, only the primary contributes. This matches
  `torus.quant.residual.ResidualTernaryPlanes` so the trainer's
  `n_planes` parameter has the same meaning at the STE level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Forward quantization: same kernel as torus.quant.ternary.ternary_quantize
# but exposed at a granularity the STE path needs (returns int8 codes
# plus the float `effective` weight). Backward is the identity w.r.t.
# the float "weight parameter" that produced the codes.
def _absmean_scale(w: np.ndarray, group_size: int, eps: float = 1e-8) -> np.ndarray:
    cols = w.shape[1]
    grouped = w.reshape(w.shape[0], cols // group_size, group_size)
    scale = np.abs(grouped).mean(axis=-1)
    return np.maximum(scale, eps)


def ternary_quantize_with_ste(
    weight: np.ndarray,
    group_size: int = 128,
    threshold: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize `weight` to ternary codes and the per-group scale.

    Returns `(codes, scale, quantized_weight)`:
        codes           : int8 in {-1, 0, +1}, same shape as `weight`.
        scale           : float32, shape `(rows, n_groups)`.
        quantized_weight: float32 effective weight (`T * s_broadcast`),
                          suitable as a drop-in replacement for `weight`
                          in a forward pass. The backward pass should
                          treat the gradient as flowing through this
                          directly (the STE).
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape {weight.shape}")
    w = weight.astype(np.float32, copy=False)
    if w.shape[1] % group_size != 0:
        raise ValueError(
            f"in_features={w.shape[1]} not divisible by group_size={group_size}"
        )
    scale = _absmean_scale(w, group_size)
    w_n = w.reshape(w.shape[0], -1, group_size) / scale[..., None]
    w_n = w_n.reshape(w.shape[0], -1)
    codes = np.clip(np.round(w_n), -1.0, 1.0)
    codes = np.where(np.abs(w_n) < threshold, 0.0, codes)
    codes_int = codes.astype(np.int8)
    s_full = np.repeat(scale, group_size, axis=-1)
    quantized = codes_int.astype(np.float32) * s_full
    return codes_int, scale, quantized


def _to_numpy(weight):
    """Convert torch Parameter / tensor / ndarray to float32 ndarray."""
    import torch as _torch
    if isinstance(weight, _torch.Tensor):
        return weight.detach().cpu().numpy()
    return np.asarray(weight)


@dataclass
class TernarySTE:
    """Stateful STE wrapper around a learnable full-precision weight.

    On `forward()` this returns the ternary-quantized effective weight.
    The trainer is responsible for applying the gradient to `weight`
    directly — the STE gives no special backward handling beyond the
    identity approximation.

    Args:
        weight: float32 2D array, the learnable parameter.
        group_size: group width for the per-group scale.
        threshold: sparsity threshold (same semantics as `ternary_quantize`).
        residual_weight: optional float32 2D array; when set, the STE
            produces a two-plane forward (primary + residual) gated
            by the `n_planes` argument to `forward()`. The residual
            weight is itself learnable; the trainer's optimizer
            updates it via the `_params_np` numpy buffer.
    """
    weight: np.ndarray
    group_size: int = 128
    threshold: float = 0.7
    residual_weight: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got shape {self.weight.shape}")
        # Auto-pick the largest power-of-two group size that fits.
        if self.weight.shape[1] % self.group_size != 0:
            n = self.weight.shape[1]
            g = 1
            while g * 2 <= n and n % (g * 2) == 0:
                g *= 2
            # If no power-of-two > 1 divides n (e.g. n is prime),
            # fall back to one big group covering the whole row.
            if g == 1:
                g = n
            # The dataclass is frozen; bypass __setattr__.
            object.__setattr__(self, "group_size", g)
        if self.residual_weight is not None:
            if self.residual_weight.shape != self.weight.shape:
                raise ValueError(
                    f"residual_weight shape {self.residual_weight.shape} "
                    f"does not match weight shape {self.weight.shape}"
                )

    def forward(
        self, n_planes: int = 1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(codes, scale, quantized_weight)`.

        When `residual_weight is None`, only the primary plane is
        computed regardless of `n_planes`. When `residual_weight` is
        present and `n_planes >= 2`, the returned `quantized_weight`
        is the sum of two independently quantized ternary weights
        (primary + residual).
        """
        w_np = _to_numpy(self.weight)
        codes, scale, q_primary = ternary_quantize_with_ste(
            w_np, group_size=self.group_size, threshold=self.threshold,
        )
        if n_planes < 2 or self.residual_weight is None:
            return codes, scale, q_primary

        r_np = _to_numpy(self.residual_weight)
        _r_codes, _r_scale, q_residual = ternary_quantize_with_ste(
            r_np, group_size=self.group_size, threshold=self.threshold,
        )
        # Return the combined quantized weight as the "primary"
        # slot; codes/scale only describe the primary plane.
        return codes, scale, q_primary + q_residual

    def params(self) -> np.ndarray:
        """Return the learnable primary weight (the only thing the
        simple optimizer updates). The trainer uses `_params_np` to
        update both `weight` and `residual_weight` directly."""
        return self.weight