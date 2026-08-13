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
    """
    weight: np.ndarray
    group_size: int = 128
    threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got shape {self.weight.shape}")
        if self.weight.shape[1] % self.group_size != 0:
            raise ValueError(
                f"in_features={self.weight.shape[1]} not divisible by group_size={self.group_size}"
            )

    def forward(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (codes, scale, quantized_weight)."""
        return ternary_quantize_with_ste(
            self.weight, group_size=self.group_size, threshold=self.threshold,
        )

    def params(self) -> np.ndarray:
        """Return the learnable weight (the only thing the optimizer updates)."""
        return self.weight
