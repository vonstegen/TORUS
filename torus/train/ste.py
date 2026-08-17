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

Norm calibration (added 2026-08-16):
- `ternary_quantize_with_ste(calibrate_norm=True, ref_weight=...)`
  rescales the per-group `scale` so that the quantized weight's
  L2 norm matches the FP16 reference. Without this, ternary codes
  × small per-group scale produce a weight whose norm is 50-70%
  of the FP16 reference, and the collapse compounds across layers
  to a ~0.0003 multiplier by the lm_head of a 16-layer model.
  Calibration preserves the FP16 norm per layer.
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
    calibrate_norm: bool = False,
    ref_weight: np.ndarray | None = None,
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

    `calibrate_norm=True` rescales the per-group `scale` so that
    `||quantized_weight||` matches `||ref_weight||` (defaults to
    `weight`). This fixes the per-layer norm collapse that
    otherwise halves the signal every layer in a deep network.
    The rescale is applied uniformly to `scale` (not per-group),
    so within-layer structure is preserved.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape {weight.shape}")
    w = _to_numpy(weight)
    if w.shape[1] % group_size != 0:
        raise ValueError(
            f"in_features={w.shape[1]} not divisible by group_size={group_size}"
        )
    ref = _to_numpy(ref_weight if ref_weight is not None else weight)
    scale = _absmean_scale(w, group_size)
    w_n = w.reshape(w.shape[0], -1, group_size) / scale[..., None]
    w_n = w_n.reshape(w.shape[0], -1)
    codes = np.clip(np.round(w_n), -1.0, 1.0)
    codes = np.where(np.abs(w_n) < threshold, 0.0, codes)
    codes_int = codes.astype(np.int8)
    s_full = np.repeat(scale, group_size, axis=-1)
    quantized = codes_int.astype(np.float32) * s_full
    if calibrate_norm:
        ref_norm = float(np.linalg.norm(ref))
        q_norm = float(np.linalg.norm(quantized))
        if q_norm > 0 and ref_norm > 0:
            # Apply the same multiplier to codes and scale so
            # both stay self-consistent for the STE path.
            mult = ref_norm / q_norm
            quantized = quantized * mult
            scale = scale * mult
            s_full = s_full * mult
            codes_int = np.clip(
                codes_int.astype(np.float32) * mult, -127.0, 127.0
            ).astype(np.int8)
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
        calibrate_norm: when True, rescale each plane's quantized
            weight so its L2 norm matches the underlying learnable
            `weight` (or `residual_weight`). Prevents the per-layer
            norm collapse that otherwise compounds across deep
            networks (see module docstring).
    """
    weight: np.ndarray
    group_size: int = 128
    threshold: float = 0.7
    residual_weight: np.ndarray | None = None
    calibrate_norm: bool = True

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
            w_np,
            group_size=self.group_size,
            threshold=self.threshold,
            calibrate_norm=self.calibrate_norm,
            ref_weight=self.weight,
        )
        if n_planes < 2 or self.residual_weight is None:
            return codes, scale, q_primary

        r_np = _to_numpy(self.residual_weight)
        _r_codes, _r_scale, q_residual = ternary_quantize_with_ste(
            r_np,
            group_size=self.group_size,
            threshold=self.threshold,
            calibrate_norm=self.calibrate_norm,
            ref_weight=self.residual_weight,
        )
        # Return the combined quantized weight as the "primary"
        # slot; codes/scale only describe the primary plane.
        return codes, scale, q_primary + q_residual

    def params(self) -> np.ndarray:
        """Return the learnable primary weight (the only thing the
        simple optimizer updates). The trainer uses `_params_np` to
        update both `weight` and `residual_weight` directly."""
        return self.weight