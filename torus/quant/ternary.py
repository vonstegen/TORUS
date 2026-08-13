"""Single-plane ternary quantization.

A ternary matrix T has values in {-1, 0, +1} with a per-group scale s.
For a row-major weight matrix W of shape (out_features, in_features), we
split along the last axis into groups of `group_size` and store:

    T   : int8 array of {-1, 0, +1}
    s   : float32 array of shape (out_features, n_groups)

The approximation is:

    W_hat = T * s_group        (broadcast)

Storage budget per element:
    T            -> 2 bits (-1, 0, +1 with 4th unused value)
    s            -> 16 bits / group_size (e.g. 128 -> ~0.125 bits/weight)
    total        -> ~1.625 + 0.125 bits/weight for group_size=128

The math here is the absmean scaling scheme used by TWN / BitNet b1.58,
adapted to a per-group layout for higher fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_VALID_TERNARY = np.array([-1, 0, 1], dtype=np.int8)


@dataclass(frozen=True)
class TernaryPlane:
    """One ternary plane: 2-bit weights plus per-group float scale."""
    codes: np.ndarray   # int8, shape (out_features, in_features), values in {-1,0,1}
    scales: np.ndarray  # float32, shape (out_features, n_groups)
    group_size: int

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.codes.shape)  # type: ignore[return-value]

    @property
    def n_groups(self) -> int:
        return int(self.scales.shape[-1])

    def effective_bits_per_weight(self) -> float:
        """Approximate bits per weight including the scale overhead."""
        out_f, in_f = self.shape
        scale_bits = 16 * out_f * self.n_groups  # FP16 per scale
        weight_bits = 2 * out_f * in_f
        return (weight_bits + scale_bits) / (out_f * in_f)

    def reconstruct(self) -> np.ndarray:
        """Return the float reconstruction W_hat = T * s_group."""
        out_f, in_f = self.shape
        if in_f % self.group_size != 0:
            raise ValueError(
                f"in_features={in_f} not divisible by group_size={self.group_size}"
            )
        s_full = np.repeat(self.scales, self.group_size, axis=-1)
        return self.codes.astype(np.float32) * s_full


def _grouped_absmean_scale(w: np.ndarray, group_size: int, eps: float = 1e-8) -> np.ndarray:
    """Compute per-group absmean scale.

    Args:
        w: float32 array of shape (rows, cols) where cols % group_size == 0.
        group_size: group width along the last axis.
        eps: numerical floor on the scale.

    Returns:
        scale: float32 array of shape (rows, n_groups).
    """
    if w.ndim != 2:
        raise ValueError(f"w must be 2D, got shape {w.shape}")
    cols = w.shape[1]
    if cols % group_size != 0:
        raise ValueError(f"cols={cols} not divisible by group_size={group_size}")
    grouped = w.reshape(w.shape[0], cols // group_size, group_size)
    scale = np.abs(grouped).mean(axis=-1)
    return np.maximum(scale, eps)


def ternary_quantize(
    weight: np.ndarray,
    group_size: int = 128,
    threshold: float = 0.7,
) -> TernaryPlane:
    """Quantize a real weight matrix to ternary {-1, 0, +1} with per-group scale.

    Pipeline per group:
        1. s = mean(|w|)  (with eps floor)
        2. w_n = w / s
        3. t = round(clip(w_n, -1, 1))
        4. t = 0  where |w_n| < threshold  (sparsity knob; >=0.5 keeps >=50% nonzeros)

    The threshold controls how aggressively zeros are introduced; higher
    threshold -> more zeros -> more sparsity -> lower compute per call.

    Args:
        weight: float32 array of shape (out_features, in_features).
        group_size: group width for the scale.
        threshold: magnitude below which weights are forced to zero.

    Returns:
        TernaryPlane holding the codes + scales.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got shape {weight.shape}")
    w = weight.astype(np.float32, copy=False)
    if w.shape[1] % group_size != 0:
        raise ValueError(
            f"in_features={w.shape[1]} not divisible by group_size={group_size}"
        )

    scales = _grouped_absmean_scale(w, group_size)              # (rows, n_groups)
    w_normalized = w.reshape(w.shape[0], -1, group_size) / scales[..., None]
    w_normalized = w_normalized.reshape(w.shape[0], -1)

    codes = np.clip(np.round(w_normalized), -1.0, 1.0)
    codes = np.where(np.abs(w_normalized) < threshold, 0.0, codes)
    codes_int = codes.astype(np.int8)

    return TernaryPlane(codes=codes_int, scales=scales.astype(np.float32),
                        group_size=group_size)
