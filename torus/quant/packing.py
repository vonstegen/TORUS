"""Packed 2-bit weight layout for ternary planes.

The reference `TernaryPlane` exposes codes as a full int8 array. A real
kernel on CPU (AVX-512) or GPU (CUDA) needs four codes packed per byte
so one 64-bit load brings in sixteen {-1, 0, +1} weights. This module
defines that layout *exactly* so Phase 3 hardware work can consume the
packed bytes directly.

Canonical 2-bit encoding (matches `bitnet.cpp`-style convention):

    code value 0  -> 0b00      (actual zero weight)
    code value +1 -> 0b01
    code value -1 -> 0b10
    0b11          -> reserved (decodes as 0)

Within each packed byte, the four codes occupy bit pairs 0, 2, 4, 6 in
little-endian order. Byte i contains codes at positions 4i, 4i+1, 4i+2,
4i+3 of the encoded (padded) array.

Inputs whose in_features is not a multiple of 4 are right-padded with
zeros; the original (unpadded) shape is preserved on `unpack()`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from torus.quant.ternary import TernaryPlane

_CODE_ZERO = 0b00
_CODE_PLUS = 0b01
_CODE_MINUS = 0b10
_CODE_PAD = 0b11  # reserved; decoder maps back to 0

_BIT_MASK = 0x3
_SLOT_SHIFTS = (0, 2, 4, 6)


def _encode_codes(codes: np.ndarray) -> np.ndarray:
    """Map int8 {-1, 0, +1} to uint8 2-bit values in {0, 1, 2, 3}."""
    encoded = np.zeros(codes.shape, dtype=np.uint8)
    encoded[codes == 1] = _CODE_PLUS
    encoded[codes == -1] = _CODE_MINUS
    return encoded


def _decode_byte(byte_u8: np.ndarray) -> np.ndarray:
    """Unpack 2-bit codes from a uint8 byte array to int8 {-1, 0, +1}.

    Inverse of the encoding step. Returns an array whose last axis is
    `byte_u8.shape[-1] * 4`. Layout within each byte is little-endian
    2-bit pairs at shifts 0, 2, 4, 6.
    """
    flat = byte_u8.reshape(-1)
    n_bytes = flat.shape[0]
    out = np.zeros(n_bytes * 4, dtype=np.int8)
    for slot, shift in enumerate(_SLOT_SHIFTS):
        code = ((flat >> shift) & _BIT_MASK).astype(np.int8)
        signed = np.where(
            code == _CODE_PLUS, np.int8(1),
            np.where(code == _CODE_MINUS, np.int8(-1), np.int8(0)),
        )
        out[slot::4] = signed
    return out.reshape(byte_u8.shape[:-1] + (byte_u8.shape[-1] * 4,))


def pack_codes(codes: np.ndarray) -> np.ndarray:
    """Pack int8 codes into uint8 bytes (4 codes per byte).

    Inputs whose last axis is not a multiple of 4 are right-padded with
    zeros. Returned shape is `codes.shape[:-1] + (ceil(last/4),)`.
    """
    if codes.shape[-1] % 4 != 0:
        pad = (-codes.shape[-1]) % 4
        pad_block = np.zeros(codes.shape[:-1] + (pad,), dtype=codes.dtype)
        padded = np.concatenate([codes, pad_block], axis=-1)
    else:
        padded = codes
    encoded = _encode_codes(padded)
    slot_len = (encoded.shape[-1] + 3) // 4
    packed = np.zeros(encoded.shape[:-1] + (slot_len,), dtype=np.uint8)
    for slot, shift in enumerate(_SLOT_SHIFTS):
        sliced = encoded[..., slot::4]
        packed[..., :sliced.shape[-1]] |= sliced.astype(np.uint8) << shift
    return packed


def unpack_codes(packed: np.ndarray, original_last: int) -> np.ndarray:
    """Inverse of `pack_codes`; truncates the last axis to `original_last`."""
    decoded = _decode_byte(packed)
    if decoded.shape[-1] > original_last:
        decoded = decoded[..., :original_last]
    return decoded


@dataclass(frozen=True)
class PackedTernaryPlane:
    """A ternary plane with weights packed 4-per-byte.

    Storage per plane:
        weight bytes = ceil(num_weights / 4)
        scales       = 16 bits * num_groups
    """
    packed_codes: np.ndarray   # uint8, shape (out, in_packed)
    scales: np.ndarray         # float32, same layout as TernaryPlane.scales
    group_size: int
    out_features: int
    in_features: int

    @property
    def num_weights(self) -> int:
        return self.out_features * self.in_features

    @property
    def bits_per_weight(self) -> float:
        scale_bits = 16 * (self.in_features // self.group_size) * self.out_features
        return (2 * self.num_weights + scale_bits) / self.num_weights

    def unpack(self) -> TernaryPlane:
        """Recover the original `TernaryPlane` representation."""
        decoded = unpack_codes(self.packed_codes, original_last=self.in_features)
        return TernaryPlane(
            codes=decoded,
            scales=self.scales,
            group_size=self.group_size,
        )

    @classmethod
    def from_plane(cls, plane: TernaryPlane) -> "PackedTernaryPlane":
        """Pack an existing `TernaryPlane` into the packed layout."""
        if plane.codes.dtype != np.int8:
            raise TypeError(f"expected int8 codes, got {plane.codes.dtype}")
        out_f, in_f = plane.codes.shape
        packed = pack_codes(plane.codes)
        return cls(
            packed_codes=packed,
            scales=plane.scales,
            group_size=plane.group_size,
            out_features=out_f,
            in_features=in_f,
        )


def pack_plane(plane: TernaryPlane) -> PackedTernaryPlane:
    """Convenience wrapper around `PackedTernaryPlane.from_plane`."""
    return PackedTernaryPlane.from_plane(plane)
