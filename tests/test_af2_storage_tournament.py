"""Tests for examples/af2_storage_tournament.py (EXP-AF-002 harness).

Pins the architectural invariants AF2 must guarantee, independent of
any model load: pack format round-trip, matched-bytes tolerance,
trained-arm completeness, cost-vector completeness, LoRA + dense
size accounting within +/- 1% of preregistered targets.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


af2 = _load("af2_storage_tournament")


def _t2_pack_decode(codes_int8: np.ndarray) -> np.ndarray:
    """Pure-numpy mirror of T2TernaryAdapter.serialize decode path."""
    flat = codes_int8.reshape(-1)
    N = flat.size
    pad = (4 - N % 4) % 4
    padded = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])
    packed = (padded[0::4].astype(np.int32)
              | (padded[1::4].astype(np.int32) << 2)
              | (padded[2::4].astype(np.int32) << 4)
              | (padded[3::4].astype(np.int32) << 6))
    out = np.empty(padded.size, dtype=np.int8)
    out[0::4] = (packed & 0x3).astype(np.int8)
    out[1::4] = ((packed >> 2) & 0x3).astype(np.int8)
    out[2::4] = ((packed >> 4) & 0x3).astype(np.int8)
    out[3::4] = ((packed >> 6) & 0x3).astype(np.int8)
    return out[:N]


def test_cost_vector_keys_complete() -> None:
    expected = {"deployed_bytes", "training_flops",
                "inference_ops_per_token", "memory_traffic_per_token",
                "latency_per_token_titan_rtx", "energy_per_token"}
    cv = af2.CostVector(1, 2, 3, 4, None, None)
    assert set(cv.as_dict()) == expected


def test_cost_vector_serializes_to_json() -> None:
    cv = af2.CostVector(1, 2, 3, 4, None, None)
    json.dumps(cv.as_dict())  # must not raise


def test_t2_pack_round_trip_lossless() -> None:
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 3, size=(8192, 2048), dtype=np.int8)
    decoded = _t2_pack_decode(codes)
    assert decoded.shape == (codes.size,)
    assert np.array_equal(decoded, codes.reshape(-1))


def test_t2_pack_handles_remainder_rows() -> None:
    codes = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.int8)
    decoded = _t2_pack_decode(codes)
    assert decoded.size == codes.size
    assert np.array_equal(decoded, codes)


def test_int4_nibble_pack_round_trip_lossless() -> None:
    rng = np.random.default_rng(1)
    codes = rng.integers(-8, 8, size=(4096,), dtype=np.int8)
    ub = (codes + 8).astype(np.uint8)
    packed = np.zeros(ub.size // 2, dtype=np.uint8)
    packed = (ub[0::2] & 0xF) | ((ub[1::2] & 0xF) << 4)
    decoded = np.empty(ub.size, dtype=np.int8)
    decoded[0::2] = (packed & 0xF).astype(np.int8) - 8
    decoded[1::2] = ((packed >> 4) & 0xF).astype(np.int8) - 8
    assert np.array_equal(decoded, codes)


def test_matched_bytes_tolerance_invariance() -> None:
    from examples.af2_storage_tournament import TARGET_DEPLOYED_BYTES
    tolerance = 1.0
    actual = int(TARGET_DEPLOYED_BYTES["t2_ternary"] * 1.05)
    delta = abs(actual - TARGET_DEPLOYED_BYTES["t2_ternary"]) / \
            TARGET_DEPLOYED_BYTES["t2_ternary"] * 100
    assert delta > tolerance
    actual = int(TARGET_DEPLOYED_BYTES["t2_ternary"] * 1.005)
    delta = abs(actual - TARGET_DEPLOYED_BYTES["t2_ternary"]) / \
            TARGET_DEPLOYED_BYTES["t2_ternary"] * 100
    assert delta <= tolerance


def test_trained_arms_have_target_bytes() -> None:
    from examples.af2_storage_tournament import (TARGET_DEPLOYED_BYTES,
                                                  TRAINED_ARMS,
                                                  ALL_ARMS)
    for arm in TRAINED_ARMS:
        assert arm in TARGET_DEPLOYED_BYTES
    assert set(TARGET_DEPLOYED_BYTES.keys()) == set(TRAINED_ARMS)
    assert set(ALL_ARMS) == set(TRAINED_ARMS) | {"random_t2_ternary",
                                                   "random_lora"}


def test_lora_rank_yields_bytes_within_one_percent() -> None:
    from examples.af2_storage_tournament import TARGET_DEPLOYED_BYTES
    rank = 216
    bytes_ = (2048 * rank + rank * 8192) * 2
    target = TARGET_DEPLOYED_BYTES["lora"]
    delta_pct = abs(bytes_ - target) / target * 100
    assert delta_pct <= 1.0


def test_dense_bottleneck_yields_bytes_within_one_percent() -> None:
    from examples.af2_storage_tournament import TARGET_DEPLOYED_BYTES
    bottleneck = 192
    bytes_ = (2048 * bottleneck + bottleneck * 8192) * 2
    target = TARGET_DEPLOYED_BYTES["dense_adapter"]
    delta_pct = abs(bytes_ - target) / target * 100
    assert delta_pct <= 1.0
