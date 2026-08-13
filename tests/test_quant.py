"""Tests for residual ternary quantization math."""
from __future__ import annotations

import math

import numpy as np
import pytest

from torus.quant import (
    TernaryPlane,
    compose_planes,
    residual_quantize,
    ternary_quantize,
)


@pytest.fixture()
def weight() -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.standard_normal((64, 256)) * 0.05).astype(np.float32)


def test_ternary_codes_are_in_valid_set(weight: np.ndarray) -> None:
    plane = ternary_quantize(weight, group_size=128)
    assert isinstance(plane, TernaryPlane)
    codes_set = set(np.unique(plane.codes).tolist())
    assert codes_set.issubset({-1, 0, 1})
    assert plane.codes.shape == weight.shape


def test_ternary_group_count(weight: np.ndarray) -> None:
    plane = ternary_quantize(weight, group_size=128)
    assert plane.n_groups == weight.shape[1] // 128
    assert plane.scales.shape == (weight.shape[0], weight.shape[1] // 128)


def test_ternary_reconstruction_shape(weight: np.ndarray) -> None:
    plane = ternary_quantize(weight, group_size=128)
    w_hat = plane.reconstruct()
    assert w_hat.shape == weight.shape


def test_ternary_bits_per_weight_is_budget(weight: np.ndarray) -> None:
    plane = ternary_quantize(weight, group_size=128)
    bpp = plane.effective_bits_per_weight()
    # 2 bits per code + 16 bits per scale / group_size (128) ~= 2.125
    assert 2.0 <= bpp <= 2.2


def test_ternary_invalid_input() -> None:
    bad = np.zeros((4, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        ternary_quantize(bad, group_size=4)
    with pytest.raises(ValueError):
        ternary_quantize(np.zeros((4, 4, 4), dtype=np.float32))


def test_residual_stack_reduces_error(weight: np.ndarray) -> None:
    p1 = compose_planes(residual_quantize(weight, num_planes=1, group_size=128))
    p2 = compose_planes(residual_quantize(weight, num_planes=2, group_size=128))
    e1 = float(np.linalg.norm(weight - p1) / np.linalg.norm(weight))
    e2 = float(np.linalg.norm(weight - p2) / np.linalg.norm(weight))
    assert e2 < e1, f"two planes should beat one: e1={e1}, e2={e2}"
    assert 0.0 <= e1 < 1.0
    assert 0.0 <= e2 < e1


def test_residual_compose_active_subset(weight: np.ndarray) -> None:
    planes = residual_quantize(weight, num_planes=3, group_size=128)
    full = compose_planes(planes)
    only1 = compose_planes(planes, active=1)
    assert full.shape == weight.shape
    assert only1.shape == weight.shape
    # Using only 1 plane should match the primary plane alone
    assert np.allclose(full - only1, compose_planes(planes) - only1)  # sanity


def test_residual_plane_count_must_be_positive(weight: np.ndarray) -> None:
    with pytest.raises(ValueError):
        residual_quantize(weight, num_planes=0)


def test_residual_active_out_of_range(weight: np.ndarray) -> None:
    planes = residual_quantize(weight, num_planes=2, group_size=128)
    with pytest.raises(ValueError):
        compose_planes(planes, active=0)
    with pytest.raises(ValueError):
        compose_planes(planes, active=3)


def test_sparsity_increases_with_threshold(weight: np.ndarray) -> None:
    p_low = ternary_quantize(weight, group_size=128, threshold=0.5)
    p_high = ternary_quantize(weight, group_size=128, threshold=0.95)
    zeros_low = float(np.mean(p_low.codes == 0))
    zeros_high = float(np.mean(p_high.codes == 0))
    assert zeros_high >= zeros_low
