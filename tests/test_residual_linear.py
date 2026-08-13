"""Tests for the reference `ResidualTernaryLinear` layer."""
from __future__ import annotations

import numpy as np
import pytest

from torus.core import ResidualTernaryLinear
from torus.core.gate import GateMode, ResidualGate
from torus.quant import residual_quantize


def _layer(num_planes: int = 2, mode: GateMode = GateMode.ALWAYS) -> ResidualTernaryLinear:
    rng = np.random.default_rng(1)
    weight = (rng.standard_normal((32, 256)) * 0.05).astype(np.float32)
    planes = residual_quantize(weight, num_planes=num_planes, group_size=128)
    gate = ResidualGate(mode=mode)
    return ResidualTernaryLinear(planes=planes, gate=gate)


def test_forward_shape() -> None:
    layer = _layer()
    x = np.random.default_rng(2).standard_normal((4, 256)).astype(np.float32)
    y, decision = layer.forward(x)
    assert y.shape == (4, 32)
    assert decision is not None


def test_always_uses_all_planes() -> None:
    layer = _layer(num_planes=3, mode=GateMode.ALWAYS)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 256)).astype(np.float32)
    y, _ = layer.forward(x)
    assert y.shape == (2, 32)


def test_never_uses_only_primary_plane() -> None:
    layer = _layer(num_planes=3, mode=GateMode.NEVER)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 256)).astype(np.float32)
    y, decision = layer.forward(x)
    assert y.shape == (2, 32)
    assert not bool(decision.activate.any())


def test_adaptive_requires_magnitude() -> None:
    layer = _layer(mode=GateMode.ADAPTIVE)
    x = np.random.default_rng(5).standard_normal((1, 256)).astype(np.float32)
    with pytest.raises(ValueError):
        layer.forward(x, depth=0)


def test_adaptive_decision_does_not_break_shape() -> None:
    layer = _layer(num_planes=2, mode=GateMode.ADAPTIVE, )
    layer.gate = ResidualGate(mode=GateMode.ADAPTIVE, threshold=0.0)
    x = np.random.default_rng(6).standard_normal((1, 256)).astype(np.float32)
    y, _ = layer.forward(x, residual_relative_magnitude=0.5, depth=0.5)
    assert y.shape == (1, 32)
