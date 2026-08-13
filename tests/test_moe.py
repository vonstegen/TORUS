"""Tests for MoE scaffolding."""
from __future__ import annotations

import numpy as np
import pytest

from torus.core import ResidualTernaryLinear
from torus.core.gate import GateMode, ResidualGate
from torus.moe import ExpertBank, TopKRouter
from torus.quant import residual_quantize


def _make_expert(out_f: int = 16, in_f: int = 256, num_planes: int = 2) -> ResidualTernaryLinear:
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((out_f, in_f)) * 0.05).astype(np.float32)
    planes = residual_quantize(w, num_planes=num_planes, group_size=128)
    return ResidualTernaryLinear(planes=planes, gate=ResidualGate(mode=GateMode.ALWAYS))


def test_expert_bank_add_get() -> None:
    bank = ExpertBank()
    bank.add(0, _make_expert())
    bank.add(1, _make_expert())
    assert 0 in bank and 1 in bank
    assert len(bank) == 2
    assert bank.get(0).out_features == 16


def test_expert_bank_missing_raises() -> None:
    bank = ExpertBank()
    with pytest.raises(KeyError):
        bank.get(7)


def test_router_picks_top_k_per_token() -> None:
    router = TopKRouter(num_experts=8, top_k=2)
    features = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    result = router.route(features)
    assert result.indices.shape == (4, 2)
    assert result.weights.shape == (4, 2)
    # Weights should sum to 1 per row.
    sums = result.weights.sum(axis=-1)
    assert np.allclose(sums, np.ones(4), atol=1e-5)
    # All experts in valid range.
    assert int(result.indices.min()) >= 0
    assert int(result.indices.max()) < 8


def test_router_top_k_bounds() -> None:
    with pytest.raises(ValueError):
        TopKRouter(num_experts=4, top_k=0)
    with pytest.raises(ValueError):
        TopKRouter(num_experts=4, top_k=5)


def test_router_features_1d_required() -> None:
    router = TopKRouter(num_experts=3, top_k=1)
    with pytest.raises(ValueError):
        router.route(np.zeros((2, 2), dtype=np.float32))
