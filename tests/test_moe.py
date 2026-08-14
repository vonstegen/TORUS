"""Tests for MoE scaffolding."""
from __future__ import annotations

import numpy as np
import pytest

from torus.core import ResidualTernaryLinear
from torus.core.gate import GateMode, ResidualGate
from torus.moe import ExpertBank, TopKRouter
from torus.quant import residual_quantize


def _make_expert_stack(out_f: int = 16, in_f: int = 256, num_planes: int = 2, expert_id: int = 0):
    from torus.moe.expert_bank import ExpertResidualStack
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((out_f, in_f)) * 0.05).astype(np.float32)
    planes = residual_quantize(w, num_planes=num_planes, group_size=128)
    return ExpertResidualStack(expert_id=expert_id, residual_planes=planes)


def _make_expert_layer(out_f: int = 16, in_f: int = 256, num_planes: int = 2) -> ResidualTernaryLinear:
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((out_f, in_f)) * 0.05).astype(np.float32)
    planes = residual_quantize(w, num_planes=num_planes, group_size=128)
    return ResidualTernaryLinear(planes=planes, gate=ResidualGate(mode=GateMode.ALWAYS))


def test_expert_bank_add_get() -> None:
    bank = ExpertBank()
    bank.add(0, _make_expert_stack())
    bank.add(1, _make_expert_stack())
    assert 0 in bank and 1 in bank
    assert len(bank) == 2
    # Stack's residual plane shape:
    assert bank.get(0).residual_planes.shape == (16, 256)


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


def test_expert_residual_stack_primary_default() -> None:
    """Without shared_primary, primary defaults to residual_planes."""
    stack = _make_expert_stack()
    assert stack.primary is stack.residual_planes
    assert stack.num_residual_planes == 2


def test_expert_bank_shared_primary() -> None:
    """All experts added after `set_shared_primary` share the same primary."""
    from torus.moe.expert_bank import ExpertResidualStack
    bank = ExpertBank()
    primary = _make_expert_stack().residual_planes
    bank.set_shared_primary(primary)
    bank.add(0, ExpertResidualStack(expert_id=0, residual_planes=primary))
    bank.add_residual(1, primary)  # adds a stack with shared_primary auto-bound
    bank.add(2, ExpertResidualStack(expert_id=2, residual_planes=primary))
    assert bank.experts_sharing_primary() == [0, 1, 2]
    assert bank.get(0).shared_primary is primary


def test_expert_bank_dict_shaped() -> None:
    """`__getitem__`, `__contains__`, `__iter__`, `__len__` still work."""
    bank = ExpertBank()
    bank.add(0, _make_expert_stack())
    bank.add(5, _make_expert_stack(expert_id=5))
    assert len(bank) == 2
    assert 0 in bank and 5 in bank and 3 not in bank
    assert bank[5].expert_id == 5
    assert list(iter(bank)) == [0, 5]


def test_router_confidence_shape_and_range() -> None:
    """`confidence()` returns a [0,1] array of shape (batch,)."""
    router = TopKRouter(num_experts=4, top_k=2)
    rr = router.route(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    conf = router.confidence(rr)
    assert conf.shape == (3,)
    # All values in [0, 1] (top-k prob mass is bounded by 1).
    assert (conf >= 0).all() and (conf <= 1.0 + 1e-6).all()


def test_gate_accepts_router_confidence() -> None:
    """The gate's decide() takes an optional `router_confidence` arg."""
    gate = ResidualGate(mode=GateMode.ADAPTIVE, threshold=0.5)
    # Low confidence (router is unsure) -> push toward activating.
    low = gate.decide(
        residual_relative_magnitude=0.0,
        depth=0.0,
        router_confidence=0.1,
    )
    # High confidence (router is sure) -> push toward NOT activating.
    high = gate.decide(
        residual_relative_magnitude=0.0,
        depth=0.0,
        router_confidence=0.9,
    )
    assert float(low.score.mean()) > float(high.score.mean())


def test_telemetry_records_per_expert() -> None:
    """`GateTelemetry.record(..., expert_id=N)` records per-expert stats."""
    from torus.core import GateTelemetry, GateDecision
    from torus.core.kernels import OpCount

    tele = GateTelemetry()
    tele.begin_layer(0)
    decision = GateDecision(
        activate=np.array([True, False, True], dtype=bool),
        score=np.array([0.9, 0.1, 0.8], dtype=np.float32),
    )
    ops = (OpCount(adds=10, subs=5, skips=2, n_rows=8, n_cols=8),)
    tele.record(decision, ops, expert_id=7)
    tele.record(decision, ops, expert_id=7)
    tele.record(decision, ops, expert_id=8)
    summary = tele.summary()
    layers = summary["layers"]
    assert len(layers) == 1
    experts = layers[0]["experts"]
    by_eid = {e["expert_id"]: e for e in experts}
    assert by_eid[7]["n_calls"] == 2
    assert by_eid[8]["n_calls"] == 1
    # expert_summary aggregates across layers.
    agg = tele.expert_summary()
    assert {e["expert_id"]: e["n_calls"] for e in agg} == {7: 2, 8: 1}
