"""Tests for the adaptive residual gate."""
from __future__ import annotations

import numpy as np
import pytest

from torus.core.gate import GateMode, ResidualGate


def _scalar(x) -> float:
    """Coerce a 0-d or 1-element ndarray to a Python float."""
    return float(np.asarray(x).reshape(-1)[0])


def test_always_mode_activates() -> None:
    g = ResidualGate(mode=GateMode.ALWAYS)
    decision = g.decide(residual_relative_magnitude=0.3, depth=0.5)
    assert bool(decision.activate.all())
    assert pytest.approx(_scalar(decision.score), abs=1e-6) == 1.0


def test_never_mode_does_not_activate() -> None:
    g = ResidualGate(mode=GateMode.NEVER)
    decision = g.decide(residual_relative_magnitude=0.99, depth=0.99)
    assert not bool(decision.activate.any())
    assert pytest.approx(_scalar(decision.score), abs=1e-6) == 0.0


def test_adaptive_high_magnitude_activates() -> None:
    g = ResidualGate(mode=GateMode.ADAPTIVE, threshold=0.5)
    decision = g.decide(residual_relative_magnitude=2.0, depth=0.5)
    assert bool(decision.activate.all())


def test_adaptive_low_magnitude_deactivates() -> None:
    g = ResidualGate(mode=GateMode.ADAPTIVE, threshold=0.5)
    decision = g.decide(residual_relative_magnitude=-2.0, depth=-2.0)
    # Very negative logits drive sigmoid -> ~0; below threshold -> False.
    assert not bool(decision.activate.any())
    assert _scalar(decision.score) < 0.5


def test_adaptive_threshold_changes_rate() -> None:
    low = ResidualGate(mode=GateMode.ADAPTIVE, threshold=0.1)
    high = ResidualGate(mode=GateMode.ADAPTIVE, threshold=0.9)
    rng = np.random.default_rng(0)
    mag = rng.uniform(-1, 1, size=128).astype(np.float32)
    depth = rng.uniform(-1, 1, size=128).astype(np.float32)
    r_low = low.activation_rate(low.decide(mag, depth))
    r_high = high.activation_rate(high.decide(mag, depth))
    assert r_low > r_high


def test_adaptive_vectorized_input() -> None:
    g = ResidualGate(mode=GateMode.ADAPTIVE, threshold=0.55)
    mag = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    depth = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    decision = g.decide(mag, depth)
    assert decision.activate.shape == (3,)
    assert not bool(decision.activate[0])
    assert bool(decision.activate[2])


def test_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        ResidualGate(threshold=-0.1)
    with pytest.raises(ValueError):
        ResidualGate(threshold=1.5)
