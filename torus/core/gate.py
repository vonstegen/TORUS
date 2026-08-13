"""Adaptive residual gate.

Decides, per call site (per token, per layer, or per expert), whether
the residual ternary plane should be activated. The gate is the
hardware-friendly signal that controls when the second ternary
datapath executes.

Design notes:

- The gate has three modes:
    * ALWAYS   -> always activate the residual plane (no quality/speed dial).
    * NEVER    -> pure primary plane (maximum efficiency).
    * ADAPTIVE -> decide per call using a lightweight scoring function.

- The scoring function takes a small feature vector describing the call
  (token entropy, layer depth, expert id, residual magnitude estimate,
  any time-varying signal) and returns a probability. A threshold turns
  the probability into a hard 0/1.

- Phase 1 ships a heuristic scoring function (magnitude of residual
  energy pre-scaled). Phase 3 will add learned gating trained jointly
  with the residual planes under capability-aware distillation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class GateMode(str, Enum):
    NEVER = "never"
    ALWAYS = "always"
    ADAPTIVE = "adaptive"


def _as_broadcastable(value) -> np.ndarray:
    """Convert a scalar or Python number to a 0-d float32 ndarray."""
    arr = np.asarray(value, dtype=np.float32)
    return arr


@dataclass(frozen=True)
class GateDecision:
    """Output of the gate for one batch element / call site."""
    activate: np.ndarray   # bool ndarray -- True means residual plane ON
    score: np.ndarray      # float ndarray -- raw probability / score


class ResidualGate:
    """Adaptive gate controlling residual plane activation.

    A call site is identified by a feature vector. The gate currently
    supports two feature dimensions:

        residual_relative_magnitude : estimated ||W - W_hat_primary|| / ||W||
        depth                        : int in [0, num_layers), normalized

    The scoring function combines these into a probability of activating
    the residual plane. A simple sigmoid blends both signals. A small
    bias lets the user tune the overall activation rate.
    """

    def __init__(
        self,
        mode: GateMode = GateMode.ADAPTIVE,
        threshold: float = 0.5,
        depth_bias: float = 0.0,
        magnitude_bias: float = 0.0,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        self.mode = mode
        self.threshold = threshold
        self.depth_bias = depth_bias
        self.magnitude_bias = magnitude_bias

    def decide(
        self,
        residual_relative_magnitude: np.ndarray | float,
        depth: np.ndarray | float | int,
    ) -> GateDecision:
        """Return a per-call-site decision to activate the residual plane.

        Args:
            residual_relative_magnitude: scalar or array; estimated
                ||W - W_hat_primary|| / ||W|| for the call site. Larger
                values indicate the primary plane is leaving more error.
            depth: scalar or array; normalized layer / call depth in [0, 1].

        Returns:
            GateDecision with .activate (bool) and .score (float).
        """
        if self.mode is GateMode.NEVER:
            score = _as_broadcastable(residual_relative_magnitude) * 0.0
            return GateDecision(activate=score.astype(bool), score=score)

        if self.mode is GateMode.ALWAYS:
            score = _as_broadcastable(residual_relative_magnitude) * 0.0 + 1.0
            return GateDecision(activate=score.astype(bool), score=score)

        # ADAPTIVE
        mag = _as_broadcastable(residual_relative_magnitude)
        d = _as_broadcastable(depth)
        # Both signals are bounded; sigmoid maps to [0,1] probability.
        logit = mag + d + self.magnitude_bias + self.depth_bias
        score = 1.0 / (1.0 + np.exp(-logit * 4.0))  # 4x amplifies sensitivity
        activate = score >= self.threshold
        return GateDecision(activate=activate.astype(bool), score=score)

    def activation_rate(self, activations: GateDecision) -> float:
        """Fraction of call sites that were activated (for telemetry)."""
        return float(np.mean(activations.activate))
