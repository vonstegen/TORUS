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
  energy pre-scaled). Phase 3 adds learned gating. Phase 4 adds a
  *router-confidence* signal: when the MoE router is unsure which
  experts to pick, the gate is biased *toward* engaging the residual
  plane because the residual plane captures exactly the kind of
  per-expert nuance the router is wavering over.
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
    return np.asarray(value, dtype=np.float32)


@dataclass(frozen=True)
class GateDecision:
    """Output of the gate for one batch element / call site."""
    activate: np.ndarray   # bool ndarray -- True means residual plane ON
    score: np.ndarray      # float ndarray -- raw probability / score


class ResidualGate:
    """Adaptive gate controlling residual plane activation.

    Three feature dimensions combine into a probability of activating
    the residual plane:

        residual_relative_magnitude : estimated ||W - W_hat_primary|| / ||W||
        depth                        : int in [0, num_layers), normalized
        router_confidence            : top-k prob mass in [0, 1] from the
                                      MoE router; LOW confidence => engage
                                      the residual plane. Phase 4 addition.
    """

    def __init__(
        self,
        mode: GateMode = GateMode.ADAPTIVE,
        threshold: float = 0.5,
        depth_bias: float = 0.0,
        magnitude_bias: float = 0.0,
        confidence_bias: float = 0.0,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}")
        self.mode = mode
        self.threshold = threshold
        self.depth_bias = depth_bias
        self.magnitude_bias = magnitude_bias
        # `confidence_bias` shifts the contribution of router_confidence
        # uniformly. Positive bias makes the gate more sensitive to
        # router uncertainty; negative bias dampens that signal.
        self.confidence_bias = confidence_bias

    def decide(
        self,
        residual_relative_magnitude: np.ndarray | float,
        depth: np.ndarray | float | int,
        router_confidence: np.ndarray | float | None = None,
    ) -> GateDecision:
        """Return a per-call-site decision to activate the residual plane.

        Args:
            residual_relative_magnitude: scalar or array; estimated
                ||W - W_hat_primary|| / ||W|| for the call site.
            depth: scalar or array; normalized layer / call depth in [0, 1].
            router_confidence: scalar or array; top-k prob mass from the
                MoE router in [0, 1]. Optional; ignored when None.
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
        if router_confidence is None:
            conf_term = np.zeros_like(mag)
        else:
            conf = _as_broadcastable(router_confidence)
            # LOW confidence should push the gate TOWARD activating the
            # residual plane. We contribute +(1 - conf) to the logit so
            # unsure tokens are more likely to engage.
            conf_term = (1.0 - conf) + self.confidence_bias
        logit = mag + d + self.magnitude_bias + self.depth_bias + conf_term
        score = 1.0 / (1.0 + np.exp(-logit * 4.0))  # 4x amplifies sensitivity
        activate = score >= self.threshold
        return GateDecision(activate=activate.astype(bool), score=score)

    def activation_rate(self, activations: GateDecision) -> float:
        """Fraction of call sites that were activated (for telemetry)."""
        return float(np.mean(activations.activate))