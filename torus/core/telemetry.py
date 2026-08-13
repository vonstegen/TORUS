"""Gate telemetry.

Records gate activation rates, op counts per layer / per plane, and
trends over time. The runtime uses this to drive the memory-tier
policy and to flag layers where the gate misfires (always-on or
always-off).
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Iterable

from torus.core.gate import GateDecision
from torus.core.kernels import OpCount


@dataclass
class LayerStats:
    """Per-layer accumulated statistics."""
    activations: int = 0
    total: int = 0
    plane_ops: list[OpCount] = field(default_factory=list)
    decisions: deque[bool] = field(default_factory=lambda: deque(maxlen=512))

    def activation_rate(self) -> float:
        return self.activations / self.total if self.total else 0.0

    def trend(self) -> float:
        """Latest-vs-oldest slope of the activation rate, in [-1, 1]."""
        if len(self.decisions) < 2:
            return 0.0
        first_half = list(self.decisions)[: len(self.decisions) // 2]
        second_half = list(self.decisions)[len(self.decisions) // 2 :]
        return float(mean(second_half) - mean(first_half))


@dataclass
class GateTelemetry:
    """Accumulates per-layer gate / kernel stats for analysis."""
    _layers: dict[int, LayerStats] = field(default_factory=lambda: defaultdict(LayerStats))
    _current_layer: int = -1

    def begin_layer(self, layer_id: int) -> None:
        self._current_layer = layer_id

    def record(
        self,
        decision: GateDecision,
        plane_ops: Iterable[OpCount],
    ) -> None:
        """Record one call.

        Args:
            decision: gate decision for the layer.
            plane_ops: an op-count per plane (1 or 2 entries, depending
                on whether the residual plane was activated).
        """
        layer = self._layers[self._current_layer]
        active = bool(decision.activate.any())
        layer.total += 1
        if active:
            layer.activations += 1
        layer.decisions.append(active)
        layer.plane_ops.extend(plane_ops)

    def layer_summary(self) -> list[dict]:
        return [
            {
                "layer_id": lid,
                "activation_rate": stats.activation_rate(),
                "trend": stats.trend(),
                "n_calls": stats.total,
                "total_adds": sum(op.adds for op in stats.plane_ops),
                "total_subs": sum(op.subs for op in stats.plane_ops),
                "total_skips": sum(op.skips for op in stats.plane_ops),
                "density": (
                    sum(op.nonzero for op in stats.plane_ops)
                    / sum(op.total for op in stats.plane_ops)
                    if any(op.total for op in stats.plane_ops)
                    else 0.0
                ),
            }
            for lid, stats in sorted(self._layers.items())
        ]

    def summary(self) -> dict:
        layers = self.layer_summary()
        if not layers:
            return {"layers": [], "average_activation": 0.0}
        return {
            "layers": layers,
            "average_activation": mean(layer["activation_rate"] for layer in layers),
            "stdev_activation": pstdev([l["activation_rate"] for l in layers])
            if len(layers) > 1
            else 0.0,
        }

    def top_layers_by_activation(self, k: int = 5) -> list[dict]:
        layers = self.layer_summary()
        return sorted(layers, key=lambda d: -d["activation_rate"])[:k]

    def flagged_layers(self, lo: float = 0.05, hi: float = 0.95) -> list[dict]:
        """Layers whose activation rate is outside [lo, hi] are flagged.

        Such layers indicate the gate is either always-off (suspect that
        the primary plane is failing to capture the layer's computation)
        or always-on (residual plane may be carrying essential
        information that the primary plane cannot replicate).
        """
        return [
            layer for layer in self.layer_summary()
            if layer["activation_rate"] <= lo or layer["activation_rate"] >= hi
        ]
