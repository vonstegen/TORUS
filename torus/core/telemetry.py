"""Gate telemetry.

Records gate activation rates, op counts per layer / per plane, and
trends over time. The runtime uses this to drive the memory-tier
policy and to flag layers where the gate misfires (always-on or
always-off).

Phase 4: per-expert stats are recorded alongside per-layer stats so
the MoE-aware residual gate can be evaluated end-to-end.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable

from torus.core.gate import GateDecision
from torus.core.kernels import OpCount


@dataclass
class ExpertStats:
    """Per-expert accumulated statistics within a layer."""
    activations: int = 0
    total: int = 0
    decisions: deque[bool] = field(default_factory=lambda: deque(maxlen=512))

    def activation_rate(self) -> float:
        return self.activations / self.total if self.total else 0.0


@dataclass
class LayerStats:
    """Per-layer accumulated statistics."""
    activations: int = 0
    total: int = 0
    plane_ops: list[OpCount] = field(default_factory=list)
    decisions: deque[bool] = field(default_factory=lambda: deque(maxlen=512))
    experts: dict[int, ExpertStats] = field(default_factory=lambda: defaultdict(ExpertStats))

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
    """Accumulates per-layer and per-expert gate / kernel stats."""
    _layers: dict[int, LayerStats] = field(default_factory=lambda: defaultdict(LayerStats))
    _current_layer: int = -1

    def begin_layer(self, layer_id: int) -> None:
        self._current_layer = layer_id

    def record(
        self,
        decision: GateDecision,
        plane_ops: Iterable[OpCount],
        expert_id: int | None = None,
    ) -> None:
        """Record one call.

        Args:
            decision: gate decision for the layer.
            plane_ops: an op-count per plane (1 or 2 entries, depending
                on whether the residual plane was activated).
            expert_id: optional MoE expert id; when set, the call is
                also recorded under that expert's per-layer stats.
        """
        layer = self._layers[self._current_layer]
        active = bool(decision.activate.any())
        layer.total += 1
        if active:
            layer.activations += 1
        layer.decisions.append(active)
        layer.plane_ops.extend(plane_ops)

        if expert_id is not None:
            exp = layer.experts[expert_id]
            exp.total += 1
            if active:
                exp.activations += 1
            exp.decisions.append(active)

    def layer_summary(self) -> list[dict]:
        out = []
        for lid, stats in sorted(self._layers.items()):
            entry = {
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
                "experts": [
                    {
                        "expert_id": eid,
                        "activation_rate": es.activation_rate(),
                        "n_calls": es.total,
                    }
                    for eid, es in sorted(stats.experts.items())
                ],
            }
            out.append(entry)
        return out

    def summary(self) -> dict:
        layers = self.layer_summary()
        if not layers:
            return {
                "average_activation": 0.0,
                "layers": [],
            }
        avg = sum(l["activation_rate"] for l in layers) / len(layers)
        return {
            "average_activation": avg,
            "layers": layers,
        }

    def top_layers_by_activation(self, k: int = 5) -> list[dict]:
        layers = self.layer_summary()
        return sorted(layers, key=lambda d: -d["activation_rate"])[:k]

    def flagged_layers(self, lo: float = 0.05, hi: float = 0.95) -> list[dict]:
        """Layers whose activation rate is outside [lo, hi] are flagged."""
        return [
            l for l in self.layer_summary()
            if l["activation_rate"] < lo or l["activation_rate"] > hi
        ]

    def expert_summary(self) -> list[dict]:
        """Per-expert aggregation across all layers."""
        agg: dict[int, dict] = defaultdict(lambda: {"activations": 0, "total": 0})
        for stats in self._layers.values():
            for eid, es in stats.experts.items():
                agg[eid]["activations"] += es.activations
                agg[eid]["total"] += es.total
        return [
            {
                "expert_id": eid,
                "activation_rate": (
                    v["activations"] / v["total"] if v["total"] else 0.0
                ),
                "n_calls": v["total"],
            }
            for eid, v in sorted(agg.items())
        ]