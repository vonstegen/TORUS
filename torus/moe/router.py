"""Top-K router for MoE-style expert selection.

Phase 1 ships a deterministic, easy-to-inspect router. It produces a
routing pattern that the residual gate can see: for "hard" tokens the
router can be biased toward activating more experts AND activating their
residual planes.

This file deliberately avoids any gradient / training assumptions: Phase 3
adds learned routing under capability-aware distillation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RouteResult:
    """Output of `TopKRouter.route`."""
    indices: np.ndarray   # int array of shape (batch, top_k)
    weights: np.ndarray   # float array of shape (batch, top_k) normalized to sum to 1
    raw_mass: np.ndarray  # float array of shape (batch,): top-k prob mass *before*
                          # renormalization. The Phase-7 gate uses this to decide
                          # how aggressively to engage residual planes.


class TopKRouter:
    """Route each token to the top-k experts by router score."""

    def __init__(self, num_experts: int, top_k: int = 2) -> None:
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if top_k < 1 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1,{num_experts}], got {top_k}")
        self.num_experts = num_experts
        self.top_k = top_k
        # Deterministic init: scaled orthonormal-ish columns. Real routers
        # learn this; Phase-3 work will replace this with a learned head.
        rng = np.random.default_rng(0)
        self._w = rng.standard_normal((num_experts,)).astype(np.float32) / np.sqrt(num_experts)

    def route(self, token_features: np.ndarray) -> RouteResult:
        """Pick the top-k experts for each token.

        Args:
            token_features: float32 array of shape (batch,). A scalar
                per token is enough for the Phase-1 heuristic; later
                phases accept richer features.

        Returns:
            RouteResult with (indices, weights) of shape (batch, top_k).
        """
        if token_features.ndim != 1:
            raise ValueError(f"token_features must be 1D, got {token_features.shape}")
        scores = np.outer(token_features.astype(np.float32), self._w)  # (batch, num_experts)
        # Softmax along the expert axis for stable weights.
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        probs = e / e.sum(axis=-1, keepdims=True)
        # Top-k indices per row.
        idx = np.argpartition(-probs, kth=self.top_k - 1, axis=-1)[:, :self.top_k]
        # Compute selected weights then renormalize them to sum to 1.
        rows = np.arange(token_features.shape[0])[:, None]
        sel = probs[rows, idx]
        # raw_mass is the pre-renormalization top-k prob mass.
        # It's the right input for the Phase-7 gate: it reflects how
        # much prob mass landed in the top-k vs the long tail.
        raw_mass = sel.sum(axis=-1).astype(np.float32)
        sel = sel / sel.sum(axis=-1, keepdims=True).clip(min=1e-8)
        return RouteResult(
            indices=idx.astype(np.int64),
            weights=sel.astype(np.float32),
            raw_mass=raw_mass,
        )

    def confidence(self, route: RouteResult) -> np.ndarray:
        """Per-token confidence = top-k prob mass.

        Returns float32 array of shape (batch,). Values near 1.0 mean
        the top-k experts cover essentially all the router's prob mass
        — easy token. Values near 0.0 mean a long tail of low-prob
        experts is competing — hard token.

        Gate policies use this signal to decide whether to engage a
        residual plane.
        """
        return route.raw_mass