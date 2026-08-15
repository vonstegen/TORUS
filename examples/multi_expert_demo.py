"""Phase 7 demo: MultiExpertRouter drives a 16-expert bank with
4-plane residual stacks per expert, using the router-confidence
signal to decide plane-count engagement.

The demo reports, for a 32-token batch:
  - Total plane activations (sum of `n_planes` across decisions).
  - The fraction of decisions that engaged each plane-count.
  - A short decision-table sample for inspection.

A "naive" baseline that always engages 4 planes is computed for
comparison.

Run with:

    python examples/multi_expert_demo.py
    python examples/multi_expert_demo.py --n-experts 64 --n-tokens 64
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from torus.moe import (
    ExpertBank,
    GatePolicy,
    MultiExpertRouter,
    TopKRouter,
)
from torus.quant import residual_quantize


def build_bank(n_experts: int, in_f: int = 32, out_f: int = 32, num_planes: int = 4) -> ExpertBank:
    bank = ExpertBank()
    rng = np.random.default_rng(0)
    for eid in range(n_experts):
        w = (rng.standard_normal((out_f, in_f)) * 0.05).astype(np.float32)
        planes = residual_quantize(w, num_planes=num_planes, group_size=in_f)
        bank.add_residual(eid, planes)
    return bank


def main(n_experts: int = 16, n_tokens: int = 32, top_k: int = 2) -> None:
    bank = build_bank(n_experts)
    router = TopKRouter(num_experts=n_experts, top_k=top_k)
    # Thresholds calibrated against TopKRouter(num_experts=16, top_k=2)
    # where raw_mass lands in ~[0.13, 0.24] (random router weights
    # spread prob mass across all 16 experts). With a *learned*
    # router these would span [0, 1]; until then, calibrate.
    policy = GatePolicy(
        confidence_low=0.13,
        confidence_high=0.24,
        n_planes_low=1,
        n_planes_high=4,
    )
    me = MultiExpertRouter(router, bank, policy=policy)

    # Mix of low / mid / high "features" so the policy has range to act on.
    features = np.concatenate([
        np.full(n_tokens // 3, 0.1, dtype=np.float32),    # low  -> 4 planes
        np.full(n_tokens // 3, 0.7, dtype=np.float32),    # mid  -> ~2 planes
        np.full(n_tokens - 2 * (n_tokens // 3), 1.0, dtype=np.float32),  # high -> 1 plane
    ])
    rng = np.random.default_rng(42)
    rng.shuffle(features)

    result = me.route(features)

    total_engaged = sum(d.n_planes for d in result.decisions)
    naive_total = len(result.decisions) * 4
    savings = naive_total - total_engaged
    pct = 100.0 * savings / naive_total if naive_total else 0.0

    print(f"experts: {n_experts}, top_k: {top_k}, num_planes per expert: 4")
    print(f"tokens routed: {features.shape[0]}")
    print(f"decisions: {len(result.decisions)}")
    print()
    print(f"  total plane activations (adaptive): {total_engaged}")
    print(f"  total plane activations (naive 4):  {naive_total}")
    print(f"  savings: {savings} ({pct:.1f}%)")
    print()

    counts: dict[int, int] = {}
    for d in result.decisions:
        counts[d.n_planes] = counts.get(d.n_planes, 0) + 1
    print("  n_planes distribution:")
    for k in sorted(counts):
        bar = "#" * counts[k]
        print(f"    {k} planes: {counts[k]:>4} {bar}")
    print()

    print("  sample decisions (first 10):")
    for d in result.decisions[:10]:
        print(
            f"    token {d.token_idx:>2} -> expert {d.expert_id:>3} "
            f"weight={d.weight:.3f} conf={d.confidence:.3f} "
            f"planes={d.n_planes}"
        )


def parse_args(argv: list[str]) -> tuple[int, int]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-experts", type=int, default=16)
    p.add_argument("--n-tokens", type=int, default=32)
    a = p.parse_args(argv)
    return a.n_experts, a.n_tokens


if __name__ == "__main__":
    n_experts, n_tokens = parse_args(sys.argv[1:])
    main(n_experts=n_experts, n_tokens=n_tokens)