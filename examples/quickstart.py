"""End-to-end smoke example for the TORUS Phase-1 reference impls.

Run with:

    python examples/quickstart.py

It demonstrates:
- single-plane ternary quantization,
- multi-plane residual ternary,
- the adaptive gate,
- a tiny MoE dispatch,
- the recursive context primitive + REPL.
"""

from __future__ import annotations

import numpy as np

from torus.core import GateMode, ResidualGate, ResidualTernaryLinear
from torus.moe import ExpertBank, TopKRouter
from torus.quant import compose_planes, residual_quantize
from torus.rlm import ContextREPL, ContextSlice, RecursiveContext


def _print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    rng = np.random.default_rng(0)

    _print_section("1. Single-plane vs residual ternary quantization")

    weight = (rng.standard_normal((128, 256)) * 0.05).astype(np.float32)
    one_plane = residual_quantize(weight, num_planes=1)
    two_planes = residual_quantize(weight, num_planes=2)
    three_planes = residual_quantize(weight, num_planes=3)

    for label, planes in [("1 plane", one_plane),
                          ("2 planes", two_planes),
                          ("3 planes", three_planes)]:
        w_hat = compose_planes(planes)
        rel = float(np.linalg.norm(weight - w_hat) / np.linalg.norm(weight))
        print(f"  {label:>10}: relative reconstruction error = {rel:.4f}")

    _print_section("2. Adaptive residual gate")

    gate_adaptive = ResidualGate(mode=GateMode.ADAPTIVE, threshold=0.5)
    gate_never = ResidualGate(mode=GateMode.NEVER)
    for name, gate in [("NEVER", gate_never), ("ADAPTIVE", gate_adaptive)]:
        decision = gate.decide(
            residual_relative_magnitude=0.6,
            depth=0.5,
        )
        rate = gate.activation_rate(decision)
        print(f"  {name:>8}: activation rate = {rate:.2f}, sample score = {float(decision.score.reshape(-1)[0]):.2f}")

    _print_section("3. Drop-in reference layer")

    planes = residual_quantize(weight, num_planes=2)
    layer = ResidualTernaryLinear(planes=planes, gate=ResidualGate(mode=GateMode.ALWAYS))
    x = rng.standard_normal((4, 256)).astype(np.float32)
    y, decision = layer.forward(x)
    print(f"  input shape:  {x.shape}")
    print(f"  output shape: {y.shape}")
    print(f"  gate activate flag: {bool(decision.activate.any())}")

    _print_section("4. Tiny MoE dispatch")

    bank = ExpertBank()
    for eid in range(3):
        w = (rng.standard_normal((32, 256)) * 0.05).astype(np.float32)
        p = residual_quantize(w, num_planes=2)
        bank.add(eid, ResidualTernaryLinear(planes=p, gate=ResidualGate(mode=GateMode.ALWAYS)))

    router = TopKRouter(num_experts=3, top_k=2)
    out = router.route(np.array([0.1, 0.4, 0.9], dtype=np.float32))
    print("  routed experts per token:")
    for i, idxs in enumerate(out.indices):
        print(f"    token {i}: ids={idxs.tolist()} weights={out.weights[i].tolist()}")

    _print_section("5. Recursive context (RLM primitive) + REPL")

    chunks = [
        "TORUS uses residual ternary planes.",
        "The recursive context handles long prompts.",
        "Adaptive gates trade speed for quality.",
        "MoE experts specialize through routing.",
        "Phase 2 will add hardware-aware kernels.",
    ]
    ctx = RecursiveContext(
        chunks,
        ask_callable=lambda s: f"<<{len(s)} chars answered>>",
    )
    aggregated = ctx.recurse_on(ContextSlice(0, len(chunks)), chunk_size=2)
    print(f"  aggregated answer (chunk_size=2): {aggregated!r}")

    repl = ContextREPL(ctx)
    stdout, value_repr = repl.run("len([c for c in context.grep('ternary')])")
    print(f"  REPL stdout: {stdout.strip()!r}")
    print(f"  REPL value:  {value_repr}")


if __name__ == "__main__":
    main()
