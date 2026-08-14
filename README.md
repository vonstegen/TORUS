# TORUS

**Ternary Optimized Recursive Unified System**

> Extreme-efficiency local LLM inference via residual ternary planes, adaptive gating, and recursive context-as-variable handling.

TORUS is a research project and reference implementation that co-designs three ideas to make frontier-class language model intelligence run on modest hardware:

1. **Residual Ternary Planes** — represent every weight as `W ≈ s₁·T₁ + s₂·T₂ + …` where each `Tᵢ ∈ {-1, 0, +1}`. The primary plane is pure ternary; residual planes can be activated on demand to recover lost capability.
2. **Adaptive Residual Gating** — a lightweight gate that decides per token, per layer (or per expert) whether to spend the extra compute. Easy tokens stay on the primary plane; hard reasoning tokens engage the residual plane.
3. **Recursive Context-as-Variable** — the long prompt lives as a `context` variable in a persistent REPL. The model writes code to inspect, chunk, and recursively query sub-portions, turning the context window from a hard limit into a programmable resource.

The name *TORUS* (a topological donut) captures the recursive, circular flow: chunks of context cycle through the model, residual planes activate on demand, and the loop continues until a complete answer is synthesized.

---

## Why TORUS?

Most local LLM users face a hard trade-off:

| Choice                 | Memory / Speed | Quality |
|------------------------|----------------|---------|
| Full-precision 70B     | Heavy          | High    |
| 4–8 bit GGUF           | Medium         | Good    |
| Pure 1.58-bit ternary  | Tiny           | Degraded |

TORUS attacks the *degraded-quality* end. It keeps the extreme efficiency of 1.58-bit weights but adds a **runtime quality–efficiency dial** through residual planes + gating, so a single model can behave like 1.58-bit for easy work and ~3–4 bit effective for hard work — without loading a second model.

When combined with the **RLM/Prime-Agent pattern** of treating context as a variable in a REPL, even a compact ternary model can serve effectively-unlimited-context tasks because no single call ever sees more than a small, relevant slice of the prompt.

---

## Design Pillars

| Pillar | Status | Notes |
|--------|:------:|-------|
| Pure ternary primary plane (`{-1, 0, +1}`)         | implemented | `torus.quant.ternary_quantize` |
| Residual ternary planes                             | implemented | `torus.quant.residual_quantize` |
| Adaptive per-layer / per-token residual gate        | implemented | `torus.core.ResidualGate` |
| Reconstruction under multi-plane weights            | implemented | `torus.quant.compose_planes` |
| CPU reference ternary GEMM kernels + op counters    | implemented | `torus.core.kernels` (dense/sparse/unrolled) |
| Compiled C ternary GEMM (portable + AVX2/AVX-512)   | implemented | `torus.kernels.simd` + `csrc/torus_kernel.c` |
| CUDA ternary GEMM with graceful fallback            | implemented | `torus.kernels.cuda` (numba) |
| Memory-hierarchy placement policy (VRAM/RAM/NVMe)   | implemented | `torus.core.memory` |
| CUDA / AVX-512 hardware kernel spec                 | specified   | `docs/KERNELS.md` |
| MoE-aware expert-specialized residual planes         | scaffolded  | `torus.moe.ExpertBank` |
| Recursive context-as-variable (RLM) primitive       | implemented | `torus.rlm.RecursiveContext` |
| REPL integration                                   | implemented | `torus.rlm.ContextREPL` |
| Capability-aware distillation loss                  | implemented | `torus.train.losses` |
| Straight-through estimator for ternary gradients   | implemented | `torus.train.TernarySTE` |
| Progressive residual-plane curriculum               | implemented | `torus.train.CurriculumSchedule` |
| QAT / distillation training loop                    | implemented | `torus.train.DistillationTrainer` |
| Hardware kernels (CUDA / AVX-512 implementation)    | planned     | Phase 2 follow-on |
| BitNet / GGUF / llama.cpp adapters                  | planned     | Phase 4 |
| Joint learned gate (replaces heuristic)             | planned     | Phase 4 |
| Native fitness + training utility scripts           | planned     | Phase 4 |
---

## Installation

```bash
git clone https://github.com/vonstegen/TORUS.git
cd TORUS
pip install -e .
```

Optional extras:

```bash
pip install -e ".[torch]"   # if you want torch tensors in addition to numpy
pip install -e ".[dev]"    # pytest + ruff + mypy
```

### Optional: GPU torch (Python 3.11 venv)

`pip install torch` from PyPI gives you the CPU build on aarch64 +
Python 3.12. To get a CUDA-enabled torch on an aarch64 host (so the
HF adapter can drive a real model on the GPU), create a side
venv with Python 3.11 and install from the CUDA index:

```bash
~/.local/bin/python3.11 -m venv .venv-py311
.venv-py311/bin/pip install -U pip
.venv-py311/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
.venv-py311/bin/pip install -e . --no-deps
.venv-py311/bin/pip install pytest numpy transformers
```

The TORUS CUDA kernel uses `numba` and works with both venvs.

---

## Quick start

```python
import numpy as np
from torus.quant import ternary_quantize, residual_quantize, compose_planes

# A realistic-ish weight matrix (e.g. a 1024x1024 attention projection)
rng = np.random.default_rng(0)
W = rng.standard_normal((1024, 1024)).astype(np.float32) * 0.05

# Pure 1.58-bit ternary baseline
T1, s1 = ternary_quantize(W, group_size=128)

# Two-plane residual ternary: primary + one residual plane
planes = residual_quantize(W, num_planes=2, group_size=128)

# Reconstruct full-precision weight (lossy by design)
W_hat = compose_planes(planes)

# Reconstruction error
err = float(np.linalg.norm(W - W_hat) / np.linalg.norm(W))
print(f"relative reconstruction error: {err:.3f}")
```

See `examples/` for full workflows and `tests/` for runnable verification.

---

## Repository layout

```
TORUS/
├── torus/                  # Python package
│   ├── core/               # adaptive gates, residual plane container
│   ├── quant/              # ternary + residual-plane quantization math
│   ├── moe/                # expert bank + routing (scaffold)
│   └── rlm/                # recursive context-as-variable primitive
├── tests/                  # pytest suite
├── docs/                   # VISION, ARCHITECTURE, ROADMAP
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Roadmap

See `docs/ROADMAP.md` for the full multi-phase plan. Short version:
- **Phase 2** — CUDA / AVX-512 kernels for ternary multi-plane GEMM on real hardware (GB10 Blackwell).
- **Phase 4** — runtime: GGUF / bitnet.cpp / custom adapter for serving ternary models locally.
- **Phase 5** (research) — FPGA/ASIC exploration: ternary-native lanes with gateable residual units.

---

## Contributing

This is an early-stage research project. Issues, design discussion, and PRs are welcome — especially around:

- residual plane scaling laws
- gating strategies (per-layer / per-token / per-expert)
- recursive context primitives
- ternary kernel design

Please open an issue before large refactors so the direction stays aligned.

---

## Citation

If you use TORUS in research, please cite this repository. A formal citable artifact will be released with Phase 2 kernels.

---

## License

Apache-2.0. See `LICENSE`.
