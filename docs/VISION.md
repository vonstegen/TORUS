# Vision

## The Problem

Running capable language models locally forces users to choose a
polarized trade-off:

| Choice                  | Memory / Speed | Quality            |
|-------------------------|----------------|--------------------|
| Full-precision 70B      | Heavy          | High               |
| 4–8 bit GGUF            | Medium         | Good               |
| Pure 1.58-bit ternary   | Tiny           | Degraded reasoning |

There is no good middle ground that lets *the same model* trade a bit of
speed for a bit of extra quality at runtime. Users either load a
different checkpoint or accept the regression.

The same problem compounds with context length: long prompts push the
context window past the model's hard limit, and the only options are
sliding-window truncation or heroic retrieval scaffolding. Both leak
information that the user expects the model to have seen.

## The TORUS Thesis

A single model can be **both** tiny *and* high-quality if you structure
its weights as a stack of ternary planes with an adaptive runtime
gate, and if you treat its long context as a programmable variable
instead of stuffing it into a fixed window.

Concretely, TORUS says three things at once:

1. **Residual Ternary Planes** — represent every weight matrix as
   `W = s₁·T₁ + s₂·T₂ + …`, where each `Tᵢ ∈ {-1, 0, +1}`. The
   primary plane is pure 1.58-bit; residual planes capture the
   approximation error and can be activated selectively.

2. **Adaptive Gating** — the gate is a hardware-friendly signal that
   decides, per token / per layer / per expert, whether to run the
   residual datapath. Easy tokens stay on the primary plane; hard
   reasoning tokens engage the residual plane. One model now behaves
   like 1.58-bit for cheap work and ~3-bit for hard work, without
   loading a second checkpoint.

3. **Recursive Context-as-Variable** — the long prompt lives in a
   persistent REPL as a `context` variable. The model writes Python to
   inspect, slice, search, and recursively query sub-portions. This
   turns the context window from a hard limit into a soft,
   programmable resource — which is exactly the RLM / Prime-Agent
   pattern, but now running on a ternary model that's cheap enough to
   be called many times per turn.

Together these three ideas form a *topological* loop — chunks of
context cycle through the model, residual planes activate when the
chunk is hard, and the loop closes when a complete answer has been
synthesized. Hence the name: a **TORUS**.

## Why a Co-Design?

Most existing work optimizes only one layer of the stack at a time:

- Pure model papers improve accuracy but still pay the cost of the
  binary silicon they happen to run on.
- Pure runtime work (llama.cpp, bitnet.cpp) optimizes inference for
  the weights it was given.
- Pure context work (RLM, Prime Agent) demonstrates the pattern on
  large frontier models that don't fit local hardware.

TORUS co-designs all three:

- The math exposes a primitive (the gate) that the runtime can
  execute as a single hardware signal.
- The runtime can run the primary datapath *without* materializing
  residual planes when the gate is low.
- The context engine keeps individual calls small enough that a
  ternary model — even one with residual planes active — can serve
  effectively-unlimited context.

A genuine improvement at one layer without the others produces a
local-optimum result. The thesis is that the gains live at the
interface.

## Success Criteria

TORUS is successful if, by the end of Phase 4, a user can:

1. Run a 70B-class model on a workstation with **48 GB VRAM + 128 GB
   RAM + fast NVMe** at interactive tokens/sec, using ternary
   weights plus the residual-plane gate when quality matters.
2. Toggle the gate to swap a *measurable* amount of speed for a
   *measurable* amount of quality on a held-out benchmark, on the
   same weights, with no reload.
3. Run the same model on a long-context task where the prompt exceeds
   the model's window by orders of magnitude, by having the model
   treat the prompt as a `context` variable and dispatching recursive
   sub-calls.
4. Inspect, fork, and re-mix the components: quantization, gating,
   routing, and recursive context are independent, replaceable
   modules.

## Non-Goals (Phase 1)

- Replacing frontier closed-weight services on quality alone.
- Pretraining 700B-class models from scratch (we use OLMoE and similar
  open bases).
- Hard-real-time guarantees.
- Vendor lock-in to a specific framework (Phase 1 ships pure-numpy
  reference impls; framework adapters arrive in Phase 2+).
