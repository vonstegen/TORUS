# Architecture Review

## Executive conclusion

TORUS is a real research prototype with three separable systems:

1. **Conditional precision:** residual ternary planes approximate a weight matrix progressively.
2. **Conditional computation:** a gate chooses whether additional planes execute.
3. **Conditional context:** recursive, persistent, indexed context is queried outside the model's attention window.

The implementation has advanced faster than scientific validation. Software correctness, efficiency demonstrations, and learned-model quality have sometimes been blended into a single notion of progress. They must be reported separately.

## What is established

- Reference quantization, residual-plane abstractions, HF adapters, training plumbing, kernels, telemetry, checkpointing, and evaluation integration exist.
- Persistent context, indexing, a REPL abstraction, sandboxing, and a model-independent Prime Agent loop are meaningful engineering work.
- A reported 2,000-chunk context demonstration showed roughly `0.041 ms` warm indexed lookup versus `49.8 ms` linear scan. This validates that workload's index, not end-to-end answer quality.
- A routing demonstration reported roughly 17% fewer plane activations than an always-four-plane baseline. It did not prove a trained gate or preserved model quality.
- Several trainer defects were found and corrected. Earlier curves affected by those defects are engineering-validation evidence, not architecture-validation evidence.

## What is not established

The central equation remains unsupported:

`TORUS quality ≈ FP/BF16 quality at materially lower measured cost.`

The strongest reported OLMo-1B evidence was negative: low-plane quantization collapsed downstream quality, training did not materially recover it, and a second residual plane did not demonstrate useful recovery. Norm calibration isolated one mechanism but did not close the gap.

## Current grades

- **Track A — Residual ternary representation: D+ / unresolved.** It has not earned expensive full-model training.
- **Track B — Adaptive precision: C+ / promising but dependent.** Gating has no value unless extra planes provide useful quality.
- **Track C — Recursive context: B / strongest subsystem.** It needs rigorous end-to-end comparison against long-context and RAG baselines.

## Architectural risks

- Uniformly ternarizing every linear layer assumes equal tolerance across `q/k/v/o`, MLP projections, and `lm_head`; that assumption is unlikely to hold.
- Current call- or batch-level activation can erase predicted savings if one difficult row activates extra planes for all rows. True row/token-granular kernels must be measured, not inferred.
- A heuristic gate is scaffolding, not evidence for learned conditional precision.
- “1.58-bit” is theoretical ternary information content, not the current physical representation. Use **ternary weights** or **2-bit packed ternary weights**, and report scale/metadata overhead.
- Packaging should discover all `torus*` packages, including training modules. Wheel installation should be tested outside the source tree; the C shared-library versus Python-extension model should be made explicit.
- Hardware documentation should not claim AVX-512 for the Ryzen Threadripper PRO 3995WX; target AVX2 for that CPU.

## Decision

Freeze major feature expansion. For the next milestone, allocate approximately 80% of effort to controlled experiments and 20% to code required to run them. Preserve Track C, but do not use it to mask failure in Tracks A or B.
