# Reconfiguration Plan

## Immediate freeze

Temporarily freeze new context features, MoE features, persistence variants, serving integrations, and custom-hardware speculation except work required by an approved experiment. Preserve and maintain existing code.

## Repository shape

Organize research artifacts conceptually as:

```text
research/
  registry/
  track-a-residual-ternary/
  track-b-adaptive-precision/
  track-c-recursive-context/
  baselines/
  reports/
artifacts/               # normally external or ignored; manifests committed
feedback/                # this steering package
```

Each experiment directory should contain a manifest, immutable result summary, metrics table, environment lock/snapshot, and links/checksums for large raw artifacts.

## Clean scientific lineage

Mark results produced before the corrected training/autograd pipeline as **engineering validation only**. Choose a new baseline revision, tag it, reproduce FP and quantized baselines, and start numbered experiments from that point. Never mix a new feature, trainer fix, benchmark change, and architectural conclusion in one experiment.

## Work allocation

- 80% controlled experiments and analysis.
- 20% minimal supporting implementation.

Assign ownership by track, not by a single integrated milestone. A shared evaluation owner maintains datasets, baseline adapters, metrics schemas, and reproducibility checks.

## First two-week queue

1. Baseline and environment reproduction.
2. Packaging/wheel smoke test outside the source tree.
3. Matrix-output suite and layer-category sampling.
4. Oracle-residual sweep.
5. Full layer-sensitivity heat map on a bounded evaluation subset.
6. Residual-plane Pareto report and Track A grade review.
7. Track C benchmark specification and first fixed-backend baseline.

## Recombination policy

- A passes, B fails: retain static heterogeneous residual precision if competitive.
- A passes, B passes: combine residual planes with adaptive execution.
- A fails, C passes: ship recursive context independently of ternary inference.
- A fails, B is automatically paused; do not call gating successful in isolation.
- All fail: archive hypotheses and preserve the engineering lessons.

Recombination is a new experiment with its own baseline; component grades do not guarantee integrated performance.
