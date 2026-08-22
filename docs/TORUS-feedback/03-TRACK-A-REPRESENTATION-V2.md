# Track A v2 --- Efficient Model Representation

> **v2.1 integration note — 2026-08-22:** Track A-F is governed by
> `10-RESIDUAL-PLANE-FALSIFICATION-SUITE-V2.md`. Where earlier guidance
> conflicts with Track A-F, document 10 takes precedence.

## Updated evidence

Corrected sequential residual-plane training materially changes the
earlier assessment. Representative KL results were approximately:

-   untrained/frozen condition: `3.42`
-   plane 1: `2.66`
-   plane 2: `1.72`
-   plane 3: `1.60`

Interpretation: plane 2 adds substantial learned capacity; plane 3 has
sharply diminishing marginal value; plane 4 is not justified without new
evidence.

Downstream improvement is much weaker. ARC-E improved only modestly and
LAMBADA remained near zero in the relevant experiments. Therefore **KL
improvement alone is not an acceptance criterion**.

## A1 --- Layer sensitivity

Test `q/k/v/o`, MLP projections, embeddings/head and representative
early/middle/late blocks independently. Record output error, cosine
similarity, logit KL, downstream task delta, physical bytes, operations
and runtime.

## A2 --- Oracle residual

For `R = W - T1`, sweep `T1 + αR`, with `α ∈ {0,.25,.5,.75,1}`. Compare
with equal-storage INT8/INT4/mixed-precision residuals. This separates
hierarchical-precision headroom from ternary residual-encoding
limitations.

## A3 --- Sequential functional correction

1.  Train or establish T1.
2.  Freeze T1.
3.  Initialize T2 conservatively (current evidence supports very small
    initialization, around `1e-3` scale).
4.  Train T2 with an independently controlled learning rate.
5.  Freeze T1+T2.
6.  Add T3 only as an ablation.
7.  Stop when marginal downstream gain per physical bit/operation falls
    below a preregistered threshold.

Later planes are evaluated as functional corrections, not solely weight
reconstruction.

## Track A-F --- Residual plane falsification gate

A3's positive result is a discovery-tier signal, not a validated mechanism.
Before T2 is considered validated --- and before any Track-B work may assume
T2 is useful --- the residual-plane claims must survive the falsification
suite defined in `10-RESIDUAL-PLANE-FALSIFICATION-SUITE-V2.md`:

-   AF1 equal-training-budget control (T2 vs. training T1 longer);
-   AF2 equal-storage/bit-budget tournament;
-   AF3 initialization robustness;
-   AF4 sequential-vs-joint training;
-   AF5 downstream-transfer gate;
-   AF6 dataset/context robustness;
-   AF7 random-capacity control;
-   AF8 mandatory clean reproduction.

The registered claims under test are A-RP-001 (T2 beats equal training
time), A-RP-002 (T2 beats equal-storage non-ternary correction), and
A-RP-003 (sequential freezing beats joint training). T2 is promoted only
when the five-condition acceptance bar in document 10 §15 is met.

## A4 --- Heterogeneous precision map

Use sensitivity data to decide which layers stay FP/INT8/INT4, which
tolerate T1, and which benefit from T1+T2. Test layer-adaptive precision
before token-adaptive precision.

## Acceptance rule

No Track-A configuration passes on KL alone. It must show:

-   task-relevant improvement;
-   reproducibility;
-   competitive physical storage/compute;
-   comparison against strong INT4/INT8 and relevant ternary baselines;
-   no provenance or evaluation artifact;
-   for any configuration relying on a trained T2: survival of the Track A-F
    falsification suite (claims A-RP-001 and A-RP-002 at minimum).
