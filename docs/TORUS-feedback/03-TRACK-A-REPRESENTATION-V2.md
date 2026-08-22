# Track A v2 --- Efficient Model Representation

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
-   no provenance or evaluation artifact.
