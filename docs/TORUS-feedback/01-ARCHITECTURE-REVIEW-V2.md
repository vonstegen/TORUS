# TORUS Architecture Review v2

## Executive conclusion

TORUS remains a real research prototype, but its three pillars must be
validated independently:

-   **conditional representation/precision** --- ternary correction
    planes and related training geometry;
-   **conditional computation** --- deciding when additional
    precision/expert computation is useful;
-   **conditional context** --- recursive/persistent/indexed context
    outside the ordinary attention window.

The v1 conclusion that residual planes had not demonstrated useful
recovery is superseded. Corrected sequential training shows that
additional ternary planes can add real functional capacity. However,
improved teacher/logit KL has not yet translated into competitive
downstream language quality, and efficiency superiority over strong
INT4/INT8/ternary baselines is not established.

## Updated provisional grades

  -----------------------------------------------------------------------
  Area                    v2 grade                Interpretation
  ----------------------- ----------------------- -----------------------
  A-R:                    B+                      Strong evidence that a
  residual/correction                             correctly trained
  capacity                                        second plane adds
                                                  capacity

  A-Q: downstream         D+/C-                   Large gap remains,
  language quality                                especially on
                                                  LAMBADA-style
                                                  prediction

  A-E: efficiency/Pareto  C / open                Must beat strong
  advantage                                       low-bit baselines on
                                                  measured cost

  Track A overall         B-                      Promising mechanism,
                                                  not yet competitive
                                                  architecture

  Track B adaptive        C+                      Now justified for
  computation                                     oracle testing because
                                                  plane 2 has measurable
                                                  value

  Track C recursive       B                       Strongest independent
  context                                         engineering subsystem;
                                                  benchmark quality still
                                                  needed

  A2 Hadamard/native      B- / open               Controlled optimization
  geometry                                        signal; native training
                                                  remains untested
  -----------------------------------------------------------------------

## Revised interpretation of residual planes

Do not assume trained planes merely reconstruct the original
floating-point matrix. The new evidence shows teacher KL improving even
when direct weight reconstruction does not. During learning, later
planes should therefore be treated as **functional correction planes**:

`W_effective = T1 + ΔT2 + ΔT3 + ...`

Their purpose is to improve model behavior under the training objective,
not necessarily minimize weight-space reconstruction error.

## Architecture rule

Do not return to monolithic development. A component enters integrated
TORUS v2 only after it:

1.  passes its isolated hypothesis test;
2.  reproduces under clean provenance;
3.  shows task-relevant value, not only proxy-metric improvement;
4.  survives comparison with simpler baselines;
5.  earns its compute/storage complexity.
