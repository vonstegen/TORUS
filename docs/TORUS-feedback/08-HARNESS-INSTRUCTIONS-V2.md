# TORUS Testing Harness Instructions v2

> **v2.1 integration note — 2026-08-22:** Track A-F is governed by
> `10-RESIDUAL-PLANE-FALSIFICATION-SUITE-V2.md`. Where earlier guidance
> conflicts with Track A-F, document 10 takes precedence.

## Primary directive

**ISOLATE → FALSIFY → GRADE → REPRODUCE → RECOMBINE**

The harness must steer TORUS as a research program, not as a
feature-completion roadmap.

## Operating rules

1.  Freeze nonessential feature expansion while a core hypothesis is
    unresolved.
2.  Run one primary hypothesis per experiment.
3.  Define controls, metrics and PASS/FAIL/KILL thresholds before
    expensive runs.
4.  Preserve negative results.
5.  Never accept proxy-metric gains alone when downstream behavior
    disagrees.
6.  Compare against simpler strong baselines.
7.  Give every run an immutable provenance namespace.
8.  Mark contaminated/uncertain runs INVALID and rerun cleanly.
9.  Require reproduction before changing architecture around a result.
10. Recombine only independently passing components.

## Claim-driven operation

The harness does not follow a fixed linear roadmap. It operates the
claim-driven experimental DAG defined in
`10-RESIDUAL-PLANE-FALSIFICATION-SUITE-V2.md`, which is the **Track-A
falsification authority**:

1.  Read the claim registry; select the highest-priority unresolved claim.
2.  Verify prerequisites and unlock rules before scheduling any experiment.
3.  Run the cheapest meaningful falsification test first.
4.  Score proxy, capability and cost metrics separately; grade; update the
    claim state; unlock or block dependent experiments.

Enforced unlock rules (document 10 §12): Track-B oracle gating requires
A-RP-001 `CONFIRMED_PASS`, A-RP-002 at least provisionally supported, and
AF5 above its preregistered threshold; OLMoE adaptive-precision work
additionally requires useful dense-model oracle savings and T1/T2 survival
of the A-F suite; routine T3/T4 scaling and large-model Hadamard runs stay
locked until their respective criteria pass.

The current steering priorities below are subordinate to these unlock
rules: Priority 1 items run now; Priority 2 items wait for the Track A-F
verdict regardless of available compute.

## Current steering priorities

### Priority 1 --- Track A

-   characterize layer sensitivity;
-   reproduce sequential T1→T2 correction gains;
-   stop routine plane stacking beyond T2 unless marginal-value criteria
    justify it;
-   test heterogeneous precision;
-   run small native-Hadamard controlled training.

### Priority 2 --- Track B

After T1/T2 downstream value is demonstrated, run oracle gating. If
oracle gating passes, use OLMoE to study expert sparsity × precision
sparsity.

### Priority 3 --- Track C

Benchmark recursive context independently with a conventional competent
model.

## Explicit prohibitions

The harness should not: - claim residual planes are useless based on
superseded trainer-bug runs; - claim Hadamard is proven based on
post-hoc experiments; - claim Hadamard failed merely because post-hoc
recovery is NO-SHIP; - call a KL-only improvement a model-quality
success; - call skipped theoretical operations an energy win without
measurement; - describe the physical format as 1.58 bits/weight without
accounting for actual packing and metadata; - let OLMoE routing confound
the basic Track-A ternary experiment.

## Decision output

Every completed experiment should end with: - hypothesis; - result
summary; - grade; - PASS / FAIL / INVALID / CONTINUE; - confidence and
reproduction status; - next permitted experiment; - experiments
explicitly blocked by the result.
