# TORUS v2 --- Residual Plane Falsification Architecture Update

**Date:** 2026-08-22\
**Applies to:** TORUS Feedback v2\
**Primary directive:** **ISOLATE → FALSIFY → GRADE → REPRODUCE →
RECOMBINE**

## Purpose

This document adds a formal residual-plane falsification pipeline to the
TORUS v2 testing architecture. The objective is not to repeat prior
training runs or search for better-looking numbers. The objective is to
determine whether the current conclusion --- that a sequentially trained
second ternary correction plane adds real, useful capacity --- survives
strong controls.

The residual-plane work becomes the first full test case for TORUS v2's
claim-driven research harness.

------------------------------------------------------------------------

## 1. Add Track A-F --- Residual Plane Falsification Suite

Place this stage after the initial Track-A representation experiments
and before Track-B adaptive-computation work.

``` text
TORUS v2 Research Harness
│
├── Track A — Representation
│   ├── A1 Layer sensitivity
│   ├── A2 Oracle residual
│   ├── A3 Sequential T1 → T2 training
│   ├── AF — RESIDUAL FALSIFICATION SUITE
│   │   ├── AF1 Equal-training-budget control
│   │   ├── AF2 Equal-storage alternative control
│   │   ├── AF3 Initialization robustness
│   │   ├── AF4 Sequential-vs-joint training
│   │   ├── AF5 Downstream-transfer test
│   │   ├── AF6 Dataset/context robustness
│   │   ├── AF7 Random-capacity control
│   │   └── AF8 Clean reproduction
│   ├── A4 Heterogeneous precision
│   └── A5 Native Hadamard ternary
│
├── Track B — Adaptive Compute
│      ↑ unlocked only after AF validates useful T2 behavior
│
└── Track C — Recursive Context
```

Track B must not assume T2 is useful merely because T2 lowers a proxy
metric. T2 must earn its way into adaptive-gating experiments.

------------------------------------------------------------------------

## 2. Make the harness claim-driven

Experiments should attack registered claims rather than merely try
configurations.

### Claim A-RP-001

> A second sequentially trained ternary correction plane provides useful
> model capacity beyond the benefit of equivalent additional training
> time.

Required evidence: AF1, AF3, AF5 and AF8.

### Claim A-RP-002

> The improvement from T2 is competitive with an equal-storage
> non-ternary correction mechanism.

Required evidence: AF2, AF7 and AF8.

### Claim A-RP-003

> Sequentially freezing T1 before learning T2 is necessary or materially
> superior to matched joint-training alternatives.

Required evidence: AF4 and AF8.

Recommended claim states:

``` text
UNTESTED
TESTING
PROVISIONAL_PASS
PROVISIONAL_FAIL
REPRODUCTION_REQUIRED
CONFIRMED_PASS
CONFIRMED_FAIL
INVALID
```

------------------------------------------------------------------------

## 3. Experiment and artifact structure

Use immutable experiment namespaces.

``` text
research/
└── track-a/
    └── residual-falsification/
        ├── claims/
        │   ├── A-RP-001.yaml
        │   ├── A-RP-002.yaml
        │   └── A-RP-003.yaml
        ├── experiments/
        │   ├── AF1/
        │   ├── AF2/
        │   ├── AF3/
        │   ├── AF4/
        │   ├── AF5/
        │   ├── AF6/
        │   ├── AF7/
        │   └── AF8/
        └── reports/
```

A run should resemble:

``` text
AF1/
├── experiment.yaml
├── runs/
│   ├── seed-001/
│   │   ├── config.json
│   │   ├── provenance.json
│   │   ├── train.jsonl
│   │   ├── eval.json
│   │   └── checkpoint.sha256
│   ├── seed-002/
│   └── seed-003/
└── verdict.md
```

Never reuse shared mutable checkpoint/result paths between experiments
or concurrent agents.

------------------------------------------------------------------------

## 4. AF1 --- Equal-training-budget control

Test whether T2 is actually better than simply training T1 longer.

Matched arms:

``` text
Arm A
T1 trained for N steps
+
T1 continued for another N steps

Arm B
T1 trained for N steps
+
freeze T1
+
T2 trained for N steps
```

Control as closely as possible for tokens, batches, data order,
optimizer budget and compute accounting.

Compare:

`Q(T1 continued)` versus `Q(T1 + T2)`.

If T2 does not materially outperform the matched T1 continuation,
downgrade the claim that the second plane itself is responsible for the
improvement.

------------------------------------------------------------------------

## 5. AF2 --- Equal-storage / bit-budget tournament

Compare T2 against alternative uses of approximately the same physical
storage budget.

Candidates should include where practical:

-   T2 ternary correction;
-   INT4 residual;
-   smaller INT8 residual;
-   low-rank correction;
-   learned group scales;
-   small dense correction/adapter.

Measure actual physical storage, including scales and metadata. Compare
downstream quality against extra bytes and runtime cost.

The relevant claim is not merely "T2 is better than no T2." It is:

> T2 is competitive on the quality-per-physical-bit /
> quality-per-compute Pareto frontier.

A T2 configuration that is dominated by a simpler equal-cost alternative
must not be promoted as the preferred representation.

------------------------------------------------------------------------

## 6. AF3 --- Initialization robustness

Automate a residual-initialization matrix, for example:

``` yaml
residual_init:
  - 0
  - 1e-4
  - 3e-4
  - 1e-3
  - 3e-3
  - 1e-2
seed:
  - 11
  - 22
  - 33
```

Aggregate mean, standard deviation, failure rate, best/worst run and
sensitivity to initialization.

Classify the method as:

-   ROBUST;
-   MODERATELY SENSITIVE;
-   FRAGILE.

A method that succeeds only in a very narrow initialization window must
receive a lower robustness assessment.

------------------------------------------------------------------------

## 7. AF4 --- Sequential versus joint training

Matched experimental arms should include:

``` text
1. T1 → freeze → T2
2. joint(T1,T2), where scientifically appropriate
3. T1 → freeze → T2 → freeze → T3
4. T1 → joint(T2,T3)
5. T1-only with matched additional optimizer/training budget
```

Report both gain per training token and gain per added physical byte.

This experiment determines whether sequential functional correction is
itself an important part of the TORUS mechanism.

------------------------------------------------------------------------

## 8. AF5 --- Downstream-transfer gate

Every representation experiment must report two metric classes.

### Proxy/model-space metrics

-   teacher/logit KL;
-   perplexity;
-   hidden-state error where applicable;
-   cosine similarity.

### Capability metrics

-   ARC-E or equivalent ranking/MC task;
-   LAMBADA or another exact next-token task;
-   at least one additional held-out language/reasoning/QA task when
    feasible.

If proxy metrics improve substantially while capability metrics do not,
the harness must label the result:

``` text
PROXY IMPROVEMENT
CAPABILITY NOT VALIDATED
```

No Track-A representation is accepted on KL alone.

------------------------------------------------------------------------

## 9. AF6 --- Dataset and context robustness

Test at least two context regimes and, where practical, two corpora.

Example:

``` text
short context:  seq_len ≈ 16
longer context: seq_len ≈ 128–256
```

The purpose is to determine whether T2's gain is a general
representation/training effect or an artifact of a narrow short-window
training distribution.

------------------------------------------------------------------------

## 10. AF7 --- Random/non-ternary capacity control

Add approximately equal trainable/storage capacity without imposing the
T2 ternary structure.

Example comparison:

``` text
T1 + ternary T2
versus
T1 + matched low-rank or dense trainable correction
```

If both improve similarly, the justified conclusion is only that
additive trainable correction helps. If T2 wins on quality-per-bit or
quality-per-compute, that provides stronger evidence for ternary
correction specifically.

------------------------------------------------------------------------

## 11. AF8 --- Mandatory clean reproduction

Important discoveries move through:

``` text
DISCOVERY PASS
      ↓
CLEAN REPRODUCTION
      ↓
CONFIRMED PASS
```

The reproduction must use a new run ID, independent artifact namespace,
frozen git SHA, fresh process, verified checkpoint hashes and
independently generated evaluation output.

If provenance is uncertain, mark the result **INVALID**, not PASS or
FAIL.

------------------------------------------------------------------------

## 12. Unlock rules

The harness should enforce dependencies rather than following a fixed
phase roadmap.

### Track-B oracle gating

Do not unlock adaptive T1/T2 gating until:

-   A-RP-001 is CONFIRMED_PASS;
-   A-RP-002 is at least provisionally supported;
-   AF5 demonstrates task-relevant T2 value above a preregistered
    threshold.

### OLMoE adaptive-precision testing

Do not launch expensive OLMoE expert-routing × precision-routing
experiments until dense-model oracle gating shows useful savings and the
T1/T2 representation has survived falsification.

### T3 production testing

Keep routine T3/T4 scaling locked unless measured marginal downstream
gain per added physical bit/operation exceeds a preregistered threshold.

### Large native-Hadamard runs

Keep large-model Hadamard experiments locked until a small controlled
native-Hadamard experiment earns a CONFIRMED_PASS.

------------------------------------------------------------------------

## 13. Discovery versus confirmation budgets

Use two compute tiers.

### Discovery

-   one seed when appropriate;
-   smaller evaluation sample;
-   shorter training budget;
-   designed to cheaply kill weak hypotheses.

### Confirmation

-   multiple seeds;
-   held-out data;
-   full downstream evaluation;
-   immutable artifacts and provenance;
-   appropriate baseline tournament.

Only promising discovery runs escalate to confirmation.

------------------------------------------------------------------------

## 14. Harness execution loop

The TORUS v2 research harness should operate approximately as follows:

``` text
1. Read claim registry.
2. Select the highest-priority unresolved claim.
3. Verify prerequisites/unlock rules.
4. Generate an immutable experiment manifest.
5. Run the cheapest meaningful falsification test.
6. Validate provenance.
7. Score proxy, capability and cost metrics separately.
8. Grade the result.
9. Reproduce important PASS/FAIL results.
10. Update claim state.
11. Unlock or block dependent experiments.
12. Select the next experiment.
```

The central research question for the harness is:

> **What is the cheapest experiment that could prove our current belief
> wrong?**

------------------------------------------------------------------------

## 15. Residual-plane acceptance bar

Before TORUS v2 promotes T2 as a validated representation mechanism, T2
should at minimum:

1.  outperform an equal-budget T1-only continuation;
2.  survive multiple seeds and reasonable initialization variation;
3.  produce held-out downstream improvement, not only KL reduction;
4.  compete with at least one equal-storage non-ternary correction
    baseline;
5.  reproduce under clean immutable provenance.

If these conditions fail, downgrade the conclusion appropriately rather
than attempting to rescue the architecture with additional complexity.

**Addendum 2026-08-30 (program-level gating order):** the
residual-plane branch closed under this bar (MECHANISM CONFIRMED /
COMPETITIVE ARCHITECTURE NOT SUPPORTED — see
`research/reports/RESIDUAL-PLANE-CLOSURE-2026-08-30.md`). For
every future TORUS mechanism, the preregistration order is:

> mechanism signal → capability check → competitive baseline →
> robustness → scale.

The competitive baselines (best matched-storage correction AND
equal-budget continuation) are evaluated at DISCOVERY tier with
frozen bars, before robustness or scale spend. Internal metrics
(KL, training loss, recovery-vs-random, conditioning) are
diagnostics regardless of their z-scores; a frozen capability-bar
FAIL closes the line with no rescue. Cross-program synthesis:
`research/reports/CROSS-PROGRAM-SYNTHESIS-2026-08-30.md`.
------------------------------------------------------------------------

## Recommended integration into TORUS Feedback v2

Add this file to the existing `TORUS-feedback-v2/` documentation set and
update:

-   `03-TRACK-A-REPRESENTATION-V2.md` to reference Track A-F;
-   `05-TRACK-B-ADAPTIVE-COMPUTE-OLMOE-V2.md` so Track B is explicitly
    gated by A-F;
-   `07-GRADING-AND-REPRODUCIBILITY-V2.md` with claim states and INVALID
    handling;
-   `08-HARNESS-INSTRUCTIONS-V2.md` with the claim-driven DAG and unlock
    rules.

Suggested repository filename:

`TORUS-feedback-v2/10-RESIDUAL-PLANE-FALSIFICATION-SUITE-V2.md`
