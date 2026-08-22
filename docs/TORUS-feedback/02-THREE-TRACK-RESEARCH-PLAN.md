# Three-Track Research Plan

## Objective

Convert TORUS from an integrated architecture-development effort into three falsifiable research tracks.

## Track ordering

### Track A: residual ternary representation

First determine whether one, two, or three residual planes preserve useful behavior, where they fail, and whether the residual encoding—not hierarchy itself—is the bottleneck.

### Track B: adaptive precision

Begin only after Track A shows a meaningful positive quality delta from additional planes. First test an oracle gate; train a gate only if the oracle proves headroom.

### Track C: recursive context

Run independently with a competent fixed model backend. Compare against conventional RAG and long-context baselines under equal budgets.

## Shared experimental contract

Every experiment must define before execution:

- experiment ID, hypothesis, owner, date, code revision, and environment;
- model, layer scope, dataset/sample IDs, seeds, and calibration method;
- baseline and variants;
- quality, cost, and storage metrics;
- pass, fail, and inconclusive thresholds;
- maximum compute budget and stopping rules;
- raw-artifact locations and a one-paragraph conclusion.

Do not change thresholds after seeing results. Corrections produce a new experiment ID. A failed experiment is retained.

## Baseline ladder

Use the same evaluation inputs for:

`FP16/BF16 → INT8 → strong modern 4-bit baseline → T1 → T1+T2 → T1+T2+T3 → adaptive TORUS`

Report quality alongside physical bytes/weight, total model memory, operations/token, latency, throughput, peak memory, plane activation rate, and—when instrumentation is trustworthy—joules/token.

## Milestone sequence

1. Reproduce FP and PTQ numbers with immutable manifests.
2. Run single-matrix output reconstruction tests on real activation distributions.
3. Run single-layer and layer-category sensitivity ablations.
4. Run oracle-residual sweeps.
5. Produce the residual-plane Pareto curve and a layer precision map.
6. If Track A passes, run oracle gating and calculate maximum attainable savings.
7. If oracle gating passes, train and calibrate a cost-sensitive gate.
8. In parallel, benchmark Track C independently.
9. Grade all tracks. Recombine only passing tracks.

## Stop conditions

- Do not scale training while cheaper representation diagnostics fail.
- Do not train a gate if `Q(T1+T2) - Q(T1)` is negligible.
- Do not claim efficiency from activation counts alone; measure end-to-end cost.
- Do not claim RLM superiority from lookup latency alone; measure answer quality and total latency/cost.
