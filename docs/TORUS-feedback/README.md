# TORUS Research Reset Feedback Package

This package converts the TORUS repository review into an executable research program.

## Governing directive

> **Isolate. Falsify. Grade. Recombine.**

Do not continue treating residual ternary weights, adaptive precision, and recursive context as one hypothesis. Test each track independently, publish failures, and recombine only components that pass preregistered gates.

## Contents

1. `01-ARCHITECTURE-REVIEW.md` — current-state assessment and evidence.
2. `02-THREE-TRACK-RESEARCH-PLAN.md` — sequencing and shared experimental rules.
3. `03-TRACK-A-RESIDUAL-TERNARY.md` — representation, oracle-residual, and layer-sensitivity experiments.
4. `04-TRACK-B-ADAPTIVE-PRECISION.md` — oracle gating, learned gating, and cost controls.
5. `05-TRACK-C-RECURSIVE-CONTEXT.md` — independent RLM/RAG benchmark plan.
6. `06-GRADING-RUBRIC.md` — track grades, thresholds, and Pareto scorecard.
7. `07-RECONFIGURATION-PLAN.md` — repository and milestone reset.
8. `08-HARNESS-INSTRUCTIONS.md` — operational instructions for an AI testing harness.

## Status captured by the review

- The engineering platform is substantial; the central low-plane model-quality hypothesis remains unproven.
- OLMo-1B FP16 results were approximately ARC Easy `0.6073`, LAMBADA `0.6095`, and WikiText perplexity `13.09`.
- Reported low-plane runs were approximately ARC Easy `0.258`, LAMBADA `0.001`, and WikiText perplexity `465,097–759,750`.
- Norm calibration reportedly improved PTQ WikiText perplexity to about `89,557`; this is useful diagnosis but still catastrophic relative to `13.09`.
- Track C is the strongest independent subsystem; Track B is meaningful only after extra planes demonstrate value.

Treat these figures as historical review inputs. Every new run must preserve raw artifacts, exact revisions, environments, datasets, and commands.

## Recommended repository location

Copy this folder to the TORUS repository as `feedback/`. Start with `08-HARNESS-INSTRUCTIONS.md`.
