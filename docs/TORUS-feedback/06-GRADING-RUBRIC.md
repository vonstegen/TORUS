# Grading Rubric and Pareto Scorecard

## Evidence classes

- **E0 — claim:** design statement only.
- **E1 — unit evidence:** software behavior verified in isolation.
- **E2 — controlled experiment:** baseline, fixed inputs, raw artifacts, and preregistered thresholds.
- **E3 — reproduced evidence:** repeated across seeds/environments or independently rerun.
- **E4 — external evidence:** independent reproduction or public benchmark audit.

No architecture claim receives an A below E3.

## Track grade definitions

- **A:** reproducible Pareto improvement or independent end-to-end advantage; limitations documented.
- **B:** strong controlled evidence and a clear useful path, but incomplete reproduction, scale, or deployment proof.
- **C:** working mechanism with plausible value; decisive quality/cost evidence missing.
- **D:** evidence is negative or severely below target, but a bounded falsifiable recovery path remains.
- **F:** key hypothesis falsified under agreed scope, or no credible cost-bounded next test remains.

Use `+/-` only to communicate proximity to the adjacent grade; it cannot replace thresholds.

## Current provisional grades

| Track | Grade | Reason |
|---|---|---|
| A: residual ternary | D+ | Catastrophic full-model quality loss; calibration helped but did not validate representation. |
| B: adaptive precision | C+ | Mechanism and telemetry work; value depends on useful extra planes and granular execution. |
| C: recursive context | B | Strongest independent subsystem; lacks rigorous end-to-end comparison. |

## Required scorecard

For every candidate, report:

| Dimension | Required measurement |
|---|---|
| Quality | Perplexity, task accuracy, logit divergence, confidence intervals |
| Representation | Physical bytes/weight including scales, metadata, padding |
| Memory | Resident model size, peak RAM/VRAM, bandwidth where measurable |
| Compute | Operations/token and realized plane activation distribution |
| Runtime | Prefill/decode throughput and p50/p95 latency |
| Energy | Joules/token with method and uncertainty, when trustworthy |
| Reliability | Seeds, failures, numerical stability, reproducibility |

Never collapse these into one opaque score. Plot Pareto frontiers, identify dominated points, and retain task-level results. The ≥90% FP16 aggregate target is a final acceptance criterion, not permission to hide efficiency tradeoffs or individual task collapse.

## Decision labels

Every experiment ends with exactly one: `PASS`, `FAIL`, `INCONCLUSIVE`, or `INVALID`. Invalid runs are corrected under new IDs; they are not silently overwritten.
