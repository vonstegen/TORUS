# Track B — Adaptive Precision

## Dependency gate

Do not optimize gating until Track A establishes that extra planes improve quality:

`DeltaQ = Q(T1+T2) - Q(T1) > meaningful preregistered threshold.`

## B1: oracle gate

For each eligible unit—row, token, token-layer, or batch—evaluate both low- and high-plane outcomes against the FP teacher or ground truth. Let an oracle activate extra precision only when it improves the defined objective. Sweep activation budgets.

This yields the upper bound on any learned gate. If the oracle cannot create a useful quality–cost frontier, stop Track B.

## B2: gate signals and granularity

Compare simple baselines before a learned network: always-one, always-all, random at matched activation rate, magnitude threshold, depth schedule, entropy/confidence threshold, and oracle. Distinguish call-level, batch-level, row-level, and token-level execution. Report the hardware-realized granularity.

## B3: learned cost-sensitive gate

Train against a task/distillation loss plus an explicit execution penalty:

`L = L_task + lambda * C_residual`.

Sweep `lambda`; publish calibration, quality, activation rate, and realized latency/energy. Prevent information leakage by separating gate-training and evaluation samples.

## B4: systems validation

Activation-rate savings are not equivalent to speed or energy savings. Measure kernel dispatch overhead, divergence, batching effects, memory traffic, tokens/sec, latency distribution, and joules/token where possible. Compare against static layer-wise precision maps from Track A.

## Pass conditions

- Oracle gating materially dominates static plane counts at one or more budgets.
- A learned or deterministic deployable gate captures a preregistered fraction of oracle gain.
- End-to-end measured cost improves, not merely theoretical operation count.
- Results hold on held-out tasks and sequence lengths.

If static layer-wise precision matches adaptive gating, retain the simpler static architecture.
