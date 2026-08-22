# Track A — Residual Ternary Representation

## Question

Can transformer matrices or blocks be represented by progressive ternary planes without destroying behavior, and at a competitive quality–cost point?

## A1: matrix-output reconstruction

For representative layers from early, middle, and late blocks, compare `W`, `T1`, `T1+T2`, and `T1+T2+T3` on real cached activations. Cover `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, and `lm_head` where present.

Record weight relative L2 error, output relative L2 error, output cosine similarity, activation-distribution shift, downstream logit KL where applicable, physical storage, and operations.

## A2: oracle residual

Construct `T1`, then the exact residual `R = W - T1`. Evaluate:

`W_hat(alpha) = T1 + alpha * R`, for `alpha ∈ {0, .25, .5, .75, 1}`.

Interpretation:

- Rapid recovery proves hierarchical precision has headroom.
- Recovery with the exact residual but not `T2` implicates residual encoding.
- No useful recovery until nearly `alpha=1` indicates the primary plane is too destructive or the layer is unsuitable.

Also compare equal-storage alternatives such as INT8/INT4 residuals or mixed precision. The goal is not to protect ternary encoding from stronger controls.

## A3: layer sensitivity

Start from FP. Quantize one layer at a time, restore it, then move to the next. Repeat for categories: attention only, MLP only, q/k, v/o, up/down, gate, embeddings, and head. Produce a heat map and rank layers by quality loss per byte saved.

## A4: precision-map search

Use sensitivity results to construct heterogeneous models: sensitive layers remain FP/INT8/INT4 while tolerant layers use one or more ternary planes. Test whether precision should vary by layer before it varies by token.

## A5: training controls

Only after PTQ diagnostics identify a plausible representation, run short QAT/distillation controls. Verify with tests that every intended parameter receives gradients and updates, loss terms are wired correctly, initialization escapes quantizer dead zones, and an FP student control can learn the same batch.

## Minimum evidence to advance

Track A advances when at least one physically measured TORUS configuration lies on or improves the quality–cost Pareto frontier versus the baseline ladder, results reproduce across seeds/sample sets, and the gain is not due to an evaluation or packing artifact.

The long-term ≥90% FP16-quality target remains useful, but it is not the sole research score. Report exact task-level deltas and uncertainty.
