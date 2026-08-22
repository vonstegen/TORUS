# Track A2 v2 --- Native Hadamard Ternary Training

## Evidence status

Two hypotheses must remain separate.

### H-POST: Post-hoc rotation and recovery

Rotate/requantize an already trained ternary model, then attempt
recovery.

**Current verdict:** **NO-SHIP at the tested budget.** Rotation creates
an admission penalty. KD recovers part of the loss but saturates below
the stock Bonsai model. Simple continued pretraining did not solve the
problem.

### H-NATIVE: Native Hadamard ternary training

Build Hadamard/rotation geometry into ternary training from
initialization or sufficiently early training.

**Current verdict:** **OPEN and motivated.** Controlled
rotated-vs-unrotated experiments showed better optimization
behavior/conditioning for the rotated arm despite the post-hoc admission
penalty.

## Next experiment --- A-H1

Use a small model first (roughly 100M--500M parameters) and matched
arms:

-   Control: standard ternary training.
-   Hadamard: identical architecture, data, optimizer, schedule and
    budget, with native rotated ternary parameterization.

Measure: - loss convergence; - teacher/logit KL if distilling; -
downstream accuracy/perplexity; - gradient conditioning; - ternary
code-flip rate; - throughput and memory traffic; - physical bits/weight
including metadata; - joules/token where measurement is trustworthy.

Preregister kill criteria before the run. Do not spend large-model GPU
budget until the small-model controlled experiment passes.
