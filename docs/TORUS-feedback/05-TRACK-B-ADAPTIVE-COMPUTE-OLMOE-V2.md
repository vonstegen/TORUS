# Track B v2 --- Adaptive Computation and OLMoE

## Updated status

The prerequisite for beginning oracle gating is provisionally satisfied:
a correctly trained T2 can provide meaningful representational
improvement over T1. This does not yet prove downstream utility, so
gating must be task-aware.

## B1 --- Oracle gating

For each candidate token/layer/expert, evaluate both T1 and T1+T2 and
determine whether the extra plane improves a task-relevant loss/output.
Calculate the maximum achievable compute saving under a perfect gate.

If an oracle cannot save substantial computation at acceptable quality,
stop learned-gate work.

## B2 --- Learned task-aware gating

Only after B1 passes, train a gate against an objective such as:

`L = L_task + λ * C_extra_precision`

Do not optimize solely for teacher KL if downstream behavior disagrees.

## B3 --- AI2 / AllenAI OLMoE test platform

Use `allenai/OLMoE-1B-7B-0125` as the principal Track-B MoE resource
once the dense Track-A representation is sufficiently stable.

Research hypothesis:

**Can MoE routing information, especially router confidence, predict how
much numerical precision an activated expert needs?**

This creates two sparsity axes:

`expert sparsity × precision sparsity`

Example: - high-confidence expert → T1 only; - uncertain/high-value
expert → T1+T2.

Controls must include: - stock OLMoE; - fixed precision for every
activated expert; - random/heuristic precision allocation at matched
activation rate; - oracle precision allocation; - learned gating.

OLMoE should not be used to prove basic ternary viability; that remains
Track A's job. This avoids MoE routing becoming a confounder.

## B4 --- Realized efficiency

Report actual kernel executions, bytes moved, tokens/sec, latency
distribution and joules/token. Theoretical skipped planes are not
sufficient evidence.
