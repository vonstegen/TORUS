# Cross-Program Synthesis — optimization/representation ≠ downstream capability

**Date:** 2026-08-30
**Status:** documented hypothesis for future TORUS designs. Not a
registered claim; no new experiment authorized by this document.

---

## The pattern

Three independent experimental branches produced the same
separation: **proxies improve, downstream capability does not
follow.** Each occurrence was measured with frozen instruments and
recorded at decision time — none of the three is a post-hoc
reading.

### Occurrence 1 — KL gains without proportional task recovery (Phase 1 / Track A2)

EXP-A-010 and the Phase-1 distillation runs: teacher/logit KL
improved monotonically (3.42 → 2.66 → 1.72 → 1.60 across planes)
while downstream acceptance was never established — the KL axis
was diagnostic-only and the eventual downstream work (AF1/AF2)
found the representation gains non-competitive at matched cost.
The program's response at the time was the metric-class rule
(proxy metrics never decide acceptance) — the first
institutionalization of this pattern.

### Occurrence 2 — Hadamard training gains vs downstream primary bar (EXP-A-H1, 2026-08-30)

Native-Hadamard rotated ternary training: the rotated arm trained
BETTER in-domain (train loss 7.231 vs 7.280; closed its 0.49-nat
step-0 PTQ deficit) and won arc_easy (+0.017) — but missed the
frozen PRIMARY bar: wikitext ppl ratio 1.045 > 0.97. The
train/downstream disagreement was recorded without rescue and the
Hadamard line closed with H-POST per CP3.2.

### Occurrence 3 — T2 correction strength vs competitive alternatives (T02 + AF-001-D, 2026-08-30)

Trained T2 produces some of the strongest recovery metrics in the
program: 20.3× ppl recovery at AF2-D, LRN z-scores to +1687σ,
held-out hellaswag +21.76σ vs the random structural prior. Against
the baselines that matter, it loses: INT8_residual beats it on all
four held-out tasks at matched storage (T02 r3: 0/4), and the
equal-budget whole-model continuation beats it on capability
(arc −10.9σ, lambada −6.9σ, AF-001-D).

## What the three share

In every case the failing comparison is the same shape: a
mechanism was validated on **internal or self-referential
metrics** (KL, training loss, recovery-vs-random, code-flip rates,
conditioning) before being tested against the **capability
baselines that define actual value** (task accuracy/perplexity at
matched storage, equal-budget continuation). The internal metrics
were real — none were artifacts — but they did not rank
architectures.

## The documented hypothesis (for future TORUS designs)

> Optimization success, representation alignment, and recovery
> strength are necessary but NOT sufficient indicators of a
> competitive precision mechanism. A mechanism that wins its
> internal comparisons can still lose to the best matched
> practical baseline on downstream capability.

Corollary constraints:

1. A new mechanism must face its competitive baselines (best
   equal-storage correction, equal-budget continuation) **early**,
   at discovery tier, with frozen bars.
2. Capability metrics decide; internal metrics are diagnostics
   regardless of their z-scores.
3. A frozen FAIL on a capability bar closes the line — no rescue
   via additional complexity (suite doc §15).

## Gating-order change (recorded)

The suite doc §15 acceptance bar now carries the 2026-08-30
addendum replacing the previous long-sequence gating philosophy:

**mechanism signal → capability check → competitive baseline →
robustness → scale.**

Under the old order (robustness/scale work preceded competitive
baselines), branches could accumulate large positive internal
evidence before their decisive negative result — the T2 branch is
the worked example: four INVALID executions, a 239-cell site
sweep, and a 32-cell held-out tournament were spent before the
two competitive baselines (INT8, continuation) delivered the
verdict in two cheap runs.

## Non-goals

This synthesis authorizes NO new experiment and NO claim state
change. It is the cross-program record the closure refers to; the
next TORUS hypothesis is chosen only after program-level
consolidation per the closure record
(`research/reports/RESIDUAL-PLANE-CLOSURE-2026-08-30.md`).
