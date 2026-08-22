# TORUS Evidence Update --- 2026-08-22

## Why v2 exists

The original feedback correctly identified that TORUS had integrated
multiple hypotheses faster than they were scientifically validated. New
tests now refine several conclusions.

## Residual-plane update

Earlier experiments suggested that additional residual planes were
ineffective. Subsequent work identified training/plumbing problems and
found that **sequentially trained correction planes can add substantial
capacity**.

Representative KL progression:

`~3.42 → ~2.66 → ~1.72 → ~1.60`

for the relevant untrained/plane-1/plane-2/plane-3 progression.

The second plane is therefore meaningful. The third plane shows much
smaller marginal value. This supersedes any blanket statement that
residual planes do not work.

However, downstream quality did not recover proportionally. ARC-E
improved modestly and LAMBADA remained extremely poor in the relevant
TORUS student experiments. The project must therefore distinguish
representation/correction capacity from useful language capability.

## Functional correction insight

Later trained planes improved teacher behavior even when direct
reconstruction of the original floating-point weights did not improve.
TORUS v2 therefore treats trained residual planes as **functional
correction planes**, not merely successive matrix-reconstruction
residuals.

## Hadamard/Bonsai update

Post-hoc Hadamard rotation of an already trained ternary Bonsai model
creates a quality penalty. Recovery by KD improved the rotated model but
saturated below the stock model at the tested budget. This path is
currently **NO-SHIP**.

At the same time, controlled rotated-vs-unrotated training produced
evidence of improved optimization behavior/conditioning. The surviving
hypothesis is therefore **native Hadamard ternary training**, tested
from initialization or early training on a small model before scaling.

## Provenance update

Concurrent experiment sessions contaminated shared temporary artifacts
and caused at least one incorrect interim verdict. Clean reruns
corrected the record. TORUS v2 makes immutable per-run provenance and
clean reproduction mandatory scientific gates.

## OLMoE role

AI2/AllenAI `OLMoE-1B-7B-0125` remains a valuable TORUS resource, but
its role is sharpened: it is the preferred Track-B platform for testing
whether MoE router information can guide adaptive precision. It should
not be used to establish basic ternary viability.

## Current steering conclusion

Do not return to the old monolithic roadmap. The strongest next
directions are:

1.  reproduce and characterize T1/T2 functional correction;
2.  test heterogeneous layer precision;
3.  test native Hadamard ternary training on a small controlled model;
4.  begin oracle gating only where T2 has demonstrated task-relevant
    value;
5.  use OLMoE for expert-routing × precision-routing research after the
    dense representation is stable;
6.  benchmark recursive context independently.
