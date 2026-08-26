# H-RPM Framework Proposal — TSP/LRN Split (2026-08-25)

This note was created in response to user feedback that the current
RPM evidence is being conflated under two distinct hypotheses with
different empirical support.

## Background

The current RPM claim registry contains:
- `A-RP-001` (PROVISIONAL_FAIL): "T2 produces measurable improvement
  over the damaged FP16 base on the calibration suite."
- `A-RP-002` (CONFIRMED_PASS): "T2 is competitive with equal-storage
  non-ternary correction on the quality-per-bit / quality-per-compute
  Pareto frontier."
- `A-RP-003` (UNTESTED): "T2 transfers to held-out tasks."

`A-RP-002` in particular **conflates two distinct hypotheses** with
different empirical support:
1. **Ternary structure as a useful prior** (TSP): even an untrained
   ternary correction captures structural information that an
   equal-budget non-ternary correction does not.
2. **Training adds value beyond the structural prior** (LRN): the
   trained T2 ternary correction adds capability beyond what an
   untrained (random) ternary correction provides.

The empirical evidence for each axis is **different**:
- TSP is positive across multiple regimes (T2 wins on capability vs
  random_lora at AF2-D TWN damage, L15 Gaussian σ=0.5 and σ=1.0).
- LRN is positive only at AF2-D TWN damage (z-scores +19σ to +262σ on
  arc_easy/lambada) and **negative or null elsewhere** (L15 Gaussian
  σ=0.20: trained ≈ random; σ=0.50: trained ≈ random; σ=1.00:
  trained < random on arc_easy and lambada_openai at >2σ).

## Proposed claim split

### H-RPM-TSP — Ternary Structural Prior

**Statement**: The T2 ternary correction structure (2-bit packed
codes + per-row fp16 scale) is a useful structural prior that
captures information an equal-budget non-ternary or random
structural correction does not.

**Falsifier**: At any site/damage/seed combination, an equal-budget
non-ternary control (e.g., random LoRA, full FP16 dense_adapter at
matched storage) outperforms T2 on the primary capability metric by
≥2σ across n=3 seeds.

**Evidence (current)**:
- **AF2-D TWN damage** (Stage 1.5 D1'-D5'): T2 beats random_lora on
  ppl, arc_easy, lambada_openai at z-scores ≥+25σ (Stage 1.5 verdict).
- **L15 Gaussian σ=0.50** (Stage 2 v3): T2 vs random_lora on wikitext
  +3.16σ (T2 wins).
- **L15 Gaussian σ=1.00** (Stage 2 v4): T2 vs random_lora on wikitext
  +3.15σ and on lambada +2.87σ (T2 wins on 2 of 3 metrics at ≥+2σ).

**Status**: **PROMISING / LIKELY-PASSING in multiple regimes.** The
TSP axis has stronger evidence than the LRN axis.

### H-RPM-LRN — Training Adds Value Beyond Random Ternary

**Statement**: Training the T2 ternary correction adds capability
beyond what a random (untrained) T2 ternary correction provides at
the same structural budget.

**Falsifier**: At any site/damage/seed combination, random T2
outperforms trained T2 by ≥+2σ on ≥1 capability metric across n=3
seeds, OR trained T2 shows no separation from random T2 across all
regimes tested.

**Evidence (current)**:
- **AF2-D TWN damage** (Stage 1.5 D1'-D5'): T2 vs random_t2 z-scores
  +19σ to +262σ on arc_easy/lambada. **STRONG POSITIVE.**
- **AF2-D D1p held-out tasks** (Stage 4 EXP-RPM-T01): T2 vs random_t2
  on hellaswag/winogrande/boolq/openbookqa — **FAIL** (max z +0.29σ,
  trained ≈ random).
- **L15 Gaussian σ=0.20** (Stage 2 v2): T2 ≈ random T2 (max z +0.18σ).
  **NULL.**
- **L15 Gaussian σ=0.50** (Stage 2 v3): T2 ≈ random T2 (max z +1.20σ).
  **NULL.**
- **L15 Gaussian σ=1.00** (Stage 2 v4): T2 < random T2 on arc_easy
  (-2.15σ) and lambada_openai (-2.26σ); T2 = random T2 within
  ±0.005 on wikitext. **NEGATIVE on 2 of 3 metrics.**

**Status**: **REGIME-DEPENDENT / LIKELY-CONFIRMED only at AF2-D TWN
damage.** The LRN axis is strong at the original Stage 1 / 1.5 site
but fails or inverts at other tested sites.

## Implications for Track B

The user's framing — that the real question becomes
`P(T₂ helps | layer, damage, token, task)` — is correct. The
**scope of T2's utility is bounded**:

- **T2 has a niche** at AF2-D TWN damage (Stage 1 / 1.5 + Stage 5
  systems): both TSP and LRN positive; T2 wins on (B, F, O, M, L)
  cost vector with E-axis uncertainty.
- **T2 may generalize for TSP** (structural prior) but **not for LRN**
  (training) outside that niche.
- **Adaptive precision gating** could use T2 reliably in regimes
  where LRN is positive; in regimes where LRN is negative, the T2
  structure adds value (TSP) but training does not (LRN fails), so
  random_t2 would be sufficient.

This is a **richer finding** than "T2 is universally good" or "T2
universally fails": T2's value is **regime-conditional**.

## Proposal: how to restructure the registry

**Option 1**: Keep `A-RP-001/002/003` and add two new claims
`A-RP-TSP` and `A-RP-LRN`. The original `A-RP-002` becomes a
composite whose current PASS is qualified by "split into TSP and
LRN: TSP confirmed; LRN under-specified."

**Option 2**: Replace `A-RP-002` with `A-RP-TSP` and `A-RP-LRN`. The
original A-RP-002 evidence gets re-tagged:
- AF2 / AF2-R PASS evidence → A-RP-TSP (T2 vs dense_adapter on
  matched storage).
- AF2-D PASS+ evidence → A-RP-LRN (T2 vs random_t2 on damaged base).
- AF2-R T2 ≈ random_t2 → A-RP-LRN UNTESTED/NULL on calibrated base.

**Option 3**: Keep the existing claims as historical artifacts and
add a new top-level framework document (`H-RPM-FRAMEWORK.md`) that
defines TSP and LRN as orthogonal axes under A-RP-002. The original
A-RP-002 PASS gets a "superseded by TSP/LRN split" annotation.

## Required user direction

Which option should I implement? The choice has cascading effects on:
- Track B gating (which requires A-RP-002 status; if A-RP-002 is
  re-tagged, the gate's input changes)
- ROADMAP checkpoint entries
- Claim history annotations
- Verdict file revisions (e.g., Stage 1.5 verdict should cite TSP/LRN
  separately)

I will not implement any restructuring without explicit user
confirmation, because the choice has substantial downstream effects
on the research program's structure.

## Next experiment after the reframe

Per user recommendation: a **boundary-mapping experiment** that
identifies the regime (damage × layer × task × budget) where
trained-vs-random T2 (LRN axis) is positive. Candidate axes to sweep:
- Damage severity (already partially explored: D1-D5 TWN at AF2-D;
  σ=0.20, 0.50, 1.00 Gaussian at L15).
- Layer type (down_proj vs q_proj vs v_proj vs gate_proj).
- Layer depth (0, 8, 15).
- Task (capability vs held-out vs efficiency-sensitive).
- Correction budget (matched-bytes vs storage-optimal).

A **damage-severity sweep at AF2-D TWN** (re-measure the LRN axis at
multiple damage levels between D5 and D1) would be the cheapest
extension: it would identify whether LRN is positive only at
"intermediate" damage (D5 = severe but not catastrophic) or whether
LRN tracks damage monotonically. This would be the natural next
experiment IF the user agrees the framework reframe is needed.

If the user prefers to keep the existing A-RP-001/002/003 registry
and add TSP/LRN as **sub-axes** within A-RP-002 (Option 3 above), the
next-experiment decision can be made without claim-registry changes.
