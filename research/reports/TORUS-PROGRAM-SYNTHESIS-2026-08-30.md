# TORUS Program-Level Research Synthesis — 2026-08-30

**Status:** the consolidation document that determines Hypothesis 2.
**Program checkpoint:** `1b271df` (residual-plane closure). **All
experimentation is FROZEN at this checkpoint** until a Hypothesis 2
is selected from §5 — this document preselects nothing.

---

## 1. What TORUS has established

### 1.1 Confirmed phenomena (positive science, frozen evidence)

| phenomenon | evidence | claim state |
|---|---|---|
| Learned correction beyond the random structural prior (LRN) | AF2-D: −226.9σ ppl / +25.1σ arc / +116.8σ lambada; band = full TWN range {0.6–1.0}, two seed sets, z to +1687σ | A-RP-LRN: CONFIRMED_AT_AF2D_TWN_BAND |
| Held-out-task learning at catastrophic damage | D5p hellaswag +21.76σ vs random T2 (base 0.426 → trained 0.585; random +0.008) | A-RP-LRN annotation (T02) |
| Ternary structural prior carries information | Stage 2 v6/v7 TSP axis, +52 to +279σ vs random LoRA | A-RP-TSP confirmed at the band |
| Recipe robustness | init σ ∈ {0..1e-2} × 3 seeds; windows {16,128,256}; wikitext↔openwebtext transfer; third reference reproduction | A-RP-002 robustness annotations |
| **Layer-0 damage locus / depth gradient** | catastrophic damage exists ONLY at layer 0 (L0-down TWN, L0-v Gaussian); deep layers robust to every tested mechanism (239-cell sweep + Stage 2 v1/v5) | discovery record |
| Mechanism specificity | T2 recovers TWN damage, ANTI-recovers Gaussian at matched magnitude | Stage 3 v1 |

### 1.2 Falsified architectural assumptions (negative architecture, decisive)

| assumption | falsifier |
|---|---|
| Correction beats continued training at equal budget | A-RP-001 CONFIRMED_FAIL (pristine start); EXP-AF-001-D (damaged start: arc −10.9σ, lambada −6.9σ) |
| Sequential curriculum is the right training structure | A-RP-003 CONFIRMED_FAIL (joint wins) |
| Ternary correction beats equal-storage alternatives | EXP-RPM-T02 r3: INT8 wins 4/4 held-out tasks |
| The correction site generalizes | EXP-RPM-SITE-DISCOVERY: NO_SECOND_SITE (0/13 candidates; 9 informative-but-mild) |
| Cross-site learning (LRN) at other categories | Stage 2 v2–v4: null at L15 Gaussian, inverted at σ=1.0 |
| Cross-mechanism recovery | Stage 3 v1: Gaussian anti-recovery |
| Hadamard rotation improves downstream | EXP-A-H1: train loss better, frozen ppl bar missed (ratio 1.045); line closed with H-POST |
| Held-out tasks are insensitive to damage | T02-PROBE: they degrade at every TWN severity; T01's null was a regime miscalibration |

### 1.3 Instrumentation discoveries (the program's actual assets)

- **The INVALID-vs-FAIL governance worked as designed**, catching
  five driver defects in A-H1 (unsaved weights, scale-feedback
  divergence, tie-overwrite, buffer-suffix substring match,
  zero-gradient STE) and one gate miscalibration in T02-PROBE that
  exposed T01's regime mismatch — each caught by an instrument
  built for that exact class (parity gate, materialize cross-check,
  conditioning proxy, grad probe, verification gate).
- **Instrument calibration findings:** lm-eval `wikitext` is
  doc-level word ppl (wikitext-2), not token-level wikitext-103
  (AF6); every "ppl" in the program is instrument-annotated.
- **Bit-reproducible damage + frozen reference cells** as
  environment pins (discovery refs bit-exact: 439.2520 / 429.55;
  probe D5p map bit-identical across runs).
- **Run hygiene:** namespaces, sha256-pinned adapters, per-run
  provenance, budget caps with kill criteria.

### 1.4 Unresolved questions

1. **Can ANY ternary representation beat INT8/INT4 at matched
   quality-cost?** Never tested as a primary question — the program
   tested the residual-plane RECIPE, not ternary-per-se. (A-H1
   tested Hadamard-rotated ternary from scratch and failed its ppl
   bar; that falsifies a recipe, not the class.)
2. **Heterogeneous layer precision** (Track A4): registered, never
   executed — but the depth gradient is direct motivating evidence:
   layers differ in damage tolerance by orders of magnitude, so a
   uniform-precision policy is provably suboptimal for the observed
   model.
3. **Track C economics** (recursive context): independent, alive.
4. The reframed question (§4).

---

## 2. Properties the next architecture must possess

Turned from the negative evidence into design constraints. A
Hypothesis-2 architecture must, at DISCOVERY tier (per the new
gating order):

- **C1 — beat INT8 at matched storage on held-out tasks**
  (falsified by T02 r3 for T2; the bar is now the comparator).
- **C2 — beat equal-budget continuation at matched training
  budget** (A-RP-001, AF-001-D).
- **C3 — transfer downstream early**: capability metrics from the
  first discovery cells; proxy metrics (KL, train loss,
  recovery-vs-random, conditioning) are diagnostics only.
- **C4 — evidence beyond a single layer/site** OR an architecture
  that exploits the depth gradient as a design input (e.g.
  heterogeneous precision quantizes deep layers hard and protects
  layer 0 — the NO_SECOND_SITE negative inverted into a feature).
- **C5 — joint training structure** (A-RP-003: no sequential
  curriculum).
- **C6 — physical accounting**: packing + metadata, never
  "1.58 bits/weight" without both.
- **C7 — a preregistered falsification at discovery tier** with
  frozen bars against C1 and C2; a capability-bar FAIL closes the
  line with no rescue (suite doc §15 + 2026-08-30 addendum).

---

## 3. Surviving independent ideas (from the original hypothesis space)

The original TORUS thesis had three pillars: residual ternary
planes, adaptive gating, recursive context-as-variable.

| idea | status | reason |
|---|---|---|
| Residual ternary planes | **CLOSED** | mechanism confirmed, architecture not supported; reopens only on genuinely new external evidence |
| Adaptive gating (Track B) | **FROZEN** | gating value depends on a competitive correction mechanism; conditions 3+4 blocked with definitive evidence |
| Hadamard-rotated training | **CLOSED** | H-POST (CP3.2) |
| Sequential curriculum | **CLOSED** | A-RP-003 |
| **Ternary/heterogeneous low-bit representation per se** | **OPEN** | never tested head-to-head vs INT8/INT4 as a primary question; the depth gradient motivates heterogeneous precision (A4) |
| **Track C recursive context** | **ALIVE** | independent of the ternary mechanism; graded B; needs benchmark quality |
| **The harness/governance machinery** | **ALIVE** | claim registry, INVALID-vs-FAIL, frozen instruments, run hygiene — the program's transferable asset |

Nothing survives merely because it was part of the original design.

---

## 4. The reframing

TORUS began as an investigation of a proposed ternary architecture.
It now has the experimental infrastructure and the negative
evidence to become a broader question:

> **What is the minimum additional information and training
> mechanism required for a ternary model to retain useful
> downstream capability while producing a genuine storage/compute
> advantage over conventional low-bit alternatives?**

Under this framing, the candidate solution space is a flat list —
no mechanism inherits priority:

1. INT8 / INT4 quantization (the baselines to beat, now also
   solution candidates in their own right),
2. base retraining / QAT-style continuation,
3. learned corrections of a different form than T2,
4. heterogeneous layer precision (A4),
5. ternary weights with alternative training structures,
6. future mechanisms yet to be proposed.

Each candidate is a competing answer to the SAME measurable
problem, tested under the SAME discovery-tier protocol: mechanism
signal → capability check → competitive baseline → robustness →
scale.

---

## 5. How Hypothesis 2 is determined

This document does not select Hypothesis 2. It fixes the selection
procedure:

1. **Candidate table.** Each §4 candidate is scored against §2
   constraints C1–C7 *on existing evidence only* — no new compute.
   A candidate that already violates C1/C2 on frozen evidence is
   out.
2. **One primary hypothesis.** Hypothesis 2 = one mechanism's
   answer to the §4 question, with a preregistered manifest whose
   DISCOVERY tier includes, in the first cells: (a) a capability
   check, (b) the C1 matched-storage baseline, (c) the C2
   continuation baseline — all with frozen bars.
3. **The user selects.** The selection decision belongs to the
   user, informed by this synthesis. The document's job is to make
   the decision space complete and the noncompetitive branches
   already closed.

**Experimentation remains frozen** at `1b271df` until that
selection exists and its manifest is preregistered.
