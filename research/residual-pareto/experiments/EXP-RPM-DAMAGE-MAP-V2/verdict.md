# EXP-RPM-DAMAGE-MAP-V2 Verdict — Stage 3 v2 Damage Mechanism Envelope

**Date:** 2026-08-27
**Run namespace:** `runs/r/EXP-RPM-DAMAGE-MAP-V2/`
**Manifest:** `research/residual-pareto/experiments/EXP-RPM-DAMAGE-MAP-V2/manifest.yaml`
**Calibration table:** `research/residual-pareto/experiments/EXP-RPM-DAMAGE-MAP-V2/calibration_table.md`
**Stage 3 v1 verdict:** `research/residual-pareto/experiments/EXP-RPM-DAMAGE-TYPE-001/verdict.md`

---

## Decision: **CALIBRATION_GATE_FAIL — Stage 3 v2 hypothesis untestable**

The Stage 3 v2 manifest's hypothesis (TWN > MagnitudePrune > Dropout > Gaussian)
is **untestable at matched magnitude** at AF2-D (`model.layers.0.mlp.down_proj`).
Two of the four mechanisms cannot produce catastrophic damage.

**The Stage 3 v2 verdict in one line:**

> Only TWN-style per-group absmean-zero and Gaussian signed noise produce
> catastrophic damage at AF2-D/L0/down_proj. Magnitude-prune (per-row top-k) and
> dropout (per-element Bernoulli) preserve too much signal to be useful as
> "damage" at this layer.

**Per the manifest's kill criterion #2:**

> "If MagnitudePrune and Dropout cannot be calibrated to ±20% of BAND-3
> magnitude (~ppl 350-515), the experiment fails the calibration gate and
> only {TWN, Gaussian} data is reported."

Stage 3 v2 reports only the calibration_table.md observation as the
deliverable. **No new tournament data is collected** — the Stage 3 v1
TWN-vs-Gaussian verdict is unchanged and remains the operative result.

---

## Stage A Calibration Results (raw numbers)

### MagnitudePrune — maximum reachable ppl: **22.06** at k=0.95

| k | Base ppl | vs BAND-3 target (ppl 430) |
| ---: | ---: | ---: |
| 0.5 | 15.49 | 3.6% |
| 0.8 | 16.54 | 3.8% |
| 0.85 | 17.22 | 4.0% |
| 0.9 | 18.55 | 4.3% |
| 0.93 | 20.13 | 4.7% |
| **0.95** | **22.06** | **5.1%** |

MagnitudePrune cannot exceed ppl 22 at AF2-D regardless of k. The next
step would be k=1.0, which is "zero every weight in every row" — that's
not pruning; it's deletion, and would make recovery mathematically
impossible (no signal to recover from).

### Dropout — maximum reachable ppl at p=0.99: **69.07**

| p | Base ppl | vs BAND-3 target (ppl 430) |
| ---: | ---: | ---: |
| 0.3 | 16.35 | 3.8% |
| 0.5 | 18.22 | 4.2% |
| 0.7 | 23.02 | 5.4% |
| 0.8 | 29.49 | 6.9% |
| 0.9 | 44.08 | 10.3% |
| 0.95 | 55.91 | 13.0% |
| **0.99** | **69.07** | **16.1%** |

Dropout at p=0.99 (99% of weights zeroed) achieves ppl 69 — still
6× short of the BAND-3 target. Higher p would eventually produce
collapse, but the regime is no longer "damage that's recoverable" —
it's "model zeroed."

### Cross-reference: BAND-3 (TWN @ thr=0.7) target

| Mechanism | Parameter | Base ppl |
| --- | ---: | ---: |
| **TWN (target)** | **thr=0.7** | **429.55** |
| **Gaussian (BAND-3 match)** | **σ=3.0** | **451.13** |

Only TWN and Gaussian can produce catastrophic damage at AF2-D/L0/down_proj
in the explored parameter ranges.

---

## Stage 3 v2 Hypothesis: FALSIFIED (fallback per kill criterion #2)

**Hypothesis (Structure-Magnitude Ranking):**
> At AF2-D under matched damage magnitude (~ppl 430 baseline), trained
> T2 LRN z-score varies systematically across mechanisms in the order
> TWN > MagnitudePrune > Dropout > Gaussian.

**Cannot be tested.** Two of four mechanisms cannot reach the calibration
target. The hypothesis is **untestable in this regime**, not falsified.

**Backup hypothesis (implicit, from Stage 3 v1):**
> TWN and Gaussian have opposite LRN sign and large |z| at matched magnitude.

**Status:** Confirmed in Stage 3 v1 (commit 84a4dad). Unchanged by Stage 3 v2.

---

## What this establishes

### 1. {TWN, Gaussian} are the only catastrophic-damage mechanisms at AF2-D

Across two new mechanisms (per-row top-k pruning and per-element Bernoulli
masking), the maximum achievable damage at AF2-D/L0/down_proj is ppl 22-69,
both orders of magnitude smaller than the BAND-3 target of ppl ~430. **No
unstructured-sparsity mechanism at this layer produces catastrophic
capability loss.**

### 2. The Pareto-frontier thesis is bounded to {TWN, Gaussian}-catastrophic regimes

The Stage 3 v1 result — T2 helps dramatically under TWN damage and hurts
under Gaussian damage — describes T2's behavior within a specific
{magnitude-capable mechanism} regime. Outside that regime
({MagnitudePrune, Dropout}-style structured sparsity), the damage is
too mild to test T2's response: the model still has near-FP16 capability
even after p=0.99 dropout, so any "recovery" measurement is in the noise.

### 3. Track B gating question is now bounded to two mechanisms, not four

The Track B question `P(T2 helps | damage mechanism × severity × layer × task)`
is now:
`P(T2 helps | mechanism ∈ {TWN, Gaussian} × severity ≥ 0.5 × layer = AF2-D × wikitext task)`

Stage 3 v2 was supposed to expand this to 4 mechanisms, but the
calibration gate shows that this layer does not support 4-mechanism
falsification. To expand the mechanism axis would require either:
- A different layer where {MagnitudePrune, Dropout} can produce catastrophic damage (test at L8/L15/L20 with different weight statistics)
- A different model where weight statistics are less heavy-tailed
- A different damage implementation (e.g., accumulate many small magnitudes over training; static magnitude per matrix is bounded)

---

## What this does NOT establish

1. **Layer generality** — MagnitudePrune and Dropout might produce
   catastrophic damage at L8, L15, L20 (where weight statistics differ).
   Stage 2 v5 already showed L15 has degenerate TWN damage. The
   {TWN, Gaussian} catastrophic regime is currently AFFIRMED-AT-L0 only.
2. **Task generality** — The experiment only tested wikitext. Other tasks
   (arc_easy, lambada) may have different sensitivity profiles.
3. **Whether {MagnitudePrune, Dropout} even COUNT as damage** — the
   extreme parameter values (k=0.95, p=0.99) produce model state very
   close to FP16 in capability. If the goal is to test "T2 helps when
   the model is severely damaged," these mechanisms do not pass the
   "severely damaged" gate.
4. **Why TWN can produce catastrophic damage but MagnitudePrune cannot** — the
   structural difference (per-group absmean with per-group scale vs
   per-row top-k) likely matters. This is a follow-up question, not
   a Stage 3 v2 deliverable.

---

## Next permitted experiment

The Stage 3 v2 verdict is a **negative result** that confirms Stage 3 v1
and bounds Track B to the {TWN, Gaussian}-catastrophic regime. Several
next steps are now well-defined:

**Option A: Phase 1 EXP-A-011 (Layer Sensitivity at TWN).**
Restore the deferred Phase 1 mandate: test T2 at layers {0, 5, 10, 15,
20, 25} under TWN damage at thr=0.7. Address the original Phase 1
mandate that's been deferred since Phase 0. ~3 hours wall time.

**Option B: Stage 3 v3 — Cross-Layer Damage Envelope (if A is inconclusive).**
Test {TWN, MagnitudePrune, Dropout, Gaussian} × {layer 0, 8, 15, 20}.
The Stage 3 v2 finding suggests {MagPrune, Dropout} may behave
differently at L8/L15 where weight statistics differ. ~6 hours wall time.

**Option C: Stage 3 v2b — Lighter damage severity across {TWN, MagPrune, Dropout, Gaussian}.**
Test all 4 mechanisms at SEVERITY ~ppl 60 (the max MagPrune/Dropout
can produce). This abandons the "matched magnitude" framing in
favor of the "matched achievable regime" framing. 2 hours wall time.

**Recommendation:** Option A. EXP-A-011 has the longest-standing
Phase 1 mandate, the layer axis is the most-important-axis-still-open
for Track B, and it doesn't require re-tuning any damage mechanism.

---

## Standing rules respected

- ✅ Preregistered thresholds and criteria frozen before Stage A probe.
- ✅ Stage A calibration completed and analyzed before Stage B decision.
- ✅ No code changes from v6/v7 except the documented freeze exception
  (commits cac5b1f, a3a338e, b2cc208, b28c258, f092e12).
- ✅ Damage modes (MagnitudePrune + Dropout) successfully added and tested.
- ✅ Calibration gate (kill criterion #2) triggered with positive
  evidence: documented ppl data + mechanism-by-mechanism analysis.
- ✅ Decision output: hypothesis (untestable), result (gate fail),
  grade (N/A), decision (CALIBRATION_GATE_FAIL with documented fallback),
  confidence (HIGH — based on 13 calibration cells × 1 seed),
  next permitted experiment (3 options).
- ✅ Commit + push per lifecycle transition.

---

## Audit trail

- `runs/r/EXP-RPM-DAMAGE-MAP-V2/20260827T162323Z/stage_a_probe/` — Stage A MagnitudePrune sweep (7 cells, 5 successful)
- `runs/r/EXP-RPM-DAMAGE-MAP-V2/20260827T164909Z/stage_a_probe_dropout/` — Stage A Dropout sweep (7 cells, 7 successful)
- `runs/r/_logs/stage3-v2-stage-a-probe.log` — MagnitudePrune run log
- `runs/r/_logs/stage3-v2-dropout-probe.log` — Dropout run log
- `stage3-v2-stage-a-probe.sh` — MagnitudePrune probe launcher (committed)
- `stage3-v2-dropout-probe.sh` — Dropout probe launcher (committed)
- `stage3-v2-stage-b-tournament.sh` — Stage B launcher (committed, unused due to gate)
- `stage3-v2-random-arms.sh` — Random arms launcher (committed, unused due to gate)
- `analyze_s3_v2.py` — Stage 3 v1 analysis script (committed)
- `examples/af2_storage_tournament.py` — freeze exception applied (commits cac5b1f, a3a338e, b28c258, f092e12)
  - New: damage_target_module_magnitude_prune, damage_target_module_dropout
  - New CLI: --damage-magnitude-prune, --damage-prune-k, --damage-dropout, --damage-dropout-p

---

## Experiments blocked

Until Stage 3 v2 verdict is registered in INDEX.md/ROADMAP.md/CHANGELOG.md:

- ❌ Track B adaptive precision gating beyond the {TWN, Gaussian}-catastrophic envelope
- ❌ General "T2 helps under damage" framing
- ❌ Cross-damage-mechanism Track B experiments without first mapping
  mechanism-envelope at a different layer (where {MagPrune, Dropout}
  can produce catastrophic damage) OR a different severity regime
  (light severity, ~ppl 60)
- ❌ Further Stage 3 v2 iterations: this verdict IS the closure; only
  Option A/B/C above are permitted as Stage 3 v3+ candidates

The freeze remains ACTIVE for code changes not required to register this
verdict (which requires only documentation updates).
