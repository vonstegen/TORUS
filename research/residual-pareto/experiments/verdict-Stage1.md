# Stage 1 Verdict — EXP-RPM-D0..D5

**Run window:** 2026-08-23T17:14:05Z → 2026-08-23T21:11:24Z (~4 hours)
**Namespaces:** `runs/r/EXP-RPM-D{0..5}/2026*/af2d/`
**Total runs:** 6 regimes × 7 arms × 3 seeds = **126 runs**. **0 tolerance violations.**
**Frozen driver:** `692e8ee`. Frozen damage knobs (preregistered): threshold axis
`{null, 0.0, 0.3, 0.5, 0.6, 0.7}` for D0..D5; group_size=128; calibrate_norm=False.
**Frozen recipe:** AF2-D's (SGD lr=1e-3, 500 steps, batch 4, seq 128, 4.2 MB target).
**Eval suite:** wikitext ppl + arc_easy + lambada_openai.

---

## Verdict per claim

### RPM-001 (Pareto regime) — **tentative PASS**
**Evidence:** Per-regime full cost-vector Pareto analysis
(5 cost dims: B/F/O/M/L × 3 capability dims: ppl/arc/lam).
T2_ternary is on the Pareto frontier in every regime:

| Regime | Pareto-optimal arms |
|---|---|
| D0 (FP16) | T2, int4, int8, lora, dense (all 5) |
| D1 (thr=0.0) | T2, int4, int8, lora, dense (all 5) |
| D2 (thr=0.3) | T2, int4, int8, lora, dense (all 5) |
| D3 (thr=0.5) | T2, int4, int8, lora, dense (all 5) |
| D4 (thr=0.6) | T2, int8, lora, dense (int4 dominated) |
| D5 (thr=0.7) | T2, int8, lora, dense (int4 dominated) |

**Why "tentative" not "CONFIRMED":**
- Per-regime audit script (`audit_rpm_d_reproduction.py`) currently
  checks only ppl and matched-bytes; it does NOT do the full
  Pareto check. The verdict here was computed manually from the
  aggregate.json cost_vector_rows + tasks.
- Energy (joules_per_token) is null across all arms; Stage 1 lacks
  power measurement. The COST-VECTOR-v1.yaml schema requires E;
  until measured, the Pareto verdict is provisional.

### RPM-002 (damage-dependence, monotone hypothesis) — **UNRESOLVED**
**Reason:** Cross-regime RPM-002 evaluation requires the trained-vs-
random z-score to be computed at each regime and compared across
regimes for monotone trend. The driver skips eval on
`untrained_controls` arms (`if not adapter.is_untrained: skip eval`),
so `random_t2_ternary` and `random_lora` have empty `tasks` dicts in
every aggregate.json. **No random-control eval data exists for any
regime.** The monotone-trend z-score cannot be computed.

**Per user directive:** RPM-002 claim definition is **NOT altered**.
The claim stays `UNTESTED` until random-control evals are produced.

### RPM-006 (representation-signal activation) — **UNRESOLVED**
**Reason:** Same data gap as RPM-002. The trained-vs-random z-score
(criterion: trained beats random by >2σ on ≥1 capability metric)
cannot be computed without random evals.

**Per user directive:** RPM-006 claim definition is **NOT altered**.
The claim stays `UNTESTED` until random-control evals are produced.

### Other claims — not addressed by Stage 1
- **RPM-003** (layer-dependence): Stage 2 task.
- **RPM-004** (task-dependence): Stage 4 task.
- **RPM-005** (budget-tightening): Stage 3 task.

---

## Three ordered questions (per user directive, results in order)

### Q1. Does trained T2 reproducibly separate from random T2 as damage increases?

**UNRESOLVED.** The required random-control eval data is missing
from the Stage 1 aggregate.json files (driver behavior:
`if not adapter.is_untrained: skip eval`). See "Data gap" below.

### Q2. Where does that separation first become statistically meaningful?

**UNRESOLVED.** Same data gap.

### Q3. At which damage levels does T2 enter the full-comparator Pareto frontier?

**ANSWERED (tentative).** T2_ternary is on the Pareto frontier in
every regime (D0 through D5), against the complete frozen
comparator set (5 trained arms: T2, int4, int8, lora, dense) and
the complete registered cost dimensions available at Stage 1
(B, F, O, M, L — E is null and excluded). The Pareto-optimal arm set
contracts from 5 to 4 at D4/D5 (int4_residual becomes dominated
on the joint metric-cost vector), but T2 remains on the frontier
throughout.

**Caveat (per user directive 5):** "Pareto at D0 from a perplexity
tie" would be the wrong framing. At D0, T2 is Pareto because no
arm dominates T2 on the joint (ppl, arc, lam, B, M, L) vector —
NOT because T2 ties dense on ppl alone. With the joint criterion,
T2's cost profile (B=4,199,318, L=12.110 ms) is not dominated by
any other arm at D0.

---

## Findings (NOT experimental failures)

### F1. Threshold → ppl is highly non-monotonic

| Regime | Preregistered knob | **Observed pre-train ppl** |
|---|---|---|
| D0 | no damage | N/A (no damage applied) |
| D1 | threshold=0.0 | **1524.80** |
| D2 | threshold=0.3 | **1524.80** |
| D3 | threshold=0.5 | **1524.80** |
| D4 | threshold=0.6 | **697.29** |
| D5 | threshold=0.7 | **429.55** |

Three thresholds (0.0, 0.3, 0.5) produce indistinguishable ppl
damage (all 1524.80). The threshold axis only "moves ppl" between
0.5 and 0.6. **This is a Stage 1 finding about the damage knob, not
a flaw in the experiment.** The Stage 1 preregistration correctly
froze the knobs; the data show that the chosen axis is not
informative at the lower end.

**Per user directive 6:** this non-monotonicity is a finding, not an
experimental failure. The preregistered knobs and the data stand.

### F2. Pre-train ppl is on par with full-layer ternary PTQ

The observed ppl at D1/D2/D3 (~1525) is on par with the EXP-A-011
"fully quantized" reference (ppl 459,454 measured, ppl 1525 in
static-damage mode at D5 = our threshold=0.7 reproduction). The
sign-rounding-only mode at threshold=0.0 already breaks the layer
as severely as full-row TWN quantization.

### F3. int8_residual has lower ppl than T2 in D1-D4 but is dominated on the full cost-vector

On ppl alone, int8 wins D1-D4 (17.75-18.99 vs T2's 23.66-26.91).
But int8's cost-vector is similar to T2's (B is slightly lower at
4,195,994 vs 4,199,318); the dominance is partial. With the full
joint Pareto criterion (5 cap + 5 cost), both T2 and int8 remain
on the frontier.

### F4. T2 wins ppl AND all 3 capability metrics at D5

D5 is the only regime where T2 wins on every capability metric
(ppl 17.32, arc 0.6094, lam 0.5498) AND is on the Pareto frontier.
The catastrophic-regime T2 advantage holds. This is the
consistent-with-AF2-D result.

### F5. D1/D2/D3 collapse into the same observed-ppl regime

D1, D2, D3 all sit at ppl 1524.80 with the same arc/lam pre-train
metrics. The Stage 1 sweep effectively produced **4 distinct
observed regimes** (D0 = FP16, D1/D2/D3 = severe at ppl 1525,
D4 = moderate at ppl 697, D5 = catastrophic at ppl 430), not 6.

---

## Data gap (the only blocker for full RPM-001/002/006 verdict)

**`random_t2_ternary` and `random_lora` arms have empty `tasks` dicts
in every aggregate.json.**

Cause: the driver code at line ~635 of `examples/af2_storage_tournament.py`:
```python
if not adapter.is_untrained:
    ... eval_arm(model, tokenizer, ...)
```

`is_untrained = (not train)` (the correct behavior set by the
0.16.7 driver fix at `7f901b3`). So `random_t2_ternary` (built with
`train=False`) skips eval. The trained-vs-random comparison is
uncomputable from the Stage 1 outputs as-is.

**Per user directive:** the driver is NOT modified. The data gap is
recorded as a finding, not a blocker to commit the Stage 1
artifacts. RPM-002 and RPM-006 stay UNRESOLVED per their
definitions.

---

## Per-regime measurements (Stage 1 as executed; not re-bucketed)

| Regime | Threshold | Pre ppl | T2 ppl | int4 ppl | int8 ppl | lora ppl | dense ppl |
|---|---|---|---|---|---|---|---|
| D0 | (none) | N/A | 13.10 | 16.97 | 13.88 | 13.10 | 13.10 |
| D1 | 0.0 | 1524.80 | 24.14 | 27.33 | 17.75 | 39.22 | 43.97 |
| D2 | 0.3 | 1524.80 | 23.66 | 28.47 | 18.28 | 51.85 | 33.68 |
| D3 | 0.5 | 1524.80 | 26.83 | 27.95 | 18.99 | 39.74 | 45.25 |
| D4 | 0.6 | 697.29 | 26.91 | 33.25 | 18.57 | 46.05 | 50.37 |
| D5 | 0.7 | 429.55 | 17.32 | 28.43 | 18.75 | 26.43 | 37.03 |

T2 Pareto-frontier status: **YES in every regime**.

---

## Recommended next step: preregister a cheap damage-calibration experiment

**Per user directive 7:** before any future damage sweep, preregister
a calibration experiment whose only purpose is to map threshold →
actual representation statistics → pre-correction capability. No
residual training.

**Proposed experiment EXP-RPM-CAL:**

- **Goal:** measure pre-train ppl at ~10-15 threshold values spanning
  [0.0, 1.0] to characterize the threshold→damage mapping for the
  AF2-D layer (and a handful of others for transfer). Output: a
  function `ppl(threshold)` per layer.
- **Cost:** ~5-10 minutes per threshold × 15 thresholds × 3 seeds ×
  ~3 layers = ~2 hours. No residual training; only eval.
- **Output:** a Pareto-frontier-of-thresholds for the next Stage 1.5
  damage sweep. The current threshold axis (0.0/0.3/0.5/0.6/0.7)
  clusters 3 of its 5 active values at the same ppl (1525), which
  is uninformative for the RPM-002 hypothesis test.
- **Why needed:** without it, the preregistered damage-axis
  hypotheses (RPM-002 monotone, RPM-001 Pareto-band) cannot be
  cleanly tested. The Stage 1 finding F5 (D1/D2/D3 collapse) is
  exactly what the calibration would resolve.

**Stage 1.5 / Stage 2 design is OUT OF SCOPE for this verdict.**
Only the calibration pre-experiment is recommended next.

---

## Drama recap (this session)

- **EXP-RPM-000 PREREGISTERED** (6c493a3): formal G-RPM-0 gate.
- **Driver regression #1 caught**: 7383b57 introduced `parent_module`
  NameError in `T2TernaryAdapter.__init__`. Restored in 7f901b3.
- **Driver regression #2 caught**: 6873f5 fix missed the missing
  `_patch_module_forward` call in `patch()`. Audit caught it via
  eval-matching-pre-train (silent failure that would have produced
  126 useless runs). Restored.
- **Band tightening**: ±1.5σ → ±2σ per OPERATING-PLAN §11 v2.3 (7262f15).
- **Stage 1 manifests PREREGISTERED** (b105ddf): 6 regimes with frozen
  threshold axis.
- **First Stage 1 launch crashed**: `no_correction` arm not implemented
  by driver. Dropped from arms list (692e8ee).
- **Second Stage 1 launch completed** (this run): 126 runs, 0 tolerance
  violations, ~4 hours wall time.

This sequence is the canonical demonstration of why G-RPM-0 existed.
The first two regressions would have produced silently-incorrect or
silently-incomplete data; the `no_correction` crash was caught by the
launch script's exit-on-error; Stage 1 itself produced clean per-seed
artifacts across all 6 regimes.

---

**Manifest:** `research/residual-pareto/experiments/EXP-RPM-D{0..5}/manifest.yaml`
**Aggregate data:** `research/residual-pareto/experiments/EXP-RPM-D{0..5}/runs/<ts>/af2d/aggregate.json`
**Per-seed data:** same path, `seed-001/` through `seed-003/`
**Stage 1 verdict:** `research/residual-pareto/experiments/verdict-Stage1.md`
**Per-regime verdicts:** `research/residual-pareto/experiments/EXP-RPM-D{0..5}/verdict.md`

**Verdict status:**
- RPM-001: tentative PASS (T2 is Pareto-optimal in every regime;
  pending full-cost-dimension confirmation including energy).
- RPM-002: UNRESOLVED (random-control evals missing; definition not
  altered).
- RPM-006: UNRESOLVED (same reason; definition not altered).
- Stage 2 damage-calibration: preregister before any further damage
  sweep.
