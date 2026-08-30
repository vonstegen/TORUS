# Track B Unlock Reassessment — 2026-08-28

**Authority:** `research/OPERATING-PLAN.md` §5 (unlock rules, v2.3 +
2026-08-25 framework restructure) and `research/ROADMAP.md` Phase 4.
**Trigger:** user steering 2026-08-28, after EXP-AF-004/004-R
(A-RP-003 CONFIRMED_FAIL), EXP-AF-003 (ROBUST), and EXP-AF-006b
(general effect) completed the A-F suite's open items.
**Verdict:** **Track B stays LOCKED.** Two of four B1 conditions are
unmet; today's results strengthen site-local validity but do not move
either blocking condition.

## B1 unlock conditions vs current evidence

| # | Condition (§5 / Phase 4) | State | Evidence |
|---|--------------------------|-------|----------|
| 1 | A-RP-002 PROVISIONAL_PASS or above on the equal-storage tournament | **MET** | A-RP-002 CONFIRMED_PASS (EXP-AF-002 + AF2-R; AF2-D PASS+; RPM-000). Robustness annotations added today: init-robust (AF3), seed-robust (AF3, AF6b reference), context/corpus-robust (AF6b) — all at the AF2-D site/recipe. |
| 2 | A-RP-LRN at least provisionally supported; operating band characterized for the site/damage in question | **MET (scoped)** | A-RP-LRN CONFIRMED at the AF2-D TWN band {0.6–1.0} (Stage 2 v6 + v7, two independent seed sets). Scope: exactly one site (L0.mlp.down_proj) × one damage mechanism (TWN). |
| 3 | AF5 task-relevant T2 value above its preregistered threshold | **NOT MET (blocking)** | EXP-RPM-T01: AF5 FAIL — T2 vs random T2 ≥+1σ on **0 of 4** held-out tasks (hellaswag −0.02σ, winogrande +0.18σ, boolq +0.09σ, openbookqa +0.29σ). The damaged base was already near FP16 on those tasks, so the correction had nothing to recover. Unchanged by today's results: AF6b demonstrated recovery generalization across windows/corpora on the *training-aligned* ladder (wikitext/arc/lambada), not on held-out commonsense/QA tasks. |
| 4 | ≥2 layer categories with Pareto-qualifying trained-vs-random separation | **NOT MET (blocking)** | Only the AF2-D site qualifies. Stage 2 v1: TWN damage degenerate at L8/L15 down_proj. Stage 2 v2–v4: L0-v and L15-Gaussian tournaments NOT QUALIFYING (trained ≈ random; at L15 σ=1.0 trained *loses* to random on 2 of 3 metrics). Stage 3 v2: catastrophic damage is bounded to {TWN, Gaussian} at AF2-D only. |

Historical-condition status for completeness: A-RP-001 is
CONFIRMED_FAIL (EXP-AF-001/001-R) — per §5 v2.3 this closes only the
equal-training-time branch and does not by itself block Track B.
A-RP-003 is CONFIRMED_FAIL (EXP-AF-004/004-R) — removes the
sequential-curriculum assumption from any future B-track design
(joint training is the evidenced recipe at the AF site).

## What today's evidence changes — and what it does not

- **Site-local validity is now unusually strong.** The AF2-D
  recovery phenomenon survives init-σ (including σ=0), fresh seeds,
  context windows {16, 128, 256}, and a corpus switch to
  openwebtext, with a third independent reproduction of the
  reference cell in AF6b. A-RP-002's CONFIRMED_PASS is not fragile
  along any of those axes at its validated site.
- **The two blockers are untouched.** Neither AF3 nor AF6b tested
  held-out tasks (condition 3) or additional sites (condition 4).
  Robustness at one site is not utility across the model, and the
  T01 failure mode (nothing-to-recover on undamaged tasks) is a
  *regime* problem, not a robustness problem.
- **AF6b's step-budget gradient is directly relevant to B1 design
  when it unlocks:** at matched tokens, recovery quality scales with
  optimizer steps (4000/500/250 → 15.1/19.0/26.7 ppl). Any future
  oracle-gating savings claim must price the correction's training
  budget, not just its inference cost.

## Cheapest paths to unblock (for the next scheduling decision)

1. **Condition 3 (AF5):** rerun held-out-task evaluation at a regime
   where the base actually loses capability on those tasks — T01's
   own diagnosis. Requires a damage regime that moves
   commonsense/QA performance (candidate: deeper TWN severity or a
   different site), then the preregistered AF5 threshold applies.
   Estimated cost: small (eval-only once a qualifying damaged base
   exists).
2. **Condition 4 (≥2 layer categories):** find a second
   CAL-qualifying site where trained T2 separates from random T2 on
   the Pareto criterion. The Stage 2/3 evidence bounds the search:
   catastrophic damage currently works only at AF2-D/L0 down_proj
   via TWN; attention v_proj at L0 was informative under Gaussian
   (CAL-qualifying) but the tournament was not. A focused
   site-discovery sweep (CAL-first, no tournaments until a site
   qualifies) is the cheapest attack.

## Decision output

- Hypothesis: "Track B (oracle gating) can be unlocked after the A-F
  suite completes." **Not supported under §5.**
- Decision: **Track B stays LOCKED** (conditions 3 and 4 unmet).
- Next permitted experiments: the two unblock paths above
  (held-out-task-damaging regime for AF5; CAL-first second-site
  discovery for condition 4), or non-Track-B work (native-Hadamard
  small-model experiment per the steering order; step-budget
  characterization).
- Experiments explicitly blocked by this reassessment: EXP-B-01x
  (oracle gating), EXP-B-02x (learned gating), EXP-B-03x (OLMoE
  adaptive precision) — all remain locked until conditions 3 and 4
  are met.

## Addendum 2026-08-30 — condition 4: second-site discovery returned NO_SECOND_SITE

EXP-RPM-SITE-DISCOVERY (run `runs/r/EXP-RPM-SITE-DISCOVERY/
20260830T175339Z`, 239/239 cells, refs bit-exact, 13/13 sites valid,
verdict `research/residual-pareto/experiments/EXP-RPM-SITE-DISCOVERY/
verdict.md`) swept the evidence-bounded grid — Gaussian ×
v_proj@{1,2,4,8,12,15} + down_proj@{1,4,8,12}, TWN × v_proj@{0,4,15}
— under the frozen candidate rule (CAL-qualifying AND damaged-base
ppl ≥ 100). Result: **0 candidates.** Nine sites qualify with
informative-but-mild damage axes (max ppl 15.3–74.0); four do not
qualify. Catastrophic damage is pinned to layer 0 (L0-down TWN,
L0-v Gaussian σ≥0.5); the untested TWN×attention pairing is
informative but never catastrophic (peak 74.0 at thr=0.9, L0).

Consequences: A-RP-002 annotated site-local within the searched
space (annotation only; CONFIRMED_PASS unchanged); RPM-006's
layer-category criterion remains unmet; **condition 4 stays
blocked** and no second-site tournament is permitted (0 candidates,
no grid expansion — exclusions frozen). Condition 3 (AF5) is now
the only unblock path with a registered attack: a damage regime
that actually moves held-out-task capability, per T01's own
diagnosis, then the preregistered AF5 threshold applies.
