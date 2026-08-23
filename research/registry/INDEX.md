# Experiment Registry

One row per experiment. Decisions are immutable; corrections get new IDs.
See `EXP-TEMPLATE.yaml` for the record format and `../OPERATING-PLAN.md`
for the lifecycle. Decision labels per v2: `PASS` / `FAIL` / `INVALID` /
`CONTINUE` (an F grade is a useful result; `INVALID` must be rerun cleanly).

**Claims** (the program beliefs under test) live in
`../track-a-residual-ternary/residual-falsification/claims/` with template
`CLAIM-TEMPLATE.yaml`. Registered: A-RP-001, A-RP-002, A-RP-003.
Current states (rev 2.3):
A-RP-001 = `PROVISIONAL_FAIL / REPRODUCTION_REQUIRED` (after EXP-AF-001, 2026-08-22);
A-RP-002 / A-RP-003 = `UNTESTED` (pending EXP-AF-002 / EXP-AF-004 respectively).
The A-F falsification suite (roadmap Phase 2) attacks them.

| ID | Track | Title | Status | Decision | Conclusion |
|---|---|---|---|---|---|
| EXP-A-000 | A | v0.16.0 overnight distillation runs (pre-regime) | DECIDED | INVALID | No preregistration; calibration + data + step-count changes mixed in one run. Engineering-validation evidence only. |
| EXP-A-000b | A | Provenance-contaminated concurrent session (interim verdict later corrected) | DECIDED | INVALID | Shared mutable `/tmp` artifacts across concurrent sessions; verdict unreliable. Motivates mandatory run namespaces (OPERATING-PLAN §3). |
| EXP-A-010 | A | Sequential T1→T2→T3 correction (KL 3.42→2.66→1.72→1.60) | retro-registered | CONTINUE | Plane 2 adds real capacity; plane 3 marginal. Diagnostic KL only — downstream acceptance untested. Must reproduce under clean provenance in Phase 1 (EXP-A-03x) before architectural action. |
| EXP-A-020 | A | Post-hoc Hadamard rotation + KD recovery (H-POST) | retro-registered | FAIL | NO-SHIP at tested budget: rotation admission penalty; KD saturates below stock model. Does not bear on H-NATIVE. |
| EXP-A-021 | A | Rotated-vs-unrotated controlled training signal (H-NATIVE motivation) | retro-registered | CONTINUE | Better optimization behavior/conditioning for the rotated arm. Motivates EXP-A-H1 (small-model native Hadamard, preregistered kill criteria). |
| EXP-A-011 | A | A1 layer sensitivity (single-layer PTQ on each Linear in OLMo-1B) | DECIDED | CONTINUE | 114/114 arms complete; FP16 reproduces to 4 dec. Per-layer wikitext ppl range 13.1 .. 9277.6 (708× spread). Early mlp_down (layers 0-1) and early attention K/Q are 100-1000× more sensitive than late layers. Mismatch with A-RP-001 wording (full-layer vs correction plane). Motivates EXP-A-011.b (paired layers) and B1 per-layer-precision oracle. |
| EXP-A-03x | A | A3 sequential functional correction (discovery): T1+T2 on worst-tolerance layer | DECIDED | PASS | Run 1 INVALID (autograd STE zero-gradient, fixed eac2c04). Run 2 clean: trained T2 on model.layers.0.mlp.down_proj recovers wikitext ppl 427.71→41.44 (10.3×, bar ≤200) and arc_easy 0.5396→0.6313 (bar ≥0.55; above FP16 0.6073 on bounded subset). KL 1.51→0.07 monotone. CP1.2 PASS → G1→2 OPEN; A-RP-001/002/003 move to TESTING. AF1 equal-budget control unblocked. |
| EXP-AF-001 | A | AF1 equal-training-budget control (T1-continued vs T1+T2, n=3) | DECIDED | FAIL | **A-RP-001 PROVISIONAL_FAIL under matched CE.** T1+T2 vs T1-continued, identical batches / optimizer / objective / seed-001,002,003, git `39be76c`. Wikitext ppl 34.81 vs 14.10 (+9.09 stderr in favor of A), arc_easy 0.635 vs 0.649 (-2.23 stderr), lambada 0.565 vs 0.618 (-6.24 stderr). A wins on all 3 capability metrics by >2 stderr. Lifecycle per OPERATING-PLAN §11 v2.3: this confirmation-tier ≥3-seed result promotes A-RP-001 `TESTING → PROVISIONAL_FAIL / REPRODUCTION_REQUIRED` in one transition. Track B stays locked. **Required next:** `EXP-AF-001-R` (clean reproduction, AF8-style) before A-RP-001 → CONFIRMED_FAIL; the equal-storage tournament `EXP-AF-002` (A-RP-002) is the now-central Track-A falsifier per OPERATING-PLAN §11 v2.3 decision-axis revision. |

## Decision log updates
- **2026-08-22 — A-RP-001 → PROVISIONAL_FAIL / REPRODUCTION_REQUIRED** (via `EXP-AF-001`, run `research/track-a-residual-ternary/residual-falsification/experiments/AF1/runs/20260822T234553Z`; git_sha `39be76c`).
  - Trigger: T1+T2 loses to T1-continued at matched CE on every capability metric (n=3 seeds).
  - Effect on other claims: A-RP-002 / A-RP-003 remain UNTESTED; `EXP-AF-002` (equal-storage tournament for A-RP-002) is now the **central** Track-A falsifier per OPERATING-PLAN §11 v2.3.
  - Effect on Track B unlock: B1 prerequisite wording is rewritten under v2.3 to allow A-RP-002 PROVISIONAL_PASS to substitute for the now-closing A-RP-001 path; Track B stays locked in either case.
  - Effect on architecture: the Track A primary decision axis is now storage/compute/energy Pareto efficiency, not matched-training-time capacity (OPERATING-PLAN §11 v2.3, ROADMAP rev 2.3).
- **2026-08-22 — OPERATING-PLAN rev 2.3, ROADMAP rev 2.3** (same commit). Reasons: (a) AF1 evidence showed the v2 claim wording chose the wrong cost axis; (b) AF1 jumped straight from `TESTING → DECIDED FAIL` without the required `REPRODUCTION_REQUIRED → CONFIRMED_FAIL` lifecycle, so the v2.3 lifecycle wording is explicit that a confirmation-tier ≥3-seed result can promote `TESTING → PROVISIONAL_FAIL` in one step, with `REPRODUCTION_REQUIRED` set at the same transition. `EXP-AF-001-R` is inserted as Roadmap §2.10 (required before A-RP-001 → CONFIRMED_FAIL).
