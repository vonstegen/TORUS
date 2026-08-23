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
| EXP-AF-001-R | A | AF1-R clean reproduction under AF8 governance (n=3) | DECIDED | REPRODUCED | **A-RP-001 -> CONFIRMED_FAIL.** Every per-seed value byte-identical to EXP-AF-001 (commit `39be76c`); arm means ± stderrs identical on every metric; (B-A) stderr-of-difference still +9.09 / -2.23 / -6.24 (identical to AF1). AF8 governance: new namespace, frozen code revision, fresh process on legion, independently generated wikitext-103 token cache (SHA-fingerprinted; identity is the expected outcome of a clean reproduction, not a violation). Equal-training-time branch closes permanently. A-RP-002 (storage tournament for the v2.3 decision axis) and A-RP-003 (sequential vs joint) remain UNTESTED. Track B stays locked under §5 v2.3 prerequisite rewrite. |
| EXP-A-03x | A | A3 sequential functional correction (discovery): T1+T2 on worst-tolerance layer | DECIDED | PASS | Run 1 INVALID (autograd STE zero-gradient, fixed eac2c04). Run 2 clean: trained T2 on model.layers.0.mlp.down_proj recovers wikitext ppl 427.71→41.44 (10.3×, bar ≤200) and arc_easy 0.5396→0.6313 (bar ≥0.55; above FP16 0.6073 on bounded subset). KL 1.51→0.07 monotone. CP1.2 PASS → G1→2 OPEN; A-RP-001/002/003 move to TESTING. AF1 equal-budget control unblocked. |
| EXP-AF-001 | A | AF1 equal-training-budget control (T1-continued vs T1+T2, n=3) | DECIDED | FAIL | **A-RP-001 PROVISIONAL_FAIL under matched CE.** T1+T2 vs T1-continued, identical batches / optimizer / objective / seed-001,002,003, git `39be76c`. Wikitext ppl 34.81 vs 14.10 (+9.09 stderr in favor of A), arc_easy 0.635 vs 0.649 (-2.23 stderr), lambada 0.565 vs 0.618 (-6.24 stderr). A wins on all 3 capability metrics by >2 stderr. Lifecycle per OPERATING-PLAN §11 v2.3: this confirmation-tier ≥3-seed result promotes A-RP-001 `TESTING → PROVISIONAL_FAIL / REPRODUCTION_REQUIRED` in one transition. Track B stays locked. **Required next:** `EXP-AF-001-R` (clean reproduction, AF8-style) before A-RP-001 → CONFIRMED_FAIL; the equal-storage tournament `EXP-AF-002` (A-RP-002) is the now-central Track-A falsifier per OPERATING-PLAN §11 v2.3 decision-axis revision. |

## Decision log updates
- **2026-08-22 — A-RP-001 → CONFIRMED_FAIL** (via `EXP-AF-001-R` clean reproduction, run `research/track-a-residual-ternary/residual-falsification/experiments/AF1-R/runs/20260822T233000Z`; git_sha `4238568` reproducing frozen revision `39be76c`).
  - Trigger: every per-seed value byte-identical to AF1; arm means ± stderrs identical; (B-A) stderr-of-difference unchanged at +9.09 / -2.23 / -6.24 in favor of A.
  - Effect: equal-training-time branch closes permanently; A-RP-002 (storage tournament for the v2.3 decision axis) and A-RP-003 (sequential vs joint) remain UNTESTED.
  - Track B: still locked. Prerequisite rewrite under OPERATING-PLAN §5 v2.3 substitutes A-RP-002 PROVISIONAL_PASS for the historical A-RP-001 prerequisite; AF5 task-relevant T2 above threshold; AF8-clean CONFIRMED on at least one of A-RP-002/003.
  - Next required experiments under the freeze: **EXP-AF-002** (storage tournament for A-RP-002; central Track-A falsifier per OPERATING-PLAN §11 v2.3); **EXP-AF-001-R audit-script correction** is already committed (commit `4238568`).
