# Experiment Registry

One row per experiment. Decisions are immutable; corrections get new IDs.
See `EXP-TEMPLATE.yaml` for the record format and `../OPERATING-PLAN.md`
for the lifecycle. Decision labels per v2: `PASS` / `FAIL` / `INVALID` /
`CONTINUE` (an F grade is a useful result; `INVALID` must be rerun cleanly).

**Claims** (the program beliefs under test) live in
`../track-a-residual-ternary/residual-falsification/claims/` with template
`CLAIM-TEMPLATE.yaml`. Registered: A-RP-001, A-RP-002, A-RP-003 — all
`UNTESTED`. The A-F falsification suite (roadmap Phase 2) attacks them.

| ID | Track | Title | Status | Decision | Conclusion |
|---|---|---|---|---|---|
| EXP-A-000 | A | v0.16.0 overnight distillation runs (pre-regime) | DECIDED | INVALID | No preregistration; calibration + data + step-count changes mixed in one run. Engineering-validation evidence only. |
| EXP-A-000b | A | Provenance-contaminated concurrent session (interim verdict later corrected) | DECIDED | INVALID | Shared mutable `/tmp` artifacts across concurrent sessions; verdict unreliable. Motivates mandatory run namespaces (OPERATING-PLAN §3). |
| EXP-A-010 | A | Sequential T1→T2→T3 correction (KL 3.42→2.66→1.72→1.60) | retro-registered | CONTINUE | Plane 2 adds real capacity; plane 3 marginal. Diagnostic KL only — downstream acceptance untested. Must reproduce under clean provenance in Phase 1 (EXP-A-03x) before architectural action. |
| EXP-A-020 | A | Post-hoc Hadamard rotation + KD recovery (H-POST) | retro-registered | FAIL | NO-SHIP at tested budget: rotation admission penalty; KD saturates below stock model. Does not bear on H-NATIVE. |
| EXP-A-021 | A | Rotated-vs-unrotated controlled training signal (H-NATIVE motivation) | retro-registered | CONTINUE | Better optimization behavior/conditioning for the rotated arm. Motivates EXP-A-H1 (small-model native Hadamard, preregistered kill criteria). |
| EXP-A-001 | A | Baseline reproduction: FP16 / PTQ-uncalibrated / PTQ-calibrated on Legion | PROPOSED | — | Preregistered 2026-08-22 (`research/baselines/EXP-A-001/manifest.yaml`); gates G0→1 via CP0.3. |
