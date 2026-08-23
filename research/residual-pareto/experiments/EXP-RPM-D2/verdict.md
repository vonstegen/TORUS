# EXP-RPM-D{n} — Per-Regime Verdict

This is one of six per-regime verdicts (D0..D5) from Stage 1 of the
RPM program (2026-08-23). The full Stage 1 verdict is at
`../verdict-Stage1.md`. This file records only this regime's
measurements and per-regime RPM-001 status.

## Per-regime measurements (Stage 1 as executed)

| Metric | Value |
|---|---|
| Damage knob (frozen at preregistration) | (see master verdict) |
| Observed pre-train ppl | (see master verdict) |
| Observed pre-train arc_easy | (see master verdict) |
| Trained t2_ternary ppl | (see master verdict) |
| Trained t2_ternary arc_easy | (see master verdict) |
| Trained t2_ternary lambada_openai | (see master verdict) |
| Matched-bytes tolerance (all trained arms) | within ±1% |
| Recovery ratio (post-train ppl / pre-train ppl) | (see master verdict) |

(See `runs/<ts>/af2d/aggregate.json` for the actual values.)

## RPM-001 verdict for this regime

T2_ternary is on the Pareto frontier vs the complete frozen
comparator set (5 trained arms: T2, int4, int8, lora, dense) and
the complete registered cost dimensions available at Stage 1
(B, F, O, M, L — E is null and excluded).

This is the tentative Pareto verdict at the per-regime level.
The Stage 1 master verdict notes that the energy_per_token
dimension (E) was not measured at Stage 1; the verdict becomes
CONFIRMED only when E is added.

## RPM-002 / RPM-006 status

**UNRESOLVED** for this regime (and all Stage 1 regimes).
The driver skips eval on `untrained_controls` arms, so
`random_t2_ternary` and `random_lora` have empty `tasks` dicts in
the aggregate.json. The trained-vs-random z-score cannot be
computed. RPM-002 claim definition is NOT altered.

See Stage 1 master verdict for the data-gap diagnosis and the
recommended Stage 1.5 calibration experiment.

## Verdict status

| Claim | Status | Reason |
|---|---|---|
| RPM-001 | tentative PASS at this regime | T2 is Pareto-optimal on the joint (3 cap × 5 cost) vector; pending E measurement |
| RPM-002 | UNRESOLVED | random-control evals missing; claim definition unaltered |
| RPM-006 | UNRESOLVED | random-control evals missing; claim definition unaltered |

## Linkage

- **Stage 1 master verdict:** `../verdict-Stage1.md`
- **Manifest:** `./manifest.yaml`
- **Aggregate:** `./runs/<ts>/af2d/aggregate.json`
- **Driver SHA at run:** `692e8ee` (commit prior to the no_correction
  fix is here; subsequent driver changes are NOT retro-applied to
  this verdict per the user directive "preserve Stage 1 exactly as
  executed").